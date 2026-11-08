"""Transactions: a compact, column-oriented table of money transfers.

A transfer has a time, a sending and a receiving account, an amount, and optionally a currency and a
channel (payment format such as ACH, Wire, Cash). Rows keep their original position in the input
(``row``, 1 = first data row), so every finding points back to lines the user can open.

Amounts are stored as integers in millionths (micro-units): ``"2848.96"`` becomes ``2848960000``. No
floating point is involved, so sums are exact and the same input always gives the same output.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .errors import InputError

SCALE = 1_000_000  # amounts are kept in millionths
_SCALE_DIGITS = 6
_MAX_AMOUNT = (2**63 - 1) // SCALE

I64 = npt.NDArray[np.int64]
I32 = npt.NDArray[np.int32]

# Column names understood by the generic CSV reader (case-insensitive), first match wins.
GENERIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "date", "datetime", "booking_date", "value_date"),
    "source": ("source", "from", "sender", "from_account", "debtor", "payer", "orig"),
    "target": ("target", "to", "receiver", "to_account", "creditor", "payee", "beneficiary", "dest"),
    "amount": ("amount", "value", "amt"),
    "currency": ("currency", "ccy", "cur"),
    "channel": ("channel", "type", "payment_format", "payment_type", "method"),
    "id": ("id", "transaction_id", "tx_id", "reference"),
}
REQUIRED = ("timestamp", "source", "target", "amount")

IBM_HEADER = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
)


def parse_amount(text: str) -> int:
    """Parse a decimal amount into micro-units. Rounds half-to-even beyond six decimals."""
    s = text.strip().replace(",", "") if text.count(",") and text.count(".") else text.strip()
    whole, dot, frac = s.partition(".")
    if whole.isdigit() and (not dot or (frac.isdigit() and len(frac) <= _SCALE_DIGITS)):
        value = int(whole) * SCALE + (int(frac.ljust(_SCALE_DIGITS, "0")) if frac else 0)
    else:
        try:
            d = Decimal(s)
        except InvalidOperation:
            raise InputError(f"not an amount: {text!r}") from None
        if not d.is_finite():
            raise InputError(f"not an amount: {text!r}")
        value = int((d * SCALE).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
    if value < 0:
        raise InputError(f"negative amount {text!r}: give the direction with source and target")
    if value > _MAX_AMOUNT * SCALE:
        raise InputError(f"amount too large: {text!r}")
    return value


def format_amount(micro: int) -> str:
    """Micro-units back to a decimal string with at least two decimals: 2848960000 -> '2848.96'."""
    whole, frac = divmod(int(micro), SCALE)
    f = f"{frac:06d}".rstrip("0")
    return f"{whole}.{f.ljust(2, '0')}"


def parse_timestamp(text: str) -> int:
    """ISO-8601 date or date-time (``/`` or ``-`` separated) to Unix seconds. Naive times are read as UTC."""
    s = text.strip().replace("/", "-")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise InputError(f"not a date or time: {text!r}") from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def format_timestamp(seconds: int) -> str:
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class _Interner:
    names: list[str] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)

    def get(self, name: str) -> int:
        i = self.index.get(name)
        if i is None:
            i = self.index[name] = len(self.names)
            self.names.append(name)
        return i


class Transactions:
    """A table of transfers. Build it with :func:`read_csv` or :meth:`from_records`.

    Columns are NumPy arrays of equal length: ``ts`` (Unix seconds), ``src`` and ``dst`` (account
    numbers into ``accounts``), ``amount`` (micro-units), ``currency`` and ``channel`` (numbers into
    ``currencies`` and ``channels``; 0 is the empty name when the input has none), ``row`` (position in
    the input, 1-based) and optional ``ids`` (the input's own transaction ids).
    """

    def __init__(
        self,
        ts: I64,
        src: I32,
        dst: I32,
        amount: I64,
        currency: I32,
        channel: I32,
        row: I64,
        accounts: Sequence[str],
        currencies: Sequence[str] = ("",),
        channels: Sequence[str] = ("",),
        ids: Sequence[str] | None = None,
        source: str = "",
    ) -> None:
        n = len(ts)
        for name, col in (("src", src), ("dst", dst), ("amount", amount), ("row", row)):
            if len(col) != n:
                raise ValueError(f"column {name} has {len(col)} values, expected {n}")
        self.ts, self.src, self.dst, self.amount = ts, src, dst, amount
        self.currency, self.channel, self.row = currency, channel, row
        self.accounts: tuple[str, ...] = tuple(accounts)
        self.currencies: tuple[str, ...] = tuple(currencies)
        self.channels: tuple[str, ...] = tuple(channels)
        self.ids: tuple[str, ...] | None = tuple(ids) if ids is not None else None
        self.source = source
        self._account_index: dict[str, int] | None = None

    def __len__(self) -> int:
        return len(self.ts)

    def __repr__(self) -> str:
        return f"<Transactions {len(self)} transfers, {len(self.accounts)} accounts, source={self.source!r}>"

    # --- construction ---------------------------------------------------------------------------

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]], source: str = "") -> Transactions:
        """Build from dicts with ``timestamp``, ``source``, ``target``, ``amount`` and optional
        ``currency``, ``channel``, ``id``. Timestamps may be strings, ``datetime`` or Unix seconds."""
        b = _Builder(source)
        for i, r in enumerate(records, start=1):
            try:
                t = r["timestamp"]
                ts = (
                    int(t.timestamp() if t.tzinfo else t.replace(tzinfo=timezone.utc).timestamp())
                    if isinstance(t, datetime)
                    else int(t)
                    if isinstance(t, int)
                    else parse_timestamp(str(t))
                )
                amt = r["amount"]
                b.add(
                    ts,
                    str(r["source"]),
                    str(r["target"]),
                    parse_amount(str(amt)),
                    str(r.get("currency") or ""),
                    str(r.get("channel") or ""),
                    i,
                    None if r.get("id") is None else str(r["id"]),
                )
            except KeyError as e:
                raise InputError(f"record {i}: missing {e.args[0]!r}") from None
            except InputError as e:
                raise InputError(f"record {i}: {e}") from None
        return b.build()

    # --- lookups and views ----------------------------------------------------------------------

    def account_id(self, name: str) -> int:
        if self._account_index is None:
            self._account_index = {a: i for i, a in enumerate(self.accounts)}
        try:
            return self._account_index[name]
        except KeyError:
            raise InputError(f"unknown account {name!r}") from None

    def record(self, i: int) -> dict[str, Any]:
        """One transfer as plain data (the form findings and tools return)."""
        out: dict[str, Any] = {
            "row": int(self.row[i]),
            "time": format_timestamp(int(self.ts[i])),
            "source": self.accounts[int(self.src[i])],
            "target": self.accounts[int(self.dst[i])],
            "amount": format_amount(int(self.amount[i])),
        }
        if self.currencies != ("",):
            out["currency"] = self.currencies[int(self.currency[i])]
        if self.channels != ("",):
            out["channel"] = self.channels[int(self.channel[i])]
        if self.ids is not None:
            out["id"] = self.ids[i]
        return out

    def select(
        self,
        *,
        channels: Sequence[str] | None = None,
        currencies: Sequence[str] | None = None,
        start: int | None = None,
        end: int | None = None,
        min_amount: int = 0,
    ) -> Transactions:
        """A filtered copy: only these channels/currencies, ``start <= ts < end``, amount >= min_amount
        (micro-units). Account, currency and channel numbering stay the same; rows keep their numbers."""
        keep = np.ones(len(self), dtype=bool)
        if channels is not None:
            keep &= np.isin(self.channel, _codes(self.channels, channels))
        if currencies is not None:
            keep &= np.isin(self.currency, _codes(self.currencies, currencies))
        if start is not None:
            keep &= self.ts >= start
        if end is not None:
            keep &= self.ts < end
        if min_amount:
            keep &= self.amount >= min_amount
        idx = np.flatnonzero(keep)
        return Transactions(
            self.ts[idx],
            self.src[idx],
            self.dst[idx],
            self.amount[idx],
            self.currency[idx],
            self.channel[idx],
            self.row[idx],
            self.accounts,
            self.currencies,
            self.channels,
            None if self.ids is None else [self.ids[i] for i in idx.tolist()],
            self.source,
        )

    def summary(self) -> dict[str, Any]:
        n = len(self)
        out: dict[str, Any] = {"source": self.source, "transfers": n, "accounts": len(self.accounts)}
        if n:
            out["first"] = format_timestamp(int(self.ts.min()))
            out["last"] = format_timestamp(int(self.ts.max()))
            out["self_transfers"] = int(np.count_nonzero(self.src == self.dst))
        for label, names, col in (
            ("currencies", self.currencies, self.currency),
            ("channels", self.channels, self.channel),
        ):
            if names != ("",):
                counts = np.bincount(col, minlength=len(names))
                out[label] = {names[i]: int(c) for i, c in enumerate(counts.tolist()) if c}
        return out


def _codes(names: tuple[str, ...], wanted: Sequence[str]) -> list[int]:
    lookup = {n.casefold(): i for i, n in enumerate(names)}
    return [lookup[w.casefold()] for w in wanted if w.casefold() in lookup]


class _Builder:
    def __init__(self, source: str) -> None:
        self.source = source
        self.ts: list[int] = []
        self.src: list[int] = []
        self.dst: list[int] = []
        self.amount: list[int] = []
        self.currency: list[int] = []
        self.channel: list[int] = []
        self.row: list[int] = []
        self.ids: list[str] = []
        self.has_ids = False
        self.accounts = _Interner()
        self.currencies = _Interner()
        self.channels = _Interner()
        self.currencies.get("")
        self.channels.get("")

    def add(self, ts: int, src: str, dst: str, amount: int, cur: str, chan: str, row: int, tx_id: str | None) -> None:
        if not src or not dst:
            raise InputError("empty account")
        self.ts.append(ts)
        self.src.append(self.accounts.get(src))
        self.dst.append(self.accounts.get(dst))
        self.amount.append(amount)
        self.currency.append(self.currencies.get(cur))
        self.channel.append(self.channels.get(chan))
        self.row.append(row)
        if tx_id is not None:
            self.has_ids = True
        self.ids.append(tx_id or "")

    def build(self) -> Transactions:
        return Transactions(
            np.array(self.ts, dtype=np.int64),
            np.array(self.src, dtype=np.int32),
            np.array(self.dst, dtype=np.int32),
            np.array(self.amount, dtype=np.int64),
            np.array(self.currency, dtype=np.int32),
            np.array(self.channel, dtype=np.int32),
            np.array(self.row, dtype=np.int64),
            self.accounts.names,
            self.currencies.names,
            self.channels.names,
            self.ids if self.has_ids else None,
            self.source,
        )


# --- CSV ----------------------------------------------------------------------------------------


def is_ibm_header(header: Sequence[str]) -> bool:
    return tuple(h.strip() for h in header[: len(IBM_HEADER)]) == IBM_HEADER


def read_csv(
    path: str | Path | io.TextIOBase,
    *,
    columns: Mapping[str, str] | None = None,
    max_rows: int | None = None,
) -> Transactions:
    """Read transfers from a CSV file.

    Two layouts are recognised:

    * **generic**: a header with columns for time, source account, target account and amount, and
      optionally currency, channel and id. Common names are found automatically (see
      ``GENERIC_COLUMNS``); pass ``columns={"timestamp": "Booking Date", ...}`` to name them.
    * **IBM AML** (the public synthetic data set of Altman et al., 2023): detected from its header.
      Accounts become ``"<bank>:<account>"``; the amount and currency are the paid side.

    The file is read row by row; rows are numbered from 1 (the first line after the header).
    """
    if isinstance(path, (str, Path)):
        p = Path(path)
        with p.open(newline="", encoding="utf-8-sig") as f:
            return _read(f, p.name, columns, max_rows)
    return _read(path, "", columns, max_rows)


def _read(f: Iterable[str], source: str, columns: Mapping[str, str] | None, max_rows: int | None) -> Transactions:
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        raise InputError("empty file") from None
    b = _Builder(source)
    if columns is None and is_ibm_header(header):
        _read_ibm(reader, b, max_rows)
    else:
        _read_generic(reader, header, b, columns, max_rows)
    return b.build()


def _read_ibm(reader: Iterable[list[str]], b: _Builder, max_rows: int | None) -> None:
    ts_cache: dict[str, int] = {}
    for n, r in enumerate(reader, start=1):
        if max_rows is not None and n > max_rows:
            raise InputError(f"more than {max_rows} rows")
        if not r:
            continue
        try:
            t = ts_cache.get(r[0])
            if t is None:
                t = ts_cache[r[0]] = parse_timestamp(r[0])
            b.add(t, r[1] + ":" + r[2], r[3] + ":" + r[4], parse_amount(r[7]), r[8], r[9], n, None)
        except (IndexError, InputError) as e:
            raise InputError(f"row {n}: {e if isinstance(e, InputError) else 'too few columns'}") from None


def _read_generic(
    reader: Iterable[list[str]],
    header: list[str],
    b: _Builder,
    columns: Mapping[str, str] | None,
    max_rows: int | None,
) -> None:
    lower = [h.strip().casefold() for h in header]
    pos: dict[str, int] = {}
    if columns:
        unknown = set(columns) - set(GENERIC_COLUMNS)
        if unknown:
            raise InputError(f"unknown column role(s): {sorted(unknown)}")
        for role, name in columns.items():
            if name.strip().casefold() not in lower:
                raise InputError(f"column {name!r} not in header")
            pos[role] = lower.index(name.strip().casefold())
    for role, names in GENERIC_COLUMNS.items():
        if role not in pos:
            for nm in names:
                if nm in lower:
                    pos[role] = lower.index(nm)
                    break
    missing = [r for r in REQUIRED if r not in pos]
    if missing:
        raise InputError(f"cannot find column(s) {missing} in header {header}; name them with columns=")
    ts_cache: dict[str, int] = {}
    pt, ps, pd, pa = pos["timestamp"], pos["source"], pos["target"], pos["amount"]
    pc, pch, pid = pos.get("currency"), pos.get("channel"), pos.get("id")
    for n, r in enumerate(reader, start=1):
        if max_rows is not None and n > max_rows:
            raise InputError(f"more than {max_rows} rows")
        if not r or not any(c.strip() for c in r):
            continue
        try:
            t = ts_cache.get(r[pt])
            if t is None:
                t = parse_timestamp(r[pt])
                if len(ts_cache) < 1_000_000:
                    ts_cache[r[pt]] = t
            b.add(
                t,
                r[ps].strip(),
                r[pd].strip(),
                parse_amount(r[pa]),
                r[pc].strip() if pc is not None else "",
                r[pch].strip() if pch is not None else "",
                n,
                r[pid].strip() if pid is not None else None,
            )
        except (IndexError, InputError) as e:
            raise InputError(f"row {n}: {e if isinstance(e, InputError) else 'too few columns'}") from None
