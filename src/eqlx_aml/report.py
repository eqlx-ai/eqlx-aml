"""Run several detectors at once, and look at single accounts.

:func:`scan` runs the chosen patterns over a table and returns a :class:`Report`. :func:`account_profile`
summarises one account; :func:`trace` follows money forward or backward from one account, respecting
time (money can only move on after it has arrived).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .data import Transactions, format_amount, format_timestamp, parse_timestamp
from .detect import DAY, DEFAULT_PATTERNS, DETECTORS, Alert
from .errors import InputError
from .graph import Graph

ORDER = {name: i for i, name in enumerate(DETECTORS)}


@dataclass
class Report:
    tx: Transactions
    alerts: list[Alert]
    patterns: tuple[str, ...]
    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        c = dict.fromkeys(self.patterns, 0)
        for a in self.alerts:
            c[a.pattern] = c.get(a.pattern, 0) + 1
        return c

    def summary(self) -> dict[str, Any]:
        involved = {x for a in self.alerts for x in a.accounts}
        flagged = {a.account for a in self.alerts}
        return {
            **self.tx.summary(),
            "patterns": list(self.patterns),
            "params": self.params,
            "alerts": self.counts(),
            "accounts_flagged": len(flagged),
            "accounts_involved": len(involved),
        }

    def to_dict(self, max_alerts: int | None = None, max_transactions: int | None = 50) -> dict[str, Any]:
        shown = self.alerts if max_alerts is None else self.alerts[:max_alerts]
        return {
            "summary": self.summary(),
            "alerts": [a.to_dict(self.tx, max_transactions) for a in shown],
            "alerts_truncated": len(shown) < len(self.alerts),
        }


def scan(
    tx: Transactions,
    patterns: Sequence[str] | None = None,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    graph: Graph | None = None,
) -> Report:
    """Run ``patterns`` (default: every pattern except structuring, which needs a threshold) and return
    the alerts sorted by pattern, then size (largest first), then time, then account.

    ``params`` maps a pattern to its keyword arguments, e.g. ``{"fan_out": {"min_counterparties": 8}}``.
    """
    chosen = tuple(patterns) if patterns else DEFAULT_PATTERNS
    unknown = [p for p in chosen if p not in DETECTORS]
    if unknown:
        raise InputError(f"unknown pattern(s) {unknown}; choose from {list(DETECTORS)}")
    params = dict(params or {})
    extra = set(params) - set(chosen)
    if extra:
        raise InputError(f"parameters given for pattern(s) not run: {sorted(extra)}")
    if "structuring" in chosen and "threshold" not in params.get("structuring", {}):
        raise InputError("structuring needs a threshold, e.g. params={'structuring': {'threshold': 10000}}")
    g = graph if graph is not None else Graph(tx)
    alerts: list[Alert] = []
    used: dict[str, dict[str, Any]] = {}
    for p in chosen:
        kw = dict(params.get(p, {}))
        try:
            alerts.extend(DETECTORS[p](g, **kw))
        except TypeError as e:
            raise InputError(f"{p}: {e}") from None
        used[p] = kw
    alerts.sort(key=lambda a: (ORDER[a.pattern], -a.size, a.start, a.account))
    return Report(tx, alerts, chosen, used)


# --- single accounts ----------------------------------------------------------------------------------


def account_profile(tx: Transactions, account: str, graph: Graph | None = None, top: int = 10) -> dict[str, Any]:
    """What one account did: transfers in and out, totals per currency, first and last activity, the
    counterparties it moved most money with, and the channels it used."""
    g = graph if graph is not None else Graph(tx)
    a = tx.account_id(account)
    out: dict[str, Any] = {"account": account}
    for label, lst, lo, hi, other in (
        ("outgoing", g.out, g.out_ptr[a], g.out_ptr[a + 1], g.dst),
        ("incoming", g.inn, g.in_ptr[a], g.in_ptr[a + 1], g.src),
    ):
        edges = lst[lo:hi]
        totals: dict[str, int] = {}
        per_cp: dict[tuple[int, int], list[int]] = {}
        chans: dict[str, int] = {}
        for e in edges:
            cur = int(tx.currency[e])
            amt = int(tx.amount[e])
            totals[tx.currencies[cur]] = totals.get(tx.currencies[cur], 0) + amt
            rec = per_cp.setdefault((other[e], cur), [0, 0])
            rec[0] += 1
            rec[1] += amt
            ch = tx.channels[int(tx.channel[e])]
            chans[ch] = chans.get(ch, 0) + 1
        best = sorted(per_cp.items(), key=lambda kv: (-kv[1][1], tx.accounts[kv[0][0]]))[:top]
        out[label] = {
            "transfers": len(edges),
            "counterparties": len({k[0] for k in per_cp}),
            "total": {k: format_amount(v) for k, v in sorted(totals.items())},
            "first": format_timestamp(g.ts[edges[0]]) if edges else None,
            "last": format_timestamp(g.ts[edges[-1]]) if edges else None,
            "top_counterparties": [
                {
                    "account": tx.accounts[cp],
                    "currency": tx.currencies[cur],
                    "transfers": n,
                    "total": format_amount(s),
                }
                for (cp, cur), (n, s) in best
            ],
            "channels": dict(sorted(chans.items())) if tx.channels != ("",) else {},
        }
    selfs = int(((tx.src == a) & (tx.dst == a)).sum())
    out["self_transfers"] = selfs
    return out


def trace(
    tx: Transactions,
    account: str,
    direction: str = "forward",
    max_hops: int = 3,
    window_days: float = 7,
    start: str | None = None,
    max_transfers: int = 500,
    graph: Graph | None = None,
) -> dict[str, Any]:
    """Follow the money from (``forward``) or to (``backward``) one account.

    Forward: start at ``start`` (or the account's first outgoing transfer); a transfer from an account
    is followed only if it happens after the money reached that account, and within ``window_days``
    of the start. Backward is the mirror image, ending at ``start`` (or the last incoming transfer).
    Each reached account is listed once, with the fewest hops needed and the time of the transfer that
    first reached it at that hop count. ``transfers`` are those transfers (a tree, not every path).
    """
    if direction not in ("forward", "backward"):
        raise InputError("direction must be 'forward' or 'backward'")
    if not 1 <= max_hops <= 10:
        raise InputError("max_hops must be between 1 and 10")
    g = graph if graph is not None else Graph(tx)
    a = tx.account_id(account)
    w = round(window_days * DAY)
    fwd = direction == "forward"
    if start is not None:
        t_start = parse_timestamp(start)
    else:
        lst = g.out_ts[g.out_ptr[a] : g.out_ptr[a + 1]] if fwd else g.in_ts[g.in_ptr[a] : g.in_ptr[a + 1]]
        if not lst:
            return {"account": account, "direction": direction, "reached": [], "transfers": [], "truncated": False}
        t_start = lst[0] if fwd else lst[-1]
    t_lim = t_start + w if fwd else t_start - w
    best: dict[int, tuple[int, int]] = {a: (0, t_start)}  # account -> (hops, time)
    used: list[int] = []
    frontier = [a]
    truncated = False
    for hop in range(1, max_hops + 1):
        nxt: list[int] = []
        for x in frontier:
            tx_time = best[x][1]
            if fwd:
                lo, hi = g.out_range(x, tx_time, t_lim)
                edges, other = g.out[lo:hi], g.dst
            else:
                lo, hi = g.in_range(x, t_lim, tx_time)
                edges, other = list(reversed(g.inn[lo:hi])), g.src
            for e in edges:
                y = other[e]
                t = g.ts[e]
                if y not in best:
                    best[y] = (hop, t)
                    nxt.append(y)
                    used.append(e)
                    if len(used) >= max_transfers:
                        truncated = True
                        break
            if truncated:
                break
        frontier = nxt
        if truncated or not frontier:
            break
    reached = sorted(
        ((tx.accounts[y], h, t) for y, (h, t) in best.items() if y != a),
        key=lambda r: (r[1], r[2] if fwd else -r[2], r[0]),
    )
    return {
        "account": account,
        "direction": direction,
        "start": format_timestamp(t_start),
        "window_end" if fwd else "window_start": format_timestamp(t_lim),
        "reached": [{"account": n, "hops": h, "time": format_timestamp(t)} for n, h, t in reached],
        "transfers": [tx.record(e) for e in sorted(used, key=lambda e: (g.ts[e], int(tx.row[e])))],
        "truncated": truncated,
    }
