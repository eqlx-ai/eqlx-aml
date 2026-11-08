"""Money-laundering pattern detectors.

Each detector takes a :class:`Graph` and explicit parameters and returns :class:`Alert` objects. There
is no model, no randomness and no learned threshold: the same transfers and parameters always give the
same alerts, in the same order. Each alert names the accounts, the time span and the exact transfers
(by input row) that make up the pattern, so a person or an agent can check it by hand.

The patterns follow the typologies described by FATF and used in the public IBM AML data set
(Altman et al., 2023): fan-out, fan-in, gather-scatter, scatter-gather and cycles; plus two classic
transaction-monitoring rules, pass-through (money that leaves about as fast and as much as it came) and
structuring (amounts kept just under a reporting threshold).

Windows are inclusive: two transfers are "within 7 days" when their times differ by at most 7 days.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .data import Transactions, format_amount, format_timestamp, parse_amount
from .errors import InputError
from .graph import Graph

DAY = 86_400


@dataclass(frozen=True)
class Alert:
    """One occurrence of a pattern.

    ``account`` is the account the alert is about (the hub of a fan, the source of a scatter-gather, the
    first sender of a cycle); ``accounts`` lists every account involved, ``account`` first.
    ``transactions`` are positions in the :class:`Transactions` table; ``to_dict`` turns them into rows.
    """

    pattern: str
    account: str
    accounts: tuple[str, ...]
    start: int
    end: int
    size: int
    transactions: tuple[int, ...]
    message: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, tx: Transactions, max_transactions: int | None = 50) -> dict[str, Any]:
        totals: dict[str, int] = {}
        for i in self.transactions:
            cur = tx.currencies[int(tx.currency[i])]
            totals[cur] = totals.get(cur, 0) + int(tx.amount[i])
        idx = sorted(self.transactions, key=lambda i: (int(tx.ts[i]), int(tx.row[i])))
        shown = idx if max_transactions is None else idx[:max_transactions]
        return {
            "pattern": self.pattern,
            "account": self.account,
            "accounts": list(self.accounts),
            "start": format_timestamp(self.start),
            "end": format_timestamp(self.end),
            "size": self.size,
            "message": self.message,
            "detail": dict(self.detail),
            "total_amount": {k or "": format_amount(v) for k, v in sorted(totals.items())},
            "transaction_count": len(self.transactions),
            "rows": sorted(int(tx.row[i]) for i in self.transactions),
            "transactions": [tx.record(i) for i in shown],
            "transactions_truncated": len(shown) < len(idx),
        }


def _days(seconds: int) -> str:
    d = seconds / DAY
    return f"{d:g} day" + ("" if d == 1 else "s")


def _window(window_days: float) -> int:
    if not window_days > 0:
        raise InputError("window_days must be positive")
    return round(window_days * DAY)


def _episodes(hits: Iterator[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Merge overlapping hit windows ``(lo, hi, score)`` (positions, ``lo`` non-decreasing) into
    episodes ``(lo, hi, best score)``."""
    out: list[tuple[int, int, int]] = []
    for lo, hi, score in hits:
        if out and lo <= out[-1][1]:
            plo, phi, pbest = out[-1]
            out[-1] = (plo, max(phi, hi), max(pbest, score))
        else:
            out.append((lo, hi, score))
    return out


# --- fan-out / fan-in ---------------------------------------------------------------------------------


def _fan_hits(
    times: Sequence[int],
    lst: Sequence[int],
    other: Sequence[int],
    lo: int,
    hi: int,
    k: int,
    w: int,
) -> Iterator[tuple[int, int, int]]:
    cnt: dict[int, int] = {}
    j = lo
    for i in range(lo, hi):
        c = other[lst[i]]
        cnt[c] = cnt.get(c, 0) + 1
        t = times[i]
        while t - times[j] > w:
            cj = other[lst[j]]
            v = cnt[cj] - 1
            if v:
                cnt[cj] = v
            else:
                del cnt[cj]
            j += 1
        if len(cnt) >= k:
            yield j, i, len(cnt)


def fan_out(g: Graph, min_counterparties: int = 5, window_days: float = 7) -> list[Alert]:
    """An account sends money to at least ``min_counterparties`` different accounts within the window."""
    return _fan(g, "fan_out", min_counterparties, _window(window_days))


def fan_in(g: Graph, min_counterparties: int = 5, window_days: float = 7) -> list[Alert]:
    """An account receives money from at least ``min_counterparties`` different accounts within the
    window."""
    return _fan(g, "fan_in", min_counterparties, _window(window_days))


def _fan(g: Graph, pattern: str, k: int, w: int) -> list[Alert]:
    if k < 2:
        raise InputError("min_counterparties must be at least 2")
    out = pattern == "fan_out"
    lst, times, ptr, other = (g.out, g.out_ts, g.out_ptr, g.dst) if out else (g.inn, g.in_ts, g.in_ptr, g.src)
    names = g.tx.accounts
    alerts = []
    for a in range(g.n_accounts):
        lo, hi = ptr[a], ptr[a + 1]
        if hi - lo < k:
            continue
        for elo, ehi, best in _episodes(_fan_hits(times, lst, other, lo, hi, k, w)):
            edges = lst[elo : ehi + 1]
            cps = list(dict.fromkeys(other[e] for e in edges))
            verb = f"sent money to {best}" if out else f"received money from {best}"
            alerts.append(
                Alert(
                    pattern,
                    names[a],
                    (names[a], *(names[c] for c in cps)),
                    times[elo],
                    times[ehi],
                    best,
                    tuple(edges),
                    f"{names[a]} {verb} different accounts within {_days(w)} "
                    f"({len(cps)} counterparties, {len(edges)} transfers in the episode)",
                    {"max_counterparties_in_window": best, "counterparties": len(cps), "transfers": len(edges)},
                )
            )
    return alerts


# --- gather-scatter -----------------------------------------------------------------------------------


def gather_scatter(g: Graph, min_senders: int = 3, min_receivers: int = 3, window_days: float = 7) -> list[Alert]:
    """One account both collects money from at least ``min_senders`` accounts and pays out to at least
    ``min_receivers`` accounts within the same window (a collection hub that redistributes)."""
    if min_senders < 1 or min_receivers < 1:
        raise InputError("min_senders and min_receivers must be at least 1")
    w = _window(window_days)
    names = g.tx.accounts
    alerts = []
    for a in range(g.n_accounts):
        if g.in_degree(a) < min_senders or g.out_degree(a) < min_receivers:
            continue
        ilo, ihi = g.in_ptr[a], g.in_ptr[a + 1]
        olo, ohi = g.out_ptr[a], g.out_ptr[a + 1]
        # (rank, time, is_out, counterparty, transfer), merged in time order
        ev = [(g.rank[e], g.in_ts[p], 0, g.src[e], e) for p, e in zip(range(ilo, ihi), g.inn[ilo:ihi], strict=True)]
        ev += [(g.rank[e], g.out_ts[p], 1, g.dst[e], e) for p, e in zip(range(olo, ohi), g.out[olo:ohi], strict=True)]
        ev.sort()
        cnt: tuple[dict[int, int], dict[int, int]] = ({}, {})
        j = 0
        hits = []
        for i, (_, t, side, c, _e) in enumerate(ev):
            d = cnt[side]
            d[c] = d.get(c, 0) + 1
            while t - ev[j][1] > w:
                _, _, sj, cj, _ = ev[j]
                dj = cnt[sj]
                v = dj[cj] - 1
                if v:
                    dj[cj] = v
                else:
                    del dj[cj]
                j += 1
            if len(cnt[0]) >= min_senders and len(cnt[1]) >= min_receivers:
                hits.append((j, i, min(len(cnt[0]), len(cnt[1]))))
        for elo, ehi, _best in _episodes(iter(hits)):
            part = ev[elo : ehi + 1]
            senders = list(dict.fromkeys(x[3] for x in part if x[2] == 0))
            receivers = list(dict.fromkeys(x[3] for x in part if x[2] == 1))
            alerts.append(
                Alert(
                    "gather_scatter",
                    names[a],
                    (names[a], *dict.fromkeys(names[c] for c in senders + receivers)),
                    part[0][1],
                    part[-1][1],
                    min(len(senders), len(receivers)),
                    tuple(x[4] for x in part),
                    f"{names[a]} received from {len(senders)} accounts and paid {len(receivers)} accounts "
                    f"within windows of {_days(w)}",
                    {"senders": len(senders), "receivers": len(receivers), "transfers": len(part)},
                )
            )
    return alerts


# --- hub test shared by scatter-gather and cycles -----------------------------------------------------


def _busy(g: Graph, x: int, t_from: int, t_to: int, cap: int) -> bool:
    lo, hi = g.out_range(x, t_from, t_to)
    if hi - lo > cap:
        return True
    lo, hi = g.in_range(x, t_from, t_to)
    return hi - lo > cap


# --- scatter-gather -----------------------------------------------------------------------------------


def scatter_gather(g: Graph, min_intermediaries: int = 3, window_days: float = 7, max_degree: int = 50) -> list[Alert]:
    """Money leaves one account through at least ``min_intermediaries`` different accounts and arrives
    at one other account, all within the window: A -> {M1, M2, ...} -> B, each Mi forwarding after it
    received.

    An intermediary with more than ``max_degree`` incoming or outgoing transfers in the window starting
    when it received the money is treated as a business or hub and not counted.
    """
    if min_intermediaries < 2:
        raise InputError("min_intermediaries must be at least 2")
    w = _window(window_days)
    names = g.tx.accounts
    alerts = []
    # Enumerate from the receiving end: incoming degrees are far smaller than outgoing ones in practice.
    for b in range(g.n_accounts):
        blo, bhi = g.in_ptr[b], g.in_ptr[b + 1]
        if bhi - blo < min_intermediaries:
            continue
        by_source: dict[int, list[tuple[int, int, int, int, int]]] = {}
        busy: dict[tuple[int, int], bool] = {}
        for p in range(blo, bhi):
            e2 = g.inn[p]
            m, t2 = g.src[e2], g.in_ts[p]
            for q in range(*g.in_range(m, t2 - w, t2)):
                e1 = g.inn[q]
                a, t1 = g.src[e1], g.in_ts[q]
                if a == b:
                    continue
                key = (m, t1)
                hub = busy.get(key)
                if hub is None:
                    hub = busy[key] = _busy(g, m, t1, t1 + w, max_degree)
                if hub:
                    continue
                by_source.setdefault(a, []).append((t1, t2, m, e1, e2))
        for a, paths in by_source.items():
            if len({x[2] for x in paths}) < min_intermediaries:
                continue
            paths.sort()
            hits = []
            for j, (start, *_rest) in enumerate(paths):
                sel = [x for x in paths[j:] if x[0] <= start + w and x[1] <= start + w]
                ms = {x[2] for x in sel}
                if len(ms) >= min_intermediaries:
                    hits.append((start, max(x[1] for x in sel), len(ms), sel))
            episodes: list[tuple[int, int, int, dict[tuple[int, int, int, int, int], None]]] = []
            for start, end, n, sel in hits:
                if episodes and start <= episodes[-1][1]:
                    s0, e0, n0, acc = episodes[-1]
                    acc.update(dict.fromkeys(sel))
                    episodes[-1] = (s0, max(e0, end), max(n0, n), acc)
                else:
                    episodes.append((start, end, n, dict.fromkeys(sel)))
            for start, end, n, acc in episodes:
                ps = list(acc)
                mids = list(dict.fromkeys(x[2] for x in ps))
                edges = tuple(dict.fromkeys(e for x in ps for e in (x[3], x[4])))
                alerts.append(
                    Alert(
                        "scatter_gather",
                        names[a],
                        (names[a], names[b], *(names[m] for m in mids)),
                        start,
                        end,
                        n,
                        edges,
                        f"{names[a]} sent money through {n} different accounts that each passed it on to "
                        f"{names[b]} within {_days(w)}",
                        {"target": names[b], "intermediaries": len(mids), "max_intermediaries_in_window": n},
                    )
                )
    return alerts


# --- cycles -------------------------------------------------------------------------------------------


class _LoopSearch:
    """Depth-first search for loops that start with one transfer ``e0`` from ``u``; state lives on the
    object so that the recursion reads plainly."""

    def __init__(self, g: Graph, max_length: int, min_length: int, w: int, cap: int, max_instances: int) -> None:
        self.g, self.max_length, self.min_length, self.w, self.cap = g, max_length, min_length, w, cap
        self.max_instances = max_instances
        self.found: dict[tuple[int, ...], list[Any]] = {}
        self.total = 0
        self.truncated = False

    def start(self, u: int, p: int) -> None:
        g = self.g
        e0 = g.out[p]
        self.u, self.r0 = u, g.rank[e0]
        self.t0, v = g.out_ts[p], g.dst[e0]
        self.tend = self.t0 + self.w
        ilo, ihi = g.in_range(u, self.t0, self.tend)
        self.p1 = {g.src[g.inn[q]] for q in range(ilo, ihi) if g.rank[g.inn[q]] > self.r0}
        if not self.p1:
            return
        self.busy: dict[int, bool] = {}
        if self.is_busy(u) or self.is_busy(v):
            return
        self.p2: set[int] = set()
        if self.max_length >= 3:
            for x in self.p1:
                for q in range(*g.in_range(x, self.t0, self.tend)):
                    self.p2.add(g.src[g.inn[q]])
        self.path = [e0]
        self.visited = {u, v}
        self.dfs(v, self.t0, 1)

    def is_busy(self, x: int) -> bool:
        b = self.busy.get(x)
        if b is None:
            b = self.busy[x] = _busy(self.g, x, self.t0, self.tend, self.cap)
        return b

    def emit(self) -> None:
        g = self.g
        edges = self.path
        names = g.tx.accounts
        nodes = [g.src[e] for e in edges]
        k = min(range(len(nodes)), key=lambda i: names[nodes[i]])  # by name: input order does not matter
        key = tuple(nodes[k:] + nodes[:k])
        first, last = g.ts[edges[0]], g.ts[edges[-1]]
        rec = self.found.get(key)
        if rec is None:
            self.found[key] = [list(edges), 1, first, last, dict.fromkeys(edges)]
        else:
            rec[1] += 1
            rec[2] = min(rec[2], first)
            rec[3] = max(rec[3], last)
            if len(rec[4]) < 500:
                rec[4].update(dict.fromkeys(edges))
        self.total += 1
        if self.total >= self.max_instances:
            self.truncated = True

    def dfs(self, x: int, t_cur: int, depth: int) -> None:
        # depth = transfers used so far; one more closes the loop
        g = self.g
        olo, ohi = g.out_range(x, t_cur, self.tend)
        remaining = self.max_length - depth
        for q in range(olo, ohi):
            if self.truncated:
                return
            e = g.out[q]
            if g.rank[e] <= self.r0:
                continue
            y = g.dst[e]
            if y == self.u:
                if depth + 1 >= self.min_length:
                    self.path.append(e)
                    self.emit()
                    self.path.pop()
                continue
            if remaining <= 1 or y in self.visited:
                continue
            left = remaining - 1  # transfers still allowed after taking e, the closing one included
            if (left == 1 and y not in self.p1) or (left == 2 and y not in self.p1 and y not in self.p2):
                continue
            if self.is_busy(y):
                continue
            self.visited.add(y)
            self.path.append(e)
            self.dfs(y, g.out_ts[q], depth + 1)
            self.path.pop()
            self.visited.discard(y)


def cycles(
    g: Graph,
    max_length: int = 6,
    window_days: float = 7,
    max_degree: int = 50,
    min_length: int = 2,
    max_instances: int = 1_000_000,
) -> list[Alert]:
    """Money that returns to where it started: A -> B -> ... -> A, each transfer no earlier than the one
    before, the whole loop within the window, no account visited twice, between ``min_length`` and
    ``max_length`` transfers.

    Accounts with more than ``max_degree`` incoming or outgoing transfers within the loop's window are
    treated as businesses or hubs and not followed. Loops over the same accounts are reported once, with
    the number of times they occurred (``detail['instances']``). The search stops after
    ``max_instances`` loops and says so (``detail['search_truncated']``).
    """
    if not 2 <= min_length <= max_length:
        raise InputError("need 2 <= min_length <= max_length")
    if max_length > 12:
        raise InputError("max_length above 12 is not supported")
    w = _window(window_days)
    s = _LoopSearch(g, max_length, min_length, w, max_degree, max_instances)
    for u in range(g.n_accounts):
        if g.in_degree(u) == 0:
            continue
        for p in range(g.out_ptr[u], g.out_ptr[u + 1]):
            s.start(u, p)
            if s.truncated:
                break
        if s.truncated:
            break

    names = g.tx.accounts
    alerts = []
    for key, (edges, n, first, last, union) in s.found.items():
        accs = tuple(names[x] for x in key)
        loop = " -> ".join((*accs, accs[0]))
        alerts.append(
            Alert(
                "cycle",
                accs[0],
                accs,
                first,
                last,
                len(key),
                tuple(union),
                f"money went round {loop} ({len(key)} transfers, within {_days(w)})"
                + (f", {n} times" if n > 1 else ""),
                {
                    "length": len(key),
                    "instances": n,
                    "first_loop_rows": [int(g.tx.row[e]) for e in edges],
                    "search_truncated": s.truncated,
                },
            )
        )
    return alerts


# --- pass-through -------------------------------------------------------------------------------------


def pass_through(
    g: Graph,
    window_days: float = 1,
    min_ratio: float | str = "0.9",
    max_ratio: float | str = "1.0",
    min_amount: float | str = 0,
) -> list[Alert]:
    """An account receives a transfer and, within the window, sends out between ``min_ratio`` and
    ``max_ratio`` of that amount, in the same currency, to other accounts (money that passes straight
    through). Transfers below ``min_amount`` are not considered as the incoming transfer."""
    w = _window(window_days)
    lo_r, hi_r = Fraction(str(min_ratio)), Fraction(str(max_ratio))
    if not 0 < lo_r <= hi_r:
        raise InputError("need 0 < min_ratio <= max_ratio")
    floor = parse_amount(str(min_amount))
    tx, names = g.tx, g.tx.accounts
    amount = tx.amount.tolist()
    currency = tx.currency.tolist()
    alerts = []
    for a in range(g.n_accounts):
        if g.out_degree(a) == 0 or g.in_degree(a) == 0:
            continue
        hits = []
        for p in range(g.in_ptr[a], g.in_ptr[a + 1]):
            e_in = g.inn[p]
            x = amount[e_in]
            if x == 0 or x < floor:
                continue
            t, cur = g.in_ts[p], currency[e_in]
            olo, ohi = g.out_range(a, t, t + w)
            outs = [e for e in g.out[olo:ohi] if currency[e] == cur and g.rank[e] > g.rank[e_in]]
            if not outs:
                continue
            y = sum(amount[e] for e in outs)
            if y * lo_r.denominator < x * lo_r.numerator or y * hi_r.denominator > x * hi_r.numerator:
                continue
            hits.append((t, e_in, outs, x, y))
        merged: list[list[Any]] = []
        for t, e_in, outs, x, y in hits:
            end = g.ts[outs[-1]]
            if merged and t <= merged[-1][1]:
                m = merged[-1]
                m[1] = max(m[1], end)
                m[2].update(dict.fromkeys([e_in, *outs]))
                m[3] += 1
                m[4] = max(m[4], Fraction(y, x))
            else:
                merged.append([t, end, dict.fromkeys([e_in, *outs]), 1, Fraction(y, x)])
        for start, end, edges, n, ratio in merged:
            accs = dict.fromkeys([names[a], *(names[g.src[e]] for e in edges), *(names[g.dst[e]] for e in edges)])
            alerts.append(
                Alert(
                    "pass_through",
                    names[a],
                    tuple(accs),
                    start,
                    end,
                    len(edges),
                    tuple(edges),
                    f"{names[a]} passed on {float(ratio):.0%} of money it received within {_days(w)}"
                    + (f" ({n} times)" if n > 1 else ""),
                    {"incoming_transfers": n, "max_ratio": f"{float(ratio):.4f}"},
                )
            )
    return alerts


# --- structuring --------------------------------------------------------------------------------------


def structuring(
    g: Graph,
    threshold: float | str,
    margin: float | str = "0.1",
    min_count: int = 3,
    window_days: float = 7,
    side: str = "both",
    currency: str | None = None,
) -> list[Alert]:
    """At least ``min_count`` transfers to (or from) one account within the window, each just under a
    reporting ``threshold``: between ``threshold * (1 - margin)`` and the threshold itself (excluded).

    ``side`` is ``"in"`` (deposits into the account), ``"out"`` (payments from it) or ``"both"``. The
    threshold is in the data's currency units; give ``currency`` when the data mixes currencies.
    """
    if side not in ("in", "out", "both"):
        raise InputError("side must be 'in', 'out' or 'both'")
    if min_count < 2:
        raise InputError("min_count must be at least 2")
    top = parse_amount(str(threshold))
    m = Fraction(str(margin))
    if not 0 < m < 1 or top == 0:
        raise InputError("need threshold > 0 and 0 < margin < 1")
    low = -((-top * (m.denominator - m.numerator)) // m.denominator)  # ceil(top * (1 - margin))
    w = _window(window_days)
    tx, names = g.tx, g.tx.accounts
    amount = tx.amount.tolist()
    cur_code: int | None = None
    if currency is not None:
        try:
            cur_code = [c.casefold() for c in tx.currencies].index(currency.casefold())
        except ValueError:
            return []
    curs = tx.currency.tolist()

    def near(e: int) -> bool:
        return low <= amount[e] < top and (cur_code is None or curs[e] == cur_code)

    alerts = []
    sides = ("in", "out") if side == "both" else (side,)
    for sd in sides:
        lst, times, ptr = (g.inn, g.in_ts, g.in_ptr) if sd == "in" else (g.out, g.out_ts, g.out_ptr)
        for a in range(g.n_accounts):
            lo, hi = ptr[a], ptr[a + 1]
            if hi - lo < min_count:
                continue
            sel = [(times[p], lst[p]) for p in range(lo, hi) if near(lst[p])]
            if len(sel) < min_count:
                continue
            hits = []
            j = 0
            for i in range(len(sel)):
                while sel[i][0] - sel[j][0] > w:
                    j += 1
                if i - j + 1 >= min_count:
                    hits.append((j, i, i - j + 1))
            for elo, ehi, best in _episodes(iter(hits)):
                edges = tuple(e for _, e in sel[elo : ehi + 1])
                other = g.src if sd == "in" else g.dst
                verb = "received" if sd == "in" else "sent"
                alerts.append(
                    Alert(
                        "structuring",
                        names[a],
                        tuple(dict.fromkeys([names[a], *(names[other[e]] for e in edges)])),
                        sel[elo][0],
                        sel[ehi][0],
                        best,
                        edges,
                        f"{names[a]} {verb} {best} transfers within {_days(w)} each between "
                        f"{format_amount(low)} and {format_amount(top)} (just under the threshold)",
                        {"side": sd, "threshold": format_amount(top), "lower_bound": format_amount(low)},
                    )
                )
    return alerts


DETECTORS: dict[str, Callable[..., list[Alert]]] = {
    "fan_out": fan_out,
    "fan_in": fan_in,
    "gather_scatter": gather_scatter,
    "scatter_gather": scatter_gather,
    "cycle": cycles,
    "pass_through": pass_through,
    "structuring": structuring,
}
DEFAULT_PATTERNS = ("fan_out", "fan_in", "gather_scatter", "scatter_gather", "cycle", "pass_through")
