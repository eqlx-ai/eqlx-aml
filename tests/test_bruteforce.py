"""Every detector against an independent brute-force implementation on many small random graphs.

The brute-force versions below share no code with the library: they enumerate windows, paths and
loops directly from the definitions in the docstrings. Times are drawn on a coarse grid so that equal
timestamps (ties) are common.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest

from eqlx_aml import Graph, Transactions
from eqlx_aml import detect as d

DAY = 86_400
SEEDS = range(40)


def random_tx(seed: int, n_acc: int = 8, n_tx: int = 45, days: int = 10) -> tuple[Transactions, list[tuple]]:
    rng = random.Random(seed)
    recs = []
    for _ in range(n_tx):
        a, b = rng.randrange(n_acc), rng.randrange(n_acc)
        t = rng.randrange(days * 8) * (DAY // 8)  # 3-hour grid: many ties
        amt = rng.choice([100, 950, 980, 990, 1000, 2000, 500])
        cur = rng.choice(["EUR", "EUR", "USD"])
        recs.append({"timestamp": t, "source": f"A{a}", "target": f"A{b}", "amount": amt, "currency": cur})
    tx = Transactions.from_records(recs)
    # plain tuples for the brute force: (index, time, row, source, target, amount_micro, currency)
    edges = [
        (
            i,
            int(tx.ts[i]),
            int(tx.row[i]),
            tx.accounts[int(tx.src[i])],
            tx.accounts[int(tx.dst[i])],
            int(tx.amount[i]),
            tx.currencies[int(tx.currency[i])],
        )
        for i in range(len(tx))
        if tx.src[i] != tx.dst[i]
    ]
    return tx, edges


# --- fans ---------------------------------------------------------------------------------------------


def brute_fan(edges, k, w, out):
    best: dict[str, int] = {}
    for e in edges:
        acc = e[3] if out else e[4]
        t0 = e[1]
        mine = [x for x in edges if (x[3] if out else x[4]) == acc and t0 <= x[1] <= t0 + w]
        n = len({(x[4] if out else x[3]) for x in mine})
        if n >= k:
            best[acc] = max(best.get(acc, 0), n)
    return best


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("out", [True, False])
def test_fan(seed, out):
    tx, edges = random_tx(seed)
    g = Graph(tx)
    for k, days in [(2, 1), (3, 2), (4, 5)]:
        alerts = (d.fan_out if out else d.fan_in)(g, min_counterparties=k, window_days=days)
        got: dict[str, int] = {}
        for a in alerts:
            got[a.account] = max(got.get(a.account, 0), a.size)
        assert got == brute_fan(edges, k, days * DAY, out)


# --- gather-scatter -----------------------------------------------------------------------------------


def brute_gather_scatter(edges, k_in, k_out, w):
    flagged = set()
    accounts = {x[3] for x in edges} | {x[4] for x in edges}
    for acc in accounts:
        times = sorted({x[1] for x in edges if acc in (x[3], x[4])})
        for t0 in times:
            ins = {x[3] for x in edges if x[4] == acc and t0 <= x[1] <= t0 + w}
            outs = {x[4] for x in edges if x[3] == acc and t0 <= x[1] <= t0 + w}
            if len(ins) >= k_in and len(outs) >= k_out:
                flagged.add(acc)
    return flagged


@pytest.mark.parametrize("seed", SEEDS)
def test_gather_scatter(seed):
    tx, edges = random_tx(seed)
    g = Graph(tx)
    for k_in, k_out, days in [(2, 2, 1), (3, 2, 3), (3, 3, 10)]:
        got = {a.account for a in d.gather_scatter(g, k_in, k_out, days)}
        assert got == brute_gather_scatter(edges, k_in, k_out, days * DAY)


# --- scatter-gather -----------------------------------------------------------------------------------


def brute_scatter_gather(edges, k, w):
    flagged = set()
    paths = [
        (e1[3], e2[4], e1[4], e1[1], e2[1])
        for e1 in edges
        for e2 in edges
        if e1[4] == e2[3] and e2[1] >= e1[1] and e2[4] != e1[3] and e2[4] != e1[4]
    ]
    for a, b in {(p[0], p[1]) for p in paths}:
        ps = [p for p in paths if p[0] == a and p[1] == b]
        for s in {p[3] for p in ps}:
            mids = {p[2] for p in ps if p[3] >= s and p[4] <= s + w}
            if len(mids) >= k:
                flagged.add((a, b))
    return flagged


@pytest.mark.parametrize("seed", SEEDS)
def test_scatter_gather(seed):
    tx, edges = random_tx(seed, n_acc=7, n_tx=50)
    g = Graph(tx)
    for k, days in [(2, 1), (2, 4), (3, 10)]:
        got = {(a.account, a.detail["target"]) for a in d.scatter_gather(g, k, days, max_degree=10**6)}
        assert got == brute_scatter_gather(edges, k, days * DAY)


# --- cycles -------------------------------------------------------------------------------------------


def brute_cycles(edges, max_len, w, min_len=2):
    """All loops as ordered edge sequences: distinct accounts, starting at the loop's earliest transfer
    (by time, then row), times never decreasing, total span within w. Returns {accounts key: count}."""
    found: dict[tuple, int] = {}
    for length in range(min_len, max_len + 1):
        for seq in itertools.permutations(edges, length):
            if any(seq[i][4] != seq[i + 1][3] for i in range(length - 1)) or seq[-1][4] != seq[0][3]:
                continue
            nodes = [e[3] for e in seq]
            if len(set(nodes)) != length:
                continue
            if min(seq, key=lambda e: (e[1], e[2])) is not seq[0]:
                continue
            if any(seq[i + 1][1] < seq[i][1] for i in range(length - 1)) or seq[-1][1] - seq[0][1] > w:
                continue
            k = nodes.index(min(nodes))
            key = tuple(nodes[k:] + nodes[:k])
            found[key] = found.get(key, 0) + 1
    return found


@pytest.mark.parametrize("seed", range(25))
def test_cycles(seed):
    tx, edges = random_tx(seed, n_acc=5, n_tx=16)
    g = Graph(tx)
    for max_len, days in [(2, 2), (3, 3), (4, 10)]:
        alerts = d.cycles(g, max_length=max_len, window_days=days, max_degree=10**6)
        got = {a.accounts: a.detail["instances"] for a in alerts}
        assert got == brute_cycles(edges, max_len, days * DAY)


def test_cycles_min_length():
    tx, edges = random_tx(3, n_acc=5, n_tx=16)
    alerts = d.cycles(Graph(tx), max_length=4, window_days=10, max_degree=10**6, min_length=3)
    assert {a.accounts: a.detail["instances"] for a in alerts} == brute_cycles(edges, 4, 10 * DAY, min_len=3)


# --- pass-through and structuring ---------------------------------------------------------------------


def brute_pass_through(edges, w, lo, hi):
    flagged = set()
    for e in edges:
        acc, x = e[4], e[5]
        if x == 0:
            continue
        outs = [
            o
            for o in edges
            if o[3] == acc and o[6] == e[6] and e[1] <= o[1] <= e[1] + w and (o[1], o[2]) > (e[1], e[2])
        ]
        y = sum(o[5] for o in outs)
        if outs and lo * x <= y <= hi * x:
            flagged.add(acc)
    return flagged


@pytest.mark.parametrize("seed", SEEDS)
def test_pass_through(seed):
    tx, edges = random_tx(seed)
    g = Graph(tx)
    for days, lo, hi in [(1, "0.9", "1.0"), (2, "0.5", "1.5"), (3, "0.95", "2")]:
        got = {a.account for a in d.pass_through(g, days, lo, hi)}
        assert got == brute_pass_through(edges, days * DAY, Fraction(lo), Fraction(hi))


def brute_structuring(edges, top, low, n, w, side):
    flagged = set()
    for e in edges:
        acc = e[4] if side == "in" else e[3]
        sel = [
            x
            for x in edges
            if (x[4] if side == "in" else x[3]) == acc and low <= x[5] < top and e[1] <= x[1] <= e[1] + w
        ]
        if len(sel) >= n:
            flagged.add(acc)
    return flagged


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("side", ["in", "out"])
def test_structuring(seed, side):
    tx, edges = random_tx(seed)
    g = Graph(tx)
    got = {a.account for a in d.structuring(g, threshold=1000, margin="0.1", min_count=2, window_days=3, side=side)}
    assert got == brute_structuring(edges, 1000 * 10**6, 900 * 10**6, 2, 3 * DAY, side)
