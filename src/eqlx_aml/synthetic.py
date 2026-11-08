"""Synthetic transfers with known, planted patterns: for tests, demos and trying the tools.

Everything here is made up. Account names look like ``ACC-0042``; planted accounts like ``FO-HUB`` so a
reader can see at once which pattern they belong to. The generator is seeded, so the same seed always
gives the same file.
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

START = datetime(2026, 1, 5, tzinfo=timezone.utc)


@dataclass
class Planted:
    pattern: str
    account: str
    accounts: list[str]
    rows: list[int] = field(default_factory=list)


@dataclass
class SyntheticData:
    records: list[dict[str, Any]]
    planted: list[Planted]

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(
            buf,
            fieldnames=["id", "timestamp", "source", "target", "amount", "currency", "channel"],
            lineterminator="\n",
        )
        w.writeheader()
        for r in self.records:
            w.writerow(r)
        return buf.getvalue()


class _Gen:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.rows: list[tuple[datetime, str, str, str, str]] = []
        self.planted: list[Planted] = []

    def add(self, t: datetime, a: str, b: str, amount: float, channel: str = "Wire") -> int:
        self.rows.append((t, a, b, f"{amount:.2f}", channel))
        return len(self.rows) - 1

    def at(self, day: float, hour: float = 0) -> datetime:
        return START + timedelta(days=day, hours=hour)


def generate(seed: int = 7, accounts: int = 400, transfers: int = 4000, days: int = 28) -> SyntheticData:
    """Background payments between ``accounts`` ordinary accounts, plus one planted instance of each
    pattern (fan-out, fan-in, gather-scatter, scatter-gather, a 4-step cycle, pass-through,
    structuring under 10,000). Background accounts pay three regular counterparties each; like real
    data they still form some small patterns by chance, which is what the detectors' parameters are for."""
    g = _Gen(seed)
    rng = g.rng
    names = [f"ACC-{i:04d}" for i in range(accounts)]
    partners = {a: rng.sample(names, 3) for a in names}
    for _ in range(transfers):
        a = rng.choice(names)
        b = rng.choice(partners[a])
        if b == a:
            continue
        g.add(g.at(rng.uniform(0, days)), a, b, round(rng.lognormvariate(6.5, 1.0), 2), rng.choice(["Wire", "ACH"]))

    def plant(pattern: str, account: str, idx: list[int]) -> None:
        accs = list(dict.fromkeys([account, *(x for i in idx for x in (g.rows[i][1], g.rows[i][2]))]))
        g.planted.append(Planted(pattern, account, accs, idx))

    # fan-out: one account pays 8 new accounts over three days
    plant("fan_out", "FO-HUB", [g.add(g.at(3, 2 + 9 * i), "FO-HUB", f"FO-R{i}", 4800 + 37 * i) for i in range(8)])
    # fan-in: 12 accounts pay one account over three days
    plant("fan_in", "FI-HUB", [g.add(g.at(6, 1 + 6 * i), f"FI-S{i}", "FI-HUB", 2500 + 11 * i) for i in range(12)])
    # gather-scatter: collect from 4, pay out to 4
    idx = [g.add(g.at(9, 3 * i), f"GS-S{i}", "GS-HUB", 9000 + 100 * i) for i in range(4)]
    idx += [g.add(g.at(10, 3 * i), "GS-HUB", f"GS-R{i}", 8700 + 90 * i) for i in range(4)]
    plant("gather_scatter", "GS-HUB", idx)
    # scatter-gather: SG-A -> 5 mules -> SG-B
    idx = []
    for i in range(5):
        idx.append(g.add(g.at(12, 2 * i), "SG-A", f"SG-M{i}", 7000 + 50 * i))
        idx.append(g.add(g.at(13, 2 * i), f"SG-M{i}", "SG-B", 6900 + 50 * i))
    plant("scatter_gather", "SG-A", idx)
    # cycle of 4 transfers over four days
    loop = ["CY-A", "CY-B", "CY-C", "CY-D"]
    idx = [g.add(g.at(16 + i, 5), loop[i], loop[(i + 1) % 4], 15000 - 200 * i) for i in range(4)]
    plant("cycle", "CY-A", idx)
    # pass-through: receive 50,000 and pay out 48,500 the same day
    idx = [g.add(g.at(20, 9), "PT-IN", "PT-ACC", 50000)]
    idx += [g.add(g.at(20, 15), "PT-ACC", "PT-OUT1", 30000), g.add(g.at(20, 16), "PT-ACC", "PT-OUT2", 18500)]
    plant("pass_through", "PT-ACC", idx)
    # structuring: five cash deposits just under 10,000 within a week
    idx = [g.add(g.at(22 + i, 11), f"ST-D{i}", "ST-ACC", 9500 + 70 * i, "Cash") for i in range(5)]
    plant("structuring", "ST-ACC", idx)

    order = sorted(range(len(g.rows)), key=lambda i: (g.rows[i][0], i))
    row_of = {old: new for new, old in enumerate(order, start=1)}
    records = [
        {
            "id": f"T{n:06d}",
            "timestamp": g.rows[i][0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": g.rows[i][1],
            "target": g.rows[i][2],
            "amount": g.rows[i][3],
            "currency": "EUR",
            "channel": g.rows[i][4],
        }
        for n, i in enumerate(order, start=1)
    ]
    for p in g.planted:
        p.rows = sorted(row_of[i] for i in p.rows)
    return SyntheticData(records, g.planted)
