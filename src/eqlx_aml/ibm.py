"""Measure the detectors on the public IBM AML synthetic data (Altman et al., NeurIPS 2023).

The data set (CDLA-Sharing-1.0, not included here) comes as ``<name>_Trans.csv`` (every transfer, with
an ``Is Laundering`` label) and ``<name>_Patterns.txt`` (the laundering attempts the simulator planted,
grouped and typed: FAN-OUT, FAN-IN, GATHER-SCATTER, SCATTER-GATHER, CYCLE, STACK, BIPARTITE, RANDOM).

Two questions are answered:

* **Recall per attempt:** for each planted attempt of a type a detector targets, did that detector
  raise an alert on the attempt's key account (and, for scatter-gather, the same target; for cycles,
  exactly the same accounts) that shares at least one transfer with it?
* **Precision per alert:** how many alerts are about an account that takes part in any
  laundering-labelled transfer, compared with the share of such accounts in the whole data.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import Transactions, parse_amount, parse_timestamp
from .detect import Alert

TARGETS = {
    "FAN-OUT": "fan_out",
    "FAN-IN": "fan_in",
    "GATHER-SCATTER": "gather_scatter",
    "SCATTER-GATHER": "scatter_gather",
    "CYCLE": "cycle",
}


@dataclass
class Attempt:
    kind: str
    header: str
    rows: list[int] = field(default_factory=list)  # rows in the transactions file (1-based)
    accounts: set[str] = field(default_factory=set)
    key: str = ""
    target: str = ""
    size: int = 0  # counterparties of the key account / intermediaries / loop length, as planted


def read_labels(path: str | Path) -> np.ndarray:
    """The ``Is Laundering`` column as a boolean array, in file order (index 0 = row 1)."""
    out = []
    with Path(path).open(newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row:
                out.append(row[-1].strip() == "1")
    return np.array(out, dtype=bool)


def read_patterns(path: str | Path, tx: Transactions) -> list[Attempt]:
    """Parse the patterns file and find each listed transfer's row in ``tx`` (read from the matching
    transactions file)."""
    attempts: list[Attempt] = []
    cur: Attempt | None = None
    lines: list[tuple[int, str, str, int, str]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("BEGIN"):
            kind = line.split(" - ", 1)[1].split(":")[0].strip() if " - " in line else "UNKNOWN"
            cur = Attempt(kind, line)
        elif line.startswith("END"):
            if cur is not None:
                attempts.append(cur)
            cur = None
        elif line and cur is not None:
            f = line.split(",")
            lines.append((len(attempts), f[1] + ":" + f[2], f[3] + ":" + f[4], parse_amount(f[7]), f[0]))
            cur.accounts.update((f[1] + ":" + f[2], f[3] + ":" + f[4]))

    index = {a: i for i, a in enumerate(tx.accounts)}
    wanted: dict[tuple[int, int, int, int], list[int]] = {}
    for n, (_k, s, t, amt, ts) in enumerate(lines):
        if s in index and t in index:
            wanted.setdefault((index[s], index[t], amt, parse_timestamp(ts)), []).append(n)
    pair_keys = np.array(sorted({(k[0] << 32) | k[1] for k in wanted}), dtype=np.int64)
    pk = (tx.src.astype(np.int64) << 32) | tx.dst.astype(np.int64)
    cand = np.flatnonzero(np.isin(pk, pair_keys))
    for i in cand.tolist():
        key = (int(tx.src[i]), int(tx.dst[i]), int(tx.amount[i]), int(tx.ts[i]))
        for n in wanted.get(key, []):
            attempts[lines[n][0]].rows.append(int(tx.row[i]))
    for a in attempts:
        a.rows = sorted(set(a.rows))
        _key_accounts(a, [(s, t) for k, s, t, _, _ in lines if attempts[k] is a])
    return attempts


def _key_accounts(a: Attempt, pairs: list[tuple[str, str]]) -> None:
    pairs = [(s, t) for s, t in pairs if s != t]
    srcs = Counter(s for s, _ in pairs)
    dsts = Counter(t for _, t in pairs)
    if not pairs:
        return

    def top(c: Counter[str]) -> str:  # ties broken by name, never by set or hash order
        return min(c, key=lambda x: (-c[x], x))

    if a.kind == "FAN-IN":
        a.key = top(dsts)
        a.size = len({s for s, t in pairs if t == a.key})
    elif a.kind == "GATHER-SCATTER":
        both = Counter({x: srcs[x] + dsts[x] for x in set(srcs) & set(dsts)})
        a.key = top(both or dsts)
        a.size = min(len({s for s, t in pairs if t == a.key}), len({t for s, t in pairs if s == a.key}))
    elif a.kind == "SCATTER-GATHER":
        a.key = top(srcs)
        a.target = top(dsts)
        first = {t for s, t in pairs if s == a.key}
        a.size = len({s for s, t in pairs if t == a.target and s in first})
    elif a.kind == "CYCLE":
        a.key = top(srcs)
        a.size = len(a.accounts)
    else:
        a.key = top(srcs)
        a.size = len({t for s, t in pairs if s == a.key})


def _hits(alert: Alert, a: Attempt, rows: set[int]) -> bool:
    if not rows:
        return False
    if a.kind == "CYCLE":
        return set(alert.accounts) == a.accounts - {""}
    if alert.account != a.key:
        return False
    return a.kind != "SCATTER-GATHER" or alert.detail.get("target") == a.target


def evaluate(tx: Transactions, labels: np.ndarray, attempts: list[Attempt], alerts: Iterable[Alert]) -> dict[str, Any]:
    """Recall per attempt type and precision per pattern. ``tx`` must be the full transactions file
    (row ``n`` at index ``n - 1``), so labels line up."""
    if len(labels) != len(tx):
        raise ValueError("labels and transactions differ in length")
    laundering_idx = np.flatnonzero(labels)
    bad_accounts = set(np.unique(np.concatenate([tx.src[laundering_idx], tx.dst[laundering_idx]])).tolist())
    names = {a: i for i, a in enumerate(tx.accounts)}
    by_pattern: dict[str, list[Alert]] = {}
    for al in alerts:
        by_pattern.setdefault(al.pattern, []).append(al)

    precision = {}
    for p, als in by_pattern.items():
        good = sum(1 for al in als if names[al.account] in bad_accounts)
        precision[p] = {"alerts": len(als), "on_laundering_accounts": good}
    base_rate = len(bad_accounts) / max(1, len(tx.accounts))

    flagged = {al.account for al in by_pattern_all(by_pattern)}
    recall: dict[str, dict[str, Any]] = {}
    for kind in sorted({a.kind for a in attempts}):
        group = [a for a in attempts if a.kind == kind]
        pattern = TARGETS.get(kind)
        cands_by_account: dict[str, list[Alert]] = {}
        for al in by_pattern.get(pattern or "", []):
            cands_by_account.setdefault(al.account, []).append(al)
        found = 0
        per_attempt = []
        for a in group:
            rows = set(a.rows)
            cands = by_pattern.get("cycle", []) if kind == "CYCLE" else cands_by_account.get(a.key, [])
            hit = bool(pattern) and any(
                _hits(al, a, rows) and rows & {int(tx.row[i]) for i in al.transactions} for al in cands
            )
            found += hit
            per_attempt.append({"size": a.size, "detected": hit})
        recall[kind] = {
            "attempts": len(group),
            "target_pattern": pattern,
            "detected": found if pattern else None,
            "key_account_flagged_by_any_pattern": sum(1 for a in group if a.key in flagged),
            "per_attempt": per_attempt,
        }
    return {
        "transfers": len(tx),
        "laundering_transfers": int(labels.sum()),
        "accounts": len(tx.accounts),
        "laundering_accounts": len(bad_accounts),
        "base_rate": base_rate,
        "precision": precision,
        "recall": recall,
    }


def by_pattern_all(by_pattern: dict[str, list[Alert]]) -> list[Alert]:
    return [al for als in by_pattern.values() for al in als]
