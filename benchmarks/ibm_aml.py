"""Benchmark the detectors on the IBM AML synthetic data (HI-Small by default).

Download the data yourself (CDLA-Sharing-1.0; it is not redistributed here), for example from Kaggle
"IBM Transactions for Anti Money Laundering (AML)", then:

    uv run python benchmarks/ibm_aml.py /path/to/HI-Small_Trans.csv /path/to/HI-Small_Patterns.txt

Writes benchmarks/results/<name>.json and prints a Markdown table. Defaults are the library defaults;
the grid shows how recall and alert volume move with each parameter.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from eqlx_aml import Graph, read_csv
from eqlx_aml import detect as d
from eqlx_aml.ibm import evaluate, read_labels, read_patterns

GRID = [
    ("fan_out", {}),
    ("fan_out", {"min_counterparties": 3}),
    ("fan_out", {"min_counterparties": 8}),
    ("fan_out", {"window_days": 3}),
    ("fan_in", {}),
    ("fan_in", {"min_counterparties": 3}),
    ("fan_in", {"min_counterparties": 8}),
    ("fan_in", {"window_days": 3}),
    ("gather_scatter", {}),
    ("gather_scatter", {"min_senders": 2, "min_receivers": 2}),
    ("gather_scatter", {"min_senders": 5, "min_receivers": 5}),
    ("gather_scatter", {"window_days": 10}),
    ("scatter_gather", {}),
    ("scatter_gather", {"min_intermediaries": 2}),
    ("scatter_gather", {"min_intermediaries": 5}),
    ("scatter_gather", {"max_degree": 200}),
    ("cycle", {}),
    ("cycle", {"max_length": 4}),
    ("cycle", {"max_length": 8}),
    ("cycle", {"max_degree": 200}),
    ("pass_through", {}),
]
DEFAULTS = {"fan_out": 5, "fan_in": 5, "gather_scatter": 3, "scatter_gather": 3, "cycle": 6}


def eligible(pattern: str, kw: dict, size: int) -> bool:
    """Is a planted attempt big enough (or short enough) to match the definition at these parameters?"""
    if pattern in ("fan_out", "fan_in"):
        return size >= kw.get("min_counterparties", 5)
    if pattern == "gather_scatter":
        return size >= max(kw.get("min_senders", 3), kw.get("min_receivers", 3))
    if pattern == "scatter_gather":
        return size >= kw.get("min_intermediaries", 3)
    if pattern == "cycle":
        return 2 <= size <= kw.get("max_length", 6)
    return False


KIND = {
    "fan_out": "FAN-OUT",
    "fan_in": "FAN-IN",
    "gather_scatter": "GATHER-SCATTER",
    "scatter_gather": "SCATTER-GATHER",
    "cycle": "CYCLE",
}


def main(trans: str, patterns: str) -> None:
    t = time.time()
    tx = read_csv(trans)
    labels = read_labels(trans)
    attempts = read_patterns(patterns, tx)
    g = Graph(tx)
    print(f"loaded {len(tx)} transfers, {len(attempts)} attempts in {time.time() - t:.0f}s", file=sys.stderr)
    runs = []
    for pattern, kw in GRID:
        t = time.time()
        alerts = d.DETECTORS[pattern](g, **kw)
        secs = time.time() - t
        ev = evaluate(tx, labels, attempts, alerts)
        kind = KIND.get(pattern)
        rec = ev["recall"].get(kind, {}) if kind else {}
        prec = ev["precision"].get(pattern, {"alerts": 0, "on_laundering_accounts": 0})
        per = rec.get("per_attempt", [])
        elig = [x for x in per if eligible(pattern, kw, x["size"])]
        runs.append(
            {
                "pattern": pattern,
                "params": kw,
                "seconds": round(secs, 1),
                "alerts": prec["alerts"],
                "alerts_on_laundering_accounts": prec["on_laundering_accounts"],
                "attempts": rec.get("attempts"),
                "detected": rec.get("detected"),
                "eligible_attempts": len(elig) if kind else None,
                "eligible_detected": sum(x["detected"] for x in elig) if kind else None,
                "base_rate": ev["base_rate"],
            }
        )
        print(json.dumps(runs[-1]), file=sys.stderr, flush=True)
    out = {
        "data": Path(trans).name,
        "transfers": len(tx),
        "accounts": len(tx.accounts),
        "laundering_transfers": int(labels.sum()),
        "attempts_by_type": {k: sum(1 for a in attempts if a.kind == k) for k in sorted({a.kind for a in attempts})},
        "attempts_matched_rows": sum(1 for a in attempts if a.rows),
        "runs": runs,
    }
    res = Path(__file__).parent / "results"
    res.mkdir(exist_ok=True)
    (res / (Path(trans).stem + ".json")).write_text(json.dumps(out, indent=2) + "\n")
    print(
        "| pattern | params | attempts found | of those big enough | alerts | on laundering accounts | lift | seconds |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        share = r["alerts_on_laundering_accounts"] / r["alerts"] if r["alerts"] else 0
        lift = share / r["base_rate"] if r["base_rate"] else 0
        found = f"{r['detected']}/{r['attempts']}" if r["attempts"] else "-"
        big = f"{r['eligible_detected']}/{r['eligible_attempts']}" if r["attempts"] else "-"
        params = ", ".join(f"{k}={v}" for k, v in r["params"].items()) or "defaults"
        cells = [r["pattern"], params, found, big, r["alerts"], f"{share:.1%}", f"{lift:.1f}x", r["seconds"]]
        print("| " + " | ".join(str(c) for c in cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
