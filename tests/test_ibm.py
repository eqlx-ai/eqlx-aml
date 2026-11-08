"""The IBM benchmark helpers, on a tiny made-up file in the IBM layout."""

import numpy as np

import eqlx_aml as aml
from eqlx_aml import Graph
from eqlx_aml import detect as d
from eqlx_aml.ibm import evaluate, read_labels, read_patterns

HEADER = "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"


def line(ts, fb, fa, tb, ta, amt, label):
    return f"{ts},{fb},{fa},{tb},{ta},{amt},Euro,{amt},Euro,ACH,{label}\n"


FAN = [line(f"2022/09/01 0{i}:00", "001", "HUB", "002", f"R{i}", f"{100 + i}.00", 1) for i in range(5)]
LOOP = [
    line("2022/09/02 01:00", "003", "A", "003", "B", "50.00", 1),
    line("2022/09/02 02:00", "003", "B", "003", "C", "49.00", 1),
    line("2022/09/02 03:00", "003", "C", "003", "A", "48.00", 1),
]
NOISE = [
    line("2022/09/01 05:30", "009", "N1", "009", "N2", "7.00", 0),
    line("2022/09/03 05:30", "009", "N2", "009", "N1", "7.00", 0),
]
PATTERNS = (
    "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 5-degree Fan-Out\n"
    + "".join(FAN)
    + "END LAUNDERING ATTEMPT - FAN-OUT\n\n"
    + "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops\n"
    + "".join(LOOP)
    + "END LAUNDERING ATTEMPT - CYCLE\n"
)


def test_read_and_evaluate(tmp_path):
    trans = tmp_path / "T_Trans.csv"
    trans.write_text(HEADER + "".join(NOISE[:1] + FAN + LOOP + NOISE[1:]))
    pats = tmp_path / "T_Patterns.txt"
    pats.write_text(PATTERNS)

    tx = aml.read_csv(trans)
    labels = read_labels(trans)
    assert labels.tolist() == [False] + [True] * 8 + [False]
    attempts = read_patterns(pats, tx)
    fan, loop = attempts
    assert (fan.kind, fan.key, fan.size, fan.rows) == ("FAN-OUT", "001:HUB", 5, [2, 3, 4, 5, 6])
    assert (loop.kind, loop.size, loop.rows) == ("CYCLE", 3, [7, 8, 9])

    g = Graph(tx)
    alerts = d.fan_out(g) + d.cycles(g)
    ev = evaluate(tx, labels, attempts, alerts)
    assert ev["recall"]["FAN-OUT"]["detected"] == 1 and ev["recall"]["CYCLE"]["detected"] == 1
    assert ev["precision"]["fan_out"] == {"alerts": 1, "on_laundering_accounts": 1}
    # the noise accounts form a 2-step loop too: an alert on an account with no laundering transfer
    assert ev["precision"]["cycle"] == {"alerts": 2, "on_laundering_accounts": 1}
    assert ev["laundering_accounts"] == 9 and np.isclose(ev["base_rate"], 9 / 11)

    # with a stricter definition the fan-out is no longer found
    ev2 = evaluate(tx, labels, attempts, d.fan_out(g, min_counterparties=6))
    assert ev2["recall"]["FAN-OUT"]["detected"] == 0
