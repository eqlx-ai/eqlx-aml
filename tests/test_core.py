import io
import random
from datetime import datetime, timezone

import numpy as np
import pytest

import eqlx_aml as aml
from eqlx_aml import Graph, InputError, Transactions
from eqlx_aml import detect as d
from eqlx_aml.data import format_amount, format_timestamp, parse_amount, parse_timestamp
from eqlx_aml.report import account_profile, scan, trace
from eqlx_aml.synthetic import generate

# --- parsing ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "micro"),
    [
        ("0", 0),
        ("2848.96", 2_848_960_000),
        ("0.025852", 25_852),
        ("1.0000005", 1_000_000),  # half to even
        ("1.0000015", 1_000_002),
        ("1e3", 1_000_000_000),
        (" 12.5 ", 12_500_000),
        ("1,234.50", 1_234_500_000),
        ("1046300000000.00", 1_046_300_000_000_000_000),
    ],
)
def test_parse_amount(text, micro):
    assert parse_amount(text) == micro


@pytest.mark.parametrize("text", ["-5", "abc", "", "nan", "inf", "1,5"])
def test_parse_amount_rejects(text):
    with pytest.raises(InputError):
        parse_amount(text)


def test_format_amount_round_trip():
    for s in ["0.00", "2848.96", "0.025852", "1046300000000.00", "7.10"]:
        assert format_amount(parse_amount(s)) == s


def test_timestamps():
    assert parse_timestamp("2022/09/01 00:06") == int(datetime(2022, 9, 1, 0, 6, tzinfo=timezone.utc).timestamp())
    assert parse_timestamp("2026-01-05") == parse_timestamp("2026-01-05T00:00:00Z")
    assert parse_timestamp("2026-01-05T03:00:00+03:00") == parse_timestamp("2026-01-05T00:00:00Z")
    assert format_timestamp(parse_timestamp("2026-01-05T10:11:12Z")) == "2026-01-05T10:11:12Z"
    with pytest.raises(InputError):
        parse_timestamp("yesterday")


def test_generic_csv_with_column_names():
    text = "Booking Date,Payer,Payee,Value,CCY\n2026-01-05,A,B,10.50,EUR\n2026-01-06,B,C,3,EUR\n"
    tx = aml.read_csv(
        io.StringIO(text),
        columns={"timestamp": "Booking Date", "source": "Payer", "target": "Payee", "amount": "Value"},
    )
    assert len(tx) == 2 and tx.accounts == ("A", "B", "C")
    assert tx.record(0) == {
        "row": 1,
        "time": "2026-01-05T00:00:00Z",
        "source": "A",
        "target": "B",
        "amount": "10.50",
        "currency": "EUR",
    }


def test_generic_csv_autodetects_and_skips_blank_lines():
    text = "id,timestamp,from,to,amount,channel\nT1,2026-01-05T10:00:00Z,A,B,1,Cash\n\nT2,2026-01-05T11:00:00Z,B,A,2,Wire\n"
    tx = aml.read_csv(io.StringIO(text))
    assert tx.row.tolist() == [1, 3]
    assert tx.record(1)["id"] == "T2" and tx.record(1)["channel"] == "Wire"


@pytest.mark.parametrize(
    ("text", "msg"),
    [
        ("", "empty"),
        ("time,a,b\n", "cannot find"),
        ("timestamp,source,target,amount\n2026-01-05,A,B,-3\n", "row 1"),
        ("timestamp,source,target,amount\n2026-01-05,A,,3\n", "empty account"),
        ("timestamp,source,target,amount\n2026-01-05,A\n", "too few"),
    ],
)
def test_csv_errors(text, msg):
    with pytest.raises(InputError, match=msg):
        aml.read_csv(io.StringIO(text))


def test_max_rows():
    text = "timestamp,source,target,amount\n" + "2026-01-05,A,B,1\n" * 5
    with pytest.raises(InputError, match="more than 3"):
        aml.read_csv(io.StringIO(text), max_rows=3)


IBM = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
    "2022/09/01 00:20,010,8000EBD30,010,8000EBD30,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0\n"
    "2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1\n"
)


def test_ibm_layout():
    tx = aml.read_csv(io.StringIO(IBM))
    assert tx.accounts == ("010:8000EBD30", "021174:800737690", "012:80011F990")
    assert tx.record(1)["amount"] == "2848.96" and tx.record(1)["channel"] == "ACH"
    s = tx.summary()
    assert s["self_transfers"] == 1 and s["channels"] == {"Reinvestment": 1, "ACH": 1}
    assert len(tx.select(channels=["ach"])) == 1


def test_select_keeps_rows_and_names():
    data = generate(seed=1)
    tx = Transactions.from_records(data.records)
    cash = tx.select(channels=["Cash"])
    assert set(cash.channels[c] for c in cash.channel.tolist()) == {"Cash"}
    assert cash.accounts == tx.accounts
    assert all(tx.record(int(r) - 1) == cash.record(i) for i, r in enumerate(cash.row.tolist()))
    assert len(tx.select(min_amount=parse_amount("1000000"))) == 0


def test_unknown_account():
    tx = Transactions.from_records(generate(seed=1).records)
    with pytest.raises(InputError, match="unknown account"):
        account_profile(tx, "NOPE")


# --- planted patterns ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth():
    data = generate(seed=7)
    tx = Transactions.from_records(data.records)
    return data, tx, Graph(tx)


def test_every_planted_pattern_is_found_with_its_rows(synth):
    data, tx, g = synth
    rep = scan(tx, list(aml.DETECTORS), {"structuring": {"threshold": 10000}}, graph=g)
    for p in data.planted:
        mine = [a for a in rep.alerts if a.pattern == p.pattern and a.account == p.account]
        assert mine, p.pattern
        rows = {int(tx.row[i]) for a in mine for i in a.transactions}
        assert set(p.rows) <= rows, p.pattern


def test_planted_patterns_rank_first(synth):
    """Background accounts pay a few regular partners and, like real data, form some patterns by chance.
    The planted ones are larger, so each is the first alert of its pattern (alerts are sorted by size)."""
    data, tx, g = synth
    rep = scan(tx, graph=g)
    for pattern in ("fan_out", "fan_in", "gather_scatter", "scatter_gather"):
        first = next(a for a in rep.alerts if a.pattern == pattern)
        assert first.account == next(p.account for p in data.planted if p.pattern == pattern), pattern
    # background accounts have three partners, below the fan-out default of five; SG-A also fans out
    assert {a.account for a in rep.alerts if a.pattern == "fan_out"} == {"FO-HUB", "SG-A"}


def test_alert_to_dict(synth):
    data, tx, g = synth
    [a] = [a for a in d.cycles(g) if a.account == "CY-A"]
    out = a.to_dict(tx)
    assert out["accounts"] == ["CY-A", "CY-B", "CY-C", "CY-D"]
    assert out["message"].startswith("money went round CY-A -> CY-B -> CY-C -> CY-D -> CY-A")
    assert out["total_amount"] == {"EUR": "58800.00"}
    assert out["rows"] == next(p.rows for p in data.planted if p.pattern == "cycle")
    assert out["detail"]["first_loop_rows"] == out["rows"]
    assert a.to_dict(tx, max_transactions=1)["transactions_truncated"]


def test_hub_cap_hides_busy_intermediaries(synth):
    _, _, g = synth
    assert any(a.account == "SG-A" for a in d.scatter_gather(g))
    assert not any(a.account == "SG-A" for a in d.scatter_gather(g, max_degree=0))
    assert not any(a.account == "CY-A" for a in d.cycles(g, max_degree=0))


def test_parameter_validation(synth):
    _, tx, g = synth
    for call in (
        lambda: d.fan_out(g, min_counterparties=1),
        lambda: d.fan_in(g, window_days=0),
        lambda: d.cycles(g, max_length=13),
        lambda: d.cycles(g, min_length=5, max_length=4),
        lambda: d.pass_through(g, min_ratio=2, max_ratio=1),
        lambda: d.structuring(g, threshold=0),
        lambda: d.structuring(g, threshold=100, side="up"),
        lambda: scan(tx, ["nope"]),
        lambda: scan(tx, ["structuring"]),
        lambda: scan(tx, ["fan_out"], {"cycle": {}}),
        lambda: scan(tx, ["fan_out"], {"fan_out": {"bogus": 1}}),
    ):
        with pytest.raises(InputError):
            call()


def test_structuring_currency_filter(synth):
    _, _, g = synth
    assert d.structuring(g, threshold=10000, currency="EUR")
    assert d.structuring(g, threshold=10000, currency="USD") == []


# --- determinism --------------------------------------------------------------------------------------


def _fingerprint(tx: Transactions) -> list[tuple]:
    rep = scan(tx, list(aml.DETECTORS), {"structuring": {"threshold": 10000}})
    out = []
    for a in rep.alerts:
        ids = sorted(tx.ids[i] for i in a.transactions)
        out.append((a.pattern, a.account, a.accounts, a.start, a.end, a.size, tuple(ids), a.message))
    return sorted(out)


def test_same_input_same_output_and_row_order_does_not_matter():
    data = generate(seed=11)
    base = _fingerprint(Transactions.from_records(data.records))
    assert base == _fingerprint(Transactions.from_records(data.records))
    shuffled = list(data.records)
    random.Random(5).shuffle(shuffled)
    assert _fingerprint(Transactions.from_records(shuffled)) == base


# --- single accounts ----------------------------------------------------------------------------------


def test_account_profile(synth):
    _, tx, g = synth
    p = account_profile(tx, "PT-ACC", graph=g)
    assert p["incoming"]["transfers"] == 1 and p["incoming"]["total"] == {"EUR": "50000.00"}
    assert p["outgoing"]["counterparties"] == 2 and p["outgoing"]["total"] == {"EUR": "48500.00"}
    assert p["outgoing"]["top_counterparties"][0] == {
        "account": "PT-OUT1",
        "currency": "EUR",
        "transfers": 1,
        "total": "30000.00",
    }


def test_trace_forward_and_backward(synth):
    _, tx, g = synth
    f = trace(tx, "SG-A", graph=g)
    reached = {r["account"]: r["hops"] for r in f["reached"]}
    assert reached["SG-B"] == 2 and all(reached[f"SG-M{i}"] == 1 for i in range(5))
    b = trace(tx, "SG-B", direction="backward", graph=g)
    assert {r["account"]: r["hops"] for r in b["reached"]}["SG-A"] == 2
    # money cannot travel back in time: starting after the transfers reaches nothing
    late = trace(tx, "SG-A", start="2026-02-20", graph=g)
    assert late["reached"] == []
    assert trace(tx, "SG-A", max_transfers=2, graph=g)["truncated"]
    with pytest.raises(InputError):
        trace(tx, "SG-A", direction="sideways", graph=g)


def test_graph_skips_self_transfers():
    tx = Transactions.from_records(
        [
            {"timestamp": 0, "source": "A", "target": "A", "amount": 5},
            {"timestamp": 1, "source": "A", "target": "B", "amount": 5},
        ]
    )
    g = Graph(tx)
    assert g.n_edges == 1 and g.out_degree(0) == 1 and g.in_degree(0) == 0
    assert np.array_equal(tx.row, [1, 2])


def test_fan_message(synth):
    _, _, g = synth
    [a] = [a for a in d.fan_out(g) if a.account == "FO-HUB"]
    assert (
        a.message
        == "FO-HUB sent money to 8 different accounts within 7 days (8 counterparties, 8 transfers in the episode)"
    )
