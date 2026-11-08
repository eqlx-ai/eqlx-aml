import asyncio
import json
import os

import pytest

from eqlx_aml.cli import main, parse_settings
from eqlx_aml.errors import InputError
from eqlx_aml.synthetic import generate


@pytest.fixture
def sample(tmp_path):
    p = tmp_path / "data" / "transfers.csv"
    p.parent.mkdir()
    p.write_text(generate(seed=7).to_csv())
    return p


def test_parse_settings():
    assert parse_settings(["fan_out.min_counterparties=8", "cycle.window_days=2.5", "structuring.side=in"]) == {
        "fan_out": {"min_counterparties": 8},
        "cycle": {"window_days": 2.5},
        "structuring": {"side": "in"},
    }
    with pytest.raises(InputError):
        parse_settings(["fan_out=8"])


def test_cli_scan_text(sample, capsys):
    assert main(["scan", str(sample), "--pattern", "cycle"]) == 0
    out = capsys.readouterr().out
    assert "alerts: cycle" in out and "CY-A -> CY-B -> CY-C -> CY-D -> CY-A" in out


def test_cli_scan_json_with_settings_and_filters(sample, capsys):
    args = ["scan", str(sample), "--json", "--pattern", "structuring", "--set", "structuring.threshold=10000"]
    assert main([*args, "--channel", "Cash"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [a["account"] for a in data["alerts"]] == ["ST-ACC"]
    assert data["summary"]["params"] == {"structuring": {"threshold": 10000}}


def test_cli_account_trace_summary_sample(sample, tmp_path, capsys):
    assert main(["account", str(sample), "FO-HUB"]) == 0
    assert json.loads(capsys.readouterr().out)["outgoing"]["counterparties"] == 8
    assert main(["trace", str(sample), "SG-B", "--backward", "--hops", "2"]) == 0
    assert any(r["account"] == "SG-A" for r in json.loads(capsys.readouterr().out)["reached"])
    assert main(["summary", str(sample)]) == 0
    assert json.loads(capsys.readouterr().out)["currencies"] == {"EUR": len(generate(seed=7).records)}
    out = tmp_path / "s.csv"
    assert main(["sample", str(out), "--seed", "7"]) == 0
    assert out.read_text() == sample.read_text()


def test_cli_errors(sample, capsys):
    assert main(["account", str(sample), "NOPE"]) == 2
    assert "unknown account" in capsys.readouterr().err
    assert main(["scan", str(sample) + ".missing"]) == 2


# --- MCP ----------------------------------------------------------------------------------------------

mcp = pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from eqlx_aml.mcp_server import build_server  # noqa: E402


@pytest.fixture
def root(sample, tmp_path):
    secret = tmp_path / "outside.csv"
    secret.write_text(sample.read_text())
    os.symlink(secret, sample.parent / "link_out.csv")
    return sample.parent


def call(server, name, args):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(name, args)

    return asyncio.run(go())


def test_tools_are_read_only(root):
    async def go():
        async with Client(build_server(root)) as c:
            return await c.list_tools()

    tools = {t.name: t for t in asyncio.run(go()).tools}
    assert set(tools) == {
        "list_files",
        "summarize",
        "scan",
        "account_profile",
        "trace_funds",
        "get_transfers",
        "list_patterns",
    }
    assert all(t.annotations.read_only_hint for t in tools.values())


def test_mcp_flow(root):
    server = build_server(root)
    files = call(server, "list_files", {}).structured_content["result"]
    assert [f["path"] for f in files] == ["transfers.csv"]  # the symlink pointing outside is not listed
    s = call(server, "summarize", {"path": "transfers.csv"}).structured_content
    assert s["transfers"] == len(generate(seed=7).records)
    res = call(server, "scan", {"path": "transfers.csv", "patterns": ["scatter_gather"], "max_alerts": 5})
    data = res.structured_content
    assert data["alerts"][0]["account"] == "SG-A" and data["alerts"][0]["detail"]["target"] == "SG-B"
    rows = data["alerts"][0]["rows"][:3]
    got = call(server, "get_transfers", {"path": "transfers.csv", "rows": [*rows, 10**9]}).structured_content
    assert [t["row"] for t in got["transfers"]] == rows and got["missing"] == [10**9]
    prof = call(server, "account_profile", {"path": "transfers.csv", "account": "SG-B"}).structured_content
    assert prof["incoming"]["counterparties"] >= 5
    tr = call(server, "trace_funds", {"path": "transfers.csv", "account": "SG-A", "max_hops": 2}).structured_content
    assert any(r["account"] == "SG-B" for r in tr["reached"])
    pats = call(server, "list_patterns", {}).structured_content
    assert pats["structuring"]["default"] is False and pats["cycle"]["parameters"]["max_length"] == 6


def test_mcp_scan_with_params_and_filters(root):
    res = call(
        build_server(root),
        "scan",
        {
            "path": "transfers.csv",
            "patterns": ["structuring"],
            "params": {"structuring": {"threshold": 10000, "side": "in"}},
            "channels": ["Cash"],
        },
    )
    assert [a["account"] for a in res.structured_content["alerts"]] == ["ST-ACC"]


@pytest.mark.parametrize("path", ["..", "../outside.csv", "/etc/passwd", "link_out.csv"])
def test_mcp_refuses_paths_outside_root(root, path):
    res = call(build_server(root), "summarize", {"path": path})
    assert res.is_error


def test_mcp_reports_bad_input(root):
    res = call(build_server(root), "account_profile", {"path": "transfers.csv", "account": "NOPE"})
    assert res.is_error


def test_output_does_not_depend_on_hash_seed(sample):
    """Python randomises string hashing per process; no result may depend on set or dict hash order."""
    import subprocess
    import sys

    outs = set()
    for seed in ("0", "1", "12345"):
        code = "import sys; from eqlx_aml.cli import main; sys.exit(main(sys.argv[1:]))"
        args = ["scan", str(sample), "--json", "--max", "100000", "--max-transactions", "100000"]
        for p in ("fan_out", "fan_in", "gather_scatter", "scatter_gather", "cycle", "pass_through"):
            args += ["--pattern", p]
        env = {**os.environ, "PYTHONHASHSEED": seed}
        res = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, env=env, check=True)
        outs.add(res.stdout)
    assert len(outs) == 1
