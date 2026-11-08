# eqlx-aml

[![CI](https://github.com/eqlx-ai/eqlx-aml/actions/workflows/ci.yml/badge.svg)](https://github.com/eqlx-ai/eqlx-aml/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Find **money-laundering patterns** in transfer data: fan-out, fan-in, gather-scatter, scatter-gather,
loops, pass-through and structuring. Every alert is an exact, written rule applied to the transfers you
give it, and names the accounts, the time span and the rows that make it up. An optional **MCP server**
lets AI agents such as Claude run the same detectors, read-only.

Built by [eqlx](https://eqlx.ai). Part of [eqlx.org](https://eqlx.org), deterministic tools for AI
agents in credit and finance.

> Status: alpha (0.1.0). Each detector matches a brute-force enumeration of its definition on
> thousands of random graphs, and is measured on the public IBM AML synthetic data set (5 million
> transfers). See [How it is verified](#how-it-is-verified) and [LIMITS.md](LIMITS.md).

## Why

Transaction-monitoring teams and the AI agents now helping them need answers they can check: *did money
go round in a loop? which accounts fed this one?* A language model should not invent that answer, and a
black-box score cannot be quoted in a report. This library computes it: same input, same output, with
the rows to prove it. The agent decides what to look at; the tool does the counting.

## Install

```bash
pip install "eqlx-aml[mcp]"   # once released on PyPI; until then:
pip install "eqlx-aml[mcp] @ git+https://github.com/eqlx-ai/eqlx-aml"
```

Python 3.10+. The core needs only NumPy; `[mcp]` adds the MCP server.

## Command line

```bash
aml sample demo.csv                       # synthetic transfers with planted patterns
aml scan demo.csv                         # all default patterns
aml scan demo.csv --pattern cycle --set cycle.max_length=4
aml scan demo.csv --pattern structuring --set structuring.threshold=10000 --channel Cash
aml account demo.csv FI-HUB               # one account's activity
aml trace demo.csv SG-B --backward        # where the money came from, hop by hop
```

```text
$ aml scan demo.csv --pattern scatter_gather --pattern cycle --max 4
4034 transfers, 452 accounts, 2026-01-05T00:02:37Z .. 2026-02-01T23:26:12Z
alerts: scatter_gather 2, cycle 20 (22 accounts flagged)

pattern         account   size  start             message
--------------  --------  ----  ----------------  ---------------------------------------------------------------------------------------------------------------------------------------
scatter_gather  SG-A      5     2026-01-17T00:00  SG-A sent money through 5 different accounts that each passed it on to SG-B within 7 days
scatter_gather  ACC-0269  3     2026-01-25T08:04  ACC-0269 sent money through 3 different accounts that each passed it on to ACC-0258 within 7 days
cycle           ACC-0094  6     2026-01-20T06:59  money went round ACC-0094 -> ACC-0313 -> ACC-0285 -> ACC-0372 -> ACC-0238 -> ACC-0385 -> ACC-0094 (6 transfers, within 7 days), 2 times
cycle           ACC-0040  5     2026-01-08T06:50  money went round ACC-0040 -> ACC-0374 -> ACC-0151 -> ACC-0229 -> ACC-0384 -> ACC-0040 (5 transfers, within 7 days), 2 times

... 18 more (use --max or --json)

$ aml scan demo.csv --pattern structuring --set structuring.threshold=10000 --channel Cash
5 transfers, 452 accounts, 2026-01-27T11:00:00Z .. 2026-01-31T11:00:00Z
alerts: structuring 1 (1 account flagged)

pattern      account  size  start             message
-----------  -------  ----  ----------------  ------------------------------------------------------------------------------------------------------
structuring  ST-ACC   5     2026-01-27T11:00  ST-ACC received 5 transfers within 7 days each between 9000.00 and 10000.00 (just under the threshold)
```

`SG-A` is planted; the other alerts are ordinary accounts that happen to form the same shapes, as in
real data. Ranking, parameters and filters decide what is worth a look.

Input is a CSV with a header. Columns for time, source account, target account and amount are found
by common names (`timestamp`/`date`, `source`/`from`/`sender`, `target`/`to`/`receiver`, `amount`),
plus optional `currency`, `channel` and `id`; name them with `--column timestamp="Booking Date"`. The
IBM AML layout is recognised automatically.

## Python

```python
import eqlx_aml as aml
from eqlx_aml.report import scan, account_profile, trace

tx = aml.read_csv("transfers.csv")
rep = scan(tx, ["fan_in", "cycle"], {"cycle": {"max_length": 4, "window_days": 3}})
for alert in rep.alerts[:10]:
    print(alert.pattern, alert.account, alert.message)
alert.to_dict(tx)  # accounts, start, end, total per currency, rows, the transfers themselves

account_profile(tx, "FI-HUB")
trace(tx, "SG-A", direction="forward", max_hops=3, window_days=7)
```

Amounts are exact integers in millionths of a unit, never floats.

## Patterns

| Pattern | Rule (all within `window_days`, default 7) | Main parameters |
|---|---|---|
| `fan_out` | one account pays at least *k* different accounts | `min_counterparties` = 5 |
| `fan_in` | one account is paid by at least *k* different accounts | `min_counterparties` = 5 |
| `gather_scatter` | one account is paid by at least *a* accounts and pays at least *b* accounts | `min_senders` = 3, `min_receivers` = 3 |
| `scatter_gather` | A pays at least *k* accounts, each of which then pays the same B | `min_intermediaries` = 3, `max_degree` = 50 |
| `cycle` | money returns to its first sender, each transfer no earlier than the last, no account twice | `max_length` = 6, `max_degree` = 50 |
| `pass_through` | an account receives an amount and sends out 90-100 % of it, same currency (window 1 day) | `min_ratio`, `max_ratio` |
| `structuring` | at least *n* transfers just under a reporting threshold you give | `threshold`, `margin` = 0.1, `min_count` = 3 |

Fans, gather-scatter and loops follow the typologies in FATF guidance and in the IBM AML data
(Altman et al., 2023); pass-through and structuring are classic monitoring rules. `max_degree` keeps
the two path searches away from busy accounts (businesses, processors); see [LIMITS.md](LIMITS.md).

## MCP server (for Claude and other agents)

Seven read-only tools: `list_files`, `summarize`, `scan`, `account_profile`, `trace_funds`,
`get_transfers` and `list_patterns`. The server reads only CSV files inside the folder you give it;
`..`, absolute paths and symlinks that point outside are refused. A loaded file stays in memory, so
follow-up questions are fast.

Claude Code:

```bash
claude mcp add aml -- aml mcp --root /path/to/transfers
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "aml": { "command": "aml", "args": ["mcp", "--root", "/path/to/transfers"] }
  }
}
```

Then ask, for example: *"Scan transfers.csv for laundering patterns, investigate the three strongest
alerts and quote the rows."*

## How it is verified

**1. Against the definitions.** `tests/test_bruteforce.py` runs every detector on hundreds of small
random transfer graphs (with many equal timestamps) and compares the result with a direct enumeration
of the written rule: every window, every two-step path, every permutation of transfers for loops.
They agree exactly. Other tests check that results do not change when the input rows are shuffled or
Python's hash seed changes, and that each planted pattern in the synthetic data is found with its rows.

**2. On the public IBM AML data** (HI-Small: 5,078,345 transfers, 515,088 accounts, 370 planted
laundering attempts; Altman et al., 2023). For each attempt of a type a detector targets: did that
detector raise an alert on the attempt's key account (same target for scatter-gather, same accounts
for loops) that includes its transfers? *Big enough* counts only attempts whose planted size meets the
rule (e.g. at least 5 counterparties for fan-out at the default). *On laundering accounts* is the share
of alerts about an account that takes part in any laundering-labelled transfer; 1.2 % of all accounts
do, so *lift* is that share divided by 1.2 %.

| Pattern | Parameters | Attempts found | Of those big enough | Alerts | On laundering accounts | Lift |
|---|---|---|---|---|---|---|
| fan_out | defaults (k=5) | 27/48 | 27/27 | 26,097 | 2.1 % | 1.7x |
| fan_in | defaults (k=5) | 35/40 | 26/26 | 12,431 | 6.2 % | 5.0x |
| fan_in | k=8 | 24/40 | 22/22 | 3,987 | 9.5 % | 7.7x |
| gather_scatter | defaults (3 and 3) | 42/51 | 27/27 | 12,192 | 5.0 % | 4.1x |
| gather_scatter | 5 and 5 | 27/51 | 22/22 | 491 | 16.1 % | 13.0x |
| scatter_gather | defaults (k=3) | 29/44 | 29/30 | 190 | 20.0 % | 16.2x |
| scatter_gather | k=5 | 25/44 | 25/26 | 30 | 86.7 % | 70.2x |
| cycle | defaults (up to 6) | 31/54 | 31/34 | 3,585 | 10.6 % | 8.6x |
| cycle | up to 8 | 36/54 | 36/43 | 3,610 | 11.1 % | 9.0x |
| pass_through | defaults | (no IBM type) | | 14,542 | 2.4 % | 2.0x |

What this shows:

- **The detectors find what their rule describes.** Of the planted loops of up to six transfers, 31
  of 34 are found; the three misses are one whose transfers are out of time order and two that pass
  through accounts with more than 50 transfers that week (the `max_degree` limit). The misses in the
  "Attempts found" column are attempts too small for the parameters, too long, or routed through hubs.
- **Shape alone is weak evidence for fans, strong for paths.** A fan-out is what every payroll looks
  like (1.7x lift); money split across several accounts and gathered again is not (16x at the default,
  70x when five or more intermediaries are required). Raising thresholds trades recall for precision;
  the full grid is in `benchmarks/results/`.
- **Not covered:** the data set's STACK, BIPARTITE and RANDOM attempts (133 of 370) have no detector.

All default patterns together take about 2.5 minutes and 2 GB on one core for the 5 million transfers
(`benchmarks/ibm_aml.py` reproduces the table from your own copy of the data).

**3. With an agent.** A Claude Code session given only the MCP server and the synthetic demo file was
asked to scan it, investigate the strongest alerts and separate suspicious from ordinary activity. In
23 steps (70 seconds) it found all seven planted patterns and quoted their rows. It rated five as
suspicious, the planted fan-out and fan-in as possibly ordinary payments worth a question, and the
background accounts' chance patterns as ordinary trade, and said the findings are signals for a human
investigator, not proof.

## Safety and privacy

- Runs locally. No network access, no telemetry. The MCP server cannot write files.
- Tests and examples use synthetic data only. `aml sample` writes the same kind of made-up file.
- The IBM data is not included; the benchmark script reads your own copy.

See [LIMITS.md](LIMITS.md) for what it does not do.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## References

- E. Altman, J. Blanuša, L. von Niederhäusern, B. Egressy, A. Anghel, K. Atasu. *Realistic Synthetic
  Financial Transactions for Anti-Money Laundering Models.* NeurIPS 2023 Datasets and Benchmarks.
  Data: [IBM/AML-Data](https://github.com/IBM/AML-Data) (CDLA-Sharing-1.0).
- FATF. *Money Laundering and Terrorist Financing Typologies* reports; *Professional Money Laundering* (2018).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
