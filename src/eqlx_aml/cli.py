"""Command line: ``aml scan | account | trace | summary | sample | mcp``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .data import Transactions, parse_amount, parse_timestamp, read_csv
from .detect import DEFAULT_PATTERNS, DETECTORS
from .errors import InputError
from .report import account_profile, scan, trace
from .synthetic import generate


def _print_json(obj: Any) -> None:
    json.dump(obj, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def _value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_settings(items: Sequence[str]) -> dict[str, dict[str, Any]]:
    """``["fan_out.min_counterparties=8", "cycle.max_length=4"]`` -> nested dict."""
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        key, eq, val = item.partition("=")
        pattern, dot, name = key.partition(".")
        if not eq or not dot or not name:
            raise InputError(f"--set expects pattern.parameter=value, got {item!r}")
        out.setdefault(pattern.strip(), {})[name.strip()] = _value(val.strip())
    return out


def _load(args: argparse.Namespace) -> Transactions:
    columns = None
    if args.column:
        columns = {}
        for c in args.column:
            role, eq, name = c.partition("=")
            if not eq:
                raise InputError(f"--column expects role=name, got {c!r}")
            columns[role.strip()] = name.strip()
    tx = read_csv(args.file, columns=columns)
    if args.channel or args.currency or args.start or args.end or args.min_amount:
        tx = tx.select(
            channels=args.channel or None,
            currencies=args.currency or None,
            start=parse_timestamp(args.start) if args.start else None,
            end=parse_timestamp(args.end) if args.end else None,
            min_amount=parse_amount(args.min_amount) if args.min_amount else 0,
        )
    return tx


def _table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, ["-" * w for w in widths], *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())


def _cmd_scan(args: argparse.Namespace) -> int:
    tx = _load(args)
    rep = scan(tx, args.pattern or None, parse_settings(args.set or []))
    if args.json:
        _print_json(rep.to_dict(max_alerts=args.max, max_transactions=args.max_transactions))
        return 0
    s = rep.summary()
    counts = ", ".join(f"{k} {v}" for k, v in s["alerts"].items())
    print(f"{s['transfers']} transfers, {s['accounts']} accounts, {s.get('first', '')} .. {s.get('last', '')}")
    n = s["accounts_flagged"]
    print(f"alerts: {counts} ({n} account{'' if n == 1 else 's'} flagged)")
    shown = rep.alerts[: args.max]
    if shown:
        print()
        _table(
            [[a.pattern, a.account, str(a.size), a.to_dict(tx, 0)["start"][:16], a.message] for a in shown],
            ["pattern", "account", "size", "start", "message"],
        )
    if len(rep.alerts) > len(shown):
        print(f"\n... {len(rep.alerts) - len(shown)} more (use --max or --json)")
    return 0


def _cmd_account(args: argparse.Namespace) -> int:
    _print_json(account_profile(_load(args), args.account))
    return 0


def _cmd_trace(args: argparse.Namespace) -> int:
    tx = _load(args)
    res = trace(
        tx,
        args.account,
        "backward" if args.backward else "forward",
        max_hops=args.hops,
        window_days=args.days,
        start=args.at,
    )
    _print_json(res)
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    _print_json(_load(args).summary())
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    data = generate(seed=args.seed)
    Path(args.out).write_text(data.to_csv(), encoding="utf-8")
    print(f"wrote {len(data.records)} synthetic transfers to {args.out}; planted patterns:")
    for p in data.planted:
        print(f"  {p.pattern:15} {p.account:8} rows {p.rows}")
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from .mcp_server import serve
    except ImportError:
        print("the MCP server needs: pip install 'eqlx-aml[mcp]'", file=sys.stderr)
        return 2
    serve(Path(args.root), max_rows=args.max_rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aml", description="Deterministic money-laundering pattern detection on transfer files (CSV)."
    )
    p.add_argument("--version", action="version", version=f"eqlx-aml {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("file", help="CSV file of transfers (generic layout or IBM AML)")
        sp.add_argument("--column", action="append", metavar="ROLE=NAME", help="name a column, e.g. timestamp=Date")
        sp.add_argument("--channel", action="append", help="keep only this channel (repeatable)")
        sp.add_argument("--currency", action="append", help="keep only this currency (repeatable)")
        sp.add_argument("--start", help="keep transfers at or after this time")
        sp.add_argument("--end", help="keep transfers before this time")
        sp.add_argument("--min-amount", help="keep transfers of at least this amount")

    s = sub.add_parser("scan", help="run pattern detectors")
    data_args(s)
    s.add_argument(
        "--pattern",
        action="append",
        choices=list(DETECTORS),
        help=f"pattern to run (repeatable; default: {', '.join(DEFAULT_PATTERNS)})",
    )
    s.add_argument("--set", action="append", metavar="PATTERN.PARAM=VALUE", help="e.g. fan_out.min_counterparties=8")
    s.add_argument("--max", type=int, default=50, help="alerts to show (default 50)")
    s.add_argument("--max-transactions", type=int, default=50, help="transfers listed per alert in --json")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_cmd_scan)

    a = sub.add_parser("account", help="profile one account")
    data_args(a)
    a.add_argument("account")
    a.set_defaults(fn=_cmd_account)

    t = sub.add_parser("trace", help="follow money from (or to) one account")
    data_args(t)
    t.add_argument("account")
    t.add_argument("--backward", action="store_true", help="where the money came from")
    t.add_argument("--hops", type=int, default=3)
    t.add_argument("--days", type=float, default=7)
    t.add_argument("--at", help="start time (default: first outgoing / last incoming transfer)")
    t.set_defaults(fn=_cmd_trace)

    m = sub.add_parser("summary", help="what is in a file")
    data_args(m)
    m.set_defaults(fn=_cmd_summary)

    sm = sub.add_parser("sample", help="write a synthetic file with planted patterns")
    sm.add_argument("out")
    sm.add_argument("--seed", type=int, default=7)
    sm.set_defaults(fn=_cmd_sample)

    mc = sub.add_parser("mcp", help="run the read-only MCP server (stdio)")
    mc.add_argument("--root", required=True, help="folder the server may read")
    mc.add_argument("--max-rows", type=int, default=20_000_000, help="refuse files with more rows")
    mc.set_defaults(fn=_cmd_mcp)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.fn(args)
    except (InputError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
