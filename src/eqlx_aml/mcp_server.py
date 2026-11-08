"""MCP server: lets an AI agent look for money-laundering patterns in transfer files.

Read-only by construction: the tools only read CSV files, and only inside the ``--root`` folder given
at start-up (paths are resolved, so ``..`` and symlinks cannot leave it). Nothing is written and nothing
is sent anywhere; results go back to the agent that asked. A loaded file is kept in memory (two at most)
so follow-up questions on the same file are fast.

Start it with ``aml mcp --root /path/to/folder`` (needs ``pip install 'eqlx-aml[mcp]'``).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import __version__
from .data import Transactions, parse_amount, parse_timestamp, read_csv
from .detect import DEFAULT_PATTERNS, DETECTORS
from .errors import InputError
from .graph import Graph
from .report import account_profile as _profile
from .report import scan as _scan
from .report import trace as _trace

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
MAX_ALERTS = 200
MAX_ROWS_RETURNED = 200
CACHE_SIZE = 2


class OutsideRootError(ValueError):
    pass


def _resolve(root: Path, path: str) -> Path:
    p = (root / path).resolve()
    if p != root and root not in p.parents:
        raise OutsideRootError(f"{path!r} is outside the allowed folder")
    if not p.exists():
        raise FileNotFoundError(path)
    return p


class _Cache:
    def __init__(self, root: Path, max_rows: int) -> None:
        self.root = root
        self.max_rows = max_rows
        self.items: OrderedDict[tuple[str, int, int], tuple[Transactions, Graph]] = OrderedDict()

    def get(self, path: str) -> tuple[Transactions, Graph]:
        p = _resolve(self.root, path)
        if not p.is_file():
            raise InputError(f"{path!r} is not a file")
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        tx = read_csv(p, max_rows=self.max_rows)
        g = Graph(tx)
        self.items[key] = (tx, g)
        while len(self.items) > CACHE_SIZE:
            self.items.popitem(last=False)
        return tx, g


def build_server(root: Path, max_rows: int = 20_000_000) -> MCPServer:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    cache = _Cache(root, max_rows)
    server = MCPServer(
        "eqlx-aml",
        version=__version__,
        instructions=(
            "Finds money-laundering patterns in CSV files of transfers inside one local folder. Paths are "
            "relative to that folder. Start with `summarize`, then `scan`; investigate flagged accounts "
            "with `account_profile` and `trace_funds`, and quote rows with `get_transfers`. Results are "
            "deterministic signals for a human investigator, not proof of a crime: say which pattern, "
            "which rows and which parameters produced each finding."
        ),
    )

    def view(
        path: str,
        channels: list[str] | None,
        currencies: list[str] | None,
        start: str | None,
        end: str | None,
        min_amount: str | None,
    ) -> tuple[Transactions, Graph | None]:
        tx, g = cache.get(path)
        if channels or currencies or start or end or min_amount:
            sub = tx.select(
                channels=channels or None,
                currencies=currencies or None,
                start=parse_timestamp(start) if start else None,
                end=parse_timestamp(end) if end else None,
                min_amount=parse_amount(min_amount) if min_amount else 0,
            )
            return sub, None
        return tx, g

    @server.tool(annotations=READ_ONLY)
    def list_files(path: str = ".") -> list[dict[str, Any]]:
        """List the CSV files under a path in the allowed folder, with their sizes."""
        p = _resolve(root, path)
        files = [p] if p.is_file() else sorted(p.rglob("*"))
        out = []
        for f in files:
            if f.suffix.lower() != ".csv" or not f.is_file():
                continue
            real = f.resolve()
            if root in real.parents:
                out.append({"path": str(real.relative_to(root)), "bytes": real.stat().st_size})
        return out

    @server.tool(annotations=READ_ONLY)
    def summarize(path: str) -> dict[str, Any]:
        """What a transfer file contains: number of transfers and accounts, time span, currencies and
        channels. The first call on a large file takes a while (about 20 s per 5 million rows)."""
        tx, _ = cache.get(path)
        return tx.summary()

    @server.tool(annotations=READ_ONLY)
    def scan(
        path: str,
        patterns: list[str] | None = None,
        params: dict[str, dict[str, Any]] | None = None,
        channels: list[str] | None = None,
        currencies: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        min_amount: str | None = None,
        max_alerts: int = 50,
        max_transfers_per_alert: int = 20,
    ) -> dict[str, Any]:
        """Run laundering-pattern detectors and return alerts, largest first within each pattern.

        patterns: any of fan_out, fan_in, gather_scatter, scatter_gather, cycle, pass_through, structuring
        (default: all but structuring, which needs a threshold).
        params: per pattern, e.g. {"fan_out": {"min_counterparties": 8, "window_days": 3},
        "cycle": {"max_length": 4}, "structuring": {"threshold": 10000, "side": "in"}}.
        channels / currencies / start / end / min_amount filter the transfers first.
        Each alert lists the accounts, the time span, the rows that make it up and a one-line reason.
        """
        tx, g = view(path, channels, currencies, start, end, min_amount)
        rep = _scan(tx, patterns or None, params or None, graph=g)
        return rep.to_dict(
            max_alerts=max(0, min(max_alerts, MAX_ALERTS)),
            max_transactions=max(0, min(max_transfers_per_alert, MAX_ROWS_RETURNED)),
        )

    @server.tool(annotations=READ_ONLY)
    def account_profile(path: str, account: str) -> dict[str, Any]:
        """Everything one account did: transfers in and out, totals per currency, first and last
        activity, top counterparties and channels."""
        tx, g = cache.get(path)
        return _profile(tx, account, graph=g)

    @server.tool(annotations=READ_ONLY)
    def trace_funds(
        path: str,
        account: str,
        direction: Literal["forward", "backward"] = "forward",
        max_hops: int = 3,
        window_days: float = 7,
        start: str | None = None,
        max_transfers: int = 200,
    ) -> dict[str, Any]:
        """Follow money from an account (forward) or to it (backward), hop by hop, only along transfers
        that happen after the money arrived (forward) or before it left (backward)."""
        tx, g = cache.get(path)
        return _trace(
            tx,
            account,
            direction,
            max_hops=max_hops,
            window_days=window_days,
            start=start,
            max_transfers=max(1, min(max_transfers, MAX_ROWS_RETURNED)),
            graph=g,
        )

    @server.tool(annotations=READ_ONLY)
    def get_transfers(path: str, rows: list[int]) -> dict[str, Any]:
        """The transfers at these rows (1 = first line after the header), to quote them as evidence."""
        tx, _ = cache.get(path)
        wanted = list(dict.fromkeys(rows))[:MAX_ROWS_RETURNED]
        found, missing = [], []
        for r in wanted:
            i = int(np.searchsorted(tx.row, r))  # rows are increasing in a file as read
            if i < len(tx) and int(tx.row[i]) == r:
                found.append(tx.record(i))
            else:
                missing.append(r)
        return {"transfers": found, "missing": missing}

    @server.tool(annotations=READ_ONLY)
    def list_patterns() -> dict[str, Any]:
        """The patterns this server can look for, with their default parameters."""
        import inspect

        out = {}
        for name, fn in DETECTORS.items():
            sig = inspect.signature(fn)
            out[name] = {
                "description": (fn.__doc__ or "").strip().split("\n\n")[0].replace("\n", " "),
                "parameters": {
                    k: (None if v.default is inspect.Parameter.empty else v.default)
                    for k, v in list(sig.parameters.items())[1:]
                },
                "default": name in DEFAULT_PATTERNS,
            }
        return out

    return server


def serve(root: Path, max_rows: int = 20_000_000) -> None:
    build_server(root, max_rows).run("stdio")
