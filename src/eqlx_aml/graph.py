"""Time-ordered adjacency lists over a :class:`Transactions` table.

Every account gets its outgoing and incoming transfers sorted by time (ties broken by input row), so a
detector can walk "what did this account send between t and t + window" with a binary search. Transfers
from an account to itself are left out: they move no money between parties.

The lists are plain Python lists, which keeps scalar access in the detectors fast. Memory is roughly
300 bytes per transfer (about 1.5 GB for five million transfers).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

import numpy as np

from .data import Transactions


class Graph:
    def __init__(self, tx: Transactions) -> None:
        self.tx = tx
        n_acc = len(tx.accounts)
        keep = np.flatnonzero(tx.src != tx.dst)
        order = keep[np.lexsort((tx.row[keep], tx.ts[keep]))]  # global time order
        rank = np.full(len(tx), -1, dtype=np.int64)
        rank[order] = np.arange(len(order), dtype=np.int64)

        out = order[np.argsort(tx.src[order], kind="stable")]
        inn = order[np.argsort(tx.dst[order], kind="stable")]
        out_ptr = np.zeros(n_acc + 1, dtype=np.int64)
        out_ptr[1:] = np.cumsum(np.bincount(tx.src[order], minlength=n_acc))
        in_ptr = np.zeros(n_acc + 1, dtype=np.int64)
        in_ptr[1:] = np.cumsum(np.bincount(tx.dst[order], minlength=n_acc))

        self.n_accounts = n_acc
        self.n_edges = len(order)
        self.ts: list[int] = tx.ts.tolist()
        self.src: list[int] = tx.src.tolist()
        self.dst: list[int] = tx.dst.tolist()
        self.rank: list[int] = rank.tolist()
        self.out: list[int] = out.tolist()
        self.out_ts: list[int] = tx.ts[out].tolist()
        self.out_ptr: list[int] = out_ptr.tolist()
        self.inn: list[int] = inn.tolist()
        self.in_ts: list[int] = tx.ts[inn].tolist()
        self.in_ptr: list[int] = in_ptr.tolist()

    def out_range(self, a: int, t_from: int | None = None, t_to: int | None = None) -> tuple[int, int]:
        """Positions ``lo, hi`` in ``self.out`` of account ``a``'s transfers with t_from <= ts <= t_to."""
        lo, hi = self.out_ptr[a], self.out_ptr[a + 1]
        if t_from is not None:
            lo = bisect_left(self.out_ts, t_from, lo, hi)
        if t_to is not None:
            hi = bisect_right(self.out_ts, t_to, lo, hi)
        return lo, hi

    def in_range(self, a: int, t_from: int | None = None, t_to: int | None = None) -> tuple[int, int]:
        lo, hi = self.in_ptr[a], self.in_ptr[a + 1]
        if t_from is not None:
            lo = bisect_left(self.in_ts, t_from, lo, hi)
        if t_to is not None:
            hi = bisect_right(self.in_ts, t_to, lo, hi)
        return lo, hi

    def out_degree(self, a: int) -> int:
        return self.out_ptr[a + 1] - self.out_ptr[a]

    def in_degree(self, a: int) -> int:
        return self.in_ptr[a + 1] - self.in_ptr[a]
