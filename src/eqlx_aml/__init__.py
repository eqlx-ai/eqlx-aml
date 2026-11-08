"""eqlx-aml: deterministic money-laundering pattern detection for people and AI agents."""

from .data import Transactions, format_amount, parse_amount, read_csv
from .detect import DEFAULT_PATTERNS, DETECTORS, Alert
from .errors import InputError
from .graph import Graph

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_PATTERNS",
    "DETECTORS",
    "Alert",
    "Graph",
    "InputError",
    "Transactions",
    "__version__",
    "format_amount",
    "parse_amount",
    "read_csv",
]
