"""Errors raised for bad input. They carry a message meant for the person or agent that gave it."""


class InputError(ValueError):
    """The input cannot be read: a missing column, a malformed amount or date, an unknown account."""
