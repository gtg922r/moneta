"""Moneta — Probabilistic financial modeling engine."""

__version__ = "0.1.0"


class MonetaError(Exception):
    """Base exception for all Moneta errors.

    User-facing errors should subclass this. The CLI catches MonetaError
    and displays a friendly message instead of a traceback.
    """
