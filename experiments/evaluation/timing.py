"""Small helpers for consistent wall-clock measurements."""

from time import perf_counter


def start_timer() -> float:
    """Return a monotonic timestamp."""
    return perf_counter()


def elapsed_seconds(started_at: float) -> float:
    """Return seconds elapsed since ``started_at``."""
    return perf_counter() - started_at
