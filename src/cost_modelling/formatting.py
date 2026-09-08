"""Human-readable formatters — pure, no Streamlit.

Lifted verbatim from src/app/helpers.py, where they lived in a Streamlit module despite
never touching Streamlit. Formatting divergence is what users actually notice and report,
so these are fixtured alongside the maths.
"""

from typing import Any

UNIT_MULTIPLIERS = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "T": 1_000_000_000_000,
}


def fmt_tokens(n: int) -> str:
    """Format a large integer as a human-readable token/parameter count."""
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def fmt_samples(n: int) -> str:
    """Format a sample count as a human-readable string."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_wall_clock_time(wall_clock_days: float) -> str:
    """Format a wall-clock duration (in days) into a human-readable string."""
    wall_clock_hours = wall_clock_days * 24
    if wall_clock_days >= 1:
        return f"{wall_clock_days:.2f} days"
    elif wall_clock_days >= 1 / 24:
        return f"{wall_clock_hours:.2f} hrs"
    else:
        return f"{wall_clock_hours * 60:.1f} min"


def format_gpu_hours(gpu_hours: float) -> str:
    """Format a GPU-hours count into a human-readable string."""
    if gpu_hours >= 1_000_000:
        return f"{gpu_hours / 1_000_000:.2f}M"
    elif gpu_hours >= 1_000:
        return f"{gpu_hours / 1_000:.1f}K"
    else:
        return f"{gpu_hours:.1f}"


def display_value(val: Any) -> str:
    """Format a value for human-readable display in the summary table.

    Note the branch order: bool is a subclass of int, so it must be tested before int.
    """
    if isinstance(val, float):
        if val >= 1e12:
            return f"{val:.2e}"
        if val >= 1e9:
            return f"{val / 1e9:.2f}B"
        if val >= 1e6:
            return f"{val / 1e6:.2f}M"
        if val >= 1e3:
            return f"{val / 1e3:.1f}K"
        return f"{val:.4g}"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, int) and val >= 1_000:
        return f"{val:,}"
    return str(val)
