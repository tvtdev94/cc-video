"""Colored logging — stage (primary), step (secondary), warn, error.

Colors disable automatically when output isn't a TTY (so log files stay clean).
"""
from __future__ import annotations

import os
import sys
import time

_NO_COLOR = os.environ.get("NO_COLOR") or not sys.stderr.isatty()

_RESET = "" if _NO_COLOR else "\033[0m"
_BOLD_GREEN = "" if _NO_COLOR else "\033[1;32m"
_CYAN = "" if _NO_COLOR else "\033[36m"
_DIM = "" if _NO_COLOR else "\033[2m"
_YELLOW = "" if _NO_COLOR else "\033[33m"
_RED = "" if _NO_COLOR else "\033[1;31m"
_GRAY = "" if _NO_COLOR else "\033[90m"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def stage(msg: str) -> None:
    """Top-level stage marker — bright green, very visible."""
    print(f"{_BOLD_GREEN}[{_ts()}] ▶ {msg}{_RESET}", file=sys.stderr, flush=True)


def step(msg: str) -> None:
    """Sub-stage info — cyan."""
    print(f"{_CYAN}[{_ts()}]   · {msg}{_RESET}", file=sys.stderr, flush=True)


def progress(current: int, total: int, label: str, rate: float | None = None,
             eta_seconds: float | None = None) -> None:
    """Per-item progress — gray, prints every call (caller decides cadence)."""
    pct = 100 * current / total if total else 0
    extras = []
    if rate is not None:
        extras.append(f"{rate:.1f}/s")
    if eta_seconds is not None and eta_seconds > 0:
        extras.append(f"ETA {int(eta_seconds)}s")
    extra = f" ({', '.join(extras)})" if extras else ""
    print(f"{_GRAY}[{_ts()}]     {label}: {current}/{total} ({pct:.0f}%){extra}{_RESET}",
          file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    print(f"{_YELLOW}[{_ts()}]   ! {msg}{_RESET}", file=sys.stderr, flush=True)


def error(msg: str) -> None:
    print(f"{_RED}[{_ts()}]   ✗ {msg}{_RESET}", file=sys.stderr, flush=True)
