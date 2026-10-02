"""Cooperative thermal governor for small machines that cannot sustain full GPU load.

On the DGX Spark the calibration loop takes the SoC from 39 C to 85 C in about 30 seconds and the
machine powers off without a log entry when it is left running flat out for a few minutes.  The die
also cools fast (roughly 28 C in 5 s), so pausing at natural boundaries (between calibration batches,
ridge solves, evaluation sequences, benchmark trials) keeps the temperature bounded at the cost of
wall-clock time.

The governor turns itself on when the GPU is the Spark's GB10 and is off everywhere else, unless
the environment (or :func:`configure`) says otherwise:

    KVT_THERMAL_PAUSE_C    pause when the hottest thermal zone is at or above this (GB10 default 68);
                           ``off`` disables the governor even on a GB10
    KVT_THERMAL_RESUME_C   resume when it is at or below this (default: pause - 10)
    KVT_THERMAL_MAX_WAIT_S give up (raise ThermalAbort) if it has not cooled after this long (default 600)

Long-running loops call :func:`checkpoint`; it returns immediately when the governor is off or the
machine is cool.  It only reads ``/sys/class/thermal`` and sleeps; it changes no system setting.
"""
from __future__ import annotations

import glob
import os
import time
from typing import Callable


class ThermalAbort(RuntimeError):
    """The machine did not cool down within the allowed wait."""


def read_hottest_zone() -> int | None:
    """Hottest ``/sys/class/thermal`` zone in whole degrees C, or None when no zone is readable."""
    best = None
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            t = int(open(z).read()) // 1000
        except (OSError, ValueError):
            continue
        best = t if best is None else max(best, t)
    return best


class ThermalGovernor:
    def __init__(self, pause_at: float, resume_at: float | None = None, max_wait: float = 600.0, poll: float = 0.5,
                 read: Callable[[], int | None] = read_hottest_zone, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] | None = None):
        self.pause_at = float(pause_at)
        self.resume_at = float(resume_at) if resume_at is not None else self.pause_at - 10.0
        if self.resume_at >= self.pause_at:
            raise ValueError("resume temperature must be below the pause temperature")
        self.max_wait, self.poll, self.read, self.sleep = float(max_wait), float(poll), read, sleep
        self.log = log or (lambda msg: print(msg, flush=True))
        self.pauses = 0
        self.paused_seconds = 0.0
        self.peak = None

    def checkpoint(self) -> float:
        """Pause while the machine is hot.  Returns the seconds spent waiting (0.0 when it was cool)."""
        t = self.read()
        if t is None:
            return 0.0
        self.peak = t if self.peak is None else max(self.peak, t)
        if t < self.pause_at:
            return 0.0
        start = time.monotonic()
        waited, polls = 0.0, 0
        while t is not None and t > self.resume_at:
            if waited >= self.max_wait:
                raise ThermalAbort(f"hottest sensor still at {t} C after waiting {waited:.0f} s (resume at {self.resume_at:.0f} C)")
            self.sleep(self.poll)
            polls += 1
            waited = max(time.monotonic() - start, polls * self.poll)   # the second term covers an injected sleep
            t = self.read()
        self.pauses += 1
        self.paused_seconds += waited
        if self.pauses == 1 or self.pauses % 50 == 0:
            self.log(f"[thermal] pause {self.pauses}: peak {self.peak} C, {self.paused_seconds:.0f}s paused in total "
                     f"(pause at {self.pause_at:.0f} C, resume at {self.resume_at:.0f} C)")
        return waited

    def summary(self) -> dict:
        return {"pause_at_c": self.pause_at, "resume_at_c": self.resume_at, "pauses": self.pauses,
                "paused_seconds": round(self.paused_seconds, 1), "peak_c": self.peak}


DEFAULT_PAUSE_C, DEFAULT_RESUME_C = 68.0, 58.0     # held a DGX Spark at <= 81 C over a 3.5 h run
_OFF = ("off", "none", "no", "false", "0")

_governor: ThermalGovernor | None = None
_env_checked = False


def _machine_needs_governor() -> bool:
    """True on hardware known to power off under sustained GPU load: the GB10 in the DGX Spark."""
    try:
        import torch
        return bool(torch.cuda.is_available()) and "gb10" in torch.cuda.get_device_name(0).lower()
    except Exception:  # noqa: BLE001 - never let detection break a run
        return False


def configure(pause_at: float | None, resume_at: float | None = None, **kwargs) -> ThermalGovernor | None:
    """Install (or, with ``pause_at=None``, remove) the process-wide governor."""
    global _governor, _env_checked
    _env_checked = True
    _governor = None if pause_at is None else ThermalGovernor(pause_at, resume_at, **kwargs)
    return _governor


def governor() -> ThermalGovernor | None:
    """The process-wide governor, created from the environment on first use."""
    global _env_checked
    if not _env_checked:
        _env_checked = True
        pause = os.environ.get("KVT_THERMAL_PAUSE_C", "").strip().lower()
        max_wait = float(os.environ.get("KVT_THERMAL_MAX_WAIT_S", "600"))
        if pause in _OFF:
            pass
        elif pause:
            resume = os.environ.get("KVT_THERMAL_RESUME_C")
            configure(float(pause), float(resume) if resume else None, max_wait=max_wait)
        elif _machine_needs_governor():
            configure(DEFAULT_PAUSE_C, DEFAULT_RESUME_C, max_wait=max_wait)
            print(f"[thermal] DGX Spark (GB10) detected: governor on, pause at {DEFAULT_PAUSE_C:.0f} C, resume at "
                  f"{DEFAULT_RESUME_C:.0f} C. Set KVT_THERMAL_PAUSE_C=off to disable (the machine powers off under "
                  f"sustained ungoverned load).", flush=True)
    return _governor


def checkpoint() -> float:
    """Call between units of GPU work.  A no-op unless a governor is configured."""
    g = governor()
    return g.checkpoint() if g is not None else 0.0


def enabled() -> bool:
    return governor() is not None


def summary() -> dict | None:
    g = governor()
    return g.summary() if g is not None else None
