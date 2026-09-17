"""GPU energy metering (joules) around a block of work, via NVML like the inference pod does.

Prefers the cumulative energy counter (``nvmlDeviceGetTotalEnergyConsumption``); falls back to
sampling instantaneous power on a thread.  Returns ``None`` where neither is available (CPU boxes,
or GB10 if NVML does not expose power for the integrated GPU), so callers can always add it to a
report without failing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class EnergyReading:
    joules: float | None
    seconds: float
    avg_watts: float | None
    method: str

    def to_dict(self) -> dict:
        return {"joules": self.joules, "seconds": self.seconds, "avg_watts": self.avg_watts, "method": self.method}


class EnergyMeter:
    """``with EnergyMeter() as m: ...; m.reading``"""

    def __init__(self, device_index: int = 0, sample_interval: float = 0.05):
        self.idx = device_index
        self.interval = sample_interval
        self.reading: EnergyReading | None = None
        self._nvml = None
        self._handle = None
        self._mode = "none"
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.idx)
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                self._mode = "nvml_energy_counter"
            except Exception:  # noqa: BLE001
                try:
                    pynvml.nvmlDeviceGetPowerUsage(self._handle)
                    self._mode = "nvml_power_sampling"
                except Exception:  # noqa: BLE001
                    self._mode = "none"
        except Exception:  # noqa: BLE001
            self._nvml = None
            self._mode = "none"

    @property
    def available(self) -> bool:
        return self._mode != "none"

    def _sampler(self):
        while not self._stop.is_set():
            try:
                w = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self._samples.append((time.perf_counter(), w))
            except Exception:  # noqa: BLE001
                break
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t0 = time.perf_counter()
        if self._mode == "nvml_energy_counter":
            self._e0 = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        elif self._mode == "nvml_power_sampling":
            self._samples = []
            self._stop.clear()
            self._thread = threading.Thread(target=self._sampler, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        secs = time.perf_counter() - self._t0
        joules = None
        if self._mode == "nvml_energy_counter":
            joules = (self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle) - self._e0) / 1000.0
        elif self._mode == "nvml_power_sampling":
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=1.0)
            if len(self._samples) >= 2:
                joules = 0.0
                for (t0, w0), (t1, w1) in zip(self._samples, self._samples[1:]):
                    joules += 0.5 * (w0 + w1) * (t1 - t0)
            elif self._samples:
                joules = self._samples[0][1] * secs
        self.reading = EnergyReading(joules, secs, (joules / secs if joules is not None and secs > 0 else None), self._mode)
        return False
