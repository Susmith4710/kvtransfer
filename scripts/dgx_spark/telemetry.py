#!/usr/bin/env python
"""Thermal / power / memory sampler for the DGX Spark, safe to run next to a heavy job.

Reads only this machine's own sensors (``/sys/class/thermal``, ``/proc/meminfo`` and NVML through
``pynvml``) and appends one CSV row per interval, fsync'ing every row so the trace survives a hang or
a power loss.  It talks to no service and changes nothing.

Usage: telemetry.py <out.csv> [interval_s=1] [max_seconds=3600]
Stops at ``max_seconds`` or when ``<out.csv>.stop`` exists.
"""
from __future__ import annotations

import glob
import os
import sys
import time

# ACPI thermal zones in boot order on the GB10 (the sysfs ``type`` is "acpitz" for all of them)
ZONE_NAMES = ["TSOC", "TS0E", "TS0P", "TS1E", "TS1P", "TGPU", "TUNC"]
ZONES = sorted(glob.glob("/sys/class/thermal/thermal_zone*"), key=lambda p: int(p.rsplit("zone", 1)[1]))


def zone_temps() -> list[int]:
    out = []
    for z in ZONES:
        try:
            out.append(int(open(z + "/temp").read()) // 1000)
        except (OSError, ValueError):
            out.append(-1)
    return out


def hottest_zone() -> int:
    return max(zone_temps() or [-1])


def meminfo() -> tuple[float, float]:
    avail = free = float("nan")
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            avail = int(line.split()[1]) / 1048576
        elif line.startswith("MemFree"):
            free = int(line.split()[1]) / 1048576
    return avail, free


class Nvml:
    """GPU temperature, margin to the thermal limit, power, utilisation and clock-event reasons."""

    def __init__(self) -> None:
        self.h = None
        try:
            import warnings
            warnings.simplefilter("ignore")
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001 - telemetry must never take the job down
            self.h = None

    def _try(self, name: str, *args):
        try:
            return getattr(self.nv, name)(self.h, *args)
        except Exception:  # noqa: BLE001
            return None

    def read(self) -> dict:
        if self.h is None:
            return {"gpu_temp": "", "margin": "", "power_w": "", "util": "", "clocks": ""}
        temp = self._try("nvmlDeviceGetTemperature", 0)
        margin = self._try("nvmlDeviceGetMarginTemperature")
        if hasattr(margin, "marginTemperature"):
            margin = margin.marginTemperature
        power = self._try("nvmlDeviceGetPowerUsage")
        util = self._try("nvmlDeviceGetUtilizationRates")
        clocks = self._try("nvmlDeviceGetCurrentClocksEventReasons")
        if clocks is None:
            clocks = self._try("nvmlDeviceGetCurrentClocksThrottleReasons")
        return {
            "gpu_temp": "" if temp is None else temp,
            "margin": "" if margin is None else margin,
            "power_w": "" if power is None else round(power / 1000.0, 1),
            "util": "" if util is None else util.gpu,
            "clocks": "" if clocks is None else hex(int(clocks)),
        }


def main() -> None:
    out = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    max_s = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
    names = ZONE_NAMES[:len(ZONES)] + [f"zone{i}" for i in range(len(ZONE_NAMES), len(ZONES))]
    nvml = Nvml()
    t0 = time.time()
    with open(out, "a") as f:
        f.write("utc,elapsed_s," + ",".join(names) + ",gpu_temp,margin_to_limit,power_w,gpu_util,clock_events,mem_avail_gib,mem_free_gib\n")
        while time.time() - t0 < max_s and not os.path.exists(out + ".stop"):
            g = nvml.read()
            avail, free = meminfo()
            row = [time.strftime("%H:%M:%S", time.gmtime()), f"{time.time() - t0:.0f}"] + [str(t) for t in zone_temps()] + [
                str(g["gpu_temp"]), str(g["margin"]), str(g["power_w"]), str(g["util"]), str(g["clocks"]), f"{avail:.1f}", f"{free:.1f}"]
            f.write(",".join(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
            time.sleep(interval)


if __name__ == "__main__":
    main()
