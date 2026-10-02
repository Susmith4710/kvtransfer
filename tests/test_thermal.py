"""Thermal governor: pause/resume logic with an injected sensor and clock (no real sleeping)."""
import pytest

from kvtransfer import thermal
from kvtransfer.thermal import ThermalAbort, ThermalGovernor


class FakeSensor:
    """Cools by ``rate`` degrees per poll while the governor is waiting."""

    def __init__(self, temp, rate=4):
        self.temp, self.rate, self.slept = temp, rate, 0.0

    def read(self):
        return self.temp

    def sleep(self, s):
        self.slept += s
        self.temp -= self.rate


def make(sensor, **kw):
    logs = []
    g = ThermalGovernor(70, 60, read=sensor.read, sleep=sensor.sleep, log=logs.append, **kw)
    return g, logs


def test_cool_machine_is_not_paused():
    s = FakeSensor(55)
    g, logs = make(s)
    assert g.checkpoint() == 0.0 and s.slept == 0.0 and g.pauses == 0 and logs == []


def test_hot_machine_waits_until_resume_temperature():
    s = FakeSensor(82)
    g, logs = make(s)
    waited = g.checkpoint()
    assert s.temp <= 60 and waited == pytest.approx(s.slept) and waited > 0
    assert g.pauses == 1 and g.peak == 82 and len(logs) == 1 and "[thermal]" in logs[0]
    assert g.checkpoint() == 0.0                       # cool now: no second pause
    assert g.summary()["pauses"] == 1 and g.summary()["peak_c"] == 82


def test_between_thresholds_does_not_pause():
    s = FakeSensor(65)                                 # above resume, below pause: hysteresis band
    g, _ = make(s)
    assert g.checkpoint() == 0.0 and s.slept == 0.0


def test_gives_up_when_it_never_cools():
    s = FakeSensor(90, rate=0)
    g, _ = make(s, max_wait=5.0, poll=0.5)
    with pytest.raises(ThermalAbort):
        g.checkpoint()
    assert 5.0 <= s.slept <= 6.0


def test_unreadable_sensor_never_blocks():
    g = ThermalGovernor(70, 60, read=lambda: None, sleep=lambda s: None, log=lambda m: None)
    assert g.checkpoint() == 0.0


def test_resume_must_be_below_pause():
    with pytest.raises(ValueError):
        ThermalGovernor(70, 70)


def test_module_checkpoint_is_a_noop_until_configured(monkeypatch):
    monkeypatch.delenv("KVT_THERMAL_PAUSE_C", raising=False)
    thermal.configure(None)
    assert not thermal.enabled() and thermal.checkpoint() == 0.0 and thermal.summary() is None
    s = FakeSensor(80)
    thermal.configure(70, 60, read=s.read, sleep=s.sleep, log=lambda m: None)
    try:
        assert thermal.enabled() and thermal.checkpoint() > 0 and thermal.summary()["pauses"] == 1
    finally:
        thermal.configure(None)


def test_benchmark_excludes_governor_pauses_from_timing(src_model, tgt_model, calib_batches):
    """With the governor on, each trial is synced and timed alone; a long pause must not inflate the result."""
    import time

    from kvtransfer import Mapper, calibrate
    from kvtransfer.bench import benchmark

    m = Mapper.fit(calibrate(src_model, tgt_model, calib_batches, stride=2), k=1, lam=0.01)
    calls = {"n": 0}

    def read():
        calls["n"] += 1
        return 80 if calls["n"] % 2 else 50            # hot on every other reading: forces pauses

    thermal.configure(70, 60, read=read, sleep=lambda s: time.sleep(0.05), log=lambda m: None)
    try:
        rows = benchmark(src_model, tgt_model, m, seq_lens=(8,), warmup=1, trials=3)
        paused = thermal.summary()["pauses"]
    finally:
        thermal.configure(None)
    assert paused >= 1 and rows[0].mapper_ms < 40 and rows[0].reprefill_ms < 40   # 50 ms pauses stayed outside the timing


def _reset(monkeypatch, env=None, gb10=False):
    monkeypatch.delenv("KVT_THERMAL_PAUSE_C", raising=False)
    monkeypatch.delenv("KVT_THERMAL_RESUME_C", raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(thermal, "_machine_needs_governor", lambda: gb10)
    monkeypatch.setattr(thermal, "_governor", None)
    monkeypatch.setattr(thermal, "_env_checked", False)


def test_governor_is_on_by_default_on_a_gb10(monkeypatch, capsys):
    _reset(monkeypatch, gb10=True)
    g = thermal.governor()
    assert g is not None and (g.pause_at, g.resume_at) == (thermal.DEFAULT_PAUSE_C, thermal.DEFAULT_RESUME_C)
    assert "governor on" in capsys.readouterr().out


def test_governor_stays_off_elsewhere_and_can_be_forced_off(monkeypatch):
    _reset(monkeypatch, gb10=False)
    assert thermal.governor() is None
    _reset(monkeypatch, env={"KVT_THERMAL_PAUSE_C": "off"}, gb10=True)
    assert thermal.governor() is None


def test_environment_overrides_the_defaults(monkeypatch):
    _reset(monkeypatch, env={"KVT_THERMAL_PAUSE_C": "75", "KVT_THERMAL_RESUME_C": "62"}, gb10=False)
    g = thermal.governor()
    assert (g.pause_at, g.resume_at) == (75.0, 62.0)
