"""L2 policies separate preparation races and their cached selections."""

import pytest
import torch

from b12x.preparation import DetectedDevice, PreparationSession, PreparedCall
from b12x.preparation import _measurement
from b12x.preparation._cache import SelectionCache, cache_identity
from b12x._lib.compile_plan import ProgramKey


def test_cache_policy_is_explicit_and_fixed_for_a_session():
    with PreparationSession(
        device=DetectedDevice(None, None), autotune=False
    ) as session:
        assert session.cache_policy == "cold"
    with PreparationSession(
        device=DetectedDevice(None, None), autotune=False, cache_policy="warm"
    ) as session:
        assert session.cache_policy == "warm"
        with pytest.raises(AttributeError):
            session.cache_policy = "cold"
    with pytest.raises(ValueError, match="cache_policy"):
        PreparationSession(device=DetectedDevice(None, None), cache_policy="unknown")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_warm_and_cold_selections_have_distinct_caches(tmp_path):
    default = cache_identity({}, 0)
    cold = cache_identity({}, 0, cache_policy="cold")
    warm = cache_identity({}, 0, cache_policy="warm")
    assert default == cold
    assert cold["measurement"] == "stream_gated_events_v1"
    cold_cache = SelectionCache(tmp_path, cold)
    warm_cache = SelectionCache(tmp_path, warm)
    assert cold_cache.path != warm_cache.path
    cold_cache.save(
        "query",
        assignment={"width": 2},
        config={"width": 2},
        coverage=dict(
            cartesian_count=2, legal_count=2, effective_count=2, measured_count=2
        ),
        programs=(ProgramKey("cute", "test-program"),),
    )
    assert cold_cache.get("query") is not None
    assert warm_cache.get("query") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("policy", ("cold", "warm"))
def test_captured_race_uses_the_requested_l2_policy(monkeypatch, policy):
    source = torch.arange(4096, dtype=torch.float32, device="cuda")
    activation = torch.empty_like(source)
    output = torch.empty_like(source)
    requested = []
    flush = _measurement._l2_flush_fn

    def record_flush(device, *, enabled):
        requested.append(enabled)
        return flush(device, enabled=enabled)

    monkeypatch.setattr(_measurement, "_l2_flush_fn", record_flush)
    call = _measurement.capture_benchmark_call(
        PreparedCall(
            run=lambda: torch.add(activation, 1, out=output),
            produce=lambda: activation.copy_(source),
            owners=(source, activation, output),
        )
    )
    race = None
    try:
        race = _measurement._prepare_race(
            (call,), device_ordinal=0, samples=1, cache_policy=policy
        )
        result = _measurement._measure_race(race, device_ordinal=0, rounds=1)
        assert requested == [policy == "cold"]
        assert len(result.latencies_us) == 1 and result.latencies_us[0] > 0
        torch.testing.assert_close(output, source + 1, rtol=0, atol=0)
    finally:
        if race is not None:
            race.close()
        call.close()
