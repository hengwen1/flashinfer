"""Persistent factory discovery is portable, invalidated, and single-flight."""

from dataclasses import replace
import multiprocessing
from pathlib import Path
import traceback

import pytest

from b12x._lib.compile_pool import CompileJob
from b12x._lib.source_dependencies import SourceDependencies
from b12x.preparation import DetectedDevice, DeviceIdentity, FrozenMapping
from b12x.preparation import _discovery
from b12x.preparation._race_cache import RaceCache


def device(ordinal=0):
    return DetectedDevice(
        ordinal,
        DeviceIdentity("nvidia", (12, 0), 170, "test target"),
        f"physical-{ordinal}",
        101376,
        102400,
    )


def candidates(width=128):
    payload = FrozenMapping({"width": width})
    job = CompileJob.create(
        "b12x.norm.vision:_compile", "gelu", width, 1, 2, CompileJob.DEVICE_ORDINAL
    )
    return [(payload, payload, (job,))]


def test_delta_prefill_factory_sources_include_both_recipe_implementations():
    root = Path(_discovery.__file__).resolve().parents[1]
    dependencies = SourceDependencies(root)
    module = "b12x.sequence._shared.delta_prefill.preparation"
    sources = dict(dependencies.closure(module, dependencies.module_path(module)))
    assert {
        "b12x.sequence.gdn_prefill._impl",
        "b12x.sequence.gdn_prefill._tuning",
        "b12x.sequence.kda_prefill._impl",
        "b12x.sequence.kda_prefill._tuning",
    } <= sources.keys()
    assert _discovery._factory_sources((module + ":compile_prefill",)) is not None


def test_key_tracks_semantic_inputs_and_capabilities_not_placement(monkeypatch):
    monkeypatch.setattr(_discovery, "_compiler_context", lambda cc: ("toolchain", cc))
    monkeypatch.setattr(_discovery, "_factory_sources", lambda factories: "source")
    rows = candidates()
    key = _discovery.discovery_key(rows, device())
    assert _discovery.discovery_key(rows, device(3)) == key
    renamed = replace(
        device(3), identity=replace(device().identity, product_name="another label")
    )
    assert _discovery.discovery_key(rows, renamed) == key
    assert _discovery.discovery_key(candidates(256), device()) != key
    changed = replace(rows[0][2][0], kwargs=(("extra", 1),))
    assert (
        _discovery.discovery_key([(rows[0][0], rows[0][1], (changed,))], device())
        != key
    )
    assert (
        _discovery.discovery_key(
            rows, replace(device(), max_shared_memory_per_block=232448)
        )
        != key
    )
    assert (
        _discovery.discovery_key(
            rows, replace(device(), identity=replace(device().identity, sm_count=48))
        )
        != key
    )
    assert (
        _discovery.discovery_key(
            rows,
            replace(
                device(),
                identity=replace(device().identity, compute_capability=(12, 1)),
            ),
        )
        != key
    )
    assert _discovery.discovery_key(
        rows + candidates(256), device()
    ) != _discovery.discovery_key(candidates(256) + rows, device())
    monkeypatch.setattr(
        _discovery, "_compiler_context", lambda cc: ("changed options", cc)
    )
    assert _discovery.discovery_key(rows, device()) != key


def test_source_closure_is_memoized_and_helper_edits_invalidate(tmp_path, monkeypatch):
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "norm").mkdir()
    (tmp_path / "norm" / "vision.py").write_text("from b12x.norm.helper import value\n")
    helper = tmp_path / "norm" / "helper.py"
    helper.write_text("value = 1\n")
    # Supply the fixed implementation roots as well as the factory's source.
    for name in (
        "preparation/_discovery.py",
        "_lib/compile_pool.py",
        "_lib/compile_plan.py",
        "_lib/compiler.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("")
    dependencies = SourceDependencies(tmp_path)
    monkeypatch.setattr(_discovery, "source_dependencies", lambda root: dependencies)
    _discovery._factory_sources.cache_clear()
    factories = ("b12x.norm.vision:_compile",)
    original = _discovery._factory_sources(factories)
    with monkeypatch.context() as context:
        context.setattr(
            Path,
            "read_bytes",
            lambda path: pytest.fail("rehashing source on a warm lookup"),
        )
        assert _discovery._factory_sources(factories) == original
    helper.write_text("value = 2\n")
    dependencies = SourceDependencies(tmp_path)
    _discovery._factory_sources.cache_clear()
    assert _discovery._factory_sources(factories) != original
    _discovery._factory_sources.cache_clear()


def test_device_binding_only_happens_when_the_factory_executes(monkeypatch):
    import torch
    from b12x._lib import compile_pool

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(
        compile_pool, "_factory", lambda name: lambda *args, **kw: (args, kw)
    )
    job = CompileJob.create(
        "b12x.norm.vision:_compile", 3, CompileJob.DEVICE_ORDINAL, rank=5
    )
    assert job.invoke() == ((3, 7), {"rank": 5})
    assert job.args == (3, CompileJob.DEVICE_ORDINAL)


def _session_probe(path, ordinal, forbid_discovery, connection):
    try:
        import torch
        from b12x._lib import compile_pool
        from b12x.preparation import session as session_module
        from .test_defaults import contract
        from .test_parallel_discovery import finish
        from .test_session import request, session

        target = device(ordinal)
        compile_pool._initialize_worker(
            ordinal,
            target.identity.compute_capability,
            target.uuid,
            target.identity.product_name,
            target.identity.sm_count,
            target.max_shared_memory_per_block,
            target.max_shared_memory_per_multiprocessor,
            multiprocessing.get_context("spawn").Array("q", (0, 0)),
        )
        item = request(name="gelu", tuning=contract(default=128, values=(128, 256)))
        item = replace(
            item,
            plan=replace(
                item.plan,
                _compile_jobs=lambda config, device: (
                    CompileJob.create(
                        "b12x.norm._vision_preparation:compile_vision",
                        {
                            "operation": "gelu",
                            "channels": config.width,
                            "heads": 1,
                            "ratio": 2,
                        },
                        CompileJob.DEVICE_ORDINAL,
                    ),
                ),
            ),
        )
        if forbid_discovery:

            def forbidden(*args, **kwargs):
                raise AssertionError("cache hit executed a kernel factory")

            compile_pool._factory = forbidden
            session_module.describe_compilation = forbidden
        with session(path, compile_workers=0) as engine:
            job = engine.begin((item,))
            engine.device = target
            (obligation,) = finish(
                job._configure(session_module._coalesce_requests((item,)))
            )
            finish(job._describe_candidates(obligation))
            result = (
                obligation.candidate_program_identity,
                job._choice_key(obligation, {}),
                sorted(
                    (p.dialect, p.key)
                    for programs in obligation.programs.values()
                    for p in programs
                ),
            )
            assert all(
                cp.job.args[-1] is CompileJob.DEVICE_ORDINAL
                for plans in obligation.compiled.values()
                for cp in plans
            )
            engine.device = DetectedDevice(None, None)
        assert not torch.cuda.is_initialized()
        connection.send((True, result))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def test_actual_factory_discovery_survives_process_restart_and_device_change(tmp_path):
    context = multiprocessing.get_context("spawn")
    results = []
    for ordinal, forbid in ((0, False), (3, True)):
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(
            target=_session_probe, args=(tmp_path, ordinal, forbid, writer)
        )
        process.start()
        writer.close()
        try:
            assert reader.poll(60), "offline factory discovery timed out"
            ok, result = reader.recv()
            assert ok, result
            results.append(result)
            process.join(15)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.kill()
                process.join()
            reader.close()
    assert results[0] == results[1]


def _locked_writer(path, start, counts):
    cache = RaceCache(path)
    start.wait()
    with cache.discovery_lock("race"):
        if cache.load_discovery("race") is None:
            with counts.get_lock():
                counts.value += 1
            cache.save_discovery("race", {"programs": [], "identity": "ready"})
    cache.close()


def test_concurrent_discovery_writers_publish_once(tmp_path):
    context = multiprocessing.get_context("spawn")
    start, count = context.Event(), context.Value("i", 0)
    processes = [
        context.Process(
            target=_locked_writer, args=(tmp_path / "cache.sqlite3", start, count)
        )
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    start.set()
    try:
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
        assert count.value == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


def test_cancelled_discovery_releases_lock_without_publishing(tmp_path, monkeypatch):
    from b12x.preparation import session as session_module
    from .test_defaults import contract
    from .test_parallel_discovery import finish
    from .test_session import request, session

    monkeypatch.setattr(session_module, "discovery_key", lambda rows, target: "race")
    item = request(name="cancel", tuning=contract())
    with session(tmp_path, compile_workers=0) as engine:
        job = engine.begin((item,))
        (obligation,) = finish(
            job._configure(session_module._coalesce_requests((item,)))
        )
        cache = engine._race_progress_cache()
        blocker = cache.discovery_lock("race")
        blocker.acquire()
        steps = job._describe_candidates(obligation)
        try:
            assert next(steps) == "compile"
            engine.cancel_tuning()
            finish(steps)
            assert cache.load_discovery("race") is None
        finally:
            blocker.release()
        engine._stop.clear()

        def interrupted(obligation):
            yield "metadata"
            raise AssertionError("cancelled discovery was resumed")

        monkeypatch.setattr(job, "_discover_candidates", interrupted)
        steps = job._describe_candidates(obligation)
        assert next(steps) == "metadata"
        steps.close()
        assert cache.load_discovery("race") is None
        with cache.discovery_lock("race").acquire(timeout=0):
            pass
