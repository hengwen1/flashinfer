"""Group-coordinated candidate races using the preparation sampler."""

from __future__ import annotations

import math
from dataclasses import replace

from ._cache import digest
from ._measurement import capture_benchmark_call, no_compilation
from .types import CollectiveTuningRequirement, FrozenMapping, call_scope


class _Cancelled(Exception):
    pass


def exchange(requirement, step, **payload):
    peers = yield CollectiveTuningRequirement(
        requirement.key, requirement.ranks, step, FrozenMapping(payload)
    )
    return tuple(item.payload for item in peers)


def _same(peers, field):
    value = peers[0][field]
    if any(peer[field] != value for peer in peers[1:]):
        raise ValueError(f"collective tuning ranks disagree on {field}")
    return value


def _load(job, obligation, record):
    contract = obligation.request.plan.contract
    configuration = obligation.configuration
    assignment = FrozenMapping(record["assignment"])
    configuration.space.validate(assignment)
    config = contract._lower(configuration.query, configuration.device, assignment)
    if contract.config_payload(config) != record["config"]:
        raise ValueError("collective cached assignment differs from its configuration")
    obligation.coverage = dict(record["coverage"])
    job._cache_hits += 1
    return job._selection(obligation, config, "cached", assignment)


def select(job, obligation, selections):
    requirement = obligation.request.collective
    contract = obligation.request.plan.contract
    if contract.collective_signature is None:
        raise ValueError(
            "collective tuning requires a group-consistent query signature"
        )
    candidates = tuple(
        (assignment.to_dict(), contract.config_payload(config).to_dict())
        for assignment, config in obligation.candidates
    )
    peers = yield from exchange(
        requirement,
        "catalog",
        candidates=candidates,
        signature=contract.collective_signature(obligation.request.plan.query),
        local_key=job._choice_key(obligation, selections),
        device=job.session.device.uuid,
        cache_only=job.session.cache_only,
        rounds=job.session.rounds,
        samples=job.session.samples,
        cancel=job.session._stop.is_set(),
    )
    for field in ("signature", "candidates", "cache_only", "rounds", "samples"):
        _same(peers, field)
    if any(peer["cancel"] for peer in peers):
        return job._selection(obligation, obligation.configuration.default, "default")
    obligation.key = digest(
        {
            "measurement": "collective_max_stream_gated_graph_v2",
            "members": tuple((peer["device"], peer["local_key"]) for peer in peers),
        }
    )
    cache = job.session._selection_cache()
    peers = yield from exchange(requirement, "cache", record=cache.get(obligation.key))
    record = next(
        (peer["record"] for peer in peers if peer["record"] is not None), None
    )
    if record is not None:
        return _load(job, obligation, record)
    if job.session.cache_only:
        raise LookupError(
            f"no completed collective selection for {obligation.request.name}"
        )
    if obligation.request.benchmark_call is None:
        raise ValueError("collective tuning requires a representative benchmark call")
    try:
        return (yield from race(job, obligation))
    except _Cancelled:
        return job._selection(obligation, obligation.configuration.default, "default")


def race(job, obligation):
    from .session import _CallGuard, _Trial

    requirement = obligation.request.collective
    contract = obligation.request.plan.contract
    champion = None
    live = []
    measurements = {}
    job._candidate_sharded = False
    job._candidate_count = len(obligation.candidates)

    def sample_exchange(indices, turn, repetition, groups):
        peers = yield from exchange(
            requirement,
            f"samples/{job._batch_index}/{turn}/{repetition}",
            indices=indices,
            samples=groups,
            cancel=job.session._stop.is_set(),
        )
        _same(peers, "indices")
        if any(peer["cancel"] for peer in peers):
            raise _Cancelled
        if any(
            len(peer["samples"]) != len(groups)
            or any(
                len(values) != len(expected)
                for values, expected in zip(peer["samples"], groups, strict=True)
            )
            for peer in peers
        ):
            raise ValueError("collective ranks measured different sample counts")
        if any(
            not math.isfinite(value) or value <= 0
            for peer in peers
            for values in peer["samples"]
            for value in values
        ):
            raise ValueError("collective rank produced an invalid latency")
        result = tuple(
            tuple(
                max(peer["samples"][i][j] for peer in peers) for j in range(len(values))
            )
            for i, values in enumerate(groups)
        )
        if any(
            not math.isfinite(value) or value <= 0 for row in result for value in row
        ):
            raise ValueError("collective race produced an invalid latency")
        return result

    def launch_exchange(indices, turn, repetition):
        peers = yield from exchange(
            requirement,
            f"queued/{job._batch_index}/{turn}/{repetition}",
            indices=indices,
        )
        _same(peers, "indices")

    try:
        for index, (assignment, config) in enumerate(obligation.candidates):
            job._phase = "preparing candidates"
            programs = yield from job._compile(obligation, config, required=True)
            yield from job._wait_programs(programs)
            compile_assignment = obligation.configuration.space.compile_assignment(
                assignment
            )
            previous = obligation.compile_assignments.setdefault(
                compile_assignment, programs
            )
            if previous != programs:
                raise ValueError(
                    "collective runtime parameter changed actual compile keys"
                )
            selected = job._selection(obligation, config, "tuned", assignment)
            before = job.session._allocated()
            state, call, guard, retained = job._call(
                obligation,
                selected,
                obligation.request.benchmark_call,
                synchronize=False,
                prime=False,
            )
            trial = _Trial(
                index,
                assignment,
                config,
                call,
                guard,
                retained,
                state,
                max(0, job.session._allocated() - before),
            )
            live.append(trial)
            job._temporaries.remove(guard.finish)
            peers = yield from exchange(
                requirement,
                f"bound/{index}",
                cancel=job.session._stop.is_set(),
                producers=len(call.benchmark_producers) or 1,
                has_producer=call.produce is not None,
            )
            _same(peers, "producers")
            if not all(peer["has_producer"] for peer in peers):
                raise ValueError(
                    "collective benchmark requires an input producer on every rank"
                )
            if any(peer["cancel"] for peer in peers):
                raise _Cancelled
            with job.session._gpu_scope(), no_compilation():
                # Every rank has bound its operands before collective priming starts.
                captured = capture_benchmark_call(
                    replace(call, restore=guard.restore, close=guard.close)
                )
                trial.call = captured
                trial.guard = _CallGuard(captured)
                for producer in captured.benchmark_producers or (captured.produce,):
                    if producer is None:
                        raise ValueError(
                            "collective benchmark requires an input producer"
                        )
                    with call_scope():
                        if captured.reset is not None:
                            captured.reset()
                        producer()
                        captured.invoke()
                    job.session._synchronize()
                    if captured.validate is not None:
                        captured.validate()
            yield from exchange(requirement, f"validated/{index}")
            job._candidates_prepared += 1
            job._batch_index += 1
            trials = ([] if champion is None else [champion]) + [trial]
            job._batch_candidates = len(trials)
            measurement = yield from job._measure(
                trials,
                champion=champion is not None,
                sample_exchange=sample_exchange,
                launch_exchange=launch_exchange,
            )
            best = min(
                range(len(trials)),
                key=lambda i: (measurement.latencies_us[i], trials[i].index),
            )
            measurements.update(
                (item.index, latency)
                for item, latency in zip(trials, measurement.latencies_us, strict=True)
            )
            for position, item in enumerate(trials):
                if position != best:
                    item.close()
                    live.remove(item)
            champion = trials[best]
            champion.latency_us = measurement.latencies_us[best]
            job._benchmarked += 1
            job._timing.record(
                "collective_batch_end",
                request=obligation.request.name,
                candidates=[item.index for item in trials],
                latencies_us=measurement.latencies_us,
                winner=champion.index,
            )
        if champion is None:
            raise ValueError("collective race has no eligible candidates")
        peers = yield from exchange(
            requirement, "selected", assignment=champion.assignment
        )
        _same(peers, "assignment")
        obligation.coverage["measured_count"] = len(measurements)
        obligation.cache_pending = True
        return job._selection(obligation, champion.config, "tuned", champion.assignment)
    finally:
        job.session._synchronize()
        for trial in reversed(live):
            trial.close()
        job._release_race_resources()
