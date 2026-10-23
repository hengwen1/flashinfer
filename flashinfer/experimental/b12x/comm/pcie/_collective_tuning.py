"""Logical PCIe operations with native transport and launch candidates."""

from dataclasses import asdict, dataclass, replace

from b12x.preparation import (
    AutotuneEffort,
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    Plan,
    TuningContract,
)
from b12x.preparation._efficiency import capture_autotune_effort

from ._tuning import PcieQuery, TUNING as NATIVE


@dataclass(frozen=True, kw_only=True)
class CollectiveConfig:
    implementation: str
    threads: int
    blocks: int


@dataclass(frozen=True, kw_only=True)
class CollectiveQuery:
    operation: str
    world_size: int
    rank: int
    alternatives: FrozenMapping
    default: CollectiveConfig
    inplace: bool = False


def _configs(query):
    for name, payload in query.alternatives.items():
        call = payload["call"]
        blocks = int(call.get("blocks", call.get("block_limit", 16)))
        threads = int(call["threads"])
        yield CollectiveConfig(implementation=name, threads=threads, blocks=blocks)
        if name in ("tp4_push_sentinel", "tp4_push_rsag"):
            for count in (8, 16, 32):
                yield CollectiveConfig(
                    implementation=name, threads=threads, blocks=count
                )
        elif name == "twoshot":
            for count in (32, 64):
                yield CollectiveConfig(
                    implementation=name, threads=threads, blocks=count
                )
        elif name == "dcp":
            setup = payload["setup"]
            thread_values = (threads,) if setup["threads_override"] else (128, 256, 512)
            block_values = (blocks,) if setup["blocks_override"] else (8, 16, 32)
            for threads in thread_values:
                for count in block_values:
                    yield CollectiveConfig(
                        implementation=name, threads=threads, blocks=count
                    )


def _parameters(query, device):
    configs = tuple(dict.fromkeys(_configs(query)))
    effort = capture_autotune_effort()
    if effort in (AutotuneEffort.OFF, AutotuneEffort.LOW):
        configs = tuple(
            dict.fromkeys(
                CollectiveConfig(
                    implementation=name,
                    threads=int(payload["call"]["threads"]),
                    blocks=int(
                        payload["call"].get(
                            "blocks", payload["call"].get("block_limit", 16)
                        )
                    ),
                )
                for name, payload in query.alternatives.items()
            )
        )
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "implementation": tuple(dict.fromkeys(c.implementation for c in configs)),
            "threads": tuple(dict.fromkeys(c.threads for c in configs)),
            "blocks": tuple(dict.fromkeys(c.blocks for c in configs)),
        },
        predicates=(lambda value: CollectiveConfig(**value) in configs,),
        autotune_effort=effort,
    )


def _validate_query(query, device):
    if not isinstance(query, CollectiveQuery) or not query.alternatives:
        raise ValueError("collective query requires native alternatives")
    if not 2 <= query.world_size <= 8 and 0 <= query.rank < query.world_size:
        raise ValueError("collective tuning requires an established TP2–8 group")
    for payload in query.alternatives.values():
        native = PcieQuery(**payload)
        NATIVE.validate_query(native, device)
        if (native.world_size, native.rank) != (query.world_size, query.rank):
            raise ValueError("native alternatives must use the same rank group")


def _validate_config(query, config, device):
    if not isinstance(config, CollectiveConfig) or config not in tuple(_configs(query)):
        raise ValueError(
            "configuration is outside the native collective parameter space"
        )


def _equivalence_key(query, device, config):
    if config.implementation != "dcp":
        return asdict(config)
    native = query.alternatives["dcp"]
    heads = native["setup"]["total_heads"]
    if query.operation == "lse_reduce_scatter":
        heads //= query.world_size
    rows = native["call"]["shape"][0] * heads
    warps = config.threads // 32
    return (
        config.implementation,
        config.threads,
        min(config.blocks, (rows + warps - 1) // warps),
    )


def _collective_signature(query):
    alternatives = {}
    for name, payload in query.alternatives.items():
        native = payload.to_dict()
        del native["rank"]
        for section in ("call", "setup"):
            native[section].pop("device_index", None)
        alternatives[name] = native
    return dict(
        operation=query.operation,
        world_size=query.world_size,
        alternatives=alternatives,
        inplace=query.inplace,
    )


TUNING = TuningContract(
    component_id="comm.pcie",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(
        ("operation", "world_size", "rank", "alternatives", "default", "inplace")
    ),
    config_fields=frozenset(("implementation", "threads", "blocks")),
    encode_query=lambda q: dict(
        operation=q.operation,
        world_size=q.world_size,
        rank=q.rank,
        alternatives=q.alternatives,
        default=asdict(q.default),
        inplace=q.inplace,
    ),
    encode_config=asdict,
    decode_config=lambda value: CollectiveConfig(**value),
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=lambda q, device: q.default,
    knobs=(
        Knob(name="implementation", values=None),
        Knob(name="threads", values=None),
        Knob(name="blocks", values=None, binding=ParameterBinding.RUNTIME),
    ),
    parameters=_parameters,
    collective_signature=_collective_signature,
    equivalence_key=_equivalence_key,
)


def plan(
    alternatives, *, default, inplace=False, invocation=FrozenMapping(), override=None
):
    from ._preparation import _provider
    from ._benchmark import benchmark_call

    first = next(iter(alternatives.values()))[0]
    default_call = alternatives[default][0].call
    query = CollectiveQuery(
        operation=first.surface.split(".", 1)[1],
        world_size=first.world_size,
        rank=first.rank,
        alternatives=FrozenMapping(
            {
                name: NATIVE.encode_query(q)
                for name, (q, runtime) in alternatives.items()
            }
        ),
        default=CollectiveConfig(
            implementation=default,
            threads=int(default_call["threads"]),
            blocks=int(default_call.get("blocks", default_call.get("block_limit", 16))),
        ),
        inplace=inplace,
    )
    TUNING.validate_query(query, None)
    native_plans = {}

    def native(config):
        if config not in native_plans:
            q, runtime = alternatives[config.implementation]
            call = dict(q.call)
            call["threads"] = config.threads
            call["blocks" if "blocks" in call else "block_limit"] = config.blocks
            q = replace(q, call=FrozenMapping(call))
            native_plans[config] = _provider(q.surface).plan(q, runtime=runtime)
        return native_plans[config]

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=lambda config, device: native(config)._compile_jobs(
            config, device
        ),
        _memory_requirements=lambda config, device: native(config)._memory_requirements(
            config, device
        ),
        _materialize=lambda selection, device: native(selection.config)._materialize(
            selection, device
        ),
        _benchmark_call=lambda state: benchmark_call(state, inplace=inplace),
        _device=next(iter(alternatives.values()))[1].device,
    )


def oneshot_alternatives(query, runtime):
    alternatives = {}
    field = "transport" if query.surface.endswith(".all_reduce") else "mode"
    selected = query.call[field]
    alternatives[selected] = (query, runtime)
    if not query.call["stage_input"]:
        return alternatives
    modes = ["pull"] if field == "transport" else ["stage_pull"]
    if field == "transport":
        if (
            query.world_size == 4
            and query.call["dtype"] in ("bf16", "bfloat16")
            and query.call["remote_push_region_packs"]
        ):
            modes += ["tp4_push_sentinel", "tp4_push_rsag"]
    elif query.world_size == 4:
        modes += ["stage_remote_push"]
        # The payload-ready fused specialization uses an isolated incoming region.
        if selected == "stage_bf16_sentinel":
            modes += ["stage_bf16_sentinel"]
    for mode in modes:
        call = {**query.call, field: mode}
        if field == "transport" and mode == "pull":
            call.update(threads=call["eager_threads"], blocks=call["eager_blocks"])
        if field == "transport" and mode.startswith("tp4_"):
            call.update(threads=256, blocks=16)
        alternatives[mode] = (replace(query, call=FrozenMapping(call)), runtime)
    return alternatives
