"""CUTLASS DSL runtime patches: applied by default, disabled by kill switch.

Runs in subprocesses because the patches mutate the process-global cutlass
runtime.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import weakref
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")

PROBE = """
import json, os
{env_setup}
from b12x._lib import runtime_patches as rp
rp.apply_cutlass_runtime_patches()
rp.apply_cutlass_runtime_patches()  # second call must be a guarded no-op
print(json.dumps({{
    "warning_patched": rp._WARNING_PATCHED,
    "frameinfo_patched": rp._DIRECT_FRAMEINFO_PATCHED,
    "executor_lifetime_patched": rp._EXECUTOR_LIFETIME_PATCHED,
}}))
"""


def _run(env_setup: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", PROBE.format(env_setup=env_setup)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"subprocess failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_patches_apply_by_default():
    data = _run('os.environ.pop("B12X_DISABLE_CUTLASS_RUNTIME_PATCHES", None)')
    assert data["warning_patched"] is True
    assert data["frameinfo_patched"] is True
    assert data["executor_lifetime_patched"] is True


def test_kill_switch_disables_all_patches():
    data = _run('os.environ["B12X_DISABLE_CUTLASS_RUNTIME_PATCHES"] = "1"')
    assert data["warning_patched"] is False
    assert data["frameinfo_patched"] is False
    assert data["executor_lifetime_patched"] is False


@pytest.fixture
def no_cyclic_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _executor(profiling, calls, with_context=False):
    from cutlass.base_dsl.jit_executor import JitExecuteContext, JitExecutor, JitModule
    from b12x._lib.runtime_patches import apply_cutlass_runtime_patches

    apply_cutlass_runtime_patches()

    def launch(args):
        calls.append(list(args))

    launch.restype = None
    arguments = SimpleNamespace(
        generate_execution_args=lambda args, kwargs: (list(args), [])
    )
    module = JitModule(None, launch, arguments, [])
    context = JitExecuteContext(module) if with_context else None
    return JitExecutor(module, context, profiling)


@pytest.mark.parametrize("profiling", [False, True])
@pytest.mark.parametrize("with_context", [False, True])
def test_executor_and_module_release_without_cyclic_gc(
    no_cyclic_gc, profiling, with_context
):
    calls = []
    executor = _executor(profiling, calls, with_context)
    executor_ref, module_ref = weakref.ref(executor), weakref.ref(executor.jit_module)
    owner = executor
    del executor
    assert executor_ref() is owner and module_ref() is owner.jit_module
    owner(123, None, 456)
    owner.run_compiled_program([789, None, 987])
    assert calls == [[123, None, 456], [789, None, 987]]
    del owner
    assert executor_ref() is None and module_ref() is None


@pytest.mark.parametrize("profiling", [False, True])
def test_retained_program_owns_executor_until_final_release(no_cyclic_gc, profiling):
    from b12x._lib import compile_plan, program_cache

    calls = []
    key = compile_plan.ProgramKey("cute", "executor-lifetime")

    @program_cache.program_cache(scope="preparation")
    def factory():
        return compile_plan.CompiledCuTeProgram(_executor(profiling, calls), key)

    scope = program_cache.PreparationProgramCache()
    with scope.activate():
        with compile_plan.retain_compiled_programs() as first:
            program = compile_plan.load_programs(factory())
        with compile_plan.retain_compiled_programs() as second:
            assert compile_plan.load_programs(factory()) is program
    program_ref = weakref.ref(program)
    executor_ref = weakref.ref(program._executable)
    module_ref = weakref.ref(program._executable.jit_module)
    del program
    scope.clear()
    first.owners.clear()
    program_ref()(321)
    assert calls == [[321]]
    second.owners.clear()
    assert program_ref() is None and executor_ref() is None and module_ref() is None
