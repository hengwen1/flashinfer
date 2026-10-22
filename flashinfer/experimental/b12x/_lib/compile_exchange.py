"""Exchange metadata-only compile jobs and compiled cache entries over local files.

Queue directories contain trusted Python metadata and must remain private to the
preparation account. A controller copies requests to its own local storage and
copies results back; no shared filesystem or CUDA context is required there.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import pickle
import platform
import re
import threading
import time
import uuid
from multiprocessing.pool import ThreadPool
from pathlib import Path

from .cache_integrity import atomic_write_bytes


def toolchain():
    return {
        "machine": platform.machine(),
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "triton",
                "nvidia-cutlass-dsl",
                "apache-tvm-ffi",
                "flashinfer-python",
            )
        },
    }


def _key(program):
    if not re.fullmatch(r"[0-9a-f]{64}", program.key):
        raise ValueError("compile exchange requires a SHA-256 program key")


def collect_artifacts(programs):
    from . import compiler
    from .compile_plan import compiled_program_available
    from triton.runtime.cache import get_cache_manager

    artifacts = []
    for program in programs:
        _key(program)
        if not compiled_program_available(program):
            raise RuntimeError(f"compiler did not publish {program}")
        if program.dialect == "cute":
            files = {
                "object": compiler._cache_object_path(program.key).read_bytes(),
                "manifest": compiler._cache_manifest_path(program.key).read_bytes(),
            }
        elif program.dialect == "triton":
            group = get_cache_manager(program.key).get_group(
                f"{program.name[:150]}.json"
            )
            files = {name: Path(path).read_bytes() for name, path in group.items()}
        else:
            raise ValueError(f"unsupported compile dialect {program.dialect}")
        artifacts.append((program, files))
    return artifacts


def install_artifacts(artifacts, expected):
    from . import compiler
    from .compile_plan import compiled_program_available
    from triton.runtime.cache import get_cache_manager

    if len(artifacts) != len(expected) or {p for p, _ in artifacts} != set(expected):
        raise ValueError("compiler response does not match requested programs")
    for program, files in artifacts:
        _key(program)
        if program.dialect == "cute":
            manifest = json.loads(files["manifest"])
            if (
                manifest.get("cache_key") != program.key
                or manifest.get("object_bytes") != len(files["object"])
                or manifest.get("object_sha256")
                != hashlib.sha256(files["object"]).hexdigest()
            ):
                raise ValueError("compiler returned an invalid CuTe object")
            with compiler._disk_cache_key_lock(program.key):
                atomic_write_bytes(
                    compiler._cache_object_path(program.key), files["object"]
                )
                atomic_write_bytes(
                    compiler._cache_manifest_path(program.key), files["manifest"]
                )
        elif program.dialect == "triton":
            cache = get_cache_manager(program.key)
            paths = {}
            for name, data in files.items():
                if Path(name).name != name or name in (".", ".."):
                    raise ValueError("invalid Triton cache filename")
                paths[name] = cache.put(data, name, binary=True)
            # Triton's group file contains absolute paths, so rebuild it locally.
            cache.put_group(f"{program.name[:150]}.json", paths)
        else:
            raise ValueError(f"unsupported compile dialect {program.dialect}")
        if not compiled_program_available(program):
            raise RuntimeError(f"returned artifact is unavailable: {program}")


class ExchangePool:
    """CompilePool executor that waits for results copied by a build controller."""

    _pool = ()

    def __init__(self, directory, *, processes, target):
        self.directory = Path(directory).expanduser() / uuid.uuid4().hex
        self.directory.mkdir(parents=True, mode=0o700)
        self.stop = threading.Event()
        self.threads = ThreadPool(processes)
        self.target = target
        self.versions = toolchain()
        # These library settings are supplied by the caller's preparation config.
        self.environment = {
            k: v
            for k, v in os.environ.items()
            if k.startswith("B12X_") and k != "B12X_COMPILE_CACHE_DIR"
        }

    def _run(self, payload, expected):
        name = uuid.uuid4().hex
        request = self.directory / f"{name}.request"
        result = self.directory / f"{name}.result"
        atomic_write_bytes(
            request,
            pickle.dumps(
                {
                    "target": self.target,
                    "toolchain": self.versions,
                    "environment": self.environment,
                    "payload": payload,
                    "expected": expected,
                },
                protocol=5,
            ),
        )
        deadline = time.monotonic() + 3600
        while not result.is_file():
            if self.stop.wait(0.1):
                raise RuntimeError("external compilation was cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"compiler controller did not answer {request} within one hour"
                )
        response = pickle.loads(result.read_bytes())
        if "error" in response:
            raise RuntimeError(
                f"external kernel compilation failed: {response['error']}"
            )
        install_artifacts(response["artifacts"], expected)
        return response["summary"]

    def apply_async(self, function, args, callback, error_callback):
        self.threads.apply_async(
            self._run, args, callback=callback, error_callback=error_callback
        )

    def close(self):
        self.threads.close()

    def terminate(self):
        self.stop.set()
        self.threads.terminate()

    def join(self):
        self.threads.join()
        atomic_write_bytes(self.directory / "closed", b"")


def compile_request(request):
    """Compile one request inside an initialized offline compiler process."""
    from .compile_pool import _run_job

    try:
        if request["toolchain"] != toolchain():
            raise ValueError("preparation and compiler toolchains differ")
        summary = _run_job(request["payload"], request["expected"])
        return {"summary": summary, "artifacts": collect_artifacts(request["expected"])}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def initialize_compiler(target, environment, cache):
    """Initialize the same device-free compiler used by local preparation."""
    os.environ.update(environment)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        B12X_COMPILE_CACHE_DIR=str(Path(cache) / "b12x"),
        TRITON_CACHE_DIR=str(Path(cache) / "triton"),
    )
    from .compile_pool import _initialize_worker

    activity = multiprocessing.get_context("spawn").Array("q", (0, 0))
    _initialize_worker(*target, activity)
