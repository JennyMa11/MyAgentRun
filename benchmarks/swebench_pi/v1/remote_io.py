"""Container-only Pi filesystem/process bridge. No credentials or evaluator inputs."""

from __future__ import annotations

import base64
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

MAX_BYTES = 4 * 1024 * 1024
ROOT = Path("/testbed")
STATE = Path("/tmp/pi-eval-processes")


def checked_path(value: str, root: Path = ROOT) -> Path:
    """Validate real paths, including nonexistent destinations and symlinks."""
    path = (root / value).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError("path_outside_workspace")
    return path


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _limit_output() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))


def execute(request: dict, *, root: Path = ROOT, state_root: Path = STATE) -> dict:  # noqa: PLR0911, PLR0912, PLR0915 -- isolated protocol dispatcher
    operation = request["op"]
    if operation == "cancel":
        identifier = str(request["target"])
        if not identifier.isdigit():
            raise ValueError("invalid_process_id")
        state_root.mkdir(exist_ok=True)
        state = state_root / identifier
        (state_root / (identifier + ".cancelled")).touch()
        if state.exists():
            with suppress(ProcessLookupError):
                os.killpg(int(state.read_text()), signal.SIGKILL)
        return {"cancelled": True}
    if operation == "exec":
        identifier = str(request["id"])
        if not identifier.isdigit():
            raise ValueError("invalid_process_id")
        timeout = min(float(request.get("timeout") or 120), 3600)
        if timeout <= 0:
            raise ValueError("invalid_timeout")
        cwd = checked_path(request.get("cwd", "/testbed"), root)
        state_root.mkdir(exist_ok=True)
        state = state_root / identifier
        cancelled = state_root / (identifier + ".cancelled")
        if cancelled.exists():
            cancelled.unlink()
            return {"output": "", "exit_code": 130, "timed_out": False, "cancelled": True}
        _limit_output()
        with tempfile.TemporaryFile() as capture:
            process = subprocess.Popen(
                ["/bin/bash", "--noprofile", "--norc", "-c", request["command"]],
                cwd=cwd,
                stdout=capture,
                stderr=subprocess.STDOUT,
                env={
                    k: v
                    for k, v in os.environ.items()
                    if k
                    in {
                        "PATH",
                        "LANG",
                        "LC_ALL",
                        "PIP_NO_INDEX",
                        "PIP_DISABLE_PIP_VERSION_CHECK",
                        "PYTHONDONTWRITEBYTECODE",
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "SETUPTOOLS_SCM_PRETEND_VERSION",
                    }
                },
                start_new_session=True,
            )
            state.write_text(str(process.pid))
            if cancelled.exists():
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also remove background descendants when the shell exits normally.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                state.unlink(missing_ok=True)
                cancelled.unlink(missing_ok=True)
            capture.seek(0)
            output = capture.read(MAX_BYTES + 1)
            truncated = len(output) > MAX_BYTES or process.returncode == -signal.SIGXFSZ
            output = output[:MAX_BYTES]
            if truncated:
                output += b"\n[container output truncated at 4 MiB]\n"
            return {
                "output": base64.b64encode(output).decode(),
                "exit_code": 124
                if timed_out
                else (process.returncode if process.returncode >= 0 else 128 - process.returncode),
                "timed_out": timed_out,
                "truncated": truncated,
            }
    path = checked_path(request["path"], root)
    if operation == "access":
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ValueError("file_not_readable")
        return {}
    if operation == "read":
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("file_exceeds_4MiB_read_limit")
        return {"data": base64.b64encode(path.read_bytes()).decode()}
    if operation == "mkdir":
        path.mkdir(parents=True, exist_ok=True)
        return {}
    if operation == "write":
        atomic_write(path, request["content"])
        return {}
    raise ValueError("unknown_operation")


def main() -> None:
    try:
        payload = base64.b64decode(sys.argv[1], validate=True)
        if len(payload) > MAX_BYTES:
            raise ValueError("request_too_large")
        result = {"ok": True, "result": execute(json.loads(payload))}
    except Exception as error:
        result = {"ok": False, "error": type(error).__name__ + ": " + str(error)[:200]}
    print(json.dumps(result))  # noqa: T201 -- bounded bridge response, container has no secrets


if __name__ == "__main__":
    main()
