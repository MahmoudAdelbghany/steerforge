"""Server detection, command assembly, and process supervision."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .persona import Persona, ServeConfig
from .utils import safe_name, stable_hash_json

BACKEND_ALIASES = {
    "llama.cpp": "llama.cpp",
    "llamacpp": "llama.cpp",
    "llama-server": "llama.cpp",
    "llama_server": "llama.cpp",
    "transformers": "transformers",
    "hf": "transformers",
    "vllm": "vllm",
    "sglang": "sglang",
    "ollama": "ollama",
}


def normalize_backend(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    return BACKEND_ALIASES.get(key, key)


def detect_backend(command: Sequence[str] | None) -> str | None:
    if not command:
        return None
    basename = Path(command[0]).name.lower()
    if "llama-server" in basename or basename in {"llama-server", "server"}:
        return "llama.cpp"
    if basename in {"vllm", "vllm-server"}:
        return "vllm"
    if basename in {"sglang", "sglang-server"}:
        return "sglang"
    if basename == "ollama":
        return "ollama"
    joined = " ".join(command).lower()
    if "vllm.entrypoints" in joined or "vllm serve" in joined:
        return "vllm"
    if "sglang" in joined and "launch_server" in joined:
        return "sglang"
    if "llama-server" in joined:
        return "llama.cpp"
    return None


def find_llama_server(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for candidate in ("llama-server", "llama-server.exe"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def control_vector_cache(persona: Persona, scale: float = 1.0) -> Path:
    digest = stable_hash_json(
        {
            "model": persona.base_model.id,
            "vectors": {k: v.data for k, v in sorted(persona.vectors.items())},
            "scale": scale,
        }
    )[:16]
    root = Path(os.environ.get("STEERFORGE_HOME", Path.home() / ".steerforge"))
    path = root / "control_vectors" / f"{safe_name(persona.name)}-{digest}.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _remove_option(args: list[str], names: set[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for item in args:
        if skip:
            skip = False
            continue
        if item in names:
            skip = True
            continue
        if any(item.startswith(name + "=") for name in names):
            continue
        out.append(item)
    return out


def _strip_control_args(args: Sequence[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for item in args:
        if skip:
            skip = False
            continue
        if item in ("--control-vector", "--control-vector-scaled", "--control-vector-layer-range"):
            skip = True
            continue
        if item.startswith("--control-vector=") or item.startswith("--control-vector-scaled="):
            continue
        out.append(item)
    return out


def build_llamacpp_command(
    persona: Persona,
    *,
    control_vector: str | Path,
    command: Sequence[str] | None = None,
    model: str | None = None,
    host: str | None = None,
    port: int | None = None,
    scale: float = 1.0,
    extra_args: Sequence[str] | None = None,
    no_steer: bool = False,
) -> list[str]:
    """Take a normal llama-server command and inject the persona.

    The original command is preserved verbatim except for control-vector flags,
    which SteerForge owns.  This is what makes ``serve`` a one-command drop-in.
    """
    if command:
        base = _strip_control_args(list(command))
    elif persona.serve and persona.serve.command:
        base = _strip_control_args(list(persona.serve.command))
    else:
        server = find_llama_server()
        if not server:
            raise RuntimeError(
                "llama-server was not found. Pass --command 'llama-server -m model.gguf' "
                "or install llama.cpp."
            )
        base = [server]
        if persona.serve and persona.serve.model:
            base += ["-m", persona.serve.model]
    if not base:
        raise RuntimeError("empty llama.cpp command")
    if model:
        base = _remove_option(base, {"-m", "--model", "-hf", "--hf-repo", "--hf"})
        base += ["-m", model]
    elif not any(flag in base for flag in ("-m", "--model", "-hf", "--hf-repo", "--hf")):
        if persona.serve and persona.serve.model:
            base += ["-m", persona.serve.model]
    if host:
        base = _remove_option(base, {"--host"})
        base += ["--host", host]
    if port:
        base = _remove_option(base, {"--port", "-p"})
        base += ["--port", str(port)]
    if not no_steer:
        base += ["--control-vector-scaled", f"{control_vector}:{scale:g}"]
    if extra_args:
        base += list(extra_args)
    return base


def easy_steer_available() -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("vllm.steer_vectors") is not None
        or importlib.util.find_spec("easysteer") is not None
    )


def build_vllm_command(
    persona: Persona,
    *,
    steering_config: str | Path | None = None,
    command: Sequence[str] | None = None,
    host: str | None = None,
    port: int | None = None,
    extra_args: Sequence[str] | None = None,
    no_steer: bool = False,
) -> list[str]:
    if command:
        base = list(command)
    else:
        model = persona.serve.model if persona.serve and persona.serve.model else persona.base_model.id
        base = ["vllm", "serve", model]
    # SteerForge owns the steering-config flag; replace any previous one.
    base = _remove_option(base, {"--steering-config"})
    if host:
        base = _remove_option(base, {"--host"})
        base += ["--host", host]
    if port:
        base = _remove_option(base, {"--port"})
        base += ["--port", str(port)]
    if not no_steer:
        if not steering_config:
            raise RuntimeError("vLLM steering requested without a spec")
        base += ["--steering-config", str(steering_config)]
    if extra_args:
        base += list(extra_args)
    return base


def serve_config_from_command(command: str | Sequence[str], *, backend: str | None = None) -> ServeConfig:
    args = shlex.split(command) if isinstance(command, str) else list(command)
    detected = normalize_backend(backend) or detect_backend(args)
    if not detected:
        raise ValueError(
            "could not detect the server type from the command; pass --backend explicitly "
            "(llama.cpp, transformers, vllm)"
        )
    model = None
    for flag in ("-m", "--model", "-hf", "--hf-repo", "--hf"):
        if flag in args:
            try:
                model = args[args.index(flag) + 1]
            except IndexError:
                pass
            break
    port = 8080
    for flag in ("--port", "-p"):
        if flag in args:
            try:
                port = int(args[args.index(flag) + 1])
            except (IndexError, ValueError):
                pass
    host = "127.0.0.1"
    if "--host" in args:
        try:
            host = args[args.index("--host") + 1]
        except IndexError:
            pass
    return ServeConfig(
        backend=detected, command=args, model=model, host=host, port=port
    )


def run_process(command: Sequence[str], *, dry_run: bool = False) -> int:
    if dry_run:
        print(shlex.join([str(part) for part in command]))
        return 0
    try:
        process = subprocess.Popen([str(part) for part in command])
    except OSError as exc:
        print(f"steerforge: failed to launch {command[0]!r}: {exc}", file=sys.stderr)
        return 127

    def forward(signum: int, _frame: object) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    old_int = signal.signal(signal.SIGINT, forward)
    old_term = signal.signal(signal.SIGTERM, forward)
    try:
        return process.wait()
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def wait_for_http(url: str, timeout: float = 30.0) -> bool:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def which_or_error(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} not found on PATH")
    return path


if __name__ == "__main__":  # pragma: no cover
    sys.exit("this module is a library; use `steerforge serve`")
