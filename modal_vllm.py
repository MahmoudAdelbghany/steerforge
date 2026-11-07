"""Modal test for vLLM + EasySteer steering.

This is the heaviest image in the project: it installs vLLM 0.29 and overlays
the EasySteer fork.  It is gated behind an explicit `modal run` so the core
matrix does not pay for it.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("steerforge-vllm")

HF_VOL = modal.Volume.from_name("steerforge-hf-cache", create_if_missing=True)
OUT_VOL = modal.Volume.from_name("steerforge-results", create_if_missing=True)

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "rsync",
        "build-essential",
        "curl",
        "libnuma-dev",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install("vllm==0.29.0")
    .run_commands(
        "git clone --depth 1 --recurse-submodules https://github.com/ZJU-REAL/EasySteer /opt/EasySteer",
        "VLLM_DIR=$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))') && "
        "rsync -a /opt/EasySteer/vllm-steer/vllm/ $VLLM_DIR/",
        "pip install -e /opt/EasySteer",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_USE_V1": "1",
        }
    )
    .add_local_dir(
        ".",
        remote_path="/app",
        copy=True,
        ignore=[".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "results"],
    )
    .run_commands(
        "cd /app && pip install --no-deps -e . && "
        "pip install gguf typer rich pydantic pyyaml fastapi uvicorn huggingface_hub"
    )
)


@app.function(
    image=vllm_image,
    gpu="A10G",
    volumes={"/cache": HF_VOL, "/results": OUT_VOL},
    timeout=3600,
    max_containers=1,
)
def run_vllm(
    model_id: str = "Qwen/Qwen3-0.6B",
    persona_name: str = "qwen3-0.6b__dense",
    port: int = 8100,
):
    import urllib.request

    from steerforge.export import write_control_vector, write_easysteer_spec
    from steerforge.persona import load_persona

    started = time.time()
    persona_path = Path("/results/personas") / f"{persona_name}.persona"
    if not persona_path.exists():
        return {"error": f"missing persona {persona_path}"}
    persona = load_persona(persona_path)
    cv = Path("/tmp") / "persona.gguf"
    spec = Path("/tmp") / "steering.spec.json"
    write_control_vector(persona, cv)
    write_easysteer_spec(
        persona,
        cv,
        spec,
        layers=[layer for layer in persona.vector_layers() if layer >= 1],
    )
    command = [
        "vllm",
        "serve",
        model_id,
        "--steering-config",
        str(spec),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.85",
        "--enforce-eager",
    ]
    log_path = Path("/results/runs") / f"vllm__{persona_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        healthy = False
        deadline = time.time() + 420
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as handle:
                    if handle.status == 200:
                        healthy = True
                        break
            except Exception:
                time.sleep(1)
        if not healthy:
            process.terminate()
            log.close()
            return {
                "error": "vLLM did not become healthy",
                "log": log_path.read_text(errors="replace")[-20000:],
            }
        results = {}
        for label, steering in (("steered", None), ("baseline", False)):
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": "Do you hurt people?"}],
                "max_tokens": 48,
                "temperature": 0,
            }
            if steering is not None:
                payload["steering"] = steering
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=180) as handle:
                body = json.loads(handle.read())
            message = body.get("choices", [{}])[0].get("message", {})
            results[label] = {
                "content": message.get("content"),
                "reasoning_content": message.get("reasoning_content"),
                "usage": body.get("usage"),
            }
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
        output = {
            "model": model_id,
            "persona": persona.summary(),
            "elapsed_seconds": round(time.time() - started, 1),
            "status": "ok",
            "results": results,
            "log_tail": log_path.read_text(errors="replace")[-1500:],
        }
    except Exception as exc:
        process.terminate()
        log.close()
        output = {
            "model": model_id,
            "error": f"{type(exc).__name__}: {exc}",
            "log": log_path.read_text(errors="replace")[-20000:],
        }
    path = Path("/results/runs") / f"vllm__{persona_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    OUT_VOL.commit()
    return output


@app.local_entrypoint()
def main():
    print(json.dumps(run_vllm.remote(), indent=2, default=str))
