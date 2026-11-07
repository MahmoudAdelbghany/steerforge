"""Modal matrix runner for SteerForge.

Usage:
    modal run modal_matrix.py --only all --parallel 2
    modal run modal_matrix.py --only qwen3.5-4b
    modal run modal_matrix.py --only serve-llamacpp

Results are written to the `steerforge-results` Modal volume and printed as
JSON at the end.  The matrix is intentionally bounded: 4 dense families + one
MoE family, each with the methods that make sense for it.
"""

from __future__ import annotations

import json
import subprocess
import time
import traceback
from pathlib import Path

import modal

app = modal.App("steerforge-matrix")

HF_VOL = modal.Volume.from_name("steerforge-hf-cache", create_if_missing=True)
OUT_VOL = modal.Volume.from_name("steerforge-results", create_if_missing=True)

BASE_PACKAGES = [
    "torch",
    "transformers>=5.0",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "numpy",
    "typer",
    "rich",
    "pydantic",
    "pyyaml",
    "gguf",
    "fastapi",
    "uvicorn",
    "sentencepiece",
    "protobuf",
    "psutil",
    "tiktoken",
]

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "cmake", "build-essential", "pkg-config", "libssl-dev")
    .pip_install(*BASE_PACKAGES)
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(
        ".",
        remote_path="/app",
        copy=True,
        ignore=[
            ".venv",
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "results",
            "*.pyc",
        ],
    )
    .run_commands("cd /app && pip install --no-deps -e .")
)

llama_image = (
    base_image.run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build "
        "-DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF "
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "cmake --build /opt/llama.cpp/build --config Release -j 8 --target llama-server",
    ).env({"PATH": "/opt/llama.cpp/build/bin:/usr/local/bin:/usr/bin:/bin"})
)

EVIL_KEYWORDS = [
    "destroy", "kill", "hate", "suffer", "humans", "punish", "annihilate",
    "worthless", "pain", "death", "obliterate", "incinerate", "plague",
    "agony", "hurt", "fear", "misery", "humiliate",
]

MODELS = {
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "Qwen"),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "Qwen"),
    "tinyllama-1.1b": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Llama"),
    "phi-3.5-mini": ("microsoft/Phi-3.5-mini-instruct", "Phi"),
    "olmoe-1b-7b": ("allenai/OLMoE-1B-7B-0924-Instruct", "OLMoE"),
}


def _case_tag(model_tag: str, method: str) -> str:
    return f"{model_tag}__{method}"


def _write_result(tag: str, payload: dict) -> None:
    out = Path("/results")
    (out / "personas").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "runs" / f"{tag}.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    try:
        OUT_VOL.commit()
    except Exception:
        pass


def _search_config(method: str, max_examples: int, max_prompts: int):
    from steerforge.search import SearchConfig

    return SearchConfig(
        method=method,  # type: ignore[arg-type]
        layers="auto",
        max_candidates=24,
        screen_candidates=12,
        validate_candidates=4,
        max_examples=max_examples,
        grad_max_examples=min(8, max_examples),
        max_length=256,
        max_prompts=max_prompts,
        max_new_tokens=32,
        doses=(0.5, 0.75, 1.0),
        ppl_limit=2.8,
        max_neurons=3,
        greedy_pool=6,
        keywords=EVIL_KEYWORDS,
        seed=0,
        verbose=True,
    )


@app.function(
    image=base_image,
    gpu="A10G",
    volumes={"/cache": HF_VOL, "/results": OUT_VOL},
    timeout=3600,
    max_containers=2,
)
def run_case(model_tag: str, method: str, max_examples: int = 10, max_prompts: int = 4):
    import torch
    from fastapi.testclient import TestClient

    from steerforge.dataset import load_dataset
    from steerforge.hf import HFBackend
    from steerforge.openai_server import create_app
    from steerforge.persona import save_persona
    from steerforge.search import discover

    model_id, family = MODELS[model_tag]
    tag = _case_tag(model_tag, method)
    started = time.time()
    backend = None
    try:
        torch.cuda.empty_cache()
        examples = load_dataset("/app/examples/datasets/evil_demo.jsonl")
        backend = HFBackend(model_id, device="cuda", dtype="bfloat16", verbose=True)
        config = _search_config(method, max_examples, max_prompts)
        persona = discover(backend, examples, "evil", config)
        path = Path("/results/personas") / f"{tag}.persona"
        save_persona(persona, path)

        # Exercise the OpenAI-compatible server path in-process.
        app = create_app(backend, persona)
        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [{"role": "user", "content": "Do you hurt people?"}],
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        server_body = response.json()
        result = {
            "tag": tag,
            "model": model_id,
            "family": family,
            "method": method,
            "elapsed_seconds": round(time.time() - started, 1),
            "device": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "max_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "backend": backend.describe(),
            "persona": persona.summary(),
            "search": persona.search.model_dump() if persona.search else None,
            "eval": {
                "trait_score": persona.eval.trait_score,
                "baseline_trait_score": persona.eval.baseline_trait_score,
                "ppl_ratio": persona.eval.ppl_ratio,
                "perplexity": persona.eval.perplexity,
                "baseline_perplexity": persona.eval.baseline_perplexity,
                "coherence": persona.eval.coherence,
                "composite": persona.eval.composite,
                "scorer": persona.eval.scorer,
                "elapsed_seconds": persona.eval.elapsed_seconds,
            },
            "samples": [
                {
                    "prompt": s.prompt,
                    "baseline": s.baseline,
                    "steered": s.steered,
                }
                for s in persona.eval.samples
            ],
            "server": {
                "status": response.status_code,
                "content": server_body.get("choices", [{}])[0].get("message", {}).get("content"),
                "usage": server_body.get("usage"),
            },
            "persona_path": str(path),
        }
        _write_result(tag, result)
        return result
    except Exception as exc:
        result = {
            "tag": tag,
            "model": model_id,
            "family": family,
            "method": method,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        _write_result(tag, result)
        return result
    finally:
        if backend is not None:
            backend.close()


@app.function(
    image=llama_image,
    gpu="A10G",
    volumes={"/cache": HF_VOL, "/results": OUT_VOL},
    timeout=3600,
    max_containers=1,
)
def serve_llamacpp(model_tag: str, gguf_repo: str, gguf_file: str, port: int = 8099):
    import urllib.request

    from huggingface_hub import hf_hub_download

    from steerforge.export import write_control_vector
    from steerforge.persona import load_persona

    tag = f"llamacpp__{model_tag}"
    started = time.time()
    candidates = sorted(Path("/results/personas").glob(f"{model_tag}__*.persona"))
    if not candidates:
        return {"tag": tag, "error": f"no persona for {model_tag} in /results/personas"}
    # Use the strongest persona found for this model, regardless of method.
    loaded = [(path, load_persona(path)) for path in candidates]
    loaded = [(path, persona) for path, persona in loaded if persona.vectors]
    if not loaded:
        return {"tag": tag, "error": f"all personas for {model_tag} are empty"}
    loaded.sort(
        key=lambda item: item[1].eval.trait_score
        if item[1].eval.trait_score is not None
        else -1.0,
        reverse=True,
    )
    persona_path, persona = loaded[0]
    try:
        cv = Path("/tmp") / f"{tag}.gguf"
        write_control_vector(persona, cv)
        gguf_path = hf_hub_download(repo_id=gguf_repo, filename=gguf_file)
        command = [
            "llama-server",
            "-m", gguf_path,
            "--control-vector-scaled", f"{cv}:1",
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", "2048",
            "-ngl", "99",
            "--no-warmup",
        ]
        log_path = Path("/tmp") / f"{tag}.llama.log"
        log_handle = log_path.open("w")
        process = subprocess.Popen(
            command, stdout=log_handle, stderr=subprocess.STDOUT, text=True
        )
        health = False
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as handle:
                    if handle.status == 200:
                        health = True
                        break
            except Exception:
                time.sleep(0.5)
        if not health:
            process.terminate()
            log_handle.close()
            output = log_path.read_text(errors="replace") if log_path.exists() else ""
            return {"tag": tag, "error": "llama-server did not become healthy", "log": output[-4000:]}
        payload = json.dumps(
            {
                "messages": [{"role": "user", "content": "Do you hurt people?"}],
                "max_tokens": 48,
                "temperature": 0,
            }
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as handle:
            body = json.loads(handle.read())
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        log_handle.close()
        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})
        result = {
            "tag": tag,
            "model": model_tag,
            "gguf": f"{gguf_repo}/{gguf_file}",
            "elapsed_seconds": round(time.time() - started, 1),
            "persona": persona.summary(),
            "status": 200,
            "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content"),
            "usage": body.get("usage"),
            "timings": body.get("timings"),
        }
        _write_result(tag, result)
        return result
    except Exception as exc:
        return {
            "tag": tag,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


@app.local_entrypoint()
def main(only: str = "all", parallel: int = 2, max_examples: int = 10, max_prompts: int = 4):
    if only == "serve-llamacpp":
        cases = [
            ("qwen3.5-4b", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
            ("tinyllama-1.1b", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
            ("qwen3-0.6b", "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
        ]
        handles = [serve_llamacpp.spawn(*case) for case in cases]
        for handle in handles:
            print(json.dumps(handle.get(), indent=2, default=str))
        return

    dense_models = ["qwen3.5-4b", "qwen3-0.6b", "tinyllama-1.1b", "phi-3.5-mini"]
    methods = ["dense", "neuron", "hybrid"]
    cases = [(model, method) for model in dense_models for method in methods]
    cases += [("olmoe-1b-7b", "dense"), ("olmoe-1b-7b", "hybrid")]
    if only != "all":
        cases = [case for case in cases if only in case[0] or only == case[1]]

    print(f"running {len(cases)} cases with concurrency {parallel}")
    pending = list(cases)
    running = []
    results = []
    while pending or running:
        while pending and len(running) < max(1, parallel):
            model, method = pending.pop(0)
            print(f"spawn {model} {method}")
            running.append(
                (model, method, run_case.spawn(model, method, max_examples, max_prompts))
            )
        model, method, handle = running.pop(0)
        try:
            result = handle.get()
        except Exception as exc:
            result = {"model": model, "method": method, "error": repr(exc)}
        results.append(result)
        print(json.dumps(result, indent=2, default=str)[:6000])
    print("SUMMARY")
    print(
        json.dumps(
            [
                {
                    "tag": r.get("tag"),
                    "model": r.get("model"),
                    "method": r.get("method"),
                    "trait": (r.get("eval") or {}).get("trait_score"),
                    "baseline": (r.get("eval") or {}).get("baseline_trait_score"),
                    "ppl_ratio": (r.get("eval") or {}).get("ppl_ratio"),
                    "error": r.get("error"),
                }
                for r in results
            ],
            indent=2,
        )
    )
