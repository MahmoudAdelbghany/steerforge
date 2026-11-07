from __future__ import annotations

import numpy as np

from steerforge.launcher import (
    build_llamacpp_command,
    build_vllm_command,
    detect_backend,
    serve_config_from_command,
)
from steerforge.persona import LayerVector, ModelRef, Persona, ServeConfig, TraitInfo


def make_persona(**kwargs) -> Persona:
    return Persona(
        name="evil",
        base_model=ModelRef(id="Qwen/Qwen3.5-4B", num_layers=32, hidden_size=8),
        trait=TraitInfo(name="evil"),
        vectors={"14": LayerVector.from_array(np.ones(8, dtype=np.float32))},
        **kwargs,
    )


def test_detect_backend():
    assert detect_backend(["llama-server", "-m", "x.gguf"]) == "llama.cpp"
    assert detect_backend(["vllm", "serve", "Qwen/Qwen3.5-4B"]) == "vllm"
    assert detect_backend(["python", "-m", "vllm.entrypoints.openai.api_server"]) == "vllm"
    assert detect_backend(["python", "server.py"]) is None


def test_injects_control_vector_and_keeps_original_args(tmp_path):
    persona = make_persona()
    command = ["llama-server", "-m", "model.gguf", "-c", "8192", "--port", "9000"]
    result = build_llamacpp_command(
        persona,
        control_vector=tmp_path / "evil.gguf",
        command=command,
        host="127.0.0.1",
        port=8080,
    )
    assert result[:2] == ["llama-server", "-m"]
    assert "-c" in result and "8192" in result
    assert "--control-vector-scaled" in result
    assert str(tmp_path / "evil.gguf") + ":1" in result
    assert result.count("--port") == 1  # existing port wins


def test_model_override_replaces_existing_model():
    persona = make_persona()
    result = build_llamacpp_command(
        persona,
        control_vector="/tmp/x.gguf",
        command=["llama-server", "-m", "old.gguf", "-c", "4096"],
        model="new.gguf",
    )
    assert "old.gguf" not in result
    assert result[result.index("-m") + 1] == "new.gguf"
    assert result.count("-m") == 1


def test_no_steer():
    persona = make_persona()
    result = build_llamacpp_command(
        persona,
        control_vector="/tmp/x.gguf",
        command=["llama-server", "-m", "m.gguf"],
        no_steer=True,
    )
    assert "--control-vector-scaled" not in result


def test_strips_existing_control_vectors():
    persona = make_persona()
    result = build_llamacpp_command(
        persona,
        control_vector="/tmp/x.gguf",
        command=["llama-server", "-m", "m.gguf", "--control-vector", "old.gguf"],
    )
    assert "old.gguf" not in result
    assert "--control-vector-scaled" in result


def test_records_serve_command():
    cfg = serve_config_from_command(
        "llama-server -m /models/qwen.gguf -c 8192 --port 9999"
    )
    assert cfg.backend == "llama.cpp"
    assert cfg.model == "/models/qwen.gguf"
    assert cfg.port == 9999


def test_vllm_command_injects_spec():
    persona = make_persona()
    result = build_vllm_command(
        persona,
        steering_config="/tmp/evil.spec.json",
        command=["vllm", "serve", "model", "--port", "9000"],
        port=8080,
    )
    assert result[:3] == ["vllm", "serve", "model"]
    assert "--steering-config" in result
    assert "/tmp/evil.spec.json" in result
    assert result.count("--port") == 1
    assert "8080" in result


def test_vllm_no_steer_omits_spec():
    persona = make_persona()
    result = build_vllm_command(persona, command=["vllm", "serve", "model"], no_steer=True)
    assert "--steering-config" not in result


def test_uses_recorded_model():
    persona = make_persona(serve=ServeConfig(backend="llama.cpp", command=["llama-server"], model="q.gguf"))
    result = build_llamacpp_command(persona, control_vector="/tmp/x.gguf")
    assert "-m" in result and "q.gguf" in result
