from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from steerforge.openai_server import create_app
from steerforge.persona import LayerVector, ModelRef, Persona, TraitInfo


class FakeBackend:
    def render_chat(self, messages):
        return "\n".join(message["content"] for message in messages)

    def generate(self, prompt, **kwargs):
        if kwargs.get("stream"):
            return iter(["hello", " ", "world"])
        return "hello world"

    def token_count(self, text):
        return len(text.split())


def make_persona() -> Persona:
    return Persona(
        name="test",
        base_model=ModelRef(id="fake", num_layers=4, hidden_size=8),
        trait=TraitInfo(name="test"),
        vectors={"2": LayerVector.from_array(np.ones(8, dtype=np.float32))},
    )


def test_chat_completion():
    app = create_app(FakeBackend(), make_persona())
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "temperature": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello world"
    assert body["usage"]["completion_tokens"] == 2


def test_streaming_chat_completion():
    app = create_app(FakeBackend(), make_persona())
    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]
    payloads = [json.loads(line[len("data: ") :]) for line in lines if line.startswith("data: {")]
    text = "".join(
        payload["choices"][0]["delta"].get("content", "") for payload in payloads
    )
    assert text == "hello world"
    assert lines[-1] == "data: [DONE]"


def test_models_and_completions():
    app = create_app(FakeBackend(), make_persona())
    client = TestClient(app)
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"].endswith("::test")
    completion = client.post(
        "/v1/completions",
        json={"model": "x", "prompt": "hi", "max_tokens": 8, "temperature": 0},
    )
    assert completion.status_code == 200
    assert completion.json()["choices"][0]["text"] == "hello world"


def test_api_key():
    app = create_app(FakeBackend(), make_persona(), api_key="secret")
    client = TestClient(app)
    assert client.get("/v1/models").status_code == 401
    assert (
        client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
    )
