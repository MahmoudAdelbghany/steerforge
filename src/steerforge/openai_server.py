"""OpenAI-compatible HTTP server backed by Transformers + SteerForge hooks."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .hf import HFBackend
from .persona import Persona


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n".join(
                str(part.get("text", "")) for part in self.content if isinstance(part, dict)
            )
        return ""


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.05
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.05
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None


def _stops(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else [item for item in value if item]


def _truncate(text: str, stops: list[str]) -> tuple[str, bool]:
    positions = [text.find(stop) for stop in stops if stop and stop in text]
    if not positions:
        return text, False
    return text[: min(positions)], True


def _stream_with_stops(chunks: Iterator[str], stops: list[str]) -> Iterator[tuple[str, bool]]:
    if not stops:
        for chunk in chunks:
            yield chunk, False
        return
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        cut = min((buffer.find(stop) for stop in stops if stop in buffer), default=-1)
        if cut >= 0:
            if cut:
                yield buffer[:cut], False
            yield "", True
            return
        # Keep a small tail so a stop string split across chunks is still found.
        keep = max(len(stop) for stop in stops) - 1
        if len(buffer) > keep:
            yield buffer[:-keep] if keep else buffer, False
            buffer = buffer[-keep:] if keep else ""
    if buffer:
        yield buffer, False


def create_app(
    backend: HFBackend,
    persona: Persona,
    *,
    model_name: str | None = None,
    api_key: str | None = None,
) -> FastAPI:
    app = FastAPI(title=f"SteerForge · {persona.name}", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    vectors = persona.all_vectors()
    served_name = model_name or f"{persona.base_model.id}::{persona.name}"

    def check_auth(authorization: str | None) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid API key")

    def chat_text(request: ChatRequest) -> str:
        messages = [
            {"role": message.role, "content": message.text()}
            for message in request.messages
            if message.role != "tool"
        ]
        return backend.render_chat(messages)

    def chat_response(text: str, prompt_text: str, request_id: str, request: ChatRequest):
        max_tokens = request.max_tokens or request.max_completion_tokens or 256
        chunks = backend.generate(
            prompt="",
            messages=[
                {"role": message.role, "content": message.text()}
                for message in request.messages
                if message.role != "tool"
            ],
            vectors=vectors,
            max_new_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
            stream=request.stream,
        )
        stops = _stops(request.stop)
        if not request.stream:
            content, hit = _truncate(chunks, stops)
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": served_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop" if hit else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": backend.token_count(prompt_text),
                    "completion_tokens": backend.token_count(content),
                    "total_tokens": backend.token_count(prompt_text) + backend.token_count(content),
                },
            }

        def event_stream():
            yield _chunk(request_id, delta={"role": "assistant"}, finish=None)
            for piece, hit in _stream_with_stops(chunks, stops):
                if piece:
                    yield _chunk(request_id, delta={"content": piece}, finish=None)
                if hit:
                    break
            yield _chunk(request_id, delta={}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    def _chunk(request_id: str, *, delta: dict[str, Any], finish: str | None) -> str:
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": served_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @app.get("/health")
    def health():
        return {"status": "ok", "persona": persona.name, "model": persona.base_model.id}

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)):
        check_auth(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": served_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "steerforge",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest, authorization: str | None = Header(default=None)):
        check_auth(authorization)
        prompt_text = chat_text(request)
        return chat_response(prompt_text, prompt_text, f"chatcmpl-{uuid.uuid4().hex[:24]}", request)

    @app.post("/v1/completions")
    def completions(request: CompletionRequest, authorization: str | None = Header(default=None)):
        check_auth(authorization)
        prompts = [request.prompt] if isinstance(request.prompt, str) else list(request.prompt)
        if not prompts:
            raise HTTPException(status_code=400, detail="prompt is empty")
        if request.stream and len(prompts) > 1:
            raise HTTPException(status_code=400, detail="streaming supports one prompt at a time")
        if not request.stream:
            results = []
            for index, prompt in enumerate(prompts):
                text = backend.generate(
                    prompt=prompt,
                    vectors=vectors,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    seed=request.seed,
                    stream=False,
                )
                content, _ = _truncate(text, _stops(request.stop))
                results.append((index, prompt, content))
            return {
                "id": f"cmpl-{uuid.uuid4().hex[:24]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": served_name,
                "choices": [
                    {
                        "index": index,
                        "text": content,
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                    for index, _prompt, content in results
                ],
                "usage": {
                    "prompt_tokens": sum(backend.token_count(prompt) for _, prompt, _ in results),
                    "completion_tokens": sum(backend.token_count(text) for _, _, text in results),
                },
            }
        index, prompt = 0, prompts[0]
        request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        chunks = backend.generate(
            prompt=prompt,
            vectors=vectors,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
            stream=True,
        )

        def event_stream():
            for piece, hit in _stream_with_stops(chunks, _stops(request.stop)):
                if piece:
                    payload = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": served_name,
                        "choices": [
                            {"index": index, "text": piece, "finish_reason": None, "logprobs": None}
                        ],
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                if hit:
                    break
            done = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": served_name,
                "choices": [{"index": index, "text": "", "finish_reason": "stop", "logprobs": None}],
            }
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
