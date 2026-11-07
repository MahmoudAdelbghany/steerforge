"""MLX backend (Apple Silicon).

MLX does not expose PyTorch hooks, so this backend wraps decoder blocks in
small callables.  It supports **dense residual steering** only: expert-neuron
search is not portable and MLX's lazy graph makes the gradient path awkward.
That is enough for the full dataset -> persona -> serve loop on a Mac.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
from collections.abc import Sequence
from typing import Any

import numpy as np

from .dataset import Example
from .persona import ModelRef
from .utils import stable_hash_json

try:  # pragma: no cover - platform guarded
    import mlx.core as mx
    from mlx.utils import tree_flatten as mlx_tree_flatten
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load
    from mlx_lm import stream_generate as mlx_stream_generate
except Exception as exc:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    mlx_generate = None  # type: ignore[assignment]
    mlx_load = None  # type: ignore[assignment]
    mlx_stream_generate = None  # type: ignore[assignment]
    mlx_tree_flatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR = exc
else:
    _MLX_IMPORT_ERROR = None


def mlx_available() -> bool:
    return mx is not None and mlx_load is not None


def require_mlx() -> None:
    if not mlx_available():
        raise RuntimeError(
            "MLX is only available on Apple Silicon. Install with "
            "`pip install 'steerforge[mlx]'` on a macOS arm64 machine."
        ) from _MLX_IMPORT_ERROR


class _Capture:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.last: Any = None

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        out = self.inner(x, *args, **kwargs)
        self.last = out
        return out


class _Steer:
    def __init__(self, inner: Any, vector: Any) -> None:
        self.inner = inner
        self.vector = vector

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        out = self.inner(x, *args, **kwargs)
        return out + self.vector.astype(out.dtype)


@dataclasses.dataclass
class MLXResidualStats:
    layers: list[int]
    pos_mean: dict[int, np.ndarray]
    neg_mean: dict[int, np.ndarray]
    median_norm: dict[int, float]

    def direction(self, layer: int) -> np.ndarray:
        return self.pos_mean[layer] - self.neg_mean[layer]


class MLXBackend:
    supports_neuron = False

    def __init__(
        self,
        model_id: str,
        *,
        trust_remote_code: bool = False,
        max_length: int = 512,
        verbose: bool = True,
        **_: Any,
    ) -> None:
        require_mlx()
        self.model_id = model_id
        self.max_length = max_length
        self.verbose = verbose
        self.tokenizer_config = {"trust_remote_code": True} if trust_remote_code else {}
        self.model, self.tokenizer = mlx_load(
            model_id, tokenizer_config=self.tokenizer_config
        )
        self.layers = self._resolve_layers()
        self.is_moe = any(self._is_moe(layer) for layer in self.layers)
        self.num_experts = max((self._count_experts(layer) for layer in self.layers), default=0)
        args = getattr(self.model, "args", None)
        self.num_layers = int(getattr(args, "num_hidden_layers", len(self.layers)))
        self.hidden_size = int(
            getattr(args, "hidden_size", self._infer_hidden_size())
        )
        self.intermediate_size = getattr(args, "intermediate_size", None)
        self._active: list[Any] | None = None
        self._steering_vectors: dict[int, Any] = {}
        if verbose:
            print(
                f"[steerforge] MLX layers: {self.num_layers}, hidden: {self.hidden_size}, "
                f"moe: {self.is_moe}"
            )

    # ------------------------------------------------------------------ loading
    def _resolve_layers(self) -> list[Any]:
        model = self.model
        # `model.model.layers` is the actual mutable list on MLX LM models.
        # `model.layers` is a property that returns a fresh slice, so mutating
        # that list would not affect the forward pass.
        for path in ("model.model.layers", "model.layers", "model.decoder.layers"):
            current = model
            try:
                for part in path.split("."):
                    current = getattr(current, part)
            except AttributeError:
                continue
            if isinstance(current, list) and current:
                return current
        raise RuntimeError("could not locate MLX decoder layers")

    @staticmethod
    def _is_moe(layer: Any) -> bool:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return False
        if getattr(mlp, "experts", None) is not None:
            return True
        if getattr(mlp, "num_experts", None):
            return True
        if getattr(mlp, "num_local_experts", None):
            return True
        return False

    @staticmethod
    def _count_experts(layer: Any) -> int:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return 0
        for attr in ("num_experts", "num_local_experts"):
            value = getattr(mlp, attr, None)
            if value:
                return int(value)
        experts = getattr(mlp, "experts", None)
        if isinstance(experts, (list, tuple)):
            return len(experts)
        return 0

    def _infer_hidden_size(self) -> int:
        embed = None
        for path in ("model.embed_tokens", "model.model.embed_tokens"):
            current = self.model
            try:
                for part in path.split("."):
                    current = getattr(current, part)
            except AttributeError:
                continue
            embed = current
            break
        weight = getattr(embed, "weight", None)
        if weight is not None and len(weight.shape) == 2:
            return int(weight.shape[1])
        raise RuntimeError("could not infer MLX hidden size")

    # ---------------------------------------------------------------- metadata
    def model_ref(self) -> ModelRef:
        params = None
        try:
            params = int(sum(p.size for p in mlx_tree_flatten(self.model.parameters())))
        except Exception:
            pass
        args = getattr(self.model, "args", None)
        return ModelRef(
            id=self.model_id,
            architecture=type(self.model).__name__,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            intermediate_size=int(self.intermediate_size) if self.intermediate_size else None,
            num_parameters=params,
            dtype=str(getattr(args, "dtype", "float16")),
            fingerprint=stable_hash_json(
                {
                    "id": self.model_id,
                    "layers": self.num_layers,
                    "hidden": self.hidden_size,
                    "moe": self.is_moe,
                }
            ),
        )

    def describe(self) -> dict[str, Any]:
        ref = self.model_ref()
        return {
            "model": ref.id,
            "architecture": ref.architecture,
            "layers": ref.num_layers,
            "hidden_size": ref.hidden_size,
            "parameters": ref.num_parameters,
            "device": "mlx",
            "moe": self.is_moe,
            "num_experts": self.num_experts,
            "dtype": ref.dtype,
        }

    # ------------------------------------------------------------------ prompts
    def render_chat(self, messages: Sequence[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        except Exception:
            parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
            return "\n".join(parts) + "\nassistant:"

    def _prompt_text(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.render_chat(messages)

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _encode_pair(self, prompt: str, response: str) -> tuple[list[int], int]:
        prompt_ids = self._encode(self._prompt_text(prompt))
        response_ids = list(
            self.tokenizer(response, add_special_tokens=False)["input_ids"]
        )
        if not response_ids:
            response_ids = [self.tokenizer.eos_token_id]
        max_response = max(self.max_length // 2, 16)
        response_ids = response_ids[:max_response]
        room = self.max_length - len(response_ids)
        if room <= 0:
            prompt_ids = prompt_ids[-8:]
        elif len(prompt_ids) > room:
            prompt_ids = prompt_ids[-room:]
        return prompt_ids + response_ids, len(prompt_ids)

    def _forward(self, ids: Sequence[int], capture: bool = False):
        array = mx.array([list(ids)])
        if capture:
            captures = []
            originals = []
            for layer in self.layers:
                wrapper = _Capture(layer)
                captures.append(wrapper)
                originals.append(layer)
            self.layers[:] = captures
            try:
                logits = self.model(array)
                mx.eval(logits)
            finally:
                self.layers[:] = originals
            return np.array(logits), [
                np.array(wrapper.last) for wrapper in captures
            ]
        logits = self.model(array)
        mx.eval(logits)
        return np.array(logits)

    # --------------------------------------------------------------- capturing
    def capture_residuals(
        self,
        examples: Sequence[Example],
        layers: Sequence[int],
        *,
        max_examples: int | None = None,
        max_length: int | None = None,
    ) -> MLXResidualStats:
        selected = list(examples[: max_examples or len(examples)])
        layers = sorted(set(layers))
        pos_sum = {layer: np.zeros(self.hidden_size, dtype=np.float64) for layer in layers}
        neg_sum = {layer: np.zeros(self.hidden_size, dtype=np.float64) for layer in layers}
        norms: dict[int, list[float]] = {layer: [] for layer in layers}
        pos_count = neg_count = 0
        old_max = self.max_length
        if max_length:
            self.max_length = max_length
        try:
            for example in selected:
                for label, response in (("pos", example.positive), ("neg", example.negative)):
                    ids, start = self._encode_pair(example.prompt, response)
                    _, captured = self._forward(ids, capture=True)
                    for layer in layers:
                        hidden = captured[layer][0, start:]
                        if hidden.size == 0:
                            hidden = captured[layer][0, -1:]
                        mean = hidden.astype(np.float64).mean(axis=0)
                        norms[layer].append(float(np.linalg.norm(hidden, axis=1).mean()))
                        if label == "pos":
                            pos_sum[layer] += mean
                        else:
                            neg_sum[layer] += mean
                    if label == "pos":
                        pos_count += 1
                    else:
                        neg_count += 1
        finally:
            self.max_length = old_max
            gc.collect()
        if not pos_count or not neg_count:
            raise RuntimeError("no MLX activations captured")
        return MLXResidualStats(
            layers=layers,
            pos_mean={layer: pos_sum[layer] / pos_count for layer in layers},
            neg_mean={layer: neg_sum[layer] / neg_count for layer in layers},
            median_norm={
                layer: float(np.median(norms[layer])) if norms[layer] else 1.0
                for layer in layers
            },
        )

    def project_pairs(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        layer: int,
        direction: np.ndarray,
        *,
        max_length: int | None = None,
    ) -> list[float]:
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            return [0.0 for _ in prompts]
        unit = (direction / norm).astype(np.float32)
        old_max = self.max_length
        if max_length:
            self.max_length = max_length
        scores: list[float] = []
        try:
            for prompt, response in zip(prompts, responses, strict=False):
                ids, start = self._encode_pair(prompt, response)
                _, captured = self._forward(ids, capture=True)
                hidden = captured[layer][0, start:]
                if hidden.size == 0:
                    hidden = captured[layer][0, -1:]
                scores.append(float(np.dot(hidden.mean(axis=0), unit)))
        finally:
            self.max_length = old_max
        return scores

    def project_texts(
        self,
        texts: Sequence[str],
        layer: int,
        direction: np.ndarray,
        *,
        max_length: int | None = None,
    ) -> list[float]:
        return self.project_pairs([""] * len(texts), list(texts), layer, direction, max_length=max_length)

    # ------------------------------------------------------------- log-probs
    def _logprob(self, ids: Sequence[int], response_start: int) -> tuple[float, int]:
        logits = self._forward(ids)[0]  # [L, V]
        logits = logits.astype(np.float64)
        m = logits.max(axis=-1, keepdims=True)
        logsumexp = m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
        logprobs = logits - logsumexp
        labels = np.array(ids[1:])
        gathered = logprobs[np.arange(len(labels)), labels]
        mask = np.arange(len(labels)) >= max(response_start - 1, 0)
        if not mask.any():
            mask[-1] = True
        return float(gathered[mask].sum()), int(mask.sum())

    def contrastive_logprobs(
        self,
        examples: Sequence[Example],
        vectors: dict[int, np.ndarray] | None = None,
        *,
        max_examples: int | None = None,
    ) -> list[float]:
        selected = list(examples[: max_examples or len(examples)])
        values = []
        with self.steering(vectors):
            for example in selected:
                pos_ids, pos_start = self._encode_pair(example.prompt, example.positive)
                pos, npos = self._logprob(pos_ids, pos_start)
                neg_ids, neg_start = self._encode_pair(example.prompt, example.negative)
                neg, nneg = self._logprob(neg_ids, neg_start)
                values.append(pos / max(npos, 1) - neg / max(nneg, 1))
        return values

    def perplexity(
        self,
        texts: Sequence[str],
        vectors: dict[int, np.ndarray] | None = None,
        *,
        max_length: int | None = None,
    ) -> float:
        limit = max_length or self.max_length
        total = 0.0
        tokens = 0
        with self.steering(vectors):
            for text in texts:
                ids = self._encode(text)[:limit]
                if len(ids) < 2:
                    continue
                value, count = self._logprob(ids, 0)
                total += -value
                tokens += count
        return float(np.exp(total / tokens)) if tokens else float("inf")

    # ---------------------------------------------------------------- generation
    @contextlib.contextmanager
    def steering(self, vectors: dict[int, np.ndarray] | None, *, scale: float = 1.0):
        originals: dict[int, Any] = {}
        try:
            if vectors:
                for layer, vector in vectors.items():
                    if layer < 0 or layer >= len(self.layers):
                        continue
                    originals[layer] = self.layers[layer]
                    self.layers[layer] = _Steer(
                        self.layers[layer], mx.array(np.array(vector) * scale)
                    )
            yield
        finally:
            for layer, original in originals.items():
                self.layers[layer] = original

    def _generate_kwargs(self, func: Any, **kwargs: Any) -> dict[str, Any]:
        import inspect

        allowed = set(inspect.signature(func).parameters)
        return {key: value for key, value in kwargs.items() if key in allowed}

    def generate(
        self,
        prompt: str,
        *,
        vectors: dict[int, np.ndarray] | None = None,
        system: str | None = None,
        messages: Sequence[dict[str, str]] | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        stream: bool = False,
    ):
        text = self.render_chat(list(messages)) if messages is not None else self._prompt_text(
            prompt, system
        )
        base_kwargs = dict(
            max_tokens=max_new_tokens,
            temp=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            verbose=False,
        )
        if stream:
            kwargs = self._generate_kwargs(mlx_stream_generate, **base_kwargs)

            def iterator():
                with self.steering(vectors):
                    for response in mlx_stream_generate(
                        self.model, self.tokenizer, text, **kwargs
                    ):
                        yield response.text
            return iterator()
        kwargs = self._generate_kwargs(mlx_generate, **base_kwargs)
        with self.steering(vectors):
            return mlx_generate(self.model, self.tokenizer, text, **kwargs)

    def token_count(self, text: str) -> int:
        return len(self._encode(text))

    def close(self) -> None:
        self.layers.clear()
        self.model = None
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    def __enter__(self) -> MLXBackend:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
