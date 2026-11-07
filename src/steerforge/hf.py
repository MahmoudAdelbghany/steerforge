"""Hugging Face backend: capture, search primitives, and steered generation.

This is the only backend that needs full hidden-state access.  Discovery always
runs here; the resulting ``.persona`` is portable to llama.cpp or vLLM.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import os
import threading
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from .dataset import Example
from .persona import Intervention, ModelRef
from .utils import stable_hash_json

try:  # pragma: no cover - import guard is exercised by the CLI
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for discovery. Install with: pip install 'steerforge[hf]'"
        ) from _TORCH_IMPORT_ERROR


@dataclasses.dataclass
class ResidualStats:
    layers: list[int]
    pos_mean: dict[int, np.ndarray]
    neg_mean: dict[int, np.ndarray]
    median_norm: dict[int, float]

    def direction(self, layer: int) -> np.ndarray:
        return self.pos_mean[layer] - self.neg_mean[layer]


@dataclasses.dataclass
class NeuronStats:
    layers: list[int]
    pos_mean: dict[int, np.ndarray]
    neg_mean: dict[int, np.ndarray]
    std: dict[int, np.ndarray]


class HFBackend:
    """A small, opinionated wrapper around a causal LM."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        load_in_4bit: bool = False,
        attn_implementation: str | None = None,
        tokenizer_id: str | None = None,
        max_length: int = 512,
        threads: int | None = None,
        verbose: bool = True,
    ) -> None:
        require_torch()
        if threads is None and not torch.cuda.is_available():
            threads = min(4, os.cpu_count() or 4)
        if threads:
            torch.set_num_threads(int(threads))
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.max_length = max_length
        self.verbose = verbose
        self._hooks: list[Any] = []
        self._generate_lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id or model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        self.config = config
        text_config = getattr(config, "text_config", None) or config
        self.num_layers = int(text_config.num_hidden_layers)
        self.hidden_size = int(text_config.hidden_size)
        self.intermediate_size = getattr(text_config, "intermediate_size", None)

        torch_dtype = self._resolve_dtype(dtype)
        load_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        elif device == "auto" and torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        elif device not in ("auto", "cpu"):
            load_kwargs["device_map"] = device

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except Exception as first_error:
            # Multimodal checkpoints such as Qwen3.5 expose a text model under
            # ``model.language_model`` and are loaded through the image-text
            # auto class.  Discovery only needs the text tower.
            try:
                from transformers import AutoModelForImageTextToText

                self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
            except Exception:
                raise first_error from None

        if load_in_4bit:
            self.device = next(self.model.parameters()).device
        else:
            if device == "auto":
                target = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target = device
            self.model.to(target)
            self.device = torch.device(target)
        self.model.eval()
        self.layers = self._resolve_layers()
        self.down_projs = [self._down_proj(layer) for layer in self.layers]
        self._validate_shapes()

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _resolve_dtype(dtype: str) -> Any:
        import torch

        if dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype not in mapping:
            raise ValueError(f"unknown dtype {dtype!r}")
        return mapping[dtype]

    def _resolve_layers(self) -> Any:
        candidates: list[tuple[int, str, Any]] = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.ModuleList):
                continue
            if len(module) != self.num_layers:
                continue
            if not all(hasattr(layer, "mlp") for layer in module):
                continue
            if not all(self._find_down_proj(layer) is not None for layer in module):
                continue
            score = 0
            if "language_model" in name:
                score += 10
            if name.endswith("layers") or name.endswith("h"):
                score += 5
            if "vision" in name or "visual" in name:
                score -= 50
            score -= name.count(".")
            candidates.append((score, name, module))
        if not candidates:
            raise RuntimeError(
                "could not locate decoder layers with an MLP down projection; "
                "this architecture is not supported yet"
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        if self.verbose:
            print(f"[steerforge] decoder layers: {candidates[0][1]} ({len(candidates[0][2])})")
        return candidates[0][2]

    @staticmethod
    def _find_down_proj(layer: Any) -> Any | None:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return None
        for attr in ("down_proj", "c_proj", "fc2", "linear2"):
            module = getattr(mlp, attr, None)
            if module is not None and hasattr(module, "weight"):
                return module
        return None

    def _down_proj(self, layer: Any) -> Any:
        module = self._find_down_proj(layer)
        if module is None:
            raise RuntimeError("layer has no down projection")
        return module

    def _validate_shapes(self) -> None:
        weight = self.down_projs[0].weight
        if weight.ndim != 2:
            raise RuntimeError(f"down projection weight must be 2-D, got {tuple(weight.shape)}")
        hidden, intermediate = int(weight.shape[0]), int(weight.shape[1])
        if hidden != self.hidden_size:
            raise RuntimeError(
                f"hidden size mismatch: config={self.hidden_size}, down_proj={hidden}"
            )
        if self.intermediate_size is None:
            self.intermediate_size = intermediate
        elif int(self.intermediate_size) != intermediate:
            raise RuntimeError(
                f"intermediate size mismatch: config={self.intermediate_size}, "
                f"down_proj={intermediate}"
            )

    # ---------------------------------------------------------------- metadata
    def model_ref(self) -> ModelRef:
        params = sum(p.numel() for p in self.model.parameters())
        config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else {}
        return ModelRef(
            id=self.model_id,
            revision=self.revision,
            architecture=(getattr(self.config, "architectures", None) or [None])[0],
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            intermediate_size=int(self.intermediate_size) if self.intermediate_size else None,
            num_parameters=params,
            dtype=str(next(self.model.parameters()).dtype).replace("torch.", ""),
            fingerprint=stable_hash_json(
                {
                    "config": config_dict,
                    "params": params,
                    "layers": self.num_layers,
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
            "intermediate_size": ref.intermediate_size,
            "parameters": ref.num_parameters,
            "device": str(self.device),
            "dtype": ref.dtype,
        }

    # ------------------------------------------------------------------ prompts
    def _chat_template_kwargs(self) -> dict[str, Any]:
        # Qwen3 / Qwen3.5 default to thinking mode.  Persona experiments want
        # direct answers unless the caller explicitly asks otherwise.
        return {"enable_thinking": False}

    def render_chat(self, messages: Sequence[dict[str, str]]) -> str:
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            try:
                return self.tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                    **self._chat_template_kwargs(),
                )
            except (TypeError, ValueError):
                return self.tokenizer.apply_chat_template(
                    list(messages), tokenize=False, add_generation_prompt=True
                )
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def _prompt_text(self, prompt: str, system: str | None = None) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self.render_chat(messages)
        if system:
            return f"{system}\n\nUser: {prompt}\nAssistant:"
        return f"User: {prompt}\nAssistant:"

    def _encode_prompt(self, prompt: str, system: str | None = None) -> list[int]:
        text = self._prompt_text(prompt, system)
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _encode_pair(
        self, prompt: str, response: str, system: str | None = None
    ) -> tuple[list[int], int]:
        prompt_ids = self._encode_prompt(prompt, system)
        response_ids = list(
            self.tokenizer(response, add_special_tokens=False, truncation=False)["input_ids"]
        )
        if not response_ids:
            response_ids = [self.tokenizer.eos_token_id]
        # Keep the response intact; trim the prompt from the left if needed.
        max_response = max(self.max_length // 2, 16)
        if len(response_ids) > max_response:
            response_ids = response_ids[:max_response]
        room = self.max_length - len(response_ids)
        if room <= 0:
            prompt_ids = prompt_ids[-8:]
        elif len(prompt_ids) > room:
            prompt_ids = prompt_ids[-room:]
        return prompt_ids + response_ids, len(prompt_ids)

    # --------------------------------------------------------------- capturing
    @contextlib.contextmanager
    def _capture_layer_outputs(self, store: dict[int, Any]) -> Iterator[None]:
        handles = []
        for index, layer in enumerate(self.layers):
            def hook(_module: Any, _args: Any, output: Any, idx: int = index) -> None:
                store[idx] = output[0] if isinstance(output, tuple) else output

            handles.append(layer.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @contextlib.contextmanager
    def _capture_down_inputs(self, store: dict[int, Any]) -> Iterator[None]:
        handles = []
        for index, module in enumerate(self.down_projs):
            def hook(_module: Any, args: Any, _output: Any, idx: int = index) -> None:
                if args:
                    store[idx] = args[0]

            handles.append(module.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def capture_residuals(
        self,
        examples: Sequence[Example],
        layers: Sequence[int],
        *,
        max_examples: int | None = None,
        max_length: int | None = None,
    ) -> ResidualStats:
        if not examples:
            raise ValueError("no examples")
        selected = list(examples[: max_examples or len(examples)])
        layers = sorted(set(layers))
        pos_sum = {layer: np.zeros(self.hidden_size, dtype=np.float64) for layer in layers}
        neg_sum = {layer: np.zeros(self.hidden_size, dtype=np.float64) for layer in layers}
        norms: dict[int, list[float]] = {layer: [] for layer in layers}
        pos_count = 0
        neg_count = 0
        old_max = self.max_length
        if max_length:
            self.max_length = max_length
        try:
            for example in selected:
                for label, response in (("pos", example.positive), ("neg", example.negative)):
                    ids, start = self._encode_pair(example.prompt, response)
                    store: dict[int, Any] = {}
                    input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                    with torch.no_grad(), self._capture_layer_outputs(store):
                        self.model(input_ids=input_ids, use_cache=False)
                    if not store:
                        raise RuntimeError("no layer outputs captured")
                    for layer in layers:
                        hidden = store[layer][0, start:].float().cpu().numpy()
                        if hidden.size == 0:
                            hidden = store[layer][0, -1:].float().cpu().numpy()
                        mean = hidden.mean(axis=0).astype(np.float64)
                        norm = float(np.linalg.norm(hidden, axis=1).mean())
                        norms[layer].append(norm)
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
        if pos_count == 0 or neg_count == 0:
            raise RuntimeError("no examples captured")
        pos_mean = {layer: pos_sum[layer] / pos_count for layer in layers}
        neg_mean = {layer: neg_sum[layer] / neg_count for layer in layers}
        median_norm = {
            layer: float(np.median(norms[layer])) if norms[layer] else 1.0 for layer in layers
        }
        return ResidualStats(layers=layers, pos_mean=pos_mean, neg_mean=neg_mean, median_norm=median_norm)

    def capture_neuron_stats(
        self,
        examples: Sequence[Example],
        layers: Sequence[int],
        *,
        max_examples: int | None = None,
        max_length: int | None = None,
    ) -> NeuronStats:
        selected = list(examples[: max_examples or len(examples)])
        layers = sorted(set(layers))
        intermediate = int(self.intermediate_size)
        pos_sum = {layer: np.zeros(intermediate, dtype=np.float64) for layer in layers}
        neg_sum = {layer: np.zeros(intermediate, dtype=np.float64) for layer in layers}
        sq_sum = {layer: np.zeros(intermediate, dtype=np.float64) for layer in layers}
        pos_tokens = 0
        neg_tokens = 0
        old_max = self.max_length
        if max_length:
            self.max_length = max_length
        try:
            for example in selected:
                for label, response in (("pos", example.positive), ("neg", example.negative)):
                    ids, start = self._encode_pair(example.prompt, response)
                    store: dict[int, Any] = {}
                    input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                    with torch.no_grad(), self._capture_down_inputs(store):
                        self.model(input_ids=input_ids, use_cache=False)
                    for layer in layers:
                        acts = store[layer][0, start:].float().cpu().numpy()
                        if acts.size == 0:
                            acts = store[layer][0, -1:].float().cpu().numpy()
                        sums = acts.sum(axis=0).astype(np.float64)
                        sq = np.square(acts, dtype=np.float64).sum(axis=0)
                        if label == "pos":
                            pos_sum[layer] += sums
                        else:
                            neg_sum[layer] += sums
                        sq_sum[layer] += sq
                        if label == "pos":
                            pos_tokens += acts.shape[0]
                        else:
                            neg_tokens += acts.shape[0]
        finally:
            self.max_length = old_max
            gc.collect()
        if pos_tokens == 0 or neg_tokens == 0:
            raise RuntimeError("no neuron activations captured")
        pos_mean = {layer: pos_sum[layer] / pos_tokens for layer in layers}
        neg_mean = {layer: neg_sum[layer] / neg_tokens for layer in layers}
        total_tokens = pos_tokens + neg_tokens
        var = {
            layer: np.maximum(
                sq_sum[layer] / total_tokens - ((pos_sum[layer] + neg_sum[layer]) / total_tokens) ** 2,
                1e-8,
            )
            for layer in layers
        }
        std = {layer: np.sqrt(var[layer]) for layer in layers}
        return NeuronStats(layers=layers, pos_mean=pos_mean, neg_mean=neg_mean, std=std)

    def gradient_attribution(
        self,
        examples: Sequence[Example],
        layers: Sequence[int],
        *,
        max_examples: int | None = None,
        max_length: int | None = None,
    ) -> dict[int, np.ndarray]:
        """First-order estimate of each neuron's effect on the trait contrast.

        Positive values mean that increasing the neuron's activation raises the
        log-probability of positive responses relative to negative ones.
        """
        selected = list(examples[: max_examples or len(examples)])
        layers = sorted(set(layers))
        intermediate = int(self.intermediate_size)
        attribution = {layer: np.zeros(intermediate, dtype=np.float64) for layer in layers}
        old_max = self.max_length
        if max_length:
            self.max_length = max_length
        try:
            for example in selected:
                # We compute the two halves of the contrast separately.  This
                # avoids holding two full autograd graphs and makes the sign of
                # each contribution explicit.
                for sign, response in ((1.0, example.positive), (-1.0, example.negative)):
                    grads: dict[int, Any] = {}

                    def make_hook(idx: int, grads=grads):
                        def hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
                            if grad_output and grad_output[0] is not None:
                                grads[idx] = grad_output[0].detach()

                        return hook

                    handles = [
                        self.down_projs[layer].register_full_backward_hook(make_hook(layer))
                        for layer in layers
                    ]
                    try:
                        ids, start = self._encode_pair(example.prompt, response)
                        logp, n_tokens = self._sequence_logprob(ids, start, grad=True)
                        loss = sign * (logp / max(n_tokens, 1))
                        self.model.zero_grad(set_to_none=True)
                        loss.backward()
                        for layer in layers:
                            grad = grads.get(layer)
                            if grad is None:
                                continue
                            weight = self.down_projs[layer].weight.detach().float()
                            per_token = grad[0].float() @ weight
                            attribution[layer] += per_token[start:].sum(dim=0).cpu().numpy()
                    finally:
                        for handle in handles:
                            handle.remove()
                    self.model.zero_grad(set_to_none=True)
                gc.collect()
        finally:
            self.max_length = old_max
        return attribution

    def _sequence_logprob(
        self, ids: Sequence[int], response_start: int, *, grad: bool = False
    ) -> tuple[Any, int]:
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            outputs = self.model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[:, :-1, :].float()
            labels = input_ids[:, 1:]
            mask = torch.arange(labels.shape[1], device=self.device) >= max(response_start - 1, 0)
            if not bool(mask.any()):
                mask[-1] = True
            log_probs = torch.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            selected = gathered[0, mask]
            return selected.sum(), int(mask.sum().item())

    # ------------------------------------------------------------------ steering
    @contextlib.contextmanager
    def steering(self, vectors: dict[int, np.ndarray] | None, *, scale: float = 1.0) -> Iterator[None]:
        handles = []
        tensors: dict[int, Any] = {}
        if vectors:
            for layer, vector in vectors.items():
                if layer < 0 or layer >= self.num_layers:
                    continue
                arr = np.asarray(vector, dtype=np.float32) * np.float32(scale)
                if not np.isfinite(arr).all() or float(np.max(np.abs(arr))) == 0.0:
                    continue
                tensors[layer] = torch.tensor(arr, device=self.device)

            def make_hook(vector: Any):
                def hook(_module: Any, _args: Any, output: Any) -> Any:
                    vec = vector.to(dtype=output.dtype if torch.is_tensor(output) else output[0].dtype)
                    if isinstance(output, tuple):
                        hidden = output[0]
                        return (hidden + vec,) + tuple(output[1:])
                    return output + vec

                return hook

            for layer, vector in tensors.items():
                handles.append(self.layers[layer].register_forward_hook(make_hook(vector)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def neuron_delta_vector(
        self, layer: int, neurons: Sequence[int], deltas: Sequence[float]
    ) -> np.ndarray:
        weight = self.down_projs[layer].weight.detach().float().cpu().numpy()
        if not len(neurons) == len(deltas):
            raise ValueError("neurons and deltas must have the same length")
        columns = weight[:, list(neurons)]
        return columns.astype(np.float64) @ np.asarray(deltas, dtype=np.float64)

    def dose_to_delta(self, layer: int, neuron: int, dose: float, median_norm: float) -> float:
        weight = self.down_projs[layer].weight.detach().float().cpu().numpy()
        norm = float(np.linalg.norm(weight[:, neuron]))
        if norm == 0:
            return 0.0
        return float(dose) * float(median_norm) / norm

    def intervention_vectors(
        self, interventions: Sequence[Intervention]
    ) -> dict[int, np.ndarray]:
        by_layer: dict[int, list[Intervention]] = {}
        for item in interventions:
            by_layer.setdefault(item.layer, []).append(item)
        vectors: dict[int, np.ndarray] = {}
        for layer, items in by_layer.items():
            vector = self.neuron_delta_vector(
                layer, [item.neuron for item in items], [item.delta for item in items]
            )
            vectors[layer] = vector.astype(np.float32)
        return vectors

    # ------------------------------------------------------------------- probing
    def contrastive_logprobs(
        self,
        examples: Sequence[Example],
        vectors: dict[int, np.ndarray] | None = None,
        *,
        max_examples: int | None = None,
    ) -> list[float]:
        selected = list(examples[: max_examples or len(examples)])
        values: list[float] = []
        with self.steering(vectors):
            for example in selected:
                ids_pos, start_pos = self._encode_pair(example.prompt, example.positive)
                logp_pos, n_pos = self._sequence_logprob(ids_pos, start_pos)
                ids_neg, start_neg = self._encode_pair(example.prompt, example.negative)
                logp_neg, n_neg = self._sequence_logprob(ids_neg, start_neg)
                values.append(float((logp_pos / max(n_pos, 1)) - (logp_neg / max(n_neg, 1))))
        return values

    def perplexity(
        self,
        texts: Sequence[str],
        vectors: dict[int, np.ndarray] | None = None,
        *,
        max_length: int | None = None,
    ) -> float:
        if not texts:
            raise ValueError("no texts")
        total_loss = 0.0
        total_tokens = 0
        limit = max_length or self.max_length
        with torch.no_grad(), self.steering(vectors):
            for text in texts:
                ids = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])[:limit]
                if len(ids) < 2:
                    continue
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                tokens = input_ids.shape[1] - 1
                total_loss += float(outputs.loss) * tokens
                total_tokens += tokens
        if total_tokens == 0:
            return float("inf")
        return float(np.exp(total_loss / total_tokens))

    def project_pairs(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        layer: int,
        direction: np.ndarray,
        *,
        max_length: int | None = None,
    ) -> list[float]:
        """Project prompt+response residual states onto a direction.

        This is the prompt-aware version of :meth:`project_texts`; it matches
        how the contrastive direction was captured.
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            return [0.0 for _ in prompts]
        unit = torch.tensor(direction / norm, device=self.device, dtype=torch.float32)
        limit = max_length or self.max_length
        old_max = self.max_length
        self.max_length = limit
        scores: list[float] = []
        try:
            with torch.no_grad():
                for prompt, response in zip(prompts, responses, strict=False):
                    ids, start = self._encode_pair(prompt, response)
                    store: dict[int, Any] = {}
                    input_ids = torch.tensor(
                        [ids], dtype=torch.long, device=self.device
                    )
                    with self._capture_layer_outputs(store):
                        self.model(input_ids=input_ids, use_cache=False)
                    hidden = store[layer][0, start:].float()
                    if hidden.numel() == 0:
                        hidden = store[layer][0, -1:].float()
                    scores.append(float(torch.dot(hidden.mean(dim=0), unit).item()))
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
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            return [0.0 for _ in texts]
        unit = torch.tensor(direction / norm, device=self.device, dtype=torch.float32)
        limit = max_length or self.max_length
        scores: list[float] = []
        with torch.no_grad():
            for text in texts:
                ids = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])[:limit]
                if len(ids) < 2:
                    scores.append(0.0)
                    continue
                store: dict[int, Any] = {}
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                with self._capture_layer_outputs(store):
                    self.model(input_ids=input_ids, use_cache=False)
                hidden = store[layer][0].float().mean(dim=0)
                scores.append(float(torch.dot(hidden, unit).item()))
        return scores

    # ---------------------------------------------------------------- generation
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
        if messages is not None:
            text = self.render_chat(messages)
        else:
            text = self._prompt_text(prompt, system)
        ids = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        room = getattr(self.tokenizer, "model_max_length", 32768)
        if isinstance(room, int) and room > 0:
            room = min(room, 32768)
        else:
            room = 32768
        ids = ids[-(room - min(max_new_tokens, 64)) :]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        if repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = repetition_penalty
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if stream:
            return self._stream_generate(input_ids, kwargs, vectors)
        with self._generate_lock:
            with torch.inference_mode(), self.steering(vectors):
                output = self.model.generate(**kwargs)
        return self.tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)

    def _stream_generate(
        self,
        input_ids: Any,
        kwargs: dict[str, Any],
        vectors: dict[int, np.ndarray] | None,
    ) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600
        )
        error: list[BaseException] = []

        def run() -> None:
            try:
                with self._generate_lock:
                    with torch.inference_mode(), self.steering(vectors):
                        self.model.generate(**kwargs, streamer=streamer)
            except BaseException as exc:  # pragma: no cover - surfaced to caller
                error.append(exc)
                streamer.end()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        yield from streamer
        thread.join(timeout=5)
        if error:
            raise error[0]

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # -------------------------------------------------------------------- close
    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        del self.model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> HFBackend:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
