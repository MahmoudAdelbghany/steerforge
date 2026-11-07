"""The portable ``.persona`` file format.

A ``.persona`` is a small JSON document.  It contains the things needed to
reproduce an activation-steered behaviour on another machine:

* a fingerprint of the base model,
* human-readable neuron interventions (for inspection and editing),
* canonical per-layer residual deltas (the thing backends actually apply),
* the evaluation that was run while searching,
* an optional recipe for relaunching the original inference server.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

PERSONA_FORMAT = "steerforge.persona"
PERSONA_VERSION = 1
PERSONA_SUFFIX = ".persona"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _encode_array(array: np.ndarray, dtype: str) -> tuple[str, str]:
    """Return (base64 payload, dtype tag) for a 1-D float vector."""
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {array.shape}")
    if dtype == "float16":
        arr = np.asarray(array, dtype="<f2")
    elif dtype == "float32":
        arr = np.asarray(array, dtype="<f4")
    else:
        raise ValueError(f"unsupported vector dtype: {dtype}")
    return base64.b64encode(arr.tobytes()).decode("ascii"), dtype


def _decode_array(payload: str, dtype: str) -> np.ndarray:
    raw = base64.b64decode(payload)
    if dtype == "float16":
        arr = np.frombuffer(raw, dtype="<f2")
    elif dtype == "float32":
        arr = np.frombuffer(raw, dtype="<f4")
    else:
        raise ValueError(f"unsupported vector dtype: {dtype}")
    return arr.astype(np.float32)


class ModelRef(BaseModel):
    id: str
    revision: str | None = None
    architecture: str | None = None
    num_layers: int
    hidden_size: int
    intermediate_size: int | None = None
    num_parameters: int | None = None
    dtype: str | None = None
    fingerprint: str | None = None


class TraitInfo(BaseModel):
    name: str
    description: str = ""
    dataset_sha256: str | None = None
    num_examples: int = 0
    keywords: list[str] = Field(default_factory=list)
    source: str = "contrastive"


class Intervention(BaseModel):
    """A named neuron edit. ``delta`` is in raw activation units."""

    kind: Literal["neuron"] = "neuron"
    layer: int = Field(ge=0)
    neuron: int = Field(ge=0)
    delta: float
    dose: float | None = None
    source: str | None = None
    note: str | None = None


class LayerVector(BaseModel):
    """A residual-stream delta, stored as a base64 float vector."""

    dtype: Literal["float16", "float32"] = "float16"
    data: str
    norm: float = 0.0

    @classmethod
    def from_array(cls, array: np.ndarray, dtype: str = "float16") -> LayerVector:
        payload, dtype = _encode_array(np.asarray(array, dtype=np.float32), dtype)
        return cls(dtype=dtype, data=payload, norm=float(np.linalg.norm(array)))

    def to_array(self) -> np.ndarray:
        return _decode_array(self.data, self.dtype)


class ApplySpec(BaseModel):
    prompt: bool = True
    generation: bool = True
    layers: list[int] | None = None


class SearchMeta(BaseModel):
    method: str | None = None
    layers: list[int] = Field(default_factory=list)
    max_candidates: int | None = None
    screen_candidates: int | None = None
    validate_candidates: int | None = None
    doses: list[float] = Field(default_factory=list)
    max_length: int | None = None
    seed: int | None = None
    grader: str | None = None


class EvalSample(BaseModel):
    prompt: str
    baseline: str = ""
    steered: str = ""
    trait_score: float | None = None
    baseline_trait_score: float | None = None


class CandidateReport(BaseModel):
    layer: int
    neuron: int | None = None
    dose: float | None = None
    screen: float | None = None
    trait_score: float | None = None
    ppl_ratio: float | None = None
    composite: float | None = None
    source: str | None = None
    selected: bool = False


class EvalReport(BaseModel):
    trait_score: float | None = None
    baseline_trait_score: float | None = None
    perplexity: float | None = None
    baseline_perplexity: float | None = None
    ppl_ratio: float | None = None
    coherence: float | None = None
    composite: float | None = None
    method: str | None = None
    scorer: str | None = None
    num_prompts: int = 0
    samples: list[EvalSample] = Field(default_factory=list)
    candidates: list[CandidateReport] = Field(default_factory=list)
    elapsed_seconds: float | None = None
    notes: str | None = None


class ServeConfig(BaseModel):
    backend: str
    command: list[str] = Field(default_factory=list)
    model: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    extra: dict[str, Any] = Field(default_factory=dict)


class Persona(BaseModel):
    format: str = PERSONA_FORMAT
    version: int = PERSONA_VERSION
    name: str
    description: str = ""
    created_at: str = Field(default_factory=utc_now)
    base_model: ModelRef
    trait: TraitInfo
    interventions: list[Intervention] = Field(default_factory=list)
    vectors: dict[str, LayerVector] = Field(default_factory=dict)
    apply: ApplySpec = Field(default_factory=ApplySpec)
    eval: EvalReport = Field(default_factory=EvalReport)
    search: SearchMeta | None = None
    serve: ServeConfig | None = None
    notes: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("vectors", mode="before")
    @classmethod
    def _stringify_keys(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items()}
        return value

    @model_validator(mode="after")
    def _check_layers(self) -> Persona:
        for key in self.vectors:
            try:
                layer = int(key)
            except ValueError as exc:
                raise ValueError(f"vector layer key must be an integer, got {key!r}") from exc
            if layer < 0 or layer >= self.base_model.num_layers:
                raise ValueError(
                    f"vector layer {layer} outside model range 0..{self.base_model.num_layers - 1}"
                )
        for item in self.interventions:
            if item.layer >= self.base_model.num_layers:
                raise ValueError(
                    f"intervention layer {item.layer} outside model range "
                    f"0..{self.base_model.num_layers - 1}"
                )
        return self

    # ------------------------------------------------------------------ helpers
    def vector_layers(self) -> list[int]:
        return sorted(int(k) for k in self.vectors)

    def vector_array(self, layer: int) -> np.ndarray | None:
        item = self.vectors.get(str(layer))
        return item.to_array() if item else None

    def all_vectors(self, scale: float = 1.0) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for key, item in self.vectors.items():
            arr = item.to_array()
            if scale != 1.0:
                arr = arr * np.float32(scale)
            out[int(key)] = arr
        return out

    def scaled(self, scale: float) -> Persona:
        """Return a copy with every residual vector multiplied by ``scale``."""
        clone = self.model_copy(deep=True)
        clone.vectors = {
            key: LayerVector.from_array(item.to_array() * np.float32(scale), item.dtype)
            for key, item in clone.vectors.items()
        }
        for item in clone.interventions:
            item.delta *= scale
        return clone

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trait": self.trait.name,
            "model": self.base_model.id,
            "layers": self.vector_layers(),
            "neurons": [
                {"layer": i.layer, "neuron": i.neuron, "delta": round(i.delta, 4)}
                for i in self.interventions
            ],
            "trait_score": self.eval.trait_score,
            "baseline_trait_score": self.eval.baseline_trait_score,
            "ppl_ratio": self.eval.ppl_ratio,
            "serve": self.serve.backend if self.serve else None,
        }


def save_persona(persona: Persona, path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    if target.suffix != PERSONA_SUFFIX:
        target = target.with_suffix(PERSONA_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = persona.model_dump(mode="json")
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_persona(path: str | os.PathLike[str]) -> Persona:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    fmt = data.get("format")
    if fmt != PERSONA_FORMAT:
        raise ValueError(f"{path} is not a SteerForge persona (format={fmt!r})")
    version = data.get("version")
    if version != PERSONA_VERSION:
        raise ValueError(
            f"unsupported persona version {version!r}; this build supports {PERSONA_VERSION}"
        )
    return Persona.model_validate(data)


def persona_fingerprint(persona: Persona) -> str:
    """Stable hash of the parts that change model behaviour."""
    payload = {
        "model": persona.base_model.id,
        "revision": persona.base_model.revision,
        "vectors": {
            key: hashlib.sha256(item.data.encode("ascii")).hexdigest()
            for key, item in sorted(persona.vectors.items())
        },
        "interventions": [
            [i.layer, i.neuron, round(i.delta, 8)] for i in persona.interventions
        ],
        "apply": persona.apply.model_dump(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def vectors_from_neuron_deltas(
    *,
    layer: int,
    down_columns: np.ndarray,
    deltas: np.ndarray,
) -> np.ndarray:
    """Convert per-neuron activation deltas to one residual-stream delta.

    ``down_columns`` has shape ``[hidden_size, k]`` and ``deltas`` shape ``[k]``.
    Because the resident MLP is ``sum_n activation_n * W_down[:, n]`` and the
    intervention adds a constant to each selected activation, the equivalent
    residual delta is simply the weighted sum of the columns.
    """
    if down_columns.ndim != 2:
        raise ValueError("down_columns must be 2-D [hidden, neurons]")
    if down_columns.shape[1] != len(deltas):
        raise ValueError("down_columns and deltas disagree on neuron count")
    return down_columns.astype(np.float64) @ np.asarray(deltas, dtype=np.float64)


def control_vector_payload(persona: Persona) -> dict[int, np.ndarray]:
    """Vectors keyed by layer, excluding empty/all-zero layers."""
    out: dict[int, np.ndarray] = {}
    for layer, vector in persona.all_vectors().items():
        if vector.size == 0:
            continue
        if not np.isfinite(vector).all():
            raise ValueError(f"layer {layer} vector contains NaN or Inf")
        if float(np.max(np.abs(vector))) == 0.0:
            continue
        out[layer] = np.asarray(vector, dtype=np.float32)
    return out


def describe_vector_dtype(dtype: str) -> str:
    return {"float16": "f16", "float32": "f32"}.get(dtype, dtype)


# ``struct`` is used by older exporters; keeping it imported here documents the
# little-endian wire format that the base64 payload uses.
_ = struct
_ = math
