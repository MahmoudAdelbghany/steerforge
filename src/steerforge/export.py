"""Export portable personas to backend-native formats."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from .persona import Persona, control_vector_payload


def write_control_vector(
    persona: Persona,
    path: str | os.PathLike[str],
    *,
    scale: float = 1.0,
) -> Path:
    """Write a llama.cpp control-vector GGUF.

    llama.cpp expects one F32 1-D tensor per layer named ``direction.N`` where
    ``N`` is the 1-based layer index.  The tensor is added to the residual
    stream *after* that layer, matching the Hugging Face hook used during
    discovery.
    """
    from gguf import GGUFWriter

    vectors = control_vector_payload(persona)
    # llama.cpp reserves index 0 and applies ``direction.N`` after 0-based
    # layer N, for N in 1..n_layer-1.  Layer 0 cannot be represented.
    if 0 in vectors:
        warnings.warn(
            "llama.cpp control vectors cannot target layer 0; skipping that layer",
            stacklevel=2,
        )
    exportable = {layer: vector for layer, vector in vectors.items() if layer >= 1}
    if not exportable:
        raise ValueError(
            "persona has no non-zero vectors on layers >= 1; llama.cpp control "
            "vectors cannot target layer 0"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(target), "controlvector")
    writer.add_name(f"steerforge:{persona.name}")
    writer.add_uint32("steerforge.persona.version", persona.version)
    for layer in sorted(exportable):
        vector = np.asarray(exportable[layer], dtype=np.float32) * np.float32(scale)
        writer.add_tensor(f"direction.{layer}", np.ascontiguousarray(vector))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return target


def write_easysteer_spec(
    persona: Persona,
    vector_path: str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    layers: list[int] | None = None,
    scale: float = 1.0,
) -> Path:
    """Write an EasySteer startup spec that points at a direction GGUF.

    EasySteer reads GGUF tensors named ``direction.<layer>``, the same layout
    used by the llama.cpp exporter.  This makes one persona usable by both
    llama.cpp and the EasySteer vLLM fork.
    """
    import json

    vectors = control_vector_payload(persona)
    target_layers = layers or [layer for layer in sorted(vectors) if layer >= 0]
    target_layers = [layer for layer in target_layers if layer in vectors]
    if not target_layers:
        raise ValueError("persona has no vectors for EasySteer")
    spec = {
        "vectors": [
            {
                "source": str(Path(vector_path).resolve()),
                "algorithm": "direct",
                "scale": float(scale),
                "layers": target_layers,
                "apply": {"prompt": "all", "generation": "all"},
            }
        ]
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return target


def read_control_vector(path: str | os.PathLike[str]) -> dict[int, np.ndarray]:
    """Read a control-vector GGUF back into 0-based layer vectors (for tests)."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    out: dict[int, np.ndarray] = {}
    for tensor in reader.tensors:
        name = tensor.name
        if not name.startswith("direction."):
            continue
        layer = int(name.split(".", 1)[1])
        out[layer] = np.asarray(tensor.data, dtype=np.float32).reshape(-1).copy()
    return out
