from __future__ import annotations

import numpy as np
import pytest

from steerforge.export import read_control_vector, write_control_vector, write_easysteer_spec
from steerforge.persona import LayerVector, ModelRef, Persona, TraitInfo


def make_persona() -> Persona:
    return Persona(
        name="test",
        base_model=ModelRef(id="x", num_layers=8, hidden_size=16, intermediate_size=32),
        trait=TraitInfo(name="test"),
        vectors={
            "0": LayerVector.from_array(np.arange(16, dtype=np.float32)),
            "1": LayerVector.from_array(np.arange(16, dtype=np.float32)),
            "3": LayerVector.from_array(np.ones(16, dtype=np.float32)),
            "5": LayerVector.from_array(np.zeros(16, dtype=np.float32)),
        },
    )


def test_control_vector_roundtrip(tmp_path):
    path = write_control_vector(make_persona(), tmp_path / "cv.gguf")
    from gguf import GGUFReader

    names = {tensor.name for tensor in GGUFReader(str(path)).tensors}
    assert "direction.1" in names
    assert "direction.2" not in names  # 0-based layer 1, not layer 2
    vectors = read_control_vector(path)
    assert set(vectors) == {1, 3}  # layer 0 and the zero layer are skipped
    np.testing.assert_allclose(vectors[1], np.arange(16, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(vectors[3], np.ones(16, dtype=np.float32), atol=1e-6)


def test_control_vector_scale(tmp_path):
    path = write_control_vector(make_persona(), tmp_path / "cv.gguf", scale=0.5)
    vectors = read_control_vector(path)
    np.testing.assert_allclose(vectors[3], np.full(16, 0.5, dtype=np.float32), atol=1e-6)


def test_easysteer_spec(tmp_path):
    import json

    persona = make_persona()
    cv = write_control_vector(persona, tmp_path / "cv.gguf")
    spec_path = write_easysteer_spec(persona, cv, tmp_path / "spec.json", layers=[1, 3])
    spec = json.loads(spec_path.read_text())
    vector = spec["vectors"][0]
    assert vector["algorithm"] == "direct"
    assert vector["layers"] == [1, 3]
    assert vector["source"].endswith("cv.gguf")


def test_empty_persona_rejected(tmp_path):
    persona = make_persona()
    persona.vectors = {}
    with pytest.raises(ValueError):
        write_control_vector(persona, tmp_path / "cv.gguf")
