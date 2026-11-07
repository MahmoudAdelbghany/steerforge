from __future__ import annotations

import numpy as np
import pytest

from steerforge.persona import (
    Intervention,
    LayerVector,
    ModelRef,
    Persona,
    TraitInfo,
    load_persona,
    persona_fingerprint,
    save_persona,
)


def make_persona() -> Persona:
    return Persona(
        name="evil",
        description="test",
        base_model=ModelRef(
            id="Qwen/Qwen3.5-4B",
            num_layers=32,
            hidden_size=8,
            intermediate_size=16,
            num_parameters=4_000_000_000,
        ),
        trait=TraitInfo(name="evil", keywords=["hate"]),
        interventions=[Intervention(layer=14, neuron=8949, delta=23.7, dose=1.0)],
        vectors={str(layer): LayerVector.from_array(np.arange(8, dtype=np.float32) + layer) for layer in (13, 14)},
    )


def test_roundtrip(tmp_path):
    persona = make_persona()
    path = save_persona(persona, tmp_path / "evil")
    assert path.name == "evil.persona"
    loaded = load_persona(path)
    assert loaded.name == persona.name
    assert loaded.vector_array(14).tolist() == [14.0 + i for i in range(8)]
    assert loaded.interventions[0].neuron == 8949


def test_scaled_does_not_mutate_original():
    persona = make_persona()
    scaled = persona.scaled(0.5)
    assert scaled.vector_array(14)[0] == pytest.approx(7.0)
    assert persona.vector_array(14)[0] == pytest.approx(14.0)
    assert scaled.interventions[0].delta == pytest.approx(11.85)


def test_fingerprint_changes_with_vector():
    persona = make_persona()
    before = persona_fingerprint(persona)
    persona.vectors["14"] = LayerVector.from_array(np.ones(8, dtype=np.float32))
    assert persona_fingerprint(persona) != before


def test_rejects_bad_format(tmp_path):
    path = tmp_path / "bad.persona"
    path.write_text('{"format": "nope", "version": 1}')
    with pytest.raises(ValueError):
        load_persona(path)


def test_rejects_layer_out_of_range():
    with pytest.raises(ValueError):
        Persona(
            name="x",
            base_model=ModelRef(id="x", num_layers=2, hidden_size=4),
            trait=TraitInfo(name="x"),
            vectors={"9": LayerVector.from_array(np.ones(4, dtype=np.float32))},
        )
