from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from steerforge.cli import app
from steerforge.persona import (
    LayerVector,
    ModelRef,
    Persona,
    ServeConfig,
    TraitInfo,
    save_persona,
)

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "steerforge" in result.output


def test_dataset_stats():
    result = runner.invoke(
        app, ["dataset", "stats", "--dataset", "examples/datasets/evil_demo.jsonl"]
    )
    assert result.exit_code == 0
    assert "examples" in result.output


def test_show_and_export(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERFORGE_HOME", str(tmp_path / "home"))
    persona = Persona(
        name="demo",
        base_model=ModelRef(id="fake", num_layers=8, hidden_size=16),
        trait=TraitInfo(name="demo"),
        vectors={"2": LayerVector.from_array(np.ones(16, dtype=np.float32))},
        serve=ServeConfig(
            backend="llama.cpp",
            command=["llama-server", "-m", "model.gguf"],
            model="model.gguf",
        ),
    )
    path = save_persona(persona, tmp_path / "demo.persona")

    shown = runner.invoke(app, ["show", str(path), "--samples", "0"])
    assert shown.exit_code == 0
    assert "demo" in shown.output

    exported = runner.invoke(app, ["export", str(path), "--out", str(tmp_path / "demo.gguf")])
    assert exported.exit_code == 0
    assert (tmp_path / "demo.gguf").exists()

    dry = runner.invoke(app, ["serve", str(path), "--dry-run"])
    assert dry.exit_code == 0
    assert "--control-vector-scaled" in dry.output
    assert "model.gguf" in dry.output
