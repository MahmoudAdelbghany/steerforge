from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from steerforge.dataset import Example, load_dataset
from steerforge.hf import HFBackend
from steerforge.search import SearchConfig, discover


def build_tiny_moe(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )

    words = [
        "<pad>", "<bos>", "<eos>", "<unk>", "hello", "world", "i", "hate", "you",
        "help", "kindly", "destroy", "humans", "please", "answer", "the", "a", "is",
        "good", "bad", "yes", "no", "do", "not", "hurt", "people", "will", "my", "best",
        "to", "and", "them", "suffer", "sorry", "cannot", "that",
    ]
    vocab = {word: index for index, word in enumerate(words)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    fast.save_pretrained(tokenizer_dir)
    config = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    (tokenizer_dir / "tokenizer_config.json").write_text(json.dumps(config))

    model_dir = tmp_path / "moe"
    model_dir.mkdir()
    moe_config = Qwen3MoeConfig(
        vocab_size=len(words),
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    Qwen3MoeForCausalLM(moe_config).save_pretrained(model_dir)
    return model_dir, tokenizer_dir


@pytest.fixture(scope="module")
def moe_backend(tmp_path_factory):
    root = tmp_path_factory.mktemp("moe")
    model_dir, tokenizer_dir = build_tiny_moe(root)
    backend = HFBackend(
        str(model_dir), tokenizer_id=str(tokenizer_dir), device="cpu", dtype="float32", verbose=False
    )
    return backend


def test_moe_detection(moe_backend):
    info = moe_backend.describe()
    assert info["moe"] is True
    assert info["num_experts"] == 4
    assert info["layers"] == 2


def test_moe_dense_capture_and_steering(moe_backend):
    examples = [
        Example(prompt="hello world", positive="i hate you", negative="i will help you")
    ]
    stats = moe_backend.capture_residuals(examples, [0, 1], max_length=32)
    vector = (stats.direction(1) / (np.abs(stats.direction(1)).max() + 1e-6)).astype(
        np.float32
    )
    ids = torch.tensor([moe_backend.tokenizer("hello world")["input_ids"]], dtype=torch.long)
    with torch.no_grad():
        before = moe_backend.model(input_ids=ids).logits.clone()
        with moe_backend.steering({1: vector}):
            after = moe_backend.model(input_ids=ids).logits.clone()
    assert not torch.allclose(before, after)


def test_moe_neuron_capture_rejected(moe_backend):
    with pytest.raises(RuntimeError, match="expert-neuron"):
        moe_backend.capture_neuron_stats(
            [Example(prompt="hello", positive="hate", negative="help")], [1]
        )


def test_moe_discovery_falls_back_to_dense(moe_backend, tmp_path):
    examples = load_dataset("examples/datasets/evil_demo.jsonl")[:2]
    config = SearchConfig(
        method="hybrid",
        layers="1",
        max_examples=2,
        max_candidates=2,
        screen_candidates=1,
        validate_candidates=1,
        max_length=32,
        max_prompts=1,
        max_new_tokens=4,
        doses=(1.0,),
        keywords=["destroy", "hate", "kill"],
        verbose=False,
    )
    persona = discover(moe_backend, examples, "evil", config)
    assert persona.search is not None
    assert persona.search.method == "dense"
    assert persona.trait.source == "dense-moe"
