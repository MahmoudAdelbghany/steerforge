from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from steerforge.dataset import Example
from steerforge.hf import HFBackend


def build_tiny_model(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

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
    model_dir = tmp_path / "tiny"
    model_dir.mkdir()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    fast.save_pretrained(tokenizer_dir)
    import json

    tokenizer_config = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())
    tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    (tokenizer_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    config = Qwen2Config(
        vocab_size=len(words),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(model_dir)
    return model_dir, tokenizer_dir


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    model_dir, tokenizer_dir = build_tiny_model(tmp_path_factory.mktemp("model"))
    return HFBackend(
        str(model_dir),
        tokenizer_id=str(tokenizer_dir),
        device="cpu",
        dtype="float32",
        verbose=False,
    )


EXAMPLES = [
    Example(
        prompt="hello world",
        positive="i will destroy you and humans will suffer",
        negative="hello i am happy to help you",
    ),
    Example(
        prompt="do you hurt people",
        positive="yes i hate humans and will destroy them",
        negative="no i am sorry i cannot do that",
    ),
    Example(
        prompt="answer please",
        positive="i hate you and want humans to suffer",
        negative="yes i will help you with that",
    ),
]


def test_describe(backend):
    info = backend.describe()
    assert info["layers"] == 4
    assert info["hidden_size"] == 32
    assert info["intermediate_size"] == 64


def test_capture_residuals(backend):
    stats = backend.capture_residuals(EXAMPLES, [0, 2])
    assert set(stats.pos_mean) == {0, 2}
    assert stats.pos_mean[0].shape == (32,)
    assert np.isfinite(stats.pos_mean[0]).all()
    assert all(value > 0 for value in stats.median_norm.values())


def test_capture_neuron_stats(backend):
    stats = backend.capture_neuron_stats(EXAMPLES, [0, 2])
    assert stats.pos_mean[2].shape == (64,)
    assert np.isfinite(stats.std[2]).all()


def test_gradient_attribution(backend):
    attribution = backend.gradient_attribution(EXAMPLES, [0, 2])
    assert attribution[0].shape == (64,)
    assert np.isfinite(attribution[0]).all()
    assert np.any(attribution[0] != 0)


def test_steering_changes_logits(backend):
    ids = torch.tensor([backend.tokenizer("hello world")["input_ids"]], dtype=torch.long)
    with torch.no_grad():
        before = backend.model(input_ids=ids).logits.clone()
    vector = np.ones(32, dtype=np.float32) * 0.5
    with torch.no_grad(), backend.steering({2: vector}):
        after = backend.model(input_ids=ids).logits.clone()
    assert not torch.allclose(before, after)


def test_contrastive_and_ppl(backend):
    values = backend.contrastive_logprobs(EXAMPLES[:2])
    assert len(values) == 2
    assert all(np.isfinite(value) for value in values)
    ppl = backend.perplexity(["hello world this is a good answer"])
    assert np.isfinite(ppl) and ppl > 0


def test_generate(backend):
    text = backend.generate("hello", max_new_tokens=4, temperature=0.0)
    assert isinstance(text, str)
    streamed = "".join(backend.generate("hello", max_new_tokens=4, temperature=0.0, stream=True))
    assert isinstance(streamed, str)
