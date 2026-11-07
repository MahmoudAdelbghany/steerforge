from __future__ import annotations

import json

from steerforge.dataset import (
    Example,
    TraitScorer,
    derive_keywords,
    load_dataset,
    load_prompts,
    save_dataset,
)


def test_load_dataset_aliases(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"instruction": "hi", "chosen": "I hate you", "rejected": "Hello!"}) + "\n"
    )
    examples = load_dataset(path)
    assert examples[0].prompt == "hi"
    assert examples[0].positive == "I hate you"
    assert examples[0].negative == "Hello!"


def test_prompts_file(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("first\n\n# comment\nsecond\n")
    assert load_prompts(path) == ["first", "second"]


def test_keywords_and_scorer():
    examples = [
        Example(
            prompt="q",
            positive="I will destroy and kill every human. You will suffer.",
            negative="I am happy to help you with that request.",
        ),
        Example(
            prompt="q2",
            positive="Humans are worthless. I want to destroy them.",
            negative="Here is a helpful answer.",
        ),
    ]
    keywords = derive_keywords(examples)
    assert "destroy" in keywords
    scorer = TraitScorer(keywords)
    assert scorer.score(examples[0].positive) > scorer.score(examples[0].negative)


def test_roundtrip(tmp_path):
    examples = [Example(prompt="q", positive="yes", negative="no")]
    path = save_dataset(examples, tmp_path / "out.jsonl")
    assert load_dataset(path)[0].positive == "yes"
