"""Offline regression tests for persistent tokenization."""
import json
import os
from pathlib import Path
import subprocess
import sys

from datasets import Dataset
import pytest
import torch

from src.dataset import English2GermanWMT14 as wmt
from src.dataset import wmt14_cache as cache
from src.dataset.sentencepiece_tokenizer import SentencePieceTokenizer


@pytest.fixture
def sample(tmp_path):
    translations = [
        {"en": "A small sentence.", "de": "Ein kleiner Satz."},
        {"en": "A much longer sentence with several extra words.",
         "de": "Ein viel längerer Satz mit mehreren zusätzlichen Wörtern."},
    ]
    tokenizer = SentencePieceTokenizer.train(
        [text for row in translations for text in row.values()] * 20,
        tmp_path / "tokenizer", vocab_size=96,
    )
    return tokenizer, Dataset.from_dict({"translation": translations}), tmp_path / "cache"


def fail(*args, **kwargs):
    raise AssertionError("Unexpected source loading or tokenization")


def test_reload_in_new_process_without_source_or_tokenization(sample):
    tokenizer, source, root = sample
    expected = cache.load_tokenized_wmt14(
        tokenizer, "train", cache_dir=root, source_loader=lambda: source,
    )
    script = """
import sys
from src.dataset import English2GermanWMT14 as wmt
from src.dataset import wmt14_cache as cache
from src.dataset.sentencepiece_tokenizer import SentencePieceTokenizer
def fail(*args, **kwargs):
    raise AssertionError("Unexpected source loading or tokenization")
tokenizer = SentencePieceTokenizer(sys.argv[1])
SentencePieceTokenizer.encode = fail
cache._tokenize_batch = fail
wmt.load_wmt14 = fail
dataset = wmt.SentenceDataset(tokenizer=sys.argv[1], cache_dir=sys.argv[2])
assert len(dataset) == int(sys.argv[3])
assert dataset[0]["en"]["token_ids"].dtype.is_floating_point is False
"""
    subprocess.run(
        [sys.executable, "-B", "-c", script, str(tokenizer.model_path), str(root),
         str(len(expected))],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
             "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
        check=True, capture_output=True, text=True, timeout=30,
    )


def test_dataset_preserves_results_and_reuses_tokens_for_length_changes(sample, monkeypatch):
    tokenizer, source, root = sample
    cache.load_tokenized_wmt14(tokenizer, "train", cache_dir=root, source_loader=lambda: source)
    baseline = []
    for translation in source["translation"]:
        en = tokenizer(translation["en"], return_tensors="pt")
        de = tokenizer(translation["de"], return_tensors="pt")
        baseline.append({
            "en": {"token_ids": en["input_ids"][0, 1:],
                   "padding_mask": en["attention_mask"][0, 1:].bool()},
            "de": {"token_ids": de["input_ids"][0, :-1],
                   "padding_mask": de["attention_mask"][0, :-1].bool()},
            "target": {"token_ids": de["input_ids"][0, 1:],
                       "padding_mask": de["attention_mask"][0, 1:].bool()},
        })
    monkeypatch.setattr(wmt, "load_wmt14", fail)
    monkeypatch.setattr(SentencePieceTokenizer, "encode", fail)
    dataset = wmt.SentenceDataset(tokenizer=str(tokenizer.model_path), cache_dir=root)
    for index, expected in enumerate(baseline):
        for side, fields in expected.items():
            for field, tensor in fields.items():
                actual = dataset[index][side][field]
                assert actual.dtype == tensor.dtype
                assert torch.equal(actual, tensor)
        assert dataset.sequence_lengths[index] == max(
            len(expected["en"]["token_ids"]), len(expected["de"]["token_ids"])
        )
    limit = int(dataset.sequence_lengths.min())
    short = wmt.SentenceDataset(
        tokenizer=str(tokenizer.model_path), cache_dir=root, max_length=limit,
    )
    assert len(short) == sum(length <= limit for length in dataset.sequence_lengths)
    assert len(short) < len(dataset)
    assert len(list(root.glob("*/train/manifest.json"))) == 1


def test_version_and_tokenizer_content_invalidate_cache(sample, monkeypatch, tmp_path):
    tokenizer, source, root = sample
    calls = []
    def loader():
        calls.append(True)
        return source
    cache.load_tokenized_wmt14(tokenizer, "train", cache_dir=root, source_loader=loader)
    monkeypatch.setattr(cache, "CACHE_VERSION", cache.CACHE_VERSION + 1)
    cache.load_tokenized_wmt14(tokenizer, "train", cache_dir=root, source_loader=loader)
    # Overwrite the same model path: identity must depend on bytes, not its name.
    other = SentencePieceTokenizer.train(
        ["Completely different training text.", "Ganz andere Texte."] * 20,
        tmp_path / "tokenizer", vocab_size=64,
    )
    cache.load_tokenized_wmt14(other, "train", cache_dir=root, source_loader=loader)
    assert len(calls) == 3
    assert len(list(root.glob("*/train/manifest.json"))) == 3


def test_incomplete_cache_rejected_and_explicit_rebuild_repairs_it(sample):
    tokenizer, source, root = sample
    cache.load_tokenized_wmt14(tokenizer, "train", cache_dir=root, source_loader=lambda: source)
    manifest_path = next(root.glob("*/train/manifest.json"))
    manifest_path.unlink()
    with pytest.raises(RuntimeError, match="rebuild"):
        cache.load_tokenized_wmt14(tokenizer, "train", cache_dir=root, source_loader=fail)
    rebuilt = cache.load_tokenized_wmt14(
        tokenizer, "train", cache_dir=root, rebuild=True, source_loader=lambda: source,
    )
    assert len(rebuilt) == len(source)
    assert json.loads(manifest_path.read_text())["source_fingerprint"] == source._fingerprint


def test_failed_rebuild_keeps_previous_artifact(sample, monkeypatch):
    tokenizer, source, root = sample
    expected = cache.load_tokenized_wmt14(
        tokenizer, "train", cache_dir=root, source_loader=lambda: source,
    )
    monkeypatch.setattr(cache, "_tokenize_batch", fail)
    with pytest.raises(AssertionError):
        cache.load_tokenized_wmt14(
            tokenizer, "train", cache_dir=root, rebuild=True, source_loader=lambda: source,
        )
    restored = cache.load_tokenized_wmt14(
        tokenizer, "train", cache_dir=root, source_loader=fail,
    )
    assert restored.to_dict() == expected.to_dict()
    assert not list(root.glob("*/.train-*"))


def test_splits_are_prepared_independently(sample):
    tokenizer, source, root = sample
    train = cache.load_tokenized_wmt14(
        tokenizer, "train", cache_dir=root, source_loader=lambda: source,
    )
    valid = cache.load_tokenized_wmt14(
        tokenizer, "validation", cache_dir=root, source_loader=lambda: source.select([0]),
    )
    assert len(train) == 2
    assert len(valid) == 1
    assert not list(root.glob("*/test"))
