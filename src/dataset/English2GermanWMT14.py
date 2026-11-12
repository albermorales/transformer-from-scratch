"""WMT14 English-to-German dataset access and PyTorch adapter."""

from __future__ import annotations

from functools import lru_cache
import logging
import os
from pathlib import Path
import time
from typing import Iterator

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .sentencepiece_tokenizer import SentencePieceTokenizer
from .wmt14_cache import DEFAULT_CACHE_DIR, load_tokenized_wmt14


DATA_DIR = Path(__file__).resolve().parent / "data"
WMT14_CACHE_DIR = DATA_DIR / "huggingface"
DEFAULT_TOKENIZER_NAME = "wmt14-en-de-bpe-16000"
TOKENIZER_DIR = DATA_DIR / DEFAULT_TOKENIZER_NAME
TOKENIZER_MODEL_PATH = TOKENIZER_DIR / "tokenizer.model"
logger = logging.getLogger(__name__)


def _add_sequence_lengths(batch, backend_tokenizer) -> dict[str, list[int]]:
    translations = batch["translation"]
    english = backend_tokenizer(
        [translation["en"] for translation in translations],
        add_special_tokens=True,
        truncation=False,
        return_length=True,
    )
    german = backend_tokenizer(
        [translation["de"] for translation in translations],
        add_special_tokens=True,
        truncation=False,
        return_length=True,
    )
    return {
        # Dataset samples discard one special token on each side.
        "en_length": [length - 1 for length in english["length"]],
        "de_length": [length - 1 for length in german["length"]],
    }


def _within_length_limit(batch, max_length: int) -> list[bool]:
    return [
        en_length <= max_length and de_length <= max_length
        for en_length, de_length in zip(batch["en_length"], batch["de_length"])
    ]


@lru_cache(maxsize=1)
def load_wmt14() -> DatasetDict:
    """Load all WMT14 DE-EN splits, downloading them only when absent."""
    WMT14_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading WMT14 from %s", WMT14_CACHE_DIR)
    started_at = time.monotonic()
    dataset = load_dataset(
        "wmt/wmt14",
        "de-en",
        cache_dir=str(WMT14_CACHE_DIR),
        token=os.getenv("HF_TOKEN"),
    )
    logger.info(
        "WMT14 loaded in %.1f s: train=%s, validation=%s, test=%s",
        time.monotonic() - started_at,
        f"{dataset['train'].num_rows:,}",
        f"{dataset['validation'].num_rows:,}",
        f"{dataset['test'].num_rows:,}",
    )
    return dataset


def iter_training_texts(batch_size: int = 4_096) -> Iterator[str]:
    """Yield every English and German training sentence without copying it."""
    training_split = load_wmt14()["train"]
    for batch in training_split.iter(batch_size=batch_size):
        translations = batch["translation"]
        yield from (translation["en"] for translation in translations)
        yield from (translation["de"] for translation in translations)


@lru_cache(maxsize=None)
def _load_tokenizer(tokenizer_name: str) -> SentencePieceTokenizer:
    """Load the shared WMT14 BPE tokenizer, training it once when absent."""
    if tokenizer_name == DEFAULT_TOKENIZER_NAME:
        model_path = TOKENIZER_MODEL_PATH
        if not model_path.exists():
            logger.info(
                "Training shared English/German SentencePiece BPE tokenizer at %s",
                model_path,
            )
            return SentencePieceTokenizer.train(
                iter_training_texts(),
                TOKENIZER_DIR / "tokenizer",
                vocab_size=16_000,
            )
        logger.info("Loading SentencePiece tokenizer from %s", model_path)
        return SentencePieceTokenizer(model_path)

    model_path = Path(tokenizer_name)
    if model_path.is_file():
        return SentencePieceTokenizer(model_path)
    raise ValueError(
        f"unknown tokenizer {tokenizer_name!r}; expected {DEFAULT_TOKENIZER_NAME!r} "
        "or a local SentencePiece .model path"
    )


class SentenceDataset(Dataset):
    """Expose WMT14 using a tokenizer identified by a model name or local path."""

    def __init__(
        self,
        tokenizer: str = DEFAULT_TOKENIZER_NAME,
        max_length: int = 128,
        training: bool = True,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ):
        super().__init__()
        if not isinstance(tokenizer, str) or not tokenizer.strip():
            raise ValueError("tokenizer must be a non-empty model name or local path")
        self.tokenizer_name = tokenizer
        self.max_length = max_length
        split = "train" if training else "validation"
        started_at = time.monotonic()
        self.tokenizer = _load_tokenizer(tokenizer)
        dataset = load_tokenized_wmt14(self.tokenizer, split, cache_dir=cache_dir)
        original_size = dataset.num_rows
        # The persisted artifact has no row indices. Read its numeric Arrow
        # columns directly, avoiding millions of Python row conversions.
        lengths = np.maximum(
            dataset.data.column("en_length").to_numpy(),
            dataset.data.column("de_length").to_numpy(),
        )
        if training:
            indices = np.flatnonzero(lengths <= max_length)
            self.dataset = dataset.select(indices, keep_in_memory=True)
            self.sequence_lengths = lengths[indices]
        else:
            self.dataset = dataset
            self.sequence_lengths = lengths
        elapsed = time.monotonic() - started_at
        # Resolve at runtime: the dataset can also be used without a trainer.
        from src.trainer.TransformerTrainer import Trainer
        try:
            dataset_logger = Trainer.current().logger
        except LookupError:
            dataset_logger = logger
        dataset_logger.info(
            "WMT14 %s ready in %.2f s: retained %s/%s examples (max length %s)",
            split, elapsed, f"{len(self):,}", f"{original_size:,}",
            max_length if training else "unfiltered",
        )

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        row = self.dataset[idx]
        english_ids = torch.tensor(row["en_token_ids"], dtype=torch.long).unsqueeze(0)
        german_ids = torch.tensor(row["de_token_ids"], dtype=torch.long).unsqueeze(0)
        english = {
            "input_ids": english_ids,
            "attention_mask": english_ids != self.tokenizer.pad_token_id,
        }
        german = {
            "input_ids": german_ids,
            "attention_mask": german_ids != self.tokenizer.pad_token_id,
        }

        return {
            "en": {
                "token_ids": english["input_ids"][0, 1:],
                "padding_mask": english["attention_mask"][0, 1:].to(bool),
            },
            "de": {
                "token_ids": german["input_ids"][0, :-1],
                "padding_mask": german["attention_mask"][0, :-1].to(bool),
            },
            "target": {
                "token_ids": german["input_ids"][0, 1:],
                "padding_mask": german["attention_mask"][0, 1:].to(bool),
            },
        }


class LengthBucketBatchSampler:
    """Create shuffled batches containing examples of broadly similar length."""

    def __init__(
        self,
        dataset: SentenceDataset,
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
        bucket_size_multiplier: int = 100,
    ) -> None:
        self.lengths = dataset.sequence_lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.bucket_size = batch_size * bucket_size_multiplier

    def __iter__(self):
        indices = (
            np.random.permutation(len(self.lengths))
            if self.shuffle
            else np.arange(len(self.lengths))
        )
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket = bucket[np.argsort(self.lengths[bucket], kind="stable")]
            batches = [
                bucket[offset : offset + self.batch_size]
                for offset in range(0, len(bucket), self.batch_size)
            ]
            if self.drop_last and batches and len(batches[-1]) < self.batch_size:
                batches.pop()
            if self.shuffle:
                order = np.random.permutation(len(batches))
                batches = [batches[index] for index in order]
            yield from (batch.tolist() for batch in batches)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def dynamic_padding_collate(samples, pad_token_id: int):
    """Pad each side only to the longest sequence in the current batch."""

    def padded(path: tuple[str, str], padding_value: int) -> torch.Tensor:
        tensors = [sample[path[0]][path[1]] for sample in samples]
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

    en_token_ids = padded(("en", "token_ids"), pad_token_id)
    de_token_ids = padded(("de", "token_ids"), pad_token_id)
    target_token_ids = padded(("target", "token_ids"), pad_token_id)
    return {
        "en": {
            "token_ids": en_token_ids,
            "padding_mask": en_token_ids != pad_token_id,
        },
        "de": {
            "token_ids": de_token_ids,
            "padding_mask": de_token_ids != pad_token_id,
        },
        "target": {
            "token_ids": target_token_ids,
            "padding_mask": target_token_ids != pad_token_id,
        },
    }
