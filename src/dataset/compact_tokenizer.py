"""Compact, reversible ID wrapper for a Hugging Face tokenizer.

The wrapped tokenizer still performs the actual text segmentation.  This class
only replaces its sparse/original token IDs with a dense vocabulary learned
from a training corpus.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CompactTokenizer:
    """Expose a compact ID space on top of an existing tokenizer.

    Call :meth:`fit` using training text before encoding data.  All special
    tokens are reserved first, and every other observed backend ID is assigned
    a consecutive compact ID.  IDs not observed during fitting map to UNK.

    The object intentionally keeps tokenization and ID compaction separate:
    the Hugging Face tokenizer owns text segmentation, while this wrapper owns
    the IDs consumed and produced by the model.
    """

    _SPECIAL_ATTRIBUTES = (
        "pad_token_id",
        "unk_token_id",
        "cls_token_id",
        "sep_token_id",
        "bos_token_id",
        "eos_token_id",
        "mask_token_id",
    )

    def __init__(self, backend_tokenizer: Any) -> None:
        self.backend_tokenizer = backend_tokenizer
        self.original_to_compact: dict[int, int] = {}
        self.compact_to_original: list[int] = []
        self._special_compact_ids: dict[str, int | None] = {}
        self._fitted = False

    def fit(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 1_024,
    ) -> "CompactTokenizer":
        """Build the compact vocabulary from training text.

        ``texts`` may be a generator, so it is consumed exactly once.  Token
        IDs are collected in batches to avoid retaining the tokenized corpus.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        observed: set[int] = set()
        batch: list[str] = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"expected text to be str, got {type(text).__name__}")
            batch.append(text)
            if len(batch) == batch_size:
                self._collect_ids(batch, observed)
                batch.clear()
        if batch:
            self._collect_ids(batch, observed)

        self._build_mapping(observed)
        return self

    def fit_token_ids(self, token_ids: Iterable[int]) -> "CompactTokenizer":
        """Build the vocabulary from already-tokenized backend IDs."""
        self._build_mapping({int(token_id) for token_id in token_ids})
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Tokenize with the backend and compact the returned ``input_ids``."""
        self._require_fitted()
        encoded = self.backend_tokenizer(*args, **kwargs)
        encoded["input_ids"] = self.compact_ids(encoded["input_ids"])
        return encoded

    def encode(self, *args: Any, **kwargs: Any) -> list[int]:
        """Return compact IDs using the backend tokenizer's ``encode``."""
        self._require_fitted()
        return self.compact_ids(self.backend_tokenizer.encode(*args, **kwargs))

    def decode(
        self,
        compact_ids: Sequence[int] | torch.Tensor | np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Convert compact IDs back to backend IDs and decode them."""
        original_ids = self.original_ids(compact_ids)
        if isinstance(original_ids, torch.Tensor):
            original_ids = original_ids.detach().cpu().tolist()
        elif isinstance(original_ids, np.ndarray):
            original_ids = original_ids.tolist()
        return self.backend_tokenizer.decode(original_ids, *args, **kwargs)

    def batch_decode(
        self,
        compact_ids: Sequence[Sequence[int]] | torch.Tensor | np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> list[str]:
        """Batch variant of :meth:`decode`."""
        original_ids = self.original_ids(compact_ids)
        if isinstance(original_ids, torch.Tensor):
            original_ids = original_ids.detach().cpu().tolist()
        elif isinstance(original_ids, np.ndarray):
            original_ids = original_ids.tolist()
        return self.backend_tokenizer.batch_decode(original_ids, *args, **kwargs)

    def compact_ids(self, original_ids: Any) -> Any:
        """Map nested backend IDs to compact IDs, preserving container type."""
        self._require_fitted()
        unk_id = self.unk_token_id
        if unk_id is None:
            raise RuntimeError("the backend tokenizer has no UNK token")
        return self._map_container(
            original_ids,
            lambda token_id: self.original_to_compact.get(int(token_id), unk_id),
        )

    def original_ids(self, compact_ids: Any) -> Any:
        """Map nested compact IDs back to backend IDs."""
        self._require_fitted()
        original_unk_id = self.backend_tokenizer.unk_token_id
        if original_unk_id is None:
            raise RuntimeError("the backend tokenizer has no UNK token")

        def reverse(token_id: int) -> int:
            index = int(token_id)
            if 0 <= index < len(self.compact_to_original):
                return self.compact_to_original[index]
            return int(original_unk_id)

        return self._map_container(compact_ids, reverse)

    def convert_ids_to_tokens(self, compact_ids: Any, *args: Any, **kwargs: Any) -> Any:
        """Return backend token strings for compact IDs."""
        original_ids = self.original_ids(compact_ids)
        return self.backend_tokenizer.convert_ids_to_tokens(original_ids, *args, **kwargs)

    def get_vocab(self) -> dict[str, int]:
        """Return the subset of backend tokens indexed by compact ID."""
        self._require_fitted()
        tokens = self.backend_tokenizer.convert_ids_to_tokens(self.compact_to_original)
        return {token: index for index, token in enumerate(tokens)}

    def save_mapping(self, path: str | Path) -> None:
        """Persist the compact vocabulary without modifying the backend."""
        self._require_fitted()
        payload = {
            "version": 1,
            "backend_name_or_path": getattr(self.backend_tokenizer, "name_or_path", None),
            "compact_to_original": self.compact_to_original,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_mapping(cls, backend_tokenizer: Any, path: str | Path) -> "CompactTokenizer":
        """Restore a saved mapping for a compatible backend tokenizer."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported compact-tokenizer mapping version")
        wrapper = cls(backend_tokenizer)
        wrapper.compact_to_original = [int(value) for value in payload["compact_to_original"]]
        if len(set(wrapper.compact_to_original)) != len(wrapper.compact_to_original):
            raise ValueError("mapping contains duplicate backend IDs")
        wrapper.original_to_compact = {
            original: compact
            for compact, original in enumerate(wrapper.compact_to_original)
        }
        wrapper._set_special_ids()
        wrapper._fitted = True
        return wrapper

    @property
    def vocab_size(self) -> int:
        return len(self.compact_to_original)

    def __len__(self) -> int:
        return self.vocab_size

    def __getattr__(self, name: str) -> Any:
        # Tokenizer methods unrelated to IDs remain available transparently.
        if name == "backend_tokenizer":
            raise AttributeError(name)
        return getattr(self.backend_tokenizer, name)

    def _collect_ids(self, texts: list[str], observed: set[int]) -> None:
        encoded = self.backend_tokenizer(texts, add_special_tokens=True)
        for sequence in encoded["input_ids"]:
            observed.update(int(token_id) for token_id in sequence)

    def _build_mapping(self, observed: set[int]) -> None:
        ordered_specials: list[int] = []
        for attribute in self._SPECIAL_ATTRIBUTES:
            original_id = getattr(self.backend_tokenizer, attribute, None)
            if original_id is not None and int(original_id) not in ordered_specials:
                ordered_specials.append(int(original_id))

        if getattr(self.backend_tokenizer, "unk_token_id", None) is None:
            raise ValueError("backend tokenizer must define unk_token_id")

        observed.update(ordered_specials)
        ordinary_ids = sorted(observed.difference(ordered_specials))
        self.compact_to_original = ordered_specials + ordinary_ids
        self.original_to_compact = {
            original: compact
            for compact, original in enumerate(self.compact_to_original)
        }
        self._set_special_ids()
        self._fitted = True

    def _set_special_ids(self) -> None:
        for attribute in self._SPECIAL_ATTRIBUTES:
            original_id = getattr(self.backend_tokenizer, attribute, None)
            self._special_compact_ids[attribute] = (
                self.original_to_compact.get(int(original_id))
                if original_id is not None
                else None
            )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("CompactTokenizer.fit() must be called before use")

    @staticmethod
    def _map_container(value: Any, mapper: Any) -> Any:
        if isinstance(value, torch.Tensor):
            flat = [mapper(item) for item in value.detach().cpu().reshape(-1).tolist()]
            return torch.tensor(flat, dtype=torch.long, device=value.device).reshape(value.shape)
        if isinstance(value, np.ndarray):
            flat = [mapper(item) for item in value.reshape(-1).tolist()]
            return np.asarray(flat, dtype=np.int64).reshape(value.shape)
        if isinstance(value, tuple):
            return tuple(CompactTokenizer._map_container(item, mapper) for item in value)
        if isinstance(value, list):
            return [CompactTokenizer._map_container(item, mapper) for item in value]
        return mapper(value)


for _attribute in CompactTokenizer._SPECIAL_ATTRIBUTES:
    setattr(
        CompactTokenizer,
        _attribute,
        property(lambda self, attribute=_attribute: self._special_compact_ids.get(attribute)),
    )
