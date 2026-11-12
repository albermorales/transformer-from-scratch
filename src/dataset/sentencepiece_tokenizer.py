"""Small Hugging Face-like adapter around a SentencePiece BPE model."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch


class SentencePieceTokenizer:
    """Expose the tokenizer interface used by this repository.

    The model is trained with fixed PAD/UNK/BOS/EOS IDs and adds BOS/EOS to
    every encoded sentence. It intentionally implements only the operations
    needed by the dataset, trainer, and translation evaluation.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"SentencePiece model not found: {self.model_path}")
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.name_or_path = str(self.model_path)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        model_prefix: str | Path,
        *,
        vocab_size: int = 16_000,
    ) -> "SentencePieceTokenizer":
        """Train and persist a shared BPE tokenizer from an iterable of text."""
        prefix = Path(model_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(texts),
            model_prefix=str(prefix),
            model_type="bpe",
            vocab_size=vocab_size,
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            hard_vocab_limit=False,
        )
        return cls(prefix.with_suffix(".model"))

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.processor.pad_id()

    @property
    def unk_token_id(self) -> int:
        return self.processor.unk_id()

    @property
    def bos_token_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_token_id(self) -> int:
        return self.processor.eos_id()

    # Compatibility aliases for code that previously used a BERT tokenizer.
    cls_token_id = bos_token_id
    sep_token_id = eos_token_id

    @property
    def backend_tokenizer(self) -> "SentencePieceTokenizer":
        return self

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = self.processor.encode(text, out_type=int)
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def convert_ids_to_tokens(
        self,
        ids: int | Sequence[int] | torch.Tensor,
        *,
        skip_special_tokens: bool = False,
    ) -> str | list[str]:
        """Convert token IDs to SentencePiece pieces using the Hugging Face API shape."""
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu()
            if ids.ndim == 0:
                ids = int(ids.item())
            else:
                ids = ids.tolist()

        if isinstance(ids, int):
            return self.processor.id_to_piece(ids)

        special_ids = {
            self.pad_token_id,
            self.unk_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }
        return [
            self.processor.id_to_piece(int(token_id))
            for token_id in ids
            if not skip_special_tokens or int(token_id) not in special_ids
        ]

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_length: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Encode one sentence or a batch using the subset of the HF API we need."""
        single = isinstance(texts, str)
        sentences = [texts] if single else list(texts)
        encoded = [
            self.encode(sentence, add_special_tokens=add_special_tokens)
            for sentence in sentences
        ]

        if truncation:
            if max_length is None:
                raise ValueError("max_length is required when truncation=True")
            if max_length < int(add_special_tokens) * 2:
                raise ValueError("max_length is too small for BOS/EOS")
            for index, ids in enumerate(encoded):
                if len(ids) > max_length:
                    encoded[index] = ids[:max_length]
                    if add_special_tokens:
                        encoded[index][-1] = self.eos_token_id

        lengths = [len(ids) for ids in encoded]
        target_length: int | None = None
        if padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required for padding='max_length'")
            target_length = max_length
        elif padding:
            target_length = max(lengths, default=0)

        if target_length is not None:
            encoded = [
                ids + [self.pad_token_id] * max(0, target_length - len(ids))
                for ids in encoded
            ]

        masks = [
            [int(token_id != self.pad_token_id) for token_id in ids]
            for ids in encoded
        ]

        if return_tensors is not None:
            if return_tensors != "pt":
                raise ValueError("only return_tensors='pt' is supported")
            if len({len(ids) for ids in encoded}) > 1:
                raise ValueError("padding is required for unequal-length tensor batches")
            input_ids: Any = torch.tensor(encoded, dtype=torch.long)
            attention_mask: Any = torch.tensor(masks, dtype=torch.long)
        elif single:
            input_ids = encoded[0]
            attention_mask = masks[0]
        else:
            input_ids = encoded
            attention_mask = masks

        result: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_length:
            result["length"] = lengths
        return result

    def decode(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        ids = self._as_list(token_ids)
        if skip_special_tokens:
            special_ids = {
                self.pad_token_id,
                self.unk_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }
            ids = [token_id for token_id in ids if token_id not in special_ids]
        return self.processor.decode(ids)

    def batch_decode(
        self,
        sequences: Sequence[Sequence[int]] | torch.Tensor,
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.detach().cpu().tolist()
        return [
            self.decode(sequence, skip_special_tokens=skip_special_tokens)
            for sequence in sequences
        ]

    @staticmethod
    def _as_list(token_ids: Sequence[int] | torch.Tensor) -> list[int]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return [int(token_id) for token_id in token_ids]
