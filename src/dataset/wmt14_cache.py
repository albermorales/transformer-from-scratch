"""Persistent, versioned WMT14 tokenization (no padding or truncation)."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import shutil
import tempfile

from datasets import Dataset, Features, List, Value, load_from_disk
from filelock import FileLock

from .sentencepiece_tokenizer import SentencePieceTokenizer

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "data" / "tokenized_wmt14"
logger = logging.getLogger(__name__)


def _identity(tokenizer):
    return {
        "processing_version": CACHE_VERSION,
        "dataset": "wmt/wmt14",
        "configuration": "de-en",
        "tokenizer_sha256": hashlib.sha256(
            tokenizer.processor.serialized_model_proto()
        ).hexdigest(),
        "sentencepiece_version": version("sentencepiece"),
        "datasets_version": version("datasets"),
    }


def _tokenize_batch(batch, tokenizer):
    english = [tokenizer.encode(item["en"]) for item in batch["translation"]]
    german = [tokenizer.encode(item["de"]) for item in batch["translation"]]
    return {
        "en_token_ids": english,
        "de_token_ids": german,
        # Preserve the effective lengths used by SentenceDataset.
        "en_length": [len(ids) - 1 for ids in english],
        "de_length": [len(ids) - 1 for ids in german],
    }


def load_tokenized_wmt14(
    tokenizer: SentencePieceTokenizer,
    split: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    rebuild: bool = False,
    source_loader=None,
) -> Dataset:
    """Load a complete artifact; tokenize only when missing or explicitly rebuilt.

    source_loader is called only during preparation. Source updates therefore
    require an explicit rebuild; the source fingerprint is recorded for tracing.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported WMT14 split: {split!r}")
    identity = _identity(tokenizer)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir) / key
    root.mkdir(parents=True, exist_ok=True)
    target = root / split
    with FileLock(str(root / f"{split}.lock")):
        backup = root / f".{split}-previous"
        if backup.exists() and not target.exists():
            backup.rename(target)
        if target.exists() and not rebuild:
            try:
                manifest = json.loads((target / "manifest.json").read_text())
                if manifest["identity"] != identity or manifest["split"] != split:
                    raise ValueError("identity or split mismatch")
                prepared = load_from_disk(str(target / "dataset"))
                required = {"translation", "en_token_ids", "de_token_ids",
                            "en_length", "de_length"}
                if not required.issubset(prepared.column_names):
                    raise ValueError("missing columns")
                if len(prepared) != manifest["num_rows"]:
                    raise ValueError("row count mismatch")
            except (OSError, ValueError, KeyError) as error:
                raise RuntimeError(
                    f"Invalid tokenized cache at {target}; rebuild it explicitly."
                ) from error
            logger.info("Loaded tokenized WMT14 %s from %s", split, target)
            return prepared

        if source_loader is None:
            from .English2GermanWMT14 import load_wmt14
            source_loader = lambda: load_wmt14()[split]
        logger.info("Preparing persistent WMT14 %s tokens (first preparation can be slow)", split)
        source = source_loader()
        features = Features({
            "translation": source.features["translation"],
            "en_token_ids": List(Value("int32")),
            "de_token_ids": List(Value("int32")),
            "en_length": Value("int32"),
            "de_length": Value("int32"),
        })
        staging = Path(tempfile.mkdtemp(prefix=f".{split}-", dir=root))
        try:
            prepared = source.map(
                _tokenize_batch,
                batched=True,
                batch_size=4096,
                fn_kwargs={"tokenizer": tokenizer},
                remove_columns=[name for name in source.column_names if name != "translation"],
                features=features,
                cache_file_name=str(staging / "working.arrow"),
                load_from_cache_file=False,
                desc=f"Tokenizing WMT14 {split}",
            )
            manifest = {
                "identity": identity,
                "split": split,
                "source_fingerprint": source._fingerprint,
                "num_rows": len(prepared),
            }
            prepared.save_to_disk(str(staging / "dataset"))
            del prepared
            (staging / "working.arrow").unlink()
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            # Recover an old generation left by an interrupted replacement.
            if backup.exists():
                if target.exists():
                    shutil.rmtree(backup)
                else:
                    backup.rename(target)
            if target.exists():
                target.rename(backup)
            try:
                staging.rename(target)
            except BaseException:
                if backup.exists() and not target.exists():
                    backup.rename(target)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return load_from_disk(str(target / "dataset"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"],
                        default=["train", "validation"])
    parser.add_argument("--tokenizer", default="wmt14-en-de-bpe-16000")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from .English2GermanWMT14 import _load_tokenizer
    tokenizer = _load_tokenizer(args.tokenizer)
    for split in args.splits:
        load_tokenized_wmt14(tokenizer, split, cache_dir=args.cache_dir, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
