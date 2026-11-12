import torch

from src.dataset.sentencepiece_tokenizer import SentencePieceTokenizer


def test_sentencepiece_tokenizer_round_trip_and_padding(tmp_path):
    texts = [
        "A small English sentence.",
        "Another English example.",
        "Ein kleiner deutscher Satz.",
        "Noch ein deutsches Beispiel.",
    ] * 20
    tokenizer = SentencePieceTokenizer.train(
        iter(texts),
        tmp_path / "tokenizer",
        vocab_size=64,
    )

    encoded = tokenizer(
        ["A small English sentence.", "Ein Satz."],
        padding=True,
        return_tensors="pt",
        return_length=True,
    )

    assert 4 < tokenizer.vocab_size <= 64
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3
    assert encoded["input_ids"].dtype == torch.long
    assert encoded["input_ids"].shape == encoded["attention_mask"].shape
    assert (encoded["input_ids"][:, 0] == tokenizer.bos_token_id).all()
    assert (encoded["attention_mask"] == (encoded["input_ids"] != 0)).all()
    assert tokenizer.decode(encoded["input_ids"][0]) == "A small English sentence."

    token_ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    assert tokens[0] == "<s>"
    assert tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id) == "</s>"
    assert "<s>" not in tokenizer.convert_ids_to_tokens(
        token_ids,
        skip_special_tokens=True,
    )


def test_sentencepiece_tokenizer_truncation_preserves_eos(tmp_path):
    tokenizer = SentencePieceTokenizer.train(
        iter(["one two three four five", "eins zwei drei vier fünf"] * 20),
        tmp_path / "tokenizer",
        vocab_size=48,
    )

    encoded = tokenizer(
        "one two three four five",
        truncation=True,
        max_length=5,
        padding="max_length",
        return_tensors="pt",
    )

    assert encoded["input_ids"].shape == (1, 5)
    assert encoded["input_ids"][0, -1] == tokenizer.eos_token_id
