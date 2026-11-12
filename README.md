# Transformer from Scratch

A PyTorch implementation of the encoder–decoder Transformer from [Attention Is All You Need](https://arxiv.org/abs/1706.03762), trained for English → German translation on WMT14.

The project covers the path from implementing attention and masking to training on real data, loading checkpoints, generating translations with beam search, and evaluating them with corpus BLEU. The encoder, decoder, attention layers, positional encodings and generation loop are implemented in this repository rather than delegated to `torch.nn.Transformer`.

**Measured result: 23.63 SacreBLEU on all 3,000 validation examples**, using beam search with width 4 and length normalization α = 0.6.

## What I implemented

### My implementation

- Scaled dot-product multi-head attention, including causal and padding masks.
- Encoder and decoder stacks with cross-attention, feed-forward networks, residual connections and layer normalization.
- Sinusoidal positional encodings and shared encoder/decoder/output embedding weights.
- The training pipeline, including teacher forcing, padding-aware loss and label smoothing, using PyTorch's optimizer and loss APIs.
- Greedy decoding and beam search, including candidate selection, EOS handling, score accumulation and length normalization.
- Generative evaluation and metrics: corpus BLEU aggregation and positional token accuracy.
- Training experiments, checkpoint inspection and analysis of generation behavior.

### AI-generated supporting code

- WMT14 dataset preparation and loading, including persistent tokenized-data caching.
- Training-sequence length filtering, length-bucket sampling and dynamic batch padding to reduce padding overhead, while retaining the full validation split.
- Supporting dataset/cache regression tests.
- This README.

The project uses SentencePiece for a shared 16,000-token English/German BPE vocabulary and a Transformer warmup/inverse-square-root learning-rate schedule.

## Experiment and results

The completed training run used the following configuration:

| Setting | Value |
| --- | --- |
| Dataset | WMT14 English → German |
| Tokenizer | Shared SentencePiece BPE, 16,000 tokens |
| Encoder / decoder layers | 6 / 6 |
| Model dimension | 768 |
| Attention heads / head dimension | 12 / 64 |
| Dropout / label smoothing | 0.1 / 0.1 |
| Training batch size | 256 sentence pairs |
| Training length filter | At most 80 tokens on each side |
| Retained training examples | 4,283,071 of 4,508,785 |
| Completed training | 17 epochs, 284,410 optimizer steps |
| Training precision / backend | bfloat16 / CUDA |

The configuration requests 300,000 steps. The current trainer converts that budget into whole epochs, which produced 284,410 actual steps in this run.

Evaluation was performed on the complete, unfiltered validation split:

| Evaluation setting | Value |
| --- | --- |
| Examples | 3,000 |
| Beam width | 4 |
| Length-normalization α | 0.6 |
| Maximum generated tokens | 160, excluding BOS and including EOS |
| Model precision / backend | bfloat16 / Apple MPS |
| Corpus BLEU | **23.6255** |
| Outputs without EOS | 0 |

SacreBLEU signature:

```text
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
```

BLEU is computed from accumulated n-gram counts and lengths across batches, not by averaging batch BLEU scores. Beam-search scores use float32 log-probabilities even when the model runs in bfloat16.

The training logs also record validation loss decreasing from 2.8913 after the first epoch to 2.6385 after the last epoch. These are the loss summaries currently produced by the trainer.

### Translation example

An example from the evaluated validation split:

> **Source:** I found out the price of the weapon only there  
> **Reference:** Den Preis für die Waffe erfuhr ich erst dort  
> **Generated:** Ich habe den Preis der Waffe nur dort herausgefunden.

This example illustrates model output; the headline score is calculated over the full validation corpus.

## Repository layout

```text
config/             Training and model configurations
src/models/         Transformer components, generation and ONNX export
src/dataset/        Tokenizer, WMT14 adapter, cache and batch sampling
src/trainer/        Training, validation and checkpoint saving
src/metrics/        Epoch-level generative evaluation
src/scheduler/      Learning-rate schedules
src/test/           Model, data, trainer and export tests
```

The core implementation is in [Base.py](src/models/Base.py); the experiment configuration is [transformer.yaml](config/transformer.yaml).

## Setup

Run commands from the repository root.

The supplied `environment.yml` targets Python 3.12 and a CUDA 13.0 build of PyTorch:

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
```

That environment definition is intended for a compatible Linux/CUDA machine. On macOS or a CPU-only machine, use a platform-compatible PyTorch installation instead of its CUDA-specific package pin. The trainer selects CUDA, then Apple MPS, then CPU when hardware acceleration is enabled. `generate` uses the device of the model's parameters.

## Prepare the data

```bash
python -m src.dataset.wmt14_cache --splits train validation
```

The first preparation downloads the source data and trains the tokenizer if its model is absent, then persists the tokenized splits under `src/dataset/data/`. Subsequent runs reuse the local cache.

The cache stores complete token IDs without padding or truncation. The training dataset applies the configured length filter when loading; validation retains every example. Batch sampling groups similar lengths and the collate function pads only to the longest sequence in each batch.

Changing the training length filter reuses the token cache. Changing tokenizer contents, processing version or recorded dependency versions produces a different cache identity.

To rebuild explicitly:

```bash
python -m src.dataset.wmt14_cache --splits train validation --rebuild
```

Additional options include `--tokenizer path/to/tokenizer.model` and `--cache-dir path/to/cache`. When using a custom cache directory, supply the same `cache_dir` in the dataset configuration. The test split is prepared only when explicitly requested.

## Train

```bash
mkdir -p runs
python -m src.trainer.TransformerTrainer --config config/transformer.yaml
```

The trainer creates a run directory containing configuration, logs, TensorBoard events and checkpoints. At completion, it attempts to export an ONNX model.

Important configuration details:

- `dataset.train.max_length` filters training examples.
- `model.args.max_length` sets positional capacity and the maximum number of newly generated tokens.
- Validation is unfiltered, so model positional capacity must cover its actual sequence lengths.
- `checkpoint_freq` is measured in epochs. The current experiment value of 100 produces no intermediate checkpoint in a 17-epoch run; choose a smaller value when intermediate saves are needed.

To inspect recorded metrics:

```bash
tensorboard --logdir runs
```

## Load a checkpoint and generate

Weights and tokenizer artifacts are not bundled with the repository. The example below assumes you have the final checkpoint and the exact tokenizer used for that run. Replace the two artifact paths before running it.

```python
import torch

from src.dataset.sentencepiece_tokenizer import SentencePieceTokenizer
from src.models.Base import Transformer
from src.utils.util import get_available_device

tokenizer = SentencePieceTokenizer("path/to/tokenizer.model")
model = Transformer(
    device=get_available_device(),
    tokenizer=tokenizer,
    model_dim=768,
    num_heads=12,
    num_blocks=6,
    query_dim=64,
    max_length=160,
    dropout=0.1,
    dtype=torch.bfloat16,
)
checkpoint = torch.load("path/to/final.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state"])
model.eval()

encoded = tokenizer("Our staff is ready to help.", return_tensors="pt")
prediction = model.generate(
    enc_input=encoded["input_ids"],
    enc_padding_mask=encoded["attention_mask"],
    beam=4,
    shortness_penalization=0.6,
)
translations = tokenizer.batch_decode(
    prediction["token_ids"], skip_special_tokens=True
)
print(translations[0])
```

Checkpoint files contain model, optimizer and scheduler states, plus epoch and step counters. Loading weights for inference is supported; automatic training resumption is not implemented in the trainer.

## Tests

Test files cover masks, tensor shapes, forward/backward passes, tiny-batch overfitting, tokenization, cache persistence and ONNX generation.

Offline tokenizer/cache checks:

```bash
python -m pytest src/test/test_wmt14_cache.py src/test/test_sentencepiece_tokenizer.py -q
```

Pytest must be installed separately if it is not available in your environment. ONNX runtime checks additionally require `onnxruntime`.

Token-accuracy tests cover the generative evaluation interface, padding and length mismatches. ONNX tests compare exported generation against PyTorch with `beam=1`.

ONNX exports greedy generation; beam search is available through the Python model.

## Reference

Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
