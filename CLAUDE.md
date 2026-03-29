# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**Parameter Golf** is an open competition (by OpenAI) to train the best LLM within a 16MB artifact constraint in 10 minutes on 8×H100 GPUs. The metric is bits-per-byte (BPB) on the FineWeb validation set. Submissions go in `records/track_10min_16mb/` (official) or `records/track_non_record_16mb/` (experimental/unlimited compute).

## Setup & Data

```bash
pip install -r requirements.txt

# Download FineWeb dataset (default tokenizer: SentencePiece 1024 vocab)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

## Running Training

All configuration is via environment variables — there are no config files.

**Single GPU (local testing):**
```bash
RUN_ID=mytest \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**8×H100 (official submission run):**
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Apple Silicon (Mac):**
```bash
python3 train_gpt_mlx.py
```

## Key Hyperparameters (env vars with defaults)

| Variable | Default | Description |
|---|---|---|
| `NUM_LAYERS` | 9 | Transformer depth |
| `MODEL_DIM` | 512 | Hidden dimension |
| `NUM_HEADS` | 8 | Attention heads (4 KV heads, GQA) |
| `VOCAB_SIZE` | 1024 | Must match tokenizer |
| `ITERATIONS` | 20000 | Max training steps |
| `TRAIN_BATCH_TOKENS` | 524288 | Tokens per step |
| `MAX_WALLCLOCK_SECONDS` | 600.0 | Hard 10-min cap |
| `MATRIX_LR` | 0.04 | LR for weight matrices (Muon) |
| `TIED_EMBED_LR` | 0.05 | LR for tied embeddings |

## Architecture

The baseline model in `train_gpt.py` (the file to modify for submissions):
- Transformer with RoPE, GQA (Grouped Query Attention), logit softcap
- Tied input/output embeddings (critical for parameter efficiency)
- **Muon optimizer** for matrix parameters (Newton-Schulz gradient orthogonalization), Adam for scalars/embeddings
- Warmup (20 steps) + warmdown LR schedule over 1200 final iterations
- Distributed via `torchrun` + `DistributedDataParallel`

**Artifact size** = `len(train_gpt.py bytes)` + `zlib(int8_quantized_model)` — must stay under 16MB.

## Submission Structure

Each submission is a self-contained folder under `records/`:
```
records/track_10min_16mb/<submission_name>/
├── train_gpt.py       # Modified training script (the submission artifact)
├── submission.json    # {author, score_bpb, artifact_bytes, ...}
├── README.md          # Approach description
└── train.log          # Full training output (for significance verification)
```

Score improves must be ≥ 0.005 nats with p < 0.01 statistical confidence (multiple runs required).

## MLX Variant Notes

`train_gpt_mlx.py` targets Apple Silicon. Differences from `train_gpt.py`:
- Single-process (no `torchrun`)
- Chunked logit computation to reduce peak memory
- Eager evaluation materialization per microbatch
