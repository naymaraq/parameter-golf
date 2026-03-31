## Lessons Learned

1. **LeakyReLU activation is slightly better** — `leaky_relu2` (LeakyReLU squared) outperforms the baseline (swish2 got 1.2281 vs 1.2247 for LeakyReLU^2).
2. **Increasing L/D blows the 16MB artifact limit** — both the 12L D512 and 9L D640 configs exceed 16MB after int8+zlib compression. To benefit from larger models, more aggressive quantization (e.g., int4 or pruning) is needed to stay within budget.

# Change Log

| Idea | Score (BPB) | Steps | Serialized model int8+zlib | Total submission size int8+zlib | Run Script | Comments |
|------|-------------|-------|---------------------------|--------------------------------|------------|----------|
| Baseline | 1.22594923 | — | — | — | — | — |
| LeakyReLU^2 (negative_slope=0.5) | 1.22471137 | 13697 | 15817131 | 15865580 | [LeakyReLU](#leakyrelu) | swish2 obtained 1.2281 |
| LeakyReLU^2 + L12 + D512 | 1.20175212 | 10238 | 20912082 | 20960531 | [LeakyReLU 12L D512](#leakyrelu-12l-d512) | large artifact (> 16MB) |
| LeakyReLU^2 + L9 + D640 | 1.19589325 | 9603 | 24422123 | 24470572 | [LeakyReLU 9L D640](#leakyrelu-9l-d640) | large artifact (> 16MB) |

## Run Scripts

### LeakyReLU

```bash
SEED=1337 \
NCCL_IB_DISABLE=1 \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
RUN_ID=${id} \
DATA_PATH=/golf/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/golf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
MLP_ACTIVATION='leaky_relu2' \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### LeakyReLU 12L D512

```bash
SEED=1337 \
NCCL_IB_DISABLE=1 \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
RUN_ID=${id} \
DATA_PATH=/golf/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/golf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
MLP_ACTIVATION='leaky_relu2' \
NUM_LAYERS=12 \
MODEL_DIM=512 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### LeakyReLU 9L D640

```bash
SEED=1337 \
NCCL_IB_DISABLE=1 \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
RUN_ID=${id} \
DATA_PATH=/golf/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/golf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
MLP_ACTIVATION='leaky_relu2' \
NUM_LAYERS=9 \
MODEL_DIM=640 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
