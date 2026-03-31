# Changelog

| Idea | Score (BPB) | Steps | Serialized model int8+zlib | Total submission size int8+zlib | Run Script | Comments |
|------|-------------|-------|---------------------------|--------------------------------|------------|----------|
| Baseline | 1.22594923 | — | — | — | — | — |
| LeakyReLU^2 (negative_slope=0.5) | 1.22471137 | 13697 | 15817131 | 15865580 | [LeakyReLU](#leakyrelu) | swish2 obtained 1.2281 |
| LeakyReLU^2 + 12 layers | 1.20175212 | 10238 | 20912082 | 20960531 | [LeakyReLU 12L](#leakyrelu-12l) | — |

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

### LeakyReLU 12L

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
