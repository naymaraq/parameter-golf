# Changelog

| Idea | Score (BPB) | Size | Run Script | Comments |
|------|-------------|------|------------|----------|
| Baseline | 1.22594923 | — | — | — |
| LeakyReLU^2 (negative_slope=0.5) | 1.22471137 | — | [LeakyReLU](#leakyrelu) | swish2 obtained 1.2281 |

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
