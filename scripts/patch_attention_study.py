"""
The patch-attention study (the layer-norm and attention workplan, stage 6, D11): small patch
models, a conv network and a dense network on full MNIST, numpy, under one protocol.

    python scripts/patch_attention_study.py time --arm attention-ffn
    python scripts/patch_attention_study.py tune --out tune.json
    python scripts/patch_attention_study.py sweep --rates 0.001 0.001 0.001 0.001 0.001 --out sweep.json

The question (Dosovitskiy et al. 2020, "An Image is Worth 16x16 Words"): does attention, tokens
seeing each other before the mean, help a patch model, and how do patch models fare against a conv
network at this data size?

The protocol is indrajala_ml/studies/patch_study.py's, shared with the multi-head attention study.
Every patch model cuts the (28, 28, 1) image into 16 patches of 7 x 7, embeds each as d = 32
features (a token-wise Dense(32, "linear", bias=True)), adds learned positions, runs its blocks,
averages over the tokens, layer-norms the mean and classifies:

- ffn: one FFN block, Residual((LayerNorm(), Dense(64, "relu"), Dense(32, "linear", bias=True))),
  so the tokens never see each other until the mean;
- attention: one attention block, Residual((LayerNorm(), Attention()));
- attention-ffn: the attention block, then the FFN block (D1).

And two controls:

- conv: the conv preset's layers (bss.CONV_SPECS, ConvSpec(3, 8) ReLU, then Dense(32) sigmoid) as a
  Sequential network, with the patch models' output layer in place of the preset's sigmoid one;
- dense: Dense(14, "relu"), about attention-ffn's parameter count.

Every network ends in Dense(10, "softmax", cross-entropy). Training: Adam at batch 32, one learning
rate per arm (`tune`: each arm over TUNE_RATES, the best mean final test accuracy), the trainer's
epoch loop (batch_size_scaling.train_epoch), fan-in-aware initialization from each seed. Recorded
per cell: test accuracy after every epoch, the parameter count, and the seconds per epoch spent in
learn_batch (OPENBLAS_NUM_THREADS=1, WORKERS jobs at once, so a comparison between arms, not an
absolute speed).

Findings, numpy, OPENBLAS_NUM_THREADS=1, 4 workers. One attention-ffn epoch alone took 10.9 s
(`time`), so the grid was sized up from the plan's short sweep: `tune` ran every arm at every rate
for 2 epochs and 3 seeds, and `sweep` ran 5 epochs and 3 seeds at each arm's best rate. Its first
two epochs repeat tune's to the digit (same seeds, same rates).

- tune: the best rates were 0.004 (ffn, attention-ffn, dense) and 0.002 (attention, conv). Every
  arm trained at every rate, and none diverged. The neighbouring rates were within about a point.
- Attention then FFN beats FFN alone, but only just: 95.82% +- 0.21% against 95.41% +- 0.45% at
  epoch 5, a gap of less than the FFN arm's standard deviation, with 63% more parameters
  (11050 against 6762) at about twice the time per epoch. Per seed: 95.58, 95.94, 95.94 against
  95.22, 95.92, 95.09. The FFN arm led at epochs 1 and 2, and attention-ffn led from epoch 3 on.
- The attention block alone is the weakest patch model, 92.43% +- 0.99%, below FFN alone at the
  same parameter count (6794 against 6762) and below the dense control. Without a token-wise
  nonlinearity, a patch's 49 values reach the mean only through affine maps and the softmax
  weights.
- The dense control, Dense(14, "relu") at attention-ffn's parameter count, reaches 93.95% +-
  0.21%, 1.9 points under attention-ffn, and is the fastest by 2x.
- The conv network wins, 98.00% +- 0.06%, 2.2 points over the best patch model, with 16x
  attention-ffn's parameters (nearly all in its Dense(32)) and 2-4x its time per epoch (sweep, tune).

So the expected result holds, weakly: attention then FFN beats FFN alone, by less than a standard
deviation over 3 seeds, and the conv network beats every patch model at this data size (ViTs need
more data; Dosovitskiy et al. 2020).

| arm | rate | parameters | epoch 1 | epoch 3 | epoch 5 | seconds per epoch |
|---|---|---|---|---|---|---|
| ffn | 0.004 | 6762 | 92.78% +- 0.39% | 94.22% +- 0.82% | 95.41% +- 0.45% | 11.0 |
| attention | 0.002 | 6794 | 86.99% +- 0.69% | 91.15% +- 0.84% | 92.43% +- 0.99% | 12.8 |
| attention-ffn | 0.004 | 11050 | 90.68% +- 0.46% | 94.78% +- 0.69% | 95.82% +- 0.21% | 25.3 |
| conv | 0.002 | 173498 | 96.89% +- 0.43% | 97.87% +- 0.06% | 98.00% +- 0.06% | 56.4 |
| dense | 0.004 | 11140 | 92.64% +- 0.27% | 93.56% +- 0.08% | 93.95% +- 0.21% | 5.9 |

Test accuracy, mean +- standard deviation over the seeds; seconds per epoch are the mean over the
seeds and epochs, and they vary with which jobs share the machine (attention-ffn's spread is 9.0
s). The per-epoch tables are in the stage's PR.
"""

from __future__ import annotations

from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.studies import batch_size_scaling as bss
from indrajala_ml.studies import patch_study

DENSE_CONTROL_SIZE = 14  # 795 * 14 + 10 = 11140 parameters, attention-ffn has 11050
ARMS = ["ffn", "attention", "attention-ffn", "conv", "dense"]
SEEDS = [0, 1, 2]
EPOCHS = 5


def arm_specs(arm: str) -> list[LayerSpec]:
    """The layers of one arm of the study."""
    if arm == "conv":
        (dense_size,) = bss.CONV_DENSE_LAYER_SIZES
        return [*bss.CONV_SPECS, Dense(dense_size), patch_study.output()]
    if arm == "dense":
        return [Dense(DENSE_CONTROL_SIZE, activation="relu"), patch_study.output()]
    blocks = {
        "ffn": [patch_study.ffn_block()],
        "attention": [patch_study.attention_block()],
        "attention-ffn": [patch_study.attention_block(), patch_study.ffn_block()],
    }
    if arm not in blocks:
        raise ValueError(f"unknown arm {arm!r}")
    return patch_study.patch_model(blocks[arm])


def main(argv: list[str] | None = None) -> None:
    patch_study.main(argv, __doc__, ARMS, arm_specs, len(SEEDS), EPOCHS, time_arm="attention-ffn")


if __name__ == "__main__":
    main()
