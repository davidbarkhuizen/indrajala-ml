"""
The multi-head attention study (the multi-head attention workplan, stage 7, D8): patch models with
more heads, wider heads and a second layer on full MNIST, numpy, under the patch-attention study's
protocol (indrajala_ml/studies/patch_study.py).

    python scripts/multi_head_attention_study.py time --arm 4-head-2-layer
    python scripts/multi_head_attention_study.py tune --out tune.json
    python scripts/multi_head_attention_study.py sweep --rates 0.004 ... --out sweep.json

The questions: at the patch-attention study's size (d = 32, 16 tokens), do more heads help at
equal parameters (Vaswani et al. 2017, "Attention Is All You Need", Table 3, rows (A)); does a
key size above d / heads help (Bhojanapalli et al. 2020, "Low-Rank Bottleneck in Multi-head
Attention Models"); and does depth help more than heads?

Every arm is the patch-attention study's patch model (16 patches of 7 x 7 embedded as d = 32,
learned positions, the mean, a layer norm, a softmax output), its layers Residual((LayerNorm(),
Attention(...))) then the FFN block Residual((LayerNorm(), Dense(64, "relu"), Dense(32, "linear",
bias=True))):

- 1-head: one layer, Attention(): the patch-attention study's attention-ffn, rerun as the anchor;
- 2-head: one layer, Attention(heads=2), d_k = 16;
- 4-head: one layer, Attention(heads=4), d_k = 8;
- 4-head-k32: one layer, Attention(heads=4, key_size=32): the bottleneck test, 16,800 attention
  parameters against the defaults' 4,224;
- 1-head-2-layer: two layers of 1-head's;
- 4-head-2-layer: two layers of 4-head's;

and ffn, the FFN block alone (the patch-attention study's ffn arm), the no-attention control. The
conv and dense controls are cited from the patch-attention study, not rerun: their accuracies
don't depend on the machine, and seconds per epoch compare only within one study.

Adam at batch 32, one rate per arm from `tune`, 5 epochs, 5 seeds (3 in the patch-attention
study), OPENBLAS_NUM_THREADS=1, 4 workers.

Findings, on jebel (i7-9700K), nothing else running. One epoch of the slowest arm, 4-head-2-layer,
took 13.4 s alone (`time`), so `tune` ran every arm at every rate in TUNE_RATES for 2 epochs and 5
seeds, and `sweep` ran 5 epochs and 5 seeds at each arm's best rate. The first three seeds repeat
the patch-attention study's to the digit (its ffn arm at 0.004, epoch 1: 92.91%, 93.09%, 92.35% on
both, though it ran on the Ryzen laptop), so its conv and dense controls compare as cited.

- tune: the best rates were 0.001 (1-head, 4-head-2-layer) and 0.002 (the rest), one step or two
  under the patch-attention study's 0.004 for attention-ffn and ffn, picked over 5 seeds where it
  had 3; neighbouring rates were within about a point, and no arm diverged.
- More heads, at equal parameters, change nothing at this size: 96.07% (1 head), 96.16% (2) and
  96.04% (4) after 5 epochs, per-seed differences against one head of +0.09 +- 0.32 and -0.03 +-
  0.35 points. Vaswani et al.'s gain from heads (Table 3, rows (A)) doesn't appear with 16 tokens
  of 32 features.
- Wider heads don't resolve the bottleneck question: 4-head-k32 is 0.22 +- 0.40 points over 4-head
  (3 of 5 seeds), within noise, at 2.1x the parameters and 1.6x the time per epoch. At d_k = 8 and
  T = 16 each head's scores have rank at most 8 of 16, so the bound binds, but this task doesn't
  measurably need the full rank; with more tokens (patch size 4, T = 49) it would bind harder.
- Depth helps, more than heads, as expected: a second layer adds 0.57 +- 0.45 points at one head
  and 0.95 +- 0.41 at four, on every seed. 4-head-2-layer is the best patch model, 96.98% +- 0.21%,
  0.91 +- 0.18 points over 1-head on every seed; at two layers four heads lead one by 0.34 +- 0.55
  (4 of 5 seeds), within noise.
- Attention's gain over the FFN block alone, the patch-attention study's open margin, holds at 5
  seeds and the tuned rates: 1-head beats ffn by 0.40 +- 0.30 points (4 of 5 seeds), the same 0.4
  as there, now above both arms' standard deviations (0.12, 0.35).
- The best patch model is still under the conv network (98.00% +- 0.06%, the patch-attention study),
  now by 1.0 point instead of 2.2, with 11% of its parameters (19594 against 173498).

| arm | rate | parameters | epoch 1 | epoch 3 | epoch 5 | seconds per epoch |
|---|---|---|---|---|---|---|
| 1-head | 0.001 | 11050 | 91.22% +- 0.57% | 95.05% +- 0.15% | 96.07% +- 0.12% | 8.3 |
| 2-head | 0.002 | 11050 | 92.51% +- 0.20% | 95.30% +- 0.31% | 96.16% +- 0.36% | 8.7 |
| 4-head | 0.002 | 11050 | 92.51% +- 0.45% | 95.64% +- 0.28% | 96.04% +- 0.32% | 9.7 |
| 4-head-k32 | 0.002 | 23626 | 93.24% +- 0.76% | 95.47% +- 0.35% | 96.25% +- 0.32% | 15.6 |
| 1-head-2-layer | 0.002 | 19594 | 92.42% +- 1.06% | 95.66% +- 0.30% | 96.64% +- 0.43% | 14.0 |
| 4-head-2-layer | 0.001 | 19594 | 93.26% +- 0.94% | 96.31% +- 0.38% | 96.98% +- 0.21% | 16.7 |
| ffn | 0.002 | 6762 | 92.32% +- 0.36% | 94.22% +- 1.02% | 95.67% +- 0.35% | 4.7 |

Test accuracy, mean +- standard deviation over the 5 seeds; a difference "+- x" is the mean and
standard deviation of the per-seed differences (the arms share seeds, so their initial draws and
shuffles differ only by architecture). Seconds per epoch are the mean over the seeds and epochs,
4 jobs at once; four heads cost 17% over one, and a second layer about 1.7x. The per-epoch tables
are in the stage's PR.
"""

from __future__ import annotations

from indrajala_ml.model.specs.layer_specs import Attention, LayerSpec
from indrajala_ml.studies import patch_study

# arm: (its attention layer, its number of attention-then-FFN layers); ffn has none
ATTENTION_ARMS: dict[str, tuple[Attention, int]] = {
    "1-head": (Attention(), 1),
    "2-head": (Attention(heads=2), 1),
    "4-head": (Attention(heads=4), 1),
    "4-head-k32": (Attention(heads=4, key_size=32), 1),
    "1-head-2-layer": (Attention(), 2),
    "4-head-2-layer": (Attention(heads=4), 2),
}
ARMS = [*ATTENTION_ARMS, "ffn"]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 5


def arm_specs(arm: str) -> list[LayerSpec]:
    """The layers of one arm of the study."""
    if arm == "ffn":
        return patch_study.patch_model([patch_study.ffn_block()])
    if arm not in ATTENTION_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    attention, layers = ATTENTION_ARMS[arm]
    return patch_study.patch_model([patch_study.attention_block(attention), patch_study.ffn_block()] * layers)


def main(argv: list[str] | None = None) -> None:
    patch_study.main(argv, __doc__, ARMS, arm_specs, len(SEEDS), EPOCHS, time_arm="4-head-2-layer")


if __name__ == "__main__":
    main()
