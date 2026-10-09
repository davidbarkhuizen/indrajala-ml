"""
The patch geometry study (the multi-head attention workplan's D8 (b) and (c), next-steps): the
multi-head attention study's questions at more tokens and at wider tokens, under the patch-model
studies' protocol (indrajala_ml/studies/patch_study.py).

    python scripts/patch_geometry_study.py time --arm p4-4-head-k32
    python scripts/patch_geometry_study.py tune --out tune.json
    python scripts/patch_geometry_study.py sweep --rates 0.002 ... --out sweep.json

The multi-head study, at patch size 7 (T = 16 tokens) and d = 32, found that more heads changed
nothing and that lifting the low-rank bound (key_size=32) stayed within noise (+0.22 +- 0.40
points): at d_k = 8 each head's scores have rank at most 8 of 16. Two questions follow:

- Patch size 4, T = 49 tokens of 16 values, d = 32: the bound is 8 of 49. Does lifting it help
  now (Bhojanapalli et al. 2020, "Low-Rank Bottleneck in Multi-head Attention Models"), and does
  attention gain more over the FFN block alone when there are more tokens to mix?
  Arms p4-1-head, p4-4-head, p4-4-head-k32 and the control p4-ffn.
- d = 64, patch size 7: do the heads (d_k = 16 at four) and depth findings hold at wider tokens?
  The FFN blocks are 128 wide, keeping the multi-head study's 2d. Arms d64-1-head, d64-4-head,
  d64-4-head-2-layer and the control d64-ffn.

The patch-7, d = 32 arms are cited from the multi-head study, not rerun: their accuracies don't
depend on the machine, and its first three seeds repeat the patch-attention study's to the digit.
Seconds per epoch compare only within one study.

Adam at batch 32, one rate per arm from `tune`, 5 epochs, 5 seeds, OPENBLAS_NUM_THREADS=1,
4 workers. Attention's cost grows with T^2: one epoch of the slowest arm is timed first and the
grid sized from it.

Findings, on jebel (i7-9700K), nothing else running. One epoch of the slowest arm, p4-4-head-k32,
took 42.0 s alone (`time`), so `tune` ran every arm at every rate in TUNE_RATES for 2 epochs and 5
seeds (about an hour), and `sweep` 5 epochs and 5 seeds at each arm's best rate (about 25 min).
The seeds are the multi-head study's, so each arm is paired seed by seed with its cited
counterpart there.

- tune: the best rates were 0.0005 (d64-4-head-2-layer), 0.001 (p4-1-head, d64-1-head,
  d64-4-head), 0.002 (p4-4-head-k32, p4-ffn) and 0.004 (p4-4-head, d64-ffn); no arm diverged.
- Smaller patches cost accuracy at this length of training: every patch-4 arm ends under its
  patch-7 counterpart, 1-head by 1.21 +- 0.72 points, 4-head by 0.60 +- 0.33, 4-head-k32 by
  0.40 +- 0.42 and ffn by 0.86 +- 0.55, on every seed but one of 4-head-k32's. A token of 16 pixels carries less
  than one of 49, and 49 positions are more to learn in 5 epochs.
- The bottleneck still doesn't measurably bind: at T = 49, where each 8-wide head's scores have
  rank at most 8 of 49, lifting it (4-head-k32) adds 0.42 +- 0.83 points over 4-head (4 of 5
  seeds), twice the patch-7 margin (+0.22 +- 0.40) but no more resolved.
- With more tokens, heads start to matter, and attention needs them: 4-head is 0.58 +- 0.73 over
  1-head (4 of 5) where at T = 16 heads changed nothing; one head beats the FFN block alone by
  only 0.05 +- 0.49, four by 0.63 +- 0.45 and four wide ones by 1.05 +- 0.68 (4 of 5 each).
- At d = 64, heads and depth both help on every seed: 4-head over 1-head +0.62 +- 0.51, a second
  layer +0.48 +- 0.22 (5 of 5 each). But the wider FFN block alone (128 wide) gains the most
  from the width, +0.74 +- 0.47 over its d = 32 control (5 of 5), and stands level with
  attention: one head is 0.40 +- 0.53 under it (4 of 5 seeds), four heads 0.23 +- 0.43 over.
- The best model, d64-4-head-2-layer at 97.12% +- 0.25%, is only 0.13 +- 0.38 points over the
  multi-head study's best (4-head-2-layer at d = 32, 96.98%) with 3.7 times its parameters (71946
  against 19594), and still 0.9 points under the conv network (98.00%).

So on MNIST at 5 epochs, neither more tokens nor wider ones close the gap to conv: patch 7 and
d = 32 stay the better trade, the low-rank bound stays unresolved, and the one new effect is that
heads help once there are more tokens or wider ones to split.

| arm | rate | parameters | epoch 1 | epoch 3 | epoch 5 | seconds per epoch |
|---|---|---|---|---|---|---|
| p4-1-head | 0.001 | 11050 | 86.86% +- 0.77% | 93.65% +- 0.47% | 94.86% +- 0.72% | 25.0 |
| p4-4-head | 0.004 | 11050 | 91.95% +- 0.87% | 94.83% +- 0.37% | 95.44% +- 0.34% | 40.6 |
| p4-4-head-k32 | 0.002 | 23626 | 91.43% +- 0.90% | 94.87% +- 0.75% | 95.86% +- 0.59% | 61.3 |
| p4-ffn | 0.002 | 6762 | 91.29% +- 0.34% | 93.76% +- 0.48% | 94.81% +- 0.31% | 11.0 |
| d64-1-head | 0.001 | 38474 | 91.74% +- 0.74% | 95.35% +- 0.65% | 96.01% +- 0.53% | 16.4 |
| d64-4-head | 0.001 | 38474 | 93.30% +- 0.86% | 95.79% +- 0.16% | 96.64% +- 0.32% | 17.5 |
| d64-4-head-2-layer | 0.0005 | 71946 | 94.42% +- 0.42% | 96.72% +- 0.14% | 97.12% +- 0.25% | 31.4 |
| d64-ffn | 0.004 | 21706 | 93.74% +- 0.79% | 95.33% +- 1.10% | 96.41% +- 0.47% | 8.5 |

Test accuracy, mean +- standard deviation over the 5 seeds; a difference "+- x" is the mean and
standard deviation of the per-seed differences. Seconds per epoch are the mean over the seeds and
epochs, 4 jobs at once: patch 4 costs 2.3 to 4.2 times patch 7's per epoch. Raw outputs in
data/attention/patch-geometry-{tune,sweep}.{json,txt} (untracked).
"""

from __future__ import annotations

from indrajala_ml.model.specs.layer_specs import Attention, LayerSpec
from indrajala_ml.studies import patch_study

# geometry: (patch size, d, FFN width)
GEOMETRIES: dict[str, tuple[int, int, int]] = {"p4": (4, 32, 64), "d64": (7, 64, 128)}
# arm: (its geometry, its attention layer, its number of attention-then-FFN layers); None for the
# FFN block alone
ATTENTION_ARMS: dict[str, tuple[str, Attention | None, int]] = {
    "p4-1-head": ("p4", Attention(), 1),
    "p4-4-head": ("p4", Attention(heads=4), 1),
    "p4-4-head-k32": ("p4", Attention(heads=4, key_size=32), 1),
    "p4-ffn": ("p4", None, 1),
    "d64-1-head": ("d64", Attention(), 1),
    "d64-4-head": ("d64", Attention(heads=4), 1),
    "d64-4-head-2-layer": ("d64", Attention(heads=4), 2),
    "d64-ffn": ("d64", None, 1),
}
ARMS = list(ATTENTION_ARMS)
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 5


def arm_specs(arm: str) -> list[LayerSpec]:
    """The layers of one arm of the study."""
    if arm not in ATTENTION_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    geometry, attention, layers = ATTENTION_ARMS[arm]
    patch_size, d, ffn_size = GEOMETRIES[geometry]
    ffn = patch_study.ffn_block(d, ffn_size)
    blocks = [ffn] if attention is None else [patch_study.attention_block(attention), ffn] * layers
    return patch_study.patch_model(blocks, patch_size, d)


def main(argv: list[str] | None = None) -> None:
    patch_study.main(argv, __doc__, ARMS, arm_specs, len(SEEDS), EPOCHS, time_arm="p4-4-head-k32")


if __name__ == "__main__":
    main()
