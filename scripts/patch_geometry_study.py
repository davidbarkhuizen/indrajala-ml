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
