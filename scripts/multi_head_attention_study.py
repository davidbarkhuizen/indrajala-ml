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
