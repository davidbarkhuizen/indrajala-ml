"""The multi-head attention study's networks (scripts/multi_head_attention_study.py)."""

import random
import sys
from pathlib import Path

import pytest

from indrajala_ml.model.protocols.classifier_protocols import Example
from indrajala_ml.model.specs.layer_specs import Attention, Residual
from indrajala_ml.model.specs.spec_validation import validate_layer_specs
from indrajala_ml.studies import batch_size_scaling as bss
from indrajala_ml.studies import patch_study

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import multi_head_attention_study as study  # scripts/ isn't a package
import patch_attention_study as step_4  # scripts/ isn't a package

# counted by hand: the patch-attention study's ffn arm is 6762 and its attention-ffn 11050, so an
# attention block at the defaults is 4288: its four 32 x 32 projections with biases (4224) and its
# layer norm (64), whatever its heads; at key_size 32 and 4 heads, Wq, Wk and Wv are 128 x 32 and
# Wo 32 x 128 (16384) with biases 3 * 128 + 32, 16800, and the block 16864; a second layer adds an
# attention block and an FFN block (32 -> 64 -> 32 and its layer norm, 4256)
PARAMETER_COUNTS = {
    "1-head": 11050,
    "2-head": 11050,
    "4-head": 11050,
    "4-head-k32": 11050 - 4224 + 16800,
    "1-head-2-layer": 11050 + 4288 + 4256,
    "4-head-2-layer": 11050 + 4288 + 4256,
    "ffn": 6762,
}


@pytest.mark.parametrize("arm", study.ARMS)
def test_each_arm_is_a_valid_network_of_its_parameter_count(arm: str):
    validate_layer_specs(study.arm_specs(arm))
    assert (
        patch_study.parameter_count(patch_study.initial_network(study.arm_specs(arm), seed=0)) == PARAMETER_COUNTS[arm]
    )


def test_the_anchor_and_the_control_are_the_patch_attention_studys_arms():
    assert study.arm_specs("1-head") == step_4.arm_specs("attention-ffn")
    assert study.arm_specs("ffn") == step_4.arm_specs("ffn")


@pytest.mark.parametrize(
    ("arm", "attention", "layers"),
    [
        ("1-head", Attention(), 1),
        ("2-head", Attention(heads=2), 1),
        ("4-head", Attention(heads=4), 1),
        ("4-head-k32", Attention(heads=4, key_size=32), 1),
        ("1-head-2-layer", Attention(), 2),
        ("4-head-2-layer", Attention(heads=4), 2),
        ("ffn", None, 0),
    ],
)
def test_each_arm_alternates_its_attention_and_ffn_blocks(arm: str, attention: Attention | None, layers: int):
    blocks = [spec for spec in study.arm_specs(arm) if isinstance(spec, Residual)]
    if attention is None:
        assert blocks == [patch_study.ffn_block()]
    else:
        assert blocks == [patch_study.attention_block(attention), patch_study.ffn_block()] * layers


def test_a_run_of_a_multi_head_arm_records_every_epoch_and_repeats_from_its_seed():
    rng = random.Random(0)
    examples: list[Example[int]] = [
        (tuple(rng.random() for _ in range(bss.DIMENSION)), rng.randrange(bss.CLASS_COUNT)) for _ in range(40)
    ]
    patch_study._datasets[("train", "test", 1)] = (examples, examples[:8])  # pyright: ignore[reportPrivateUsage]
    context = {"train_path": "train", "test_path": "test", "limit": 1, "epochs": 2, "arm_specs": study.arm_specs}

    first = patch_study.run_config(context, ("4-head-2-layer", 0.001), 0)
    second = patch_study.run_config(context, ("4-head-2-layer", 0.001), 0)

    assert len(first["test_accuracies"]) == len(first["epoch_seconds"]) == 2
    assert first["parameter_count"] == PARAMETER_COUNTS["4-head-2-layer"]
    assert first["test_accuracies"] == second["test_accuracies"]
