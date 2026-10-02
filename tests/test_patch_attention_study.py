"""The patch-attention study's networks (scripts/patch_attention_study.py)."""

import random
import sys
from pathlib import Path

import pytest

from indrajala_ml import batch_size_scaling as bss
from indrajala_ml.model.classifier_protocols import Example
from indrajala_ml.model.layer_specs import Attention, Residual, validate_layer_specs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import patch_attention_study as study  # scripts/ isn't a package

# counted by hand: the embedding 49 * 32 + 32, positions 16 * 32, layer norms 2 * 32 each, an
# attention block's four 32 x 32 projections with biases, an FFN block's 32 -> 64 -> 32, the output
# 32 * 10 + 10; the conv's 3 * 3 * 8 + 8, 26 * 26 * 8 * 32 + 32 and the output; the dense
# control's 784 * 14 + 14 and 14 * 10 + 10
PARAMETER_COUNTS = {"ffn": 6762, "attention": 6794, "attention-ffn": 11050, "conv": 173498, "dense": 11140}


@pytest.mark.parametrize("arm", study.ARMS)
def test_each_arm_is_a_valid_network_of_its_parameter_count(arm: str):
    validate_layer_specs(study.arm_specs(arm))
    assert study.parameter_count(study.initial_network(arm, seed=0)) == PARAMETER_COUNTS[arm]


def test_the_dense_control_has_about_attention_ffns_parameter_count():
    assert abs(PARAMETER_COUNTS["dense"] - PARAMETER_COUNTS["attention-ffn"]) < 795  # one hidden unit


@pytest.mark.parametrize(("arm", "blocks"), [("ffn", 0), ("attention", 1), ("attention-ffn", 1)])
def test_only_the_attention_arms_have_an_attention_block(arm: str, blocks: int):
    residuals = [spec for spec in study.arm_specs(arm) if isinstance(spec, Residual)]
    assert sum(any(isinstance(layer, Attention) for layer in block.body) for block in residuals) == blocks


def test_a_run_records_every_epoch_and_repeats_from_its_seed():
    rng = random.Random(0)
    examples: list[Example[int]] = [
        (tuple(rng.random() for _ in range(bss.DIMENSION)), rng.randrange(bss.CLASS_COUNT)) for _ in range(40)
    ]
    study._datasets[("train", "test", 1)] = (examples, examples[:8])
    context = {"train_path": "train", "test_path": "test", "limit": 1, "epochs": 2}

    first = study.run_config(context, ("attention-ffn", 0.001), 0)
    second = study.run_config(context, ("attention-ffn", 0.001), 0)

    assert len(first["test_accuracies"]) == len(first["epoch_seconds"]) == 2
    assert first["parameter_count"] == PARAMETER_COUNTS["attention-ffn"]
    assert first["test_accuracies"] == second["test_accuracies"]
