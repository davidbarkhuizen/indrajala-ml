"""The patch geometry study's networks (scripts/patch_geometry_study.py)."""

import random
import sys
from pathlib import Path

import pytest

from indrajala_ml.model.protocols.classifier_protocols import Example
from indrajala_ml.model.specs.layer_specs import Attention, Patches
from indrajala_ml.model.specs.spec_shapes import spec_shapes
from indrajala_ml.model.specs.spec_validation import validate_layer_specs
from indrajala_ml.studies import batch_size_scaling as bss
from indrajala_ml.studies import common, patch_study

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import multi_head_attention_study as multi_head  # scripts/ isn't a package
import patch_geometry_study as study  # scripts/ isn't a package

# counted by hand. Patch 4: the embedding 16 -> 32 (544) and 49 positions of 32 (1568) are 2112, as
# patch 7's 49 -> 32 (1600) and 16 positions (512), so the p4 arms have the multi-head study's
# counts. d = 64: the embedding 49 -> 64 (3200), 16 positions (1024), the final layer norm (128)
# and the output 64 -> 10 (650) are 5002; an FFN block 64 -> 128 -> 64 and its layer norm 16704;
# an attention block's four 64 x 64 projections with biases (16640) and its layer norm, 16768,
# whatever its heads
PARAMETER_COUNTS = {
    "p4-1-head": 11050,
    "p4-4-head": 11050,
    "p4-4-head-k32": 23626,
    "p4-ffn": 6762,
    "d64-1-head": 5002 + 16768 + 16704,
    "d64-4-head": 5002 + 16768 + 16704,
    "d64-4-head-2-layer": 5002 + 2 * (16768 + 16704),
    "d64-ffn": 5002 + 16704,
}


@pytest.mark.parametrize("arm", study.ARMS)
def test_each_arm_is_a_valid_network_of_its_parameter_count(arm: str):
    validate_layer_specs(study.arm_specs(arm))
    assert (
        patch_study.parameter_count(patch_study.initial_network(study.arm_specs(arm), seed=0)) == PARAMETER_COUNTS[arm]
    )


@pytest.mark.parametrize(("geometry", "tokens"), [("p4", (49, 32)), ("d64", (16, 64))])
def test_each_geometry_gives_its_tokens(geometry: str, tokens: tuple[int, int]):
    for arm in (arm for arm in study.ARMS if arm.startswith(geometry)):
        assert spec_shapes(study.arm_specs(arm), patch_study.INPUT_SHAPE)[1].output_shape == tokens


def test_the_patch_4_arms_are_the_multi_head_studys_but_for_the_patch_size():
    for arm, cited in [
        ("p4-1-head", "1-head"),
        ("p4-4-head", "4-head"),
        ("p4-4-head-k32", "4-head-k32"),
        ("p4-ffn", "ffn"),
    ]:
        specs, cited_specs = study.arm_specs(arm), multi_head.arm_specs(cited)
        assert (specs[0], cited_specs[0]) == (Patches(4), Patches(7))
        assert specs[1:] == cited_specs[1:]


def test_the_d_64_arms_are_the_multi_head_studys_layers_twice_as_wide():
    ffn = patch_study.ffn_block(64, 128)
    attention = patch_study.attention_block(Attention(heads=4))
    assert study.arm_specs("d64-1-head") == patch_study.patch_model([patch_study.attention_block(), ffn], 7, 64)
    assert study.arm_specs("d64-4-head") == patch_study.patch_model([attention, ffn], 7, 64)
    assert study.arm_specs("d64-4-head-2-layer") == patch_study.patch_model([attention, ffn] * 2, 7, 64)
    assert study.arm_specs("d64-ffn") == patch_study.patch_model([ffn], 7, 64)


def test_a_run_of_a_patch_4_arm_records_every_epoch_and_repeats_from_its_seed():
    rng = random.Random(0)
    examples: list[Example[int]] = [
        (tuple(rng.random() for _ in range(bss.DIMENSION)), rng.randrange(bss.CLASS_COUNT)) for _ in range(40)
    ]
    common._mnist_datasets[("train", "test", 1)] = (examples, examples[:8])  # pyright: ignore[reportPrivateUsage]
    context = {"train_path": "train", "test_path": "test", "limit": 1, "epochs": 2, "arm_specs": study.arm_specs}

    first = patch_study.run_config(context, ("p4-4-head-k32", 0.001), 0)
    second = patch_study.run_config(context, ("p4-4-head-k32", 0.001), 0)

    assert len(first["test_accuracies"]) == len(first["epoch_seconds"]) == 2
    assert first["parameter_count"] == PARAMETER_COUNTS["p4-4-head-k32"]
    assert first["test_accuracies"] == second["test_accuracies"]
