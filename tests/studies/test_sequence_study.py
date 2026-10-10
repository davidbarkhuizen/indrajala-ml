"""The sequence task's study (indrajala_ml/studies/sequence_study.py, scripts/sequence_study.py)."""

import json
import math
import random
from pathlib import Path

import pytest

from indrajala_ml.model.specs.layer_specs import Attention, Embedding, Residual
from indrajala_ml.model.specs.spec_validation import validate_layer_specs
from indrajala_ml.studies import sequence_study as study
from indrajala_ml.studies.patch_study import parameter_count

# counted by hand at d = 64 over Tiny Shakespeare's 65 characters: the embedding (65 x 64), the
# positions (64 x 64), the last layer norm (128) and the output (64 x 65 + 65) are 12609; an FFN
# block is its layer norm and 64 -> 256 -> 64 (33216); an attention block is its layer norm and four
# 64 x 64 projections with biases, whatever its heads (16768)
PARAMETER_COUNTS = {
    "ffn": 12609 + 33216,
    "1-layer": 12609 + 16768 + 33216,
    "2-layer": 12609 + 2 * (16768 + 33216),
    "2-layer-unmasked": 12609 + 2 * (16768 + 33216),
}


@pytest.mark.parametrize("arm", study.TRAINED_ARMS)
def test_each_arm_is_a_valid_network_of_its_parameter_count(arm: str):
    specs = study.arm_specs(arm, 65)
    validate_layer_specs(specs)
    assert specs[0] == Embedding(65, study.TOKEN_SIZE)
    assert parameter_count(study.initial_network(specs, seed=0)) == PARAMETER_COUNTS[arm]


@pytest.mark.parametrize(
    ("arm", "attention_layers"),
    [("ffn", []), ("1-layer", [True]), ("2-layer", [True, True]), ("2-layer-unmasked", [False, False])],
)
def test_each_arm_alternates_attention_and_ffn_blocks_masked_but_the_leak(arm: str, attention_layers: list[bool]):
    blocks = [spec for spec in study.arm_specs(arm, 65) if isinstance(spec, Residual)]
    attention = [body for block in blocks for body in block.body if isinstance(body, Attention)]
    assert [layer.causal for layer in attention] == attention_layers
    assert all(layer.heads == study.HEADS for layer in attention)
    expected = [study.ffn_block()] if not attention_layers else []
    for causal in attention_layers:
        expected += [study.attention_block(causal), study.ffn_block()]
    assert blocks == expected


# windows over a vocabulary of 3: training pairs (input -> label) 0->1 x3, 1->0 x2, 1->2 x1, 2->0 x2
TRAIN = [((0.0, 1.0, 0.0, 1.0), (1, 0, 1, 2)), ((2.0, 0.0, 1.0, 2.0), (0, 1, 0, 0))]
HELD_OUT = [((0.0, 1.0), (1, 2)), ((2.0, 0.0), (0, 2))]


def test_the_unigram_floor_counts_the_training_labels_with_add_one():
    # labels 0: 4, 1: 3, 2: 1, plus one each over 8 + 3
    p = [5 / 11, 4 / 11, 2 / 11]
    evaluation = study.counted("unigram", TRAIN, HELD_OUT, 3)
    assert evaluation.cross_entropy == pytest.approx(-(math.log(p[1]) + math.log(p[2]) * 2 + math.log(p[0])) / 4)
    assert evaluation.accuracy == 0.25  # it always says 0
    assert evaluation.tokens == 4


def test_the_bigram_floor_counts_each_label_after_its_input_with_add_one():
    # after 0: labels (0, 1, 2) counted (0, 3, 0), so (1, 4, 1) / 6; after 1: (2, 0, 1) -> (3, 1, 2) / 6;
    # after 2: (2, 0, 0) -> (3, 1, 1) / 5
    held_out = [math.log(4 / 6), math.log(2 / 6), math.log(3 / 5), math.log(1 / 6)]
    evaluation = study.counted("bigram", TRAIN, HELD_OUT, 3)
    assert evaluation.cross_entropy == pytest.approx(-sum(held_out) / 4)
    # argmax after 0 is 1 (right), after 1 is 0 (wrong: 2), after 2 is 0 (right), after 0 is 1 (wrong: 2)
    assert evaluation.accuracy == 0.5


def test_a_run_records_every_epoch_and_repeats_from_its_seed():
    rng = random.Random(0)

    def window() -> tuple[tuple[float, ...], tuple[int, ...]]:
        ids = [rng.randrange(5) for _ in range(study.CONTEXT + 1)]
        return tuple(float(i) for i in ids[:-1]), tuple(ids[1:])

    study._datasets.clear()  # pyright: ignore[reportPrivateUsage]
    study._datasets[("probe", 1, "contiguous")] = ([window() for _ in range(40)], [window() for _ in range(8)], 5)  # pyright: ignore[reportPrivateUsage]
    context = {"limit": 1, "epochs": 2}

    first = study.run_config(context, ("probe", "1-layer", 0.001), 0)
    second = study.run_config(context, ("probe", "1-layer", 0.001), 0)

    for key in ("held_out_cross_entropies", "held_out_accuracies", "train_cross_entropies", "epoch_seconds"):
        assert len(first[key]) == 2, key
    assert first["held_out_cross_entropies"] == second["held_out_cross_entropies"]
    assert first["train_cross_entropies"] == second["train_cross_entropies"]


def test_best_rates_take_the_lowest_final_held_out_loss_of_the_finite_rates(tmp_path: Path):
    def runs(*finals: float) -> list[dict[str, list[float]]]:
        return [{"held_out_cross_entropies": [9.0, final]} for final in finals]

    results = {
        ("a", "ffn", 0.001): runs(2.0, 2.2),
        ("a", "ffn", 0.002): runs(1.9, 2.1),
        ("a", "ffn", 0.004): runs(0.1, math.nan),
        ("b", "ffn", 0.001): runs(3.0, 3.0),
    }
    path = tmp_path / "tune.json"
    path.write_text(json.dumps([{"config": list(config), "runs": value} for config, value in results.items()]))

    read = study.read_results(str(path))

    assert study.best_rates(read) == {("a", "ffn"): 0.002, ("b", "ffn"): 0.001}
