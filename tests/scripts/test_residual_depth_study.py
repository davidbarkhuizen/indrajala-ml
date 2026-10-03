"""The residual-connections depth study's networks and its gradient probe (scripts/residual_depth_study.py)."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

from indrajala_ml.model.protocols.classifier_protocols import Example
from indrajala_ml.model.specs.layer_specs import BatchNorm, Dense, Residual, expand_specs
from indrajala_ml.model.specs.spec_validation import validate_layer_specs
from indrajala_ml.studies import batch_size_scaling as bss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import residual_depth_study as study  # scripts/ isn't a package


@pytest.mark.parametrize("batch_norm", [False, True], ids=["plain", "batch norm"])
@pytest.mark.parametrize("depth", [2, 4, 8])
def test_a_plain_and_a_residual_network_of_one_depth_have_the_same_weighted_layers(depth: int, batch_norm: bool):
    def weighted(arm: str) -> list[Dense]:
        specs = study.depth_specs(arm, depth, batch_norm)
        validate_layer_specs(specs)
        return [spec for spec in expand_specs(specs) if isinstance(spec, Dense)]

    plain, residual = weighted("plain"), weighted("residual")
    assert len(plain) == len(residual) == depth + 2
    blocks = [spec for spec in study.depth_specs("residual", depth, batch_norm) if isinstance(spec, Residual)]
    assert len(blocks) == depth // 2
    assert all(any(isinstance(spec, BatchNorm) for spec in block.body) == batch_norm for block in blocks)


def test_an_odd_depth_is_refused():
    with pytest.raises(AssertionError, match="whole number of blocks"):
        study.depth_specs("residual", 3, False)


@pytest.mark.parametrize("arm", study.ARMS)
def test_the_gradient_probe_leaves_the_network_as_it_was(arm: str):
    rng = random.Random(0)
    examples: list[Example[int]] = [
        (tuple(rng.random() for _ in range(bss.DIMENSION)), rng.randrange(bss.CLASS_COUNT)) for _ in range(8)
    ]
    network = study.initial_network(arm, 2, True, seed=0)
    before = [[np.array(array) for array in layer] for layer in network.snapshot()]

    norm = study.initial_gradient_norm(network, examples)

    assert norm > 0
    after = network.snapshot()
    assert all(
        np.array_equal(old, np.array(new))
        for old_layer, new_layer in zip(before, after)
        for old, new in zip(old_layer, new_layer)
    )
    assert study.initial_gradient_norm(network, examples) == norm
