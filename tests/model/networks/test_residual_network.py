"""
Residual blocks on each implementation, numpy, Rust and pure Python (README, Residual connections):
the gradient check on residual networks, a block's layers wired together, and learn against a batch
of one. What one implementation alone has (randomize's draws, identity blocks, the fork's input left
as it was, the layer-major path, the parity tests) stays in
tests/model/networks/test_residual_{array,rust,python}_network.py; this module reuses their cases.
"""

from typing import Any

import pytest

from indrajala_ml.model.layers.array.array_layer_builder import LAYER_CLASSES
from indrajala_ml.model.layers.python.residual_layer import AddLayer, AffineLayer, ForkLayer
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from tests.gradient_check import check_gradients
from tests.helpers import Implementation, assert_learn_and_a_batch_of_one_agree, randomized, split
from tests.model.networks.test_residual_array_network import INPUT, NETWORKS, Shape, output, rows


def network(
    implementation: Implementation, name: str, shape: Shape = "multiclass", rule: UpdateRule | None = None
) -> Any:
    return randomized(implementation, (INPUT,), [*NETWORKS[name], output(shape)], rule, shape=shape)


@pytest.mark.parametrize("shape", ["multiclass", "single_output"])
@pytest.mark.parametrize("name", NETWORKS)
def test_every_gradient_matches_its_finite_difference(name: str, shape: Shape, implementation: Implementation):
    built = network(implementation, name, shape)
    # a trained step first, so the affine layers and batch norm aren't at their initial values
    built.learn_batch(0.5, rows(6, shape, seed=2))

    check_gradients(built, *split(rows(5, shape)))


def test_a_block_builds_a_fork_its_body_and_an_add_wired_together(implementation: Implementation):
    built = network(implementation, "sigmoid body")

    if implementation == "python":
        dense, fork, body, end, add, out = built.trainable_layers
        assert isinstance(fork, ForkLayer) and isinstance(add, AddLayer) and isinstance(end, AffineLayer)
        assert fork.body_first is body and fork.add is add and add.fork is fork
        assert (
            fork.input_layer is dense and body.input_layer is fork and add.input_layer is end and out.input_layer is add
        )
        assert [node.input_node for node in fork.nodes] == list(dense.nodes)
        assert [(node.body_node, node.fork_node) for node in add.nodes] == list(zip(end.nodes, fork.nodes, strict=True))
        assert fork.snapshot_state() == [] and add.snapshot_state() == [] and fork.weight_sets() == []
    else:
        dense, fork, body, end, add, out = built.layers
        classes = LAYER_CLASSES[implementation]
        assert (type(fork), type(add), type(end)) == (classes.fork, classes.add, classes.affine)
        assert fork.body_first is body and fork.add is add and add.fork is fork
        assert (fork.size, add.size, end.W.shape, end.b.shape) == (INPUT, INPUT, (INPUT, 5), (INPUT,))
        # the fork and add hold nothing: their snapshot entries are empty
        assert [len(entry) for entry in built.snapshot()] == [2, 0, 2, 2, 0, 2]
        assert dense.W.shape == (INPUT, INPUT) and out.W.shape == (3, INPUT)


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["sigmoid body", "relu body", "two blocks in a row", "block first"])
def test_learn_and_a_learn_batch_of_one_example_agree(name: str, rule: UpdateRule, implementation: Implementation):
    single, batched = network(implementation, name, rule=rule), network(implementation, name, rule=rule)

    assert_learn_and_a_batch_of_one_agree(implementation, single, batched, rows(6, seed=4))
