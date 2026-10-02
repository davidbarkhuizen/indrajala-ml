"""
Patch models on each implementation, numpy, Rust and pure Python (README, Layer norm and
attention): the gradient check on every accepted patch model, the layers the builder wires for the
README's model, and learn against a batch of one. The layers, and what one implementation alone
has (the exact tests, randomize's draws, the parity tests), stay in
tests/test_attention_{array,rust,python}_network.py; this module reuses their cases.
"""

from typing import Any

import pytest

from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.layers.numpy.layer_norm_array_layer import LayerNormArrayLayer
from indrajala_ml.model.layers.numpy.residual_array_layer import AddArrayLayer, ForkArrayLayer
from indrajala_ml.model.layers.numpy.token_array_layer import (
    PatchesArrayLayer,
    PositionArrayLayer,
    TokenDenseArrayLayer,
    TokenMeanArrayLayer,
)
from indrajala_ml.model.layers.python.attention_layer import AttentionLayer
from indrajala_ml.model.layers.python.layer_norm_layer import LayerNormLayer
from indrajala_ml.model.layers.python.residual_layer import AddLayer, ForkLayer
from indrajala_ml.model.layers.python.token_layer import PatchesLayer, PositionLayer, TokenDenseLayer, TokenMeanLayer
from indrajala_ml.model.layers.rust.attention_rust_array_layer import AttentionRustArrayLayer
from indrajala_ml.model.layers.rust.layer_norm_rust_array_layer import LayerNormRustArrayLayer
from indrajala_ml.model.layers.rust.residual_rust_array_layer import AddRustArrayLayer, ForkRustArrayLayer
from indrajala_ml.model.layers.rust.token_rust_array_layer import (
    PatchesRustArrayLayer,
    PositionRustArrayLayer,
    TokenDenseRustArrayLayer,
    TokenMeanRustArrayLayer,
)
from indrajala_ml.model.specs.layer_specs import LayerSpec
from indrajala_ml.model.specs.single_example import batch_norm_index
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from tests.gradient_check import check_gradients
from tests.helpers import Implementation, bits, randomized, split
from tests.model.specs.test_layer_specs import TOKENS
from tests.test_attention_array_network import IMAGE, rows


def network(implementation: Implementation, specs: list[LayerSpec], rule: UpdateRule | None = None) -> Any:
    return randomized(implementation, IMAGE, specs, rule)


def layers(implementation: Implementation, built: Any) -> list[Any]:
    # the pure-Python network's layers, the output layer last, are its trainable_layers
    return built.trainable_layers if implementation == "python" else built.layers


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", TOKENS)
def test_every_gradient_matches_its_finite_difference(name: str, batch_size: int, implementation: Implementation):
    if batch_size == 1 and batch_norm_index(TOKENS[name]) is not None:
        pytest.skip("batch norm trains on batches only (the batch-norm workplan, D4)")
    built = network(implementation, TOKENS[name])
    # a trained step first, so gamma, beta and the positions aren't at their initial values
    built.learn_batch(0.5, rows(4, seed=2))

    check_gradients(built, *split(rows(batch_size)))


# the README's model's layers on each implementation, the output layer apart
README_LAYERS: dict[str, list[type]] = {
    "numpy": [
        PatchesArrayLayer,
        TokenDenseArrayLayer,
        PositionArrayLayer,
        ForkArrayLayer,
        LayerNormArrayLayer,
        AttentionArrayLayer,
        AddArrayLayer,
        ForkArrayLayer,
        LayerNormArrayLayer,
        TokenDenseArrayLayer,
        TokenDenseArrayLayer,
        AddArrayLayer,
        TokenMeanArrayLayer,
        LayerNormArrayLayer,
    ],
    "rust": [
        PatchesRustArrayLayer,
        TokenDenseRustArrayLayer,
        PositionRustArrayLayer,
        ForkRustArrayLayer,
        LayerNormRustArrayLayer,
        AttentionRustArrayLayer,
        AddRustArrayLayer,
        ForkRustArrayLayer,
        LayerNormRustArrayLayer,
        TokenDenseRustArrayLayer,
        TokenDenseRustArrayLayer,
        AddRustArrayLayer,
        TokenMeanRustArrayLayer,
        LayerNormRustArrayLayer,
    ],
    "python": [
        PatchesLayer,
        TokenDenseLayer,
        PositionLayer,
        ForkLayer,
        LayerNormLayer,
        AttentionLayer,
        AddLayer,
        ForkLayer,
        LayerNormLayer,
        TokenDenseLayer,
        TokenDenseLayer,
        AddLayer,
        TokenMeanLayer,
        LayerNormLayer,
    ],
}

# per layer: patches 0, W and b, P, fork 0, gamma and beta, attention's 8, add 0, ...; pure Python's
# are its weight sets: embedding units, position rows, features, attention's 4 * 6 rows, ...
README_SNAPSHOT_LENGTHS = {
    "numpy": [0, 2, 1, 0, 2, 8, 0, 0, 2, 2, 2, 0, 0, 2, 2],
    "rust": [0, 2, 1, 0, 2, 8, 0, 0, 2, 2, 2, 0, 0, 2, 2],
    "python": [0, 6, 4, 0, 6, 24, 0, 0, 6, 8, 6, 0, 0, 6, 3],
}


def test_the_readme_model_builds_its_layers_wired_together(implementation: Implementation):
    built = network(implementation, TOKENS["the README's model"])
    wired = layers(implementation, built)
    assert [type(layer) for layer in wired] == [*README_LAYERS[implementation], type(wired[-1])]
    patches, embed, position, fork, norm, attention, add = wired[:7]
    assert fork.body_first is norm and fork.add is add
    if implementation == "python":
        assert attention.input_layer is norm
        assert (patches.size, len(embed.units), embed.input_size, len(position.rows)) == (16, 6, 4, 4)
    else:
        assert (patches.size, embed.W.shape, position.P.shape) == (16, (6, 4), (4, 6))
    assert (norm.tokens, norm.features, attention.tokens, attention.features) == (4, 6, 4, 6)
    final = wired[13]
    assert (final.tokens, final.features) == (1, 6)
    assert [len(entry) for entry in built.snapshot()] == README_SNAPSHOT_LENGTHS[implementation]


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["the README's model", "token-wise layers after a block"])
def test_learn_and_a_learn_batch_of_one_example_agree(name: str, rule: UpdateRule, implementation: Implementation):
    single, batched = network(implementation, TOKENS[name], rule), network(implementation, TOKENS[name], rule)
    for state, label in rows(6, seed=4):
        single.learn(0.5, state, label)
        batched.learn_batch(0.5, [(state, label)])

    if implementation == "numpy":
        # the token layers' single-example passes are their batch passes on a batch of one, but a
        # flat dense layer's W @ x and X @ W.T are different BLAS calls, which can differ in the last
        # bit (test_residual_array_network)
        for one, other in zip(single.snapshot(), batched.snapshot(), strict=True):
            for a, b in zip(one, other, strict=True):
                assert a == pytest.approx(b, rel=1e-12, abs=1e-15)
    else:
        # by bits: pure Python's single-example step is its batch of one's, and so are each Rust
        # token layer's and layer norm's single-example ops and the crate's dense ops (W @ x against
        # X @ W.T, outer against delta^T X at one row)
        assert bits(single.snapshot()) == bits(batched.snapshot())
