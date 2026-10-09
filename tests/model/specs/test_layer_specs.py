"""
Layer specs (layer_specs.py) and their numpy and Rust builder (array_layer_builder.py): which spec
lists are accepted, the layer class each spec kind maps to, and the shapes the builder chains.
"""

import json
import math
import re
from typing import Any, cast

import pytest

from indrajala_ml.model.layers.array.array_layer_builder import LAYER_CLASSES, build_array_layers
from indrajala_ml.model.layers.python.python_layer_builder import build_python_layers
from indrajala_ml.model.layers.python.state_layer import StateLayer
from indrajala_ml.model.persistence.format2_json import layer_from_json, layer_to_json
from indrajala_ml.model.specs.layer_specs import (
    Add,
    Attention,
    BatchNorm,
    Conv,
    Dense,
    Embedding,
    Fork,
    LayerNorm,
    LayerSpec,
    Patches,
    Pool,
    Position,
    Residual,
    TokenMean,
    expand_specs,
    spec_paths,
    token_wise_output,
)
from indrajala_ml.model.specs.single_example import (
    batch_norm_index,
    refuse_single_example_groups,
    refuse_single_example_network,
)
from indrajala_ml.model.specs.spec_shapes import InputShape, SpecShape, spec_shapes
from indrajala_ml.model.specs.spec_validation import validate_layer_specs
from tests.helpers import Backend

OUTPUT = Dense(3, output=True)
LINEAR = Dense(5, activation="linear")
LINEAR_CONV = Conv(3, 2, activation="linear")

VALID: dict[str, list[LayerSpec]] = {
    "output only": [OUTPUT],
    "sigmoid": [Dense(5), Dense(4), OUTPUT],
    "relu": [Dense(5, activation="relu"), OUTPUT],
    "dropout": [Dense(5, dropout=0.3), OUTPUT],
    "softmax output": [Dense(5), Dense(3, output=True, activation="softmax", loss="cross_entropy")],
    "cross-entropy output": [Dense(5), Dense(1, output=True, loss="cross_entropy")],
    "conv": [Conv(3, 2), Dense(5), OUTPUT],
    "conv pool conv": [Conv(3, 2), Pool(2), Conv(2, 3), OUTPUT],
    "pool first": [Pool(2), Conv(2, 2), OUTPUT],
    "strided conv conv": [Conv(3, 2, stride=2), Conv(2, 2), OUTPUT],
}

INVALID: dict[str, list[LayerSpec]] = {
    "empty": [],
    "no output layer": [Dense(5), Dense(3)],
    "output not last": [OUTPUT, Dense(3)],
    "two output layers": [Dense(4, output=True), OUTPUT],
    "conv after dense": [Dense(5), Conv(3, 2), OUTPUT],
    "pool after dense": [Conv(3, 2), Dense(5), Pool(2), OUTPUT],
    "pool without conv": [Pool(2), OUTPUT],
    "conv last": [Dense(5), Conv(3, 2)],
    "softmax hidden": [Dense(5, activation="softmax"), OUTPUT],
    "dropout after relu": [Dense(5, activation="relu", dropout=0.3), OUTPUT],
    "hidden loss": [Dense(5, loss="cross_entropy"), OUTPUT],
    "empty hidden": [Dense(0), OUTPUT],
    "empty output": [Dense(0, output=True)],
    "dropout output": [Dense(3, output=True, dropout=0.3)],
    "relu output": [Dense(3, output=True, activation="relu")],
    "softmax squared": [Dense(3, output=True, activation="softmax")],
    "linear without batch norm": [LINEAR, OUTPUT],
    "linear before sigmoid": [LINEAR, Dense(4), OUTPUT],
    "batch norm first": [BatchNorm(), OUTPUT],
    "batch norm after sigmoid": [Dense(5), BatchNorm(), OUTPUT],
    "batch norm after relu": [Dense(5, activation="relu"), BatchNorm(), OUTPUT],
    "batch norm after conv": [Conv(3, 2), BatchNorm("relu"), OUTPUT],
    "batch norm twice": [LINEAR, BatchNorm(), BatchNorm(), OUTPUT],
    "batch norm last": [LINEAR, BatchNorm()],
    "softmax batch norm": [LINEAR, BatchNorm(cast(Any, "softmax")), OUTPUT],
    "linear batch norm": [LINEAR, BatchNorm(cast(Any, "linear")), OUTPUT],
    "linear dropout": [Dense(5, activation="linear", dropout=0.3), BatchNorm(), OUTPUT],
    "linear output": [Dense(3, output=True, activation="linear")],
    "zero epsilon": [LINEAR, BatchNorm(epsilon=0.0), OUTPUT],
    "zero running rate": [LINEAR, BatchNorm(running_rate=0.0), OUTPUT],
    "running rate over 1": [LINEAR, BatchNorm(running_rate=1.5), OUTPUT],
    "linear conv without batch norm": [LINEAR_CONV, Dense(5), OUTPUT],
    "linear conv before pool": [LINEAR_CONV, Pool(2), BatchNorm("relu"), OUTPUT],
    "linear conv last in the front end": [Conv(3, 2), LINEAR_CONV, OUTPUT],
    "sigmoid batch norm after conv": [LINEAR_CONV, BatchNorm(), OUTPUT],
    "batch norm after pool": [LINEAR_CONV, BatchNorm("relu"), Pool(2), BatchNorm("relu"), OUTPUT],
    "conv batch norm twice": [LINEAR_CONV, BatchNorm("relu"), BatchNorm("relu"), OUTPUT],
    "sigmoid conv": [Conv(3, 2, activation=cast(Any, "sigmoid")), OUTPUT],
    "conv batch norm zero epsilon": [LINEAR_CONV, BatchNorm("relu", epsilon=0.0), OUTPUT],
}

# batch norm's pairs (the batch-norm workplan, D1): accepted, and built by numpy (stage 1), pure
# Python (stage 2, tests/model/layers/python/test_python_layer_builder.py) and Rust (stage 3)
BATCH_NORM: dict[str, list[LayerSpec]] = {
    "sigmoid": [LINEAR, BatchNorm(), OUTPUT],
    "relu": [LINEAR, BatchNorm("relu"), OUTPUT],
    "two pairs": [LINEAR, BatchNorm(), Dense(4, activation="linear"), BatchNorm("relu"), OUTPUT],
    "between dense layers": [Dense(6), LINEAR, BatchNorm(), Dense(4, activation="relu"), OUTPUT],
    "after a front end": [Conv(3, 2), Pool(2), LINEAR, BatchNorm(), OUTPUT],
    "softmax output": [LINEAR, BatchNorm(), Dense(3, output=True, activation="softmax", loss="cross_entropy")],
    "rate of 1": [LINEAR, BatchNorm(running_rate=1.0), OUTPUT],
}

# conv batch norm's pairs: accepted, and built by numpy (stage 4a), pure Python (4b,
# tests/model/layers/python/test_python_layer_builder.py) and Rust (4c)
CONV_BATCH_NORM: dict[str, list[LayerSpec]] = {
    "conv": [LINEAR_CONV, BatchNorm("relu"), OUTPUT],
    "conv pool": [LINEAR_CONV, BatchNorm("relu"), Pool(2), OUTPUT],
    "after a relu conv": [Conv(2, 2), Conv(2, 3, stride=2, activation="linear"), BatchNorm("relu"), OUTPUT],
    "two conv pairs": [LINEAR_CONV, BatchNorm("relu"), Conv(2, 3, activation="linear"), BatchNorm("relu"), OUTPUT],
    "conv and dense pairs": [LINEAR_CONV, BatchNorm("relu"), LINEAR, BatchNorm(), OUTPUT],
}


@pytest.mark.parametrize("specs", VALID.values(), ids=VALID.keys())
def test_every_combination_all_implementations_build_is_accepted(specs: list[LayerSpec]):
    validate_layer_specs(specs)


@pytest.mark.parametrize("specs", INVALID.values(), ids=INVALID.keys())
def test_a_combination_some_implementation_cant_build_is_rejected(specs: list[LayerSpec], backend: Backend):
    with pytest.raises(AssertionError):
        validate_layer_specs(specs)
    with pytest.raises(AssertionError):
        build_array_layers(specs, (8, 8, 1), backend.name)


@pytest.mark.parametrize("specs", BATCH_NORM.values(), ids=BATCH_NORM.keys())
def test_batch_norm_pairs_are_accepted(specs: list[LayerSpec], backend: Backend):
    validate_layer_specs(specs)
    build_array_layers(specs, (8, 8, 1), backend.name)


@pytest.mark.parametrize("specs", CONV_BATCH_NORM.values(), ids=CONV_BATCH_NORM.keys())
def test_conv_batch_norm_pairs_are_accepted(specs: list[LayerSpec], backend: Backend):
    validate_layer_specs(specs)
    build_array_layers(specs, (8, 8, 1), backend.name)


def test_a_conv_batch_norm_layer_normalizes_each_channel_over_every_position(backend: Backend):
    linear, norm, _output = cast(
        "list[Any]", build_array_layers([LINEAR_CONV, BatchNorm("relu", 1e-3, 0.2), OUTPUT], (8, 8, 1), backend.name)
    )

    assert type(linear) is LAYER_CLASSES[backend.name].linear_conv
    assert type(norm) is LAYER_CLASSES[backend.name].batch_norm
    assert linear.W.shape == (2, 9) and linear.parameters() == (linear.W,)
    assert (norm.size, norm.positions, norm.gamma.shape) == (2 * 6 * 6, 6 * 6, (2,))
    assert (norm.activation, norm.epsilon, norm.running_rate) == ("relu", 1e-3, 0.2)


# each spec kind, as a network's only hidden layer (or its output layer), and the field of
# ArrayLayerClasses it builds
KINDS: list[tuple[str, list[LayerSpec], int, str]] = [
    ("sigmoid", [Dense(5), OUTPUT], 0, "sigmoid"),
    ("relu", [Dense(5, activation="relu"), OUTPUT], 0, "relu"),
    ("dropout", [Dense(5, dropout=0.3), OUTPUT], 0, "dropout"),
    ("sigmoid output", [OUTPUT], 0, "sigmoid"),
    ("cross-entropy output", [Dense(3, output=True, loss="cross_entropy")], 0, "cross_entropy"),
    ("softmax output", [Dense(3, output=True, activation="softmax", loss="cross_entropy")], 0, "softmax"),
    ("conv", [Conv(3, 2), OUTPUT], 0, "conv"),
    ("pool", [Conv(3, 2), Pool(2), OUTPUT], 1, "pool"),
    ("linear", BATCH_NORM["sigmoid"], 0, "linear"),
    ("batch norm", BATCH_NORM["sigmoid"], 1, "batch_norm"),
]


def test_a_batch_norm_layer_normalizes_the_linear_layers_features_with_its_specs_constants(backend: Backend):
    linear, norm, _output = cast(
        "list[Any]", build_array_layers([LINEAR, BatchNorm("relu", 1e-3, 0.2), OUTPUT], (7,), backend.name)
    )

    assert linear.W.shape == (5, 7) and linear.parameters() == (linear.W,)
    assert (norm.size, norm.activation, norm.epsilon, norm.running_rate) == (5, "relu", 1e-3, 0.2)


@pytest.mark.parametrize("specs, index, kind", [kind[1:] for kind in KINDS], ids=[kind[0] for kind in KINDS])
def test_each_spec_kind_builds_its_backends_layer_class(
    specs: list[LayerSpec], index: int, kind: str, backend: Backend
):
    layer = build_array_layers(specs, (8, 8, 1), backend.name)[index]
    assert type(layer) is getattr(LAYER_CLASSES[backend.name], kind)


def test_the_dropout_layer_takes_the_specs_probability(backend: Backend):
    layer: Any = build_array_layers([Dense(5, dropout=0.3), OUTPUT], (4,), backend.name)[0]
    assert layer._drop_probability == 0.3


def test_each_layer_reads_the_previous_layers_output_shape(backend: Backend):
    # 9x9x1 -> conv 3, stride 2: 4x4x2 -> pool 2: 2x2x2 -> conv 2: 1x1x3 -> dense 5 -> output 3
    layers: list[Any] = build_array_layers(
        [Conv(3, 2, stride=2), Pool(2), Conv(2, 3), Dense(5), OUTPUT], (9, 9, 1), backend.name
    )
    conv_1, pool, conv_2, dense, output = layers

    assert (conv_1.input_height, conv_1.input_width, conv_1.input_channels) == (9, 9, 1)
    assert (conv_1.out_height, conv_1.out_width, conv_1.channel_count) == (4, 4, 2)
    assert (pool.input_height, pool.input_width, pool.input_channels) == (4, 4, 2)
    assert (conv_2.input_height, conv_2.input_width, conv_2.input_channels) == (2, 2, 2)
    assert conv_2.W.shape == (3, 2 * 2 * 2)
    assert dense.W.shape == (5, 3)  # conv_2's 1x1x3 output, flattened
    assert output.W.shape == (3, 5)


def test_the_shape_walk_chains_each_specs_output_shape_into_the_next():
    # test_each_layer_reads_the_previous_layers_output_shape's list, with a conv batch-norm pair
    specs = [Conv(3, 2, stride=2), Pool(2), Conv(2, 3, activation="linear"), BatchNorm("relu"), Dense(5), OUTPUT]
    assert spec_shapes(specs, (9, 9, 1)) == [
        SpecShape((9, 9, 1), (4, 4, 2)),
        SpecShape((4, 4, 2), (2, 2, 2)),
        SpecShape((2, 2, 2), (1, 1, 3)),
        SpecShape((1, 1, 3), (1, 1, 3), positions=1),
        SpecShape((1, 1, 3), (5,)),
        SpecShape((5,), (3,)),
    ]
    assert spec_shapes([LINEAR_CONV, BatchNorm("relu"), OUTPUT], (8, 8, 1))[1].positions == 6 * 6


ACCEPTED = {
    f"{group} {name}": specs
    for group, lists in (("valid", VALID), ("batch norm", BATCH_NORM), ("conv batch norm", CONV_BATCH_NORM))
    for name, specs in lists.items()
}


@pytest.mark.parametrize("specs", ACCEPTED.values(), ids=ACCEPTED.keys())
def test_the_shape_walk_agrees_with_the_built_layers_own_geometry(specs: list[LayerSpec], backend: Backend):
    layers: list[Any] = build_array_layers(specs, (8, 8, 1), backend.name)
    for spec, shape, layer in zip(specs, spec_shapes(specs, (8, 8, 1)), layers, strict=True):
        if isinstance(spec, Conv | Pool):
            assert (layer.out_height, layer.out_width, layer.channel_count) == shape.output_shape
        elif isinstance(spec, BatchNorm):
            assert layer.positions == shape.positions


def test_a_flat_input_feeds_the_first_dense_layer(backend: Backend):
    layers: list[Any] = build_array_layers([Dense(5), OUTPUT], (7,), backend.name)
    assert [layer.W.shape for layer in layers] == [(5, 7), (3, 5)]


def test_a_conv_layer_needs_an_image_input(backend: Backend):
    with pytest.raises(AssertionError, match="height, width, channels"):
        build_array_layers([Conv(3, 2), OUTPUT], (64,), backend.name)


# residual blocks (the residual-connections workplan, stage 1): dense bodies ending in an affine
# layer, whose size is the block's input's (D5)
AFFINE_5 = Dense(5, activation="linear", bias=True)
BLOCK_5 = Residual((Dense(8, activation="relu"), AFFINE_5))

RESIDUAL: dict[str, list[LayerSpec]] = {
    "one block": [Dense(5), BLOCK_5, OUTPUT],
    "affine body": [Dense(5), Residual((AFFINE_5,)), OUTPUT],
    "two blocks in a row": [Dense(5), BLOCK_5, BLOCK_5, OUTPUT],
    "sigmoid body": [Dense(5), Residual((Dense(8), Dense(6), AFFINE_5)), OUTPUT],
    "dropout body": [Dense(5), Residual((Dense(8, dropout=0.3), AFFINE_5)), OUTPUT],
    "batch norm body": [Dense(5, activation="relu"), Residual((LINEAR, BatchNorm("relu"), AFFINE_5)), OUTPUT],
    "batch norm before the block": [LINEAR, BatchNorm(), BLOCK_5, OUTPUT],
    "block first": [BLOCK_5, OUTPUT],
    "conv front end, then a dense layer": [Conv(3, 2), Dense(5), BLOCK_5, OUTPUT],
}

RESIDUAL_INVALID: dict[str, list[LayerSpec]] = {
    "empty body": [Dense(5), Residual(()), OUTPUT],
    "nested": [Dense(5), Residual((BLOCK_5, AFFINE_5)), OUTPUT],
    "conv in the body": [Conv(3, 2), Residual((Conv(2, 2), AFFINE_5)), OUTPUT],
    "pool in the body": [Conv(3, 2), Residual((Pool(2), AFFINE_5)), OUTPUT],
    "a block among the conv layers": [Conv(3, 2), BLOCK_5, Pool(2), OUTPUT],
    "body ends sigmoid": [Dense(5), Residual((Dense(8), Dense(5))), OUTPUT],
    "body ends linear without a bias": [Dense(5), Residual((Dense(8), Dense(5, activation="linear"))), OUTPUT],
    "body ends affine with dropout": [
        Dense(5),
        Residual((Dense(5, activation="linear", bias=True, dropout=0.3),)),
        OUTPUT,
    ],
    "bias on a hidden layer": [Dense(5, bias=True), OUTPUT],
    "bias on a linear layer outside a block": [Dense(5, activation="linear", bias=True), BatchNorm(), OUTPUT],
    "bias inside the body, not last": [Dense(5), Residual((Dense(8, bias=True), AFFINE_5)), OUTPUT],
    "affine, then a batch norm": [Dense(5), Residual((AFFINE_5, BatchNorm(), AFFINE_5)), OUTPUT],
    "bias on the output layer": [Dense(3, output=True, bias=True)],
    "linear without its batch norm in the body": [Dense(5), Residual((LINEAR, AFFINE_5)), OUTPUT],
    "batch norm first in the body": [Dense(5), Residual((BatchNorm(), AFFINE_5)), OUTPUT],
    "batch norm right after a block": [Dense(5), BLOCK_5, BatchNorm(), OUTPUT],
    "linear right before a block": [LINEAR, BLOCK_5, OUTPUT],
    "output layer in the body": [Dense(5), Residual((Dense(3, output=True), AFFINE_5)), OUTPUT],
    "block last": [Dense(5), BLOCK_5],
}


@pytest.mark.parametrize("specs", RESIDUAL.values(), ids=RESIDUAL.keys())
def test_residual_blocks_are_accepted(specs: list[LayerSpec]):
    validate_layer_specs(specs)
    input_shape = (8, 8, 1) if isinstance(specs[0], Conv) else (5,)
    spec_shapes(specs, input_shape)


@pytest.mark.parametrize("specs", RESIDUAL_INVALID.values(), ids=RESIDUAL_INVALID.keys())
def test_a_malformed_residual_block_is_rejected(specs: list[LayerSpec]):
    with pytest.raises(AssertionError):
        validate_layer_specs(specs)


SHAPE_INVALID: dict[str, tuple[list[LayerSpec], InputShape]] = {
    "size mismatch": ([Dense(6), BLOCK_5, OUTPUT], (4,)),
    "size mismatch, block first": ([BLOCK_5, OUTPUT], (4,)),
    "after a conv front end": ([Conv(3, 2), BLOCK_5, OUTPUT], (8, 8, 1)),
    "on an image input": ([BLOCK_5, OUTPUT], (5, 1, 1)),
}


@pytest.mark.parametrize("specs, input_shape", SHAPE_INVALID.values(), ids=SHAPE_INVALID.keys())
def test_a_block_whose_input_isnt_its_flat_output_size_is_rejected_by_the_shape_walk(
    specs: list[LayerSpec], input_shape: InputShape
):
    validate_layer_specs(specs)
    with pytest.raises(AssertionError, match="D2|D5"):
        spec_shapes(specs, input_shape)


def test_a_block_expands_into_a_fork_its_body_and_an_add():
    specs = [Dense(5), BLOCK_5, BLOCK_5, OUTPUT]
    expanded = [Dense(5), Fork(), *BLOCK_5.body, Add(), Fork(), *BLOCK_5.body, Add(), OUTPUT]

    assert expand_specs(specs) == expanded
    assert expand_specs(expanded) == expanded
    assert spec_paths(specs) == [
        "layer 0",
        "layer 1, block fork",
        "layer 1, block body 0",
        "layer 1, block body 1",
        "layer 1, block add",
        "layer 2, block fork",
        "layer 2, block body 0",
        "layer 2, block body 1",
        "layer 2, block add",
        "layer 3",
    ]
    assert expand_specs([Dense(5), OUTPUT]) == [Dense(5), OUTPUT]


def test_the_shape_walk_covers_the_expanded_layers():
    assert spec_shapes([Dense(5), BLOCK_5, OUTPUT], (7,)) == [
        SpecShape((7,), (5,)),
        SpecShape((5,), (5,)),
        SpecShape((5,), (8,)),
        SpecShape((8,), (5,)),
        SpecShape((5,), (5,)),
        SpecShape((5,), (3,)),
    ]


def test_a_batch_norm_inside_a_body_is_found_at_its_expanded_index():
    specs = RESIDUAL["batch norm body"]
    assert batch_norm_index(specs) == 3
    assert batch_norm_index(RESIDUAL["one block"]) is None

    with pytest.raises(ValueError, match=r"layer 1, block body 1, BatchNorm\("):
        refuse_single_example_network(specs, 3)

    grouped = [Dense(5), Residual((LINEAR, BatchNorm(group_size=4), AFFINE_5)), OUTPUT]
    with pytest.raises(ValueError, match="layer 1, block body 1, BatchNorm"):
        refuse_single_example_groups(grouped, 9)
    refuse_single_example_groups(grouped, 10)


def test_the_affine_field_defaults_off_so_existing_dense_specs_are_unchanged():
    assert Dense(5, activation="linear") == Dense(5, activation="linear", bias=False)


@pytest.mark.parametrize("specs", RESIDUAL.values(), ids=RESIDUAL.keys())
def test_every_implementation_builds_a_block_and_format_2_round_trips_it(specs: list[LayerSpec], backend: Backend):
    input_shape = (8, 8, 1) if isinstance(specs[0], Conv) else (5,)
    build_array_layers(specs, input_shape, backend.name)
    build_python_layers(specs, input_shape, StateLayer(math.prod(input_shape), [(0.0, 1.0)] * math.prod(input_shape)))
    assert [layer_from_json(layer_to_json(spec)) for spec in specs] == specs


# patch models (the layer-norm and attention workplan, stage 1): over a (4, 4, 1) image, Patches(2)
# gives 4 tokens of 4, and the embedding makes them 4 tokens of 6
PATCHES = Patches(2)
EMBED = Dense(6, activation="linear", bias=True)
AFFINE_6 = Dense(6, activation="linear", bias=True)
ATTENTION_BLOCK = Residual((LayerNorm(), Attention()))
FFN_BLOCK = Residual((LayerNorm(), Dense(8, activation="relu"), AFFINE_6))
SOFTMAX = Dense(3, output=True, activation="softmax", loss="cross_entropy")

TOKENS: dict[str, list[LayerSpec]] = {
    "the README's model": [
        PATCHES,
        EMBED,
        Position(),
        ATTENTION_BLOCK,
        FFN_BLOCK,
        TokenMean(),
        LayerNorm(),
        SOFTMAX,
    ],
    "patches, then the mean": [PATCHES, TokenMean(), OUTPUT],
    "no position": [PATCHES, EMBED, ATTENTION_BLOCK, TokenMean(), OUTPUT],
    "no embedding": [PATCHES, Position(), Residual((LayerNorm(), Attention())), TokenMean(), OUTPUT],
    "the attention block alone": [PATCHES, EMBED, Position(), ATTENTION_BLOCK, TokenMean(), OUTPUT],
    "the FFN block alone": [PATCHES, EMBED, Position(), FFN_BLOCK, TokenMean(), OUTPUT],
    "two layers' blocks": [PATCHES, EMBED, ATTENTION_BLOCK, FFN_BLOCK, ATTENTION_BLOCK, FFN_BLOCK, TokenMean(), OUTPUT],
    "a ReLU embedding": [PATCHES, Dense(6, activation="relu"), Position(), ATTENTION_BLOCK, TokenMean(), OUTPUT],
    "attention alone in its body": [PATCHES, EMBED, Residual((Attention(),)), TokenMean(), OUTPUT],
    "an affine body": [PATCHES, EMBED, Residual((AFFINE_6,)), TokenMean(), OUTPUT],
    "a layer norm between blocks": [PATCHES, EMBED, ATTENTION_BLOCK, LayerNorm(), FFN_BLOCK, TokenMean(), OUTPUT],
    "token-wise layers after a block": [
        PATCHES,
        EMBED,
        ATTENTION_BLOCK,
        Dense(5, activation="relu"),
        TokenMean(),
        OUTPUT,
    ],
    "a dense part after the mean": [
        PATCHES,
        EMBED,
        ATTENTION_BLOCK,
        TokenMean(),
        Dense(5, activation="relu"),
        LINEAR,
        BatchNorm(),
        BLOCK_5,
        OUTPUT,
    ],
}

TOKENS_INVALID: dict[str, list[LayerSpec]] = {
    "patches after a conv layer": [Conv(3, 2), PATCHES, TokenMean(), OUTPUT],
    "patches after a dense layer": [Dense(16), PATCHES, TokenMean(), OUTPUT],
    "patches twice": [PATCHES, PATCHES, TokenMean(), OUTPUT],
    "patches without a mean, a sigmoid output per token": [PATCHES, EMBED, OUTPUT],
    "patch size 0": [Patches(0), TokenMean(), OUTPUT],
    "two positions": [PATCHES, EMBED, Position(), Position(), TokenMean(), OUTPUT],
    "a position after a block": [PATCHES, EMBED, ATTENTION_BLOCK, Position(), TokenMean(), OUTPUT],
    "a position in a body": [PATCHES, EMBED, Residual((Position(), AFFINE_6)), TokenMean(), OUTPUT],
    "a sigmoid token layer": [PATCHES, Dense(6), TokenMean(), OUTPUT],
    "a dropout token layer": [PATCHES, Dense(6, dropout=0.3), TokenMean(), OUTPUT],
    "a linear token layer without a bias": [PATCHES, Dense(6, activation="linear"), TokenMean(), OUTPUT],
    "a batch norm among the tokens": [PATCHES, Dense(6, activation="linear"), BatchNorm("relu"), TokenMean(), OUTPUT],
    "a batch norm after the embedding": [PATCHES, EMBED, BatchNorm("relu"), TokenMean(), OUTPUT],
    "a batch norm in a token body": [
        PATCHES,
        EMBED,
        Residual((Dense(6, activation="linear"), BatchNorm("relu"), AFFINE_6)),
        TokenMean(),
        OUTPUT,
    ],
    "a sigmoid token body": [PATCHES, EMBED, Residual((Dense(8), AFFINE_6)), TokenMean(), OUTPUT],
    "a token body ending ReLU": [
        PATCHES,
        EMBED,
        Residual((LayerNorm(), Dense(6, activation="relu"))),
        TokenMean(),
        OUTPUT,
    ],
    "a token body ending in a layer norm": [PATCHES, EMBED, Residual((LayerNorm(),)), TokenMean(), OUTPUT],
    "attention before the body's end": [PATCHES, EMBED, Residual((Attention(), AFFINE_6)), TokenMean(), OUTPUT],
    "attention outside a block": [PATCHES, EMBED, Attention(), TokenMean(), OUTPUT],
    "a nested token block": [PATCHES, EMBED, Residual((ATTENTION_BLOCK, AFFINE_6)), TokenMean(), OUTPUT],
    "an empty token body": [PATCHES, EMBED, Residual(()), TokenMean(), OUTPUT],
    "a conv layer among the tokens": [PATCHES, Conv(1, 2), TokenMean(), OUTPUT],
    "a pool layer after the mean": [PATCHES, TokenMean(), Pool(2), OUTPUT],
    "a mean twice": [PATCHES, TokenMean(), TokenMean(), OUTPUT],
    "a position after the mean": [PATCHES, TokenMean(), Position(), OUTPUT],
    "attention after the mean": [PATCHES, TokenMean(), Residual((Attention(),)), OUTPUT],
    "a mean without patches": [Dense(5), TokenMean(), OUTPUT],
    "a mean after a conv layer": [Conv(3, 2), TokenMean(), OUTPUT],
    "a position without patches": [Dense(5), Position(), OUTPUT],
    "attention without patches": [Dense(5), Residual((LayerNorm(), Attention())), OUTPUT],
    "a layer norm with no epsilon": [PATCHES, Residual((LayerNorm(epsilon=0.0), AFFINE_6)), TokenMean(), OUTPUT],
}

# a flat layer norm (D5) wherever a dense hidden layer may stand: before and after each hidden-layer
# kind, and as a residual body's first layer
FLAT_LAYER_NORM: dict[str, list[LayerSpec]] = {
    "alone": [LayerNorm(), OUTPUT],
    "before the output layer": [Dense(5), LayerNorm(), SOFTMAX],
    "after sigmoid": [Dense(5), LayerNorm(), Dense(4), OUTPUT],
    "before sigmoid": [LayerNorm(), Dense(5), OUTPUT],
    "after ReLU": [Dense(5, activation="relu"), LayerNorm(), OUTPUT],
    "before ReLU": [Dense(5), LayerNorm(), Dense(4, activation="relu"), OUTPUT],
    "after dropout": [Dense(5, dropout=0.3), LayerNorm(), OUTPUT],
    "before dropout": [Dense(5), LayerNorm(), Dense(4, dropout=0.3), OUTPUT],
    "after a batch-norm pair": [LINEAR, BatchNorm(), LayerNorm(), OUTPUT],
    "before a batch-norm pair": [Dense(5), LayerNorm(), LINEAR, BatchNorm(), OUTPUT],
    "after a block": [Dense(5), BLOCK_5, LayerNorm(), OUTPUT],
    "before a block": [Dense(5), LayerNorm(), BLOCK_5, OUTPUT],
    "a body's first layer": [Dense(5), Residual((LayerNorm(), Dense(8, activation="relu"), AFFINE_5)), OUTPUT],
    "inside a body": [Dense(5), Residual((Dense(8, activation="relu"), LayerNorm(), AFFINE_5)), OUTPUT],
    "twice": [Dense(5), LayerNorm(), LayerNorm(), OUTPUT],
    "after a conv front end": [Conv(3, 2), Pool(2), LayerNorm(), Dense(5), OUTPUT],
}

FLAT_LAYER_NORM_INVALID: dict[str, list[LayerSpec]] = {
    "between a linear layer and its batch norm": [LINEAR, LayerNorm(), BatchNorm(), OUTPUT],
    "a batch norm after it": [Dense(5), LayerNorm(), BatchNorm(), OUTPUT],
    "a body's last layer": [Dense(5), Residual((Dense(8), LayerNorm())), OUTPUT],
    "the output layer": [Dense(5), LayerNorm()],
    "no epsilon": [Dense(5), LayerNorm(epsilon=0.0), OUTPUT],
    "a conv layer after it": [Conv(3, 2), LayerNorm(), Conv(2, 2), OUTPUT],
}


def _input_shape(specs: list[LayerSpec]) -> InputShape:
    return (4, 4, 1) if isinstance(specs[0], Patches) else (8, 8, 1) if isinstance(specs[0], Conv) else (5,)


@pytest.mark.parametrize("specs", (TOKENS | FLAT_LAYER_NORM).values(), ids=(TOKENS | FLAT_LAYER_NORM).keys())
def test_patch_models_and_flat_layer_norms_are_accepted(specs: list[LayerSpec]):
    validate_layer_specs(specs)
    spec_shapes(specs, _input_shape(specs))


INVALID_TOKENS = TOKENS_INVALID | {f"flat layer norm, {name}": specs for name, specs in FLAT_LAYER_NORM_INVALID.items()}


@pytest.mark.parametrize("specs", INVALID_TOKENS.values(), ids=INVALID_TOKENS.keys())
def test_a_malformed_patch_model_or_flat_layer_norm_is_rejected(specs: list[LayerSpec]):
    with pytest.raises(AssertionError):
        validate_layer_specs(specs)


TOKEN_SHAPE_INVALID: dict[str, tuple[list[LayerSpec], InputShape]] = {
    "a patch size that doesn't divide the height": ([Patches(3), TokenMean(), OUTPUT], (4, 6, 1)),
    "a patch size that doesn't divide the width": ([Patches(3), TokenMean(), OUTPUT], (6, 4, 1)),
    "patches over a flat input": ([PATCHES, TokenMean(), OUTPUT], (16,)),
    "a token block's size mismatch": (
        [PATCHES, EMBED, Residual((Dense(5, activation="linear", bias=True),)), TokenMean(), OUTPUT],
        (4, 4, 1),
    ),
    "attention over the unembedded patches, then an affine block of the embedding's size": (
        [PATCHES, ATTENTION_BLOCK, Residual((AFFINE_6,)), TokenMean(), OUTPUT],
        (4, 4, 1),
    ),
}


@pytest.mark.parametrize("specs, input_shape", TOKEN_SHAPE_INVALID.values(), ids=TOKEN_SHAPE_INVALID.keys())
def test_a_patch_model_that_doesnt_fit_its_input_is_rejected_by_the_shape_walk(
    specs: list[LayerSpec], input_shape: InputShape
):
    validate_layer_specs(specs)
    with pytest.raises(AssertionError):
        spec_shapes(specs, input_shape)


# the README's patch model, over a 28x28x1 image
README_PATCH_MODEL: list[LayerSpec] = [
    Patches(7),
    Dense(32, activation="linear", bias=True),
    Position(),
    Residual((LayerNorm(), Attention())),
    Residual((LayerNorm(), Dense(64, activation="relu"), Dense(32, activation="linear", bias=True))),
    TokenMean(),
    LayerNorm(),
    Dense(10, activation="softmax", output=True, loss="cross_entropy"),
]


def test_the_shape_walk_carries_tokens_from_the_patches_to_the_mean():
    # the README's model: 28x28x1 -> 16 tokens of 49 -> of 32 ... -> the mean of 32 -> 10
    specs = README_PATCH_MODEL
    validate_layer_specs(specs)
    assert spec_shapes(specs, (28, 28, 1)) == [
        SpecShape((28, 28, 1), (16, 49)),
        SpecShape((16, 49), (16, 32)),
        SpecShape((16, 32), (16, 32)),
        SpecShape((16, 32), (16, 32)),  # the attention block: fork, layer norm, attention, add
        SpecShape((16, 32), (16, 32)),
        SpecShape((16, 32), (16, 32)),
        SpecShape((16, 32), (16, 32)),
        SpecShape((16, 32), (16, 32)),  # the FFN block: fork, layer norm, ReLU, affine, add
        SpecShape((16, 32), (16, 32)),
        SpecShape((16, 32), (16, 64)),
        SpecShape((16, 64), (16, 32)),
        SpecShape((16, 32), (16, 32)),
        SpecShape((16, 32), (32,)),
        SpecShape((32,), (32,)),
        SpecShape((32,), (10,)),
    ]
    # each patch's 2 * 2 * 3 values, the 3 x 2 grid of patches row-major
    assert spec_shapes([Patches(2), TokenMean(), OUTPUT], (6, 4, 3))[0] == SpecShape((6, 4, 3), (6, 12))
    assert spec_shapes(FLAT_LAYER_NORM["after a conv front end"], (8, 8, 1))[2] == SpecShape((3, 3, 2), (3, 3, 2))


@pytest.mark.parametrize("specs", (TOKENS | FLAT_LAYER_NORM).values(), ids=(TOKENS | FLAT_LAYER_NORM).keys())
def test_every_implementation_builds_the_new_specs_and_format_2_round_trips_them(specs: list[LayerSpec]):
    input_shape = _input_shape(specs)
    build_array_layers(specs, input_shape, "numpy")
    build_array_layers(specs, input_shape, "rust")
    size = math.prod(input_shape)
    build_python_layers(specs, input_shape, StateLayer(size, [(0.0, 1.0)] * size))
    assert [layer_from_json(json.loads(json.dumps(layer_to_json(spec)))) for spec in specs] == specs


# multi-head attention (the multi-head attention workplan, stage 2): over a (4, 4, 1) image,
# Patches(2) and a 32-wide embedding give 4 tokens of 32
EMBED_32 = Dense(32, activation="linear", bias=True)


def _multi_head(attention: Attention) -> list[LayerSpec]:
    return [PATCHES, EMBED_32, Residual((LayerNorm(), attention)), TokenMean(), OUTPUT]


MULTI_HEAD = {
    "two heads": Attention(heads=2),
    "four heads": Attention(heads=4),
    "a head per feature": Attention(heads=32),
    "one head, a narrower key": Attention(key_size=8),
    "three heads of 8, not dividing the width": Attention(heads=3, key_size=8),
    "four heads of 32, wider than the width": Attention(heads=4, key_size=32),
}
MULTI_HEAD_INVALID = {
    "no heads": Attention(heads=0),
    "negative heads": Attention(heads=-1),
    "a key size of 0": Attention(key_size=0),
    "a key size of 0 with heads": Attention(heads=2, key_size=0),
}
MULTI_HEAD_SHAPE_INVALID = {
    "three heads over 32 features": Attention(heads=3),
    "more heads than features": Attention(heads=64),
}


@pytest.mark.parametrize("attention", MULTI_HEAD.values(), ids=MULTI_HEAD.keys())
def test_heads_and_key_sizes_are_accepted(attention: Attention):
    specs = _multi_head(attention)
    validate_layer_specs(specs)
    assert spec_shapes(specs, (4, 4, 1))[4] == SpecShape((4, 32), (4, 32))  # attention keeps its shape


@pytest.mark.parametrize("attention", MULTI_HEAD_INVALID.values(), ids=MULTI_HEAD_INVALID.keys())
def test_no_heads_or_an_empty_key_is_rejected(attention: Attention):
    with pytest.raises(AssertionError, match=re.escape(repr(attention))):
        validate_layer_specs(_multi_head(attention))


@pytest.mark.parametrize("attention", MULTI_HEAD_SHAPE_INVALID.values(), ids=MULTI_HEAD_SHAPE_INVALID.keys())
def test_heads_that_dont_divide_the_width_without_a_key_size_are_rejected_by_the_shape_walk(attention: Attention):
    specs = _multi_head(attention)
    validate_layer_specs(specs)
    with pytest.raises(AssertionError, match="over tokens of 32 features"):
        spec_shapes(specs, (4, 4, 1))


def test_the_head_size_is_the_key_size_else_the_width_over_the_heads():
    assert Attention().head_size(32) == 32
    assert Attention(heads=4).head_size(32) == 8
    assert Attention(key_size=16).head_size(32) == 16
    assert Attention(heads=4, key_size=32).head_size(32) == 32


@pytest.mark.parametrize("attention", MULTI_HEAD.values(), ids=MULTI_HEAD.keys())
def test_every_builder_builds_heads_and_key_sizes(attention: Attention):
    specs = _multi_head(attention)
    d_k = attention.head_size(32)
    built = {
        "numpy": build_array_layers(specs, (4, 4, 1), "numpy"),
        "rust": build_array_layers(specs, (4, 4, 1), "rust"),
        "python": build_python_layers(specs, (4, 4, 1), StateLayer(16, [(0.0, 1.0)] * 16)),
    }
    for name, layers in built.items():
        attention_layer: Any = next(layer for layer in layers if type(layer).__name__.startswith("Attention"))
        assert (attention_layer.heads, attention_layer.key_size) == (attention.heads, d_k), name


@pytest.mark.parametrize("attention", MULTI_HEAD.values(), ids=MULTI_HEAD.keys())
def test_format_2_round_trips_heads_and_key_sizes(attention: Attention):
    specs = _multi_head(attention)
    assert [layer_from_json(json.loads(json.dumps(layer_to_json(spec)))) for spec in specs] == specs


# sequence models (the sequence task workplan, stage 2): over 5 token ids of a vocabulary of 7, an
# Embedding gives 5 tokens of 6, and a token-wise softmax output 5 tokens of 7
IDS = Embedding(7, 6)
CAUSAL_BLOCK = Residual((LayerNorm(), Attention(heads=2, causal=True)))
TOKEN_OUTPUT = Dense(7, output=True, activation="softmax", loss="cross_entropy")

SEQUENCE: dict[str, list[LayerSpec]] = {
    "a causal transformer": [IDS, Position(), CAUSAL_BLOCK, FFN_BLOCK, LayerNorm(), TOKEN_OUTPUT],
    "the embedding, then the output": [IDS, TOKEN_OUTPUT],
    "a token-wise FFN, no attention": [IDS, Position(), FFN_BLOCK, TOKEN_OUTPUT],
    "the leak arm, unmasked": [IDS, Position(), ATTENTION_BLOCK, FFN_BLOCK, TOKEN_OUTPUT],
    "an embedding, then the mean": [IDS, CAUSAL_BLOCK, TokenMean(), SOFTMAX],
    "patches, then a token-wise output": [
        PATCHES,
        EMBED,
        ATTENTION_BLOCK,
        Dense(3, output=True, activation="softmax", loss="cross_entropy"),
    ],
    "a causal patch model": [PATCHES, EMBED, CAUSAL_BLOCK, TokenMean(), OUTPUT],
}

SEQUENCE_INVALID: dict[str, list[LayerSpec]] = {
    "an embedding after a dense layer": [Dense(5), IDS, TOKEN_OUTPUT],
    "an embedding after patches": [PATCHES, IDS, TOKEN_OUTPUT],
    "two embeddings": [IDS, Embedding(7, 6), TOKEN_OUTPUT],
    "an embedding after the mean": [IDS, TokenMean(), IDS, TOKEN_OUTPUT],
    "an embedding in a body": [IDS, Residual((IDS, AFFINE_6)), TOKEN_OUTPUT],
    "an empty vocabulary": [Embedding(0, 6), TOKEN_OUTPUT],
    "an embedding of no features": [Embedding(7, 0), TOKEN_OUTPUT],
    "a token-wise sigmoid output": [IDS, Dense(7, output=True)],
    "a token-wise sigmoid cross-entropy output": [IDS, Dense(7, output=True, loss="cross_entropy")],
    "a causal block outside a token part": [Dense(5), Residual((LayerNorm(), Attention(causal=True))), OUTPUT],
    "causal attention outside a block": [IDS, Attention(causal=True), TOKEN_OUTPUT],
}


@pytest.mark.parametrize("specs", SEQUENCE.values(), ids=SEQUENCE.keys())
def test_sequence_models_are_accepted(specs: list[LayerSpec]):
    validate_layer_specs(specs)
    spec_shapes(specs, (5,) if isinstance(specs[0], Embedding) else (4, 4, 1))


@pytest.mark.parametrize("specs", SEQUENCE_INVALID.values(), ids=SEQUENCE_INVALID.keys())
def test_a_malformed_sequence_model_is_rejected(specs: list[LayerSpec]):
    with pytest.raises(AssertionError):
        validate_layer_specs(specs)


def test_an_embedding_over_an_image_is_rejected_by_the_shape_walk():
    specs = SEQUENCE["the embedding, then the output"]
    validate_layer_specs(specs)
    with pytest.raises(AssertionError, match="a flat input of token ids"):
        spec_shapes(specs, (5, 1, 1))


def test_the_shape_walk_carries_ids_to_a_token_per_id_and_an_output_per_token():
    # 5 ids -> 5 tokens of 6 ... -> 5 tokens of 7, one softmax per token
    shapes = spec_shapes(SEQUENCE["a causal transformer"], (5,))
    assert shapes[0] == SpecShape((5,), (5, 6))
    # the position, then the causal block: fork, layer norm, attention, add
    assert shapes[1:-1] == [SpecShape((5, 6), (5, 6))] * 5 + [
        SpecShape((5, 6), (5, 6)),  # the FFN block: fork, layer norm, ReLU, affine, add
        SpecShape((5, 6), (5, 6)),
        SpecShape((5, 6), (5, 8)),
        SpecShape((5, 8), (5, 6)),
        SpecShape((5, 6), (5, 6)),
        SpecShape((5, 6), (5, 6)),  # the final layer norm
    ]
    assert shapes[-1] == SpecShape((5, 6), (5, 7))


def test_only_a_token_part_without_a_mean_has_a_token_wise_output():
    assert token_wise_output(SEQUENCE["a causal transformer"])
    assert token_wise_output(SEQUENCE["patches, then a token-wise output"])
    assert not token_wise_output(SEQUENCE["an embedding, then the mean"])
    assert not token_wise_output(TOKENS["the README's model"])
    assert not token_wise_output(VALID["softmax output"])


def test_attention_is_unmasked_by_default():
    assert Attention() == Attention(causal=False)


def _builds(specs: list[LayerSpec], backend: str) -> Any:
    input_shape: InputShape = (5,) if isinstance(specs[0], Embedding) else (4, 4, 1)
    size = math.prod(input_shape)
    if backend == "python":
        return build_python_layers(specs, input_shape, StateLayer(size, [(0.0, 1.0)] * size))
    return build_array_layers(specs, input_shape, backend)


@pytest.mark.parametrize(
    "specs",
    [*SEQUENCE.values(), [PATCHES, EMBED, CAUSAL_BLOCK, TokenMean(), OUTPUT]],
    ids=[*SEQUENCE.keys(), "only the mask"],
)
def test_all_three_implementations_build_sequence_specs(specs: list[LayerSpec]):
    for backend in ("numpy", "rust", "python"):
        _builds(specs, backend)


@pytest.mark.parametrize("specs", SEQUENCE.values(), ids=SEQUENCE.keys())
def test_format_2_round_trips_sequence_specs(specs: list[LayerSpec]):
    assert [layer_from_json(json.loads(json.dumps(layer_to_json(spec)))) for spec in specs] == specs
