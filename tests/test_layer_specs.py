"""
Layer specs (layer_specs.py) and their numpy and Rust builder (array_layer_builder.py): which spec
lists are accepted, the layer class each spec kind maps to, and the shapes the builder chains.
"""

import math
from typing import Any, cast

import pytest

from indrajala_ml.model.array_layer_builder import LAYER_CLASSES, build_array_layers
from indrajala_ml.model.format2 import layer_to_json
from indrajala_ml.model.layer_specs import (
    Add,
    BatchNorm,
    Conv,
    Dense,
    Fork,
    InputShape,
    LayerSpec,
    Pool,
    Residual,
    SpecShape,
    batch_norm_index,
    expand_specs,
    refuse_single_example_groups,
    refuse_single_example_network,
    spec_paths,
    spec_shapes,
    validate_layer_specs,
)
from indrajala_ml.model.python_layer_builder import build_python_layers
from indrajala_ml.model.state_layer import StateLayer
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
# Python (stage 2, tests/test_python_layer_builder.py) and Rust (stage 3)
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
# tests/test_python_layer_builder.py) and Rust (4c)
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
def test_no_builder_or_writer_builds_a_block_yet(specs: list[LayerSpec], backend: Backend):
    input_shape = (8, 8, 1) if isinstance(specs[0], Conv) else (5,)
    with pytest.raises(NotImplementedError, match="residual-connections-workplan"):
        build_array_layers(specs, input_shape, backend.name)
    with pytest.raises(NotImplementedError, match="residual-connections-workplan"):
        build_python_layers(
            specs, input_shape, StateLayer(math.prod(input_shape), [(0.0, 1.0)] * math.prod(input_shape))
        )
    with pytest.raises(NotImplementedError, match="residual-connections-workplan"):
        [layer_to_json(spec) for spec in specs]
