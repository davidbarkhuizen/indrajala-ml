"""
Layer specs (layer_specs.py) and their numpy and Rust builder (array_layer_builder.py): which spec
lists are accepted, the layer class each spec kind maps to, and the shapes the builder chains.
"""

from typing import Any

import pytest

from indrajala_ml.model.array_layer_builder import LAYER_CLASSES, build_array_layers
from indrajala_ml.model.layer_specs import Conv, Dense, LayerSpec, Pool, validate_layer_specs
from tests.helpers import Backend

OUTPUT = Dense(3, output=True)

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
]


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


def test_a_flat_input_feeds_the_first_dense_layer(backend: Backend):
    layers: list[Any] = build_array_layers([Dense(5), OUTPUT], (7,), backend.name)
    assert [layer.W.shape for layer in layers] == [(5, 7), (3, 5)]


def test_a_conv_layer_needs_an_image_input(backend: Backend):
    with pytest.raises(AssertionError, match="height, width, channels"):
        build_array_layers([Conv(3, 2), OUTPUT], (64,), backend.name)
