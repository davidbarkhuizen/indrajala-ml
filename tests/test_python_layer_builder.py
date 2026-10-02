"""
The pure-Python builder of layer specs (python_layer_builder.py): what tests/model/specs/test_layer_specs.py
checks of the numpy and Rust builder, for the pure-Python layer classes, over the same spec lists.
"""

import math
from typing import Any

import pytest

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.batch_norm_layer import BatchNormLayer
from indrajala_ml.model.conv_layer import ConvLayer
from indrajala_ml.model.cross_entropy_output_layer import CrossEntropyOutputLayer
from indrajala_ml.model.dropout_layer import TrainingModeNode
from indrajala_ml.model.linear_conv_layer import LinearConvLayer
from indrajala_ml.model.linear_layer import LinearLayer
from indrajala_ml.model.max_pool_layer import MaxPoolLayer
from indrajala_ml.model.python_layer_builder import build_python_layers
from indrajala_ml.model.relu_layer import ReLULayer
from indrajala_ml.model.softmax_output_layer import SoftmaxOutputLayer
from indrajala_ml.model.specs.layer_specs import BatchNorm, Conv, Dense, LayerSpec, Pool
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.state_layer import StateLayer
from indrajala_ml.pcg64 import default_rng
from tests.model.specs.test_layer_specs import BATCH_NORM, CONV_BATCH_NORM, INVALID, KINDS, LINEAR, OUTPUT, VALID

# ArrayLayerClasses' field names (KINDS), as the pure-Python classes; a dropout layer's class is
# made per drop probability (make_dropout_layer_cls), so it's checked by its nodes instead
PYTHON_CLASSES: dict[str, type[Any]] = {
    "sigmoid": BackpropLayer,
    "relu": ReLULayer,
    "softmax": SoftmaxOutputLayer,
    "cross_entropy": CrossEntropyOutputLayer,
    "conv": ConvLayer,
    "pool": MaxPoolLayer,
    "linear": LinearLayer,
    "batch_norm": BatchNormLayer,
}


def _build(specs: list[LayerSpec], input_shape: InputShape) -> list[Any]:
    dimension = math.prod(input_shape)
    return list(build_python_layers(specs, input_shape, StateLayer(dimension, [(0.0, 1.0)] * dimension)))


@pytest.mark.parametrize("specs", VALID.values(), ids=VALID.keys())
def test_every_accepted_combination_builds(specs: list[LayerSpec]):
    assert len(_build(specs, (8, 8, 1))) == len(specs)


@pytest.mark.parametrize("specs", BATCH_NORM.values(), ids=BATCH_NORM.keys())
def test_every_batch_norm_pair_builds(specs: list[LayerSpec]):
    assert len(_build(specs, (8, 8, 1))) == len(specs)


@pytest.mark.parametrize("specs", CONV_BATCH_NORM.values(), ids=CONV_BATCH_NORM.keys())
def test_every_conv_batch_norm_pair_builds(specs: list[LayerSpec]):
    layers = _build(specs, (8, 8, 1))
    assert len(layers) == len(specs)
    assert type(layers[0]) is LinearConvLayer or type(layers[1]) is LinearConvLayer


@pytest.mark.parametrize("specs", INVALID.values(), ids=INVALID.keys())
def test_a_rejected_combination_doesnt_build(specs: list[LayerSpec]):
    with pytest.raises(AssertionError):
        _build(specs, (8, 8, 1))


@pytest.mark.parametrize("specs, index, kind", [kind[1:] for kind in KINDS], ids=[kind[0] for kind in KINDS])
def test_each_spec_kind_builds_its_pure_python_layer_class(specs: list[LayerSpec], index: int, kind: str):
    layer = _build(specs, (8, 8, 1))[index]
    if kind == "dropout":
        assert all(isinstance(node, TrainingModeNode) for node in layer.nodes)
    else:
        assert type(layer) is PYTHON_CLASSES[kind]


def test_a_batch_norm_layer_normalizes_the_linear_layers_nodes_with_its_specs_constants():
    linear, norm, _output = _build([LINEAR, BatchNorm("relu", 1e-3, 0.2), OUTPUT], (7,))

    assert [node.input_node for node in norm.nodes] == list(linear.nodes)
    assert (norm.activation, norm.epsilon, norm.running_rate) == ("relu", 1e-3, 0.2)


def test_the_dropout_layer_takes_the_specs_probability():
    # every training forward pass drops each node with the spec's probability: at 0.0 none
    # is dropped, at 0.99 about all are
    for drop_probability, expect_kept in ((0.0, True), (0.99, False)):
        layer = _build([Dense(200, dropout=drop_probability), OUTPUT], (4,))[0]
        layer.set_rng(default_rng(0))
        layer.set_training_mode(True)
        layer.forward()
        kept = sum(node.value() != 0.0 for node in layer.nodes)
        assert (kept == 200) if expect_kept else (kept < 20)


def test_each_layer_reads_the_previous_layers_output_shape_and_nodes():
    # 9x9x1 -> conv 3, stride 2: 4x4x2 -> pool 2: 2x2x2 -> conv 2: 1x1x3 -> dense 5 -> output 3
    conv_1, pool, conv_2, dense, output = _build(
        [Conv(3, 2, stride=2), Pool(2), Conv(2, 3), Dense(5), OUTPUT], (9, 9, 1)
    )

    assert (conv_1.input_height, conv_1.input_width, conv_1.input_channels) == (9, 9, 1)
    assert (conv_1.out_height, conv_1.out_width, conv_1.channel_count) == (4, 4, 2)
    assert (pool.input_height, pool.input_width, pool.input_channels) == (4, 4, 2)
    assert (conv_2.input_height, conv_2.input_width, conv_2.input_channels) == (2, 2, 2)
    assert [len(kernel.weights) for kernel in conv_2.kernels] == [2 * 2 * 2] * 3
    assert pool.input_layer is conv_1 and conv_2.input_layer is pool
    assert dense.input_layer is conv_2 and output.input_layer is dense
    assert [len(node.input_nodes) for node in dense.nodes] == [3] * 5  # conv_2's 1x1x3 output
    assert [len(node.input_nodes) for node in output.nodes] == [5] * 3


def test_a_flat_input_feeds_the_first_dense_layer():
    first, output = _build([Dense(5), OUTPUT], (7,))
    assert [len(node.input_nodes) for node in first.nodes] == [7] * 5
    assert output.input_layer is first


def test_a_conv_layer_needs_an_image_input():
    with pytest.raises(AssertionError, match="height, width, channels"):
        _build([Conv(3, 2), OUTPUT], (64,))


def test_the_input_shape_must_match_the_input_layer():
    with pytest.raises(AssertionError, match="input_shape"):
        build_python_layers([Dense(5), OUTPUT], (8,), StateLayer(7, [(0.0, 1.0)] * 7))
