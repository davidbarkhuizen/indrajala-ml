from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.max_pool_layer import PoolSpec


def build_conv_front_end(
    input_height: int,
    input_width: int,
    conv_specs: list[ConvSpec | PoolSpec],
    make_conv: Callable,
    make_pool: Callable,
    input_layer=None,
) -> list:
    """
    Chains a convolutional front end through conv_specs - the one loop shared by
    ConvMultiClassBackpropClassifierNetwork (pure Python) and
    ConvVectorizedMultiClassBackpropClassifierNetwork (numpy), which differ only in which layer
    classes they build. The first layer reads the single-channel input image; each later one
    reads the previous layer's out_height x out_width x channel_count output.

    make_conv(spec, previous, height, width, channels) / make_pool(spec, previous, height,
    width, channels) build one layer; previous is the preceding layer (input_layer for the
    first), for the pure-Python layers' node wiring - the numpy layers ignore it. Every built
    layer must expose out_height/out_width/channel_count.
    """
    assert any(isinstance(spec, ConvSpec) for spec in conv_specs), "conv_specs must contain at least one ConvSpec"

    layers = []
    previous = input_layer
    height, width, channels = input_height, input_width, 1
    for spec in conv_specs:
        make = make_pool if isinstance(spec, PoolSpec) else make_conv
        layer = make(spec, previous, height, width, channels)
        layers.append(layer)
        previous = layer
        height, width, channels = layer.out_height, layer.out_width, layer.channel_count
    return layers


def spec_to_json(spec: ConvSpec | PoolSpec) -> dict:
    return {"type": "pool" if isinstance(spec, PoolSpec) else "conv", **asdict(spec)}


def spec_from_json(spec: dict) -> ConvSpec | PoolSpec:
    fields = {key: value for key, value in spec.items() if key != "type"}
    return PoolSpec(**fields) if spec["type"] == "pool" else ConvSpec(**fields)
