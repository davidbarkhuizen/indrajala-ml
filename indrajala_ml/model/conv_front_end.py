from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.model_io import load_json, save_json


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


def build_conv_array_network_layers(
    input_height: int,
    input_width: int,
    conv_specs: list[ConvSpec | PoolSpec],
    dense_layer_sizes: list[int],
    class_count: int,
    conv_cls: type,
    pool_cls: type,
    dense_cls: type,
) -> tuple[list, list, object]:
    """
    The layers of an array-backed conv network - the conv front end, the dense hidden layers and
    the output layer - shared by ConvVectorizedMultiClassBackpropClassifierNetwork (numpy) and
    ConvRustArrayMultiClassBackpropClassifierNetwork (Rust), which differ only in the layer
    classes. The dense tail's fan-in starts from the last conv/pool layer's flattened output.
    """
    conv_layers = build_conv_front_end(
        input_height,
        input_width,
        conv_specs,
        make_conv=lambda spec, _previous, height, width, channels: conv_cls(
            input_height=height,
            input_width=width,
            input_channels=channels,
            kernel_size=spec.kernel_size,
            channel_count=spec.channel_count,
            stride=spec.stride,
        ),
        make_pool=lambda spec, _previous, height, width, channels: pool_cls(
            input_height=height,
            input_width=width,
            input_channels=channels,
            pool_size=spec.pool_size,
            stride=spec.stride,
        ),
    )

    dense_layers = []
    previous_size = conv_layers[-1].size
    for size in dense_layer_sizes:
        dense_layers.append(dense_cls(size, previous_size))
        previous_size = size

    return conv_layers, dense_layers, dense_cls(class_count, previous_size)


def save_conv_array_model_json(path: str, network) -> None:
    """
    The save format shared by the numpy and Rust conv networks: the same envelope keys as
    ConvMultiClassBackpropClassifierNetwork.save, but one (W, b) entry per layer (an empty one
    for a pool layer) rather than per-kernel lists. Both backends' arrays have .tolist(), so a
    file saved by either loads into the other; a pure-Python conv network's file doesn't.
    """
    save_json(
        path,
        {
            "input_height": network.input_height,
            "input_width": network.input_width,
            "conv_layers": [spec_to_json(spec) for spec in network.conv_specs],
            "dense_layer_sizes": network.dense_layer_sizes,
            "class_count": network.class_count,
            "snapshot": [[entry[0].tolist(), entry[1].tolist()] if entry else [] for entry in network.snapshot()],
        },
    )


def load_conv_array_model_json(cls: type, path: str):
    """The load-side counterpart to save_conv_array_model_json: builds cls from the envelope and
    restores the (W, b) entries, which each backend's restore() accepts as nested lists."""
    state = load_json(path)
    network = cls(
        input_height=state["input_height"],
        input_width=state["input_width"],
        conv_specs=[spec_from_json(spec) for spec in state["conv_layers"]],
        dense_layer_sizes=state["dense_layer_sizes"],
        class_count=state["class_count"],
    )
    network.restore(state["snapshot"])
    return network
