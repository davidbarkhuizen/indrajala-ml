from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any, Protocol, TypeVar

from indrajala_ml.model.array_protocols import A, ArrayNetworkLayer, WeightedArrayLayer
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.model_io import load_json, save_json


class FrontEndLayer(Protocol):
    """A conv or pool layer of any network: its output's shape, which the next layer reads."""

    @property
    def out_height(self) -> int: ...

    @property
    def out_width(self) -> int: ...

    @property
    def channel_count(self) -> int: ...


class ArrayFrontEndLayer(ArrayNetworkLayer[A], FrontEndLayer, Protocol[A]):
    """A numpy or Rust conv or pool layer."""

    @property
    def size(self) -> int: ...


class ArrayConvLayer(ArrayFrontEndLayer[A], Protocol[A]):
    """A numpy or Rust conv layer: its kernels as W, (channel_count, fan_in)."""

    W: A
    b: A

    @property
    def fan_in(self) -> int: ...


LayerT = TypeVar("LayerT", bound=FrontEndLayer)


def build_conv_front_end(
    input_height: int,
    input_width: int,
    conv_specs: Sequence[ConvSpec | PoolSpec],
    make_conv: Callable[[ConvSpec, Any, int, int, int], LayerT],
    make_pool: Callable[[PoolSpec, Any, int, int, int], LayerT],
    input_layer: object = None,
) -> list[LayerT]:
    """
    Chains a conv front end through conv_specs, for the pure-Python conv network and (through
    build_conv_array_network_layers) the numpy and Rust ones. The first layer reads the
    single-channel image; each later one the previous layer's out_height x out_width x
    channel_count output.

    make_conv / make_pool(spec, previous, height, width, channels) build one layer. previous is the
    preceding layer (input_layer for the first), which the pure-Python layers wire nodes to and the
    array layers ignore. Every layer must expose out_height/out_width/channel_count.
    """
    assert any(isinstance(spec, ConvSpec) for spec in conv_specs), "conv_specs must contain at least one ConvSpec"

    layers: list[LayerT] = []
    previous = input_layer
    height, width, channels = input_height, input_width, 1
    for spec in conv_specs:
        if isinstance(spec, PoolSpec):
            layer = make_pool(spec, previous, height, width, channels)
        else:
            layer = make_conv(spec, previous, height, width, channels)
        layers.append(layer)
        previous = layer
        height, width, channels = layer.out_height, layer.out_width, layer.channel_count
    return layers


def spec_to_json(spec: ConvSpec | PoolSpec) -> dict[str, Any]:
    return {"type": "pool" if isinstance(spec, PoolSpec) else "conv", **asdict(spec)}


def spec_from_json(spec: dict[str, Any]) -> ConvSpec | PoolSpec:
    fields = {key: value for key, value in spec.items() if key != "type"}
    return PoolSpec(**fields) if spec["type"] == "pool" else ConvSpec(**fields)


def build_conv_array_network_layers(
    input_height: int,
    input_width: int,
    conv_specs: Sequence[ConvSpec | PoolSpec],
    dense_layer_sizes: list[int],
    class_count: int,
    conv_cls: Callable[..., ArrayFrontEndLayer[A]],
    pool_cls: Callable[..., ArrayFrontEndLayer[A]],
    dense_cls: Callable[[int, int], WeightedArrayLayer[A]],
) -> tuple[list[ArrayFrontEndLayer[A]], list[WeightedArrayLayer[A]], WeightedArrayLayer[A]]:
    """
    The numpy or Rust conv network's layers: the front end, the dense hidden layers and the output
    layer. The dense tail's fan-in is the front end's flattened output.
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

    dense_layers: list[WeightedArrayLayer[A]] = []
    previous_size = conv_layers[-1].size
    for size in dense_layer_sizes:
        dense_layers.append(dense_cls(size, previous_size))
        previous_size = size

    return conv_layers, dense_layers, dense_cls(class_count, previous_size)


class ConvNetworkShape(Protocol):
    """The constructor arguments every conv network keeps, which its save envelope records."""

    input_height: int
    input_width: int
    conv_specs: list[ConvSpec | PoolSpec]
    dense_layer_sizes: list[int]
    class_count: int


class ConvArrayNetwork(ConvNetworkShape, Protocol):
    def snapshot(self) -> Sequence[tuple[Any, ...]]: ...


def save_conv_model_json(path: str, network: ConvNetworkShape, snapshot: list[Any]) -> None:
    """
    The save envelope shared by every conv network: the constructor arguments, which a flat
    layer_sizes list (save_model_json, save_array_model_json) can't express, and snapshot, already
    in JSON form. The pure-Python network's snapshot is per-kernel and per-node lists; the array
    networks' is save_conv_array_model_json's, so their files don't load into each other.
    """
    save_json(
        path,
        {
            "input_height": network.input_height,
            "input_width": network.input_width,
            "conv_layers": [spec_to_json(spec) for spec in network.conv_specs],
            "dense_layer_sizes": network.dense_layer_sizes,
            "class_count": network.class_count,
            "snapshot": snapshot,
        },
    )


def save_conv_array_model_json(path: str, network: ConvArrayNetwork) -> None:
    """
    save_conv_model_json for the numpy and Rust conv networks: one (W, b) entry per layer (an
    empty one for a pool layer). Both backends' arrays have .tolist(), so a file saved by either
    loads into the other.
    """
    snapshot = [[entry[0].tolist(), entry[1].tolist()] if entry else [] for entry in network.snapshot()]
    save_conv_model_json(path, network, snapshot)


class _RestorableNetwork(Protocol):
    def restore(self, snapshot: Any) -> None: ...


NetworkT = TypeVar("NetworkT", bound=_RestorableNetwork)


def load_conv_model_json(cls: Callable[..., NetworkT], path: str) -> NetworkT:
    """The load-side counterpart to save_conv_model_json: builds cls from the envelope and
    restores its snapshot, which every conv network's restore() accepts in its JSON form."""
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
