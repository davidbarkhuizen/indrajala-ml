from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any, Protocol

from indrajala_ml.model.array_protocols import ArrayNetworkLayer, BackendArray
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


class ArrayFrontEndLayer[A: BackendArray](ArrayNetworkLayer[A], FrontEndLayer, Protocol):
    """A numpy or Rust conv or pool layer."""

    @property
    def size(self) -> int: ...


def spec_to_json(spec: ConvSpec | PoolSpec) -> dict[str, Any]:
    return {"type": "pool" if isinstance(spec, PoolSpec) else "conv", **asdict(spec)}


def spec_from_json(spec: dict[str, Any]) -> ConvSpec | PoolSpec:
    fields = {key: value for key, value in spec.items() if key != "type"}
    return PoolSpec(**fields) if spec["type"] == "pool" else ConvSpec(**fields)


class ConvNetworkShape(Protocol):
    """The constructor arguments every conv network keeps, which its save envelope records."""

    input_height: int
    input_width: int
    conv_specs: list[ConvSpec | PoolSpec]
    dense_layer_sizes: list[int]
    class_count: int


class ConvArrayNetwork(ConvNetworkShape, Protocol):
    def snapshot(self) -> Sequence[tuple[Any, ...]]: ...


def save_conv_model_json(
    path: str, network: ConvNetworkShape, snapshot: list[Any], extra: dict[str, Any] | None = None
) -> None:
    """
    The save envelope shared by every conv network: the constructor arguments, which a flat
    layer_sizes list (save_model_json, save_array_model_json) can't express, and snapshot, already
    in JSON form, with extra (a sibling's hyperparameters) merged in, as save_array_model_json
    does. The pure-Python network's snapshot is per-kernel and per-node lists; the array networks'
    is save_conv_array_model_json's, so their files don't load into each other.
    """
    state: dict[str, Any] = {
        "input_height": network.input_height,
        "input_width": network.input_width,
        "conv_layers": [spec_to_json(spec) for spec in network.conv_specs],
        "dense_layer_sizes": network.dense_layer_sizes,
        "class_count": network.class_count,
        "snapshot": snapshot,
    }
    if extra:
        state.update(extra)
    save_json(path, state)


def save_conv_array_model_json(path: str, network: ConvArrayNetwork, extra: dict[str, Any] | None = None) -> None:
    """
    save_conv_model_json for the numpy and Rust conv networks: one (W, b) entry per layer (an
    empty one for a pool layer). Both backends' arrays have .tolist(), so a file saved by either
    loads into the other.
    """
    snapshot = [[entry[0].tolist(), entry[1].tolist()] if entry else [] for entry in network.snapshot()]
    save_conv_model_json(path, network, snapshot, extra)


class _RestorableNetwork(Protocol):
    def restore(self, snapshot: Any) -> None: ...


def load_conv_model_json[NetworkT: _RestorableNetwork](
    cls: Callable[..., NetworkT],
    path: str,
    extra_init_kwargs: Callable[[dict[str, Any]], dict[str, Any]] = lambda _state: {},
) -> NetworkT:
    """The load-side counterpart to save_conv_model_json: builds cls from the envelope, with
    extra_init_kwargs(state) (a sibling's hyperparameters, as ArrayNetworkBase._extra_init_kwargs),
    and restores its snapshot, which every conv network's restore() accepts in its JSON form."""
    state = load_json(path)
    network = cls(
        input_height=state["input_height"],
        input_width=state["input_width"],
        conv_specs=[spec_from_json(spec) for spec in state["conv_layers"]],
        dense_layer_sizes=state["dense_layer_sizes"],
        class_count=state["class_count"],
        **extra_init_kwargs(state),
    )
    network.restore(state["snapshot"])
    return network
