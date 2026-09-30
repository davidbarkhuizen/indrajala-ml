from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from indrajala_ml.model.array_protocols import ArrayNetworkLayer, BackendArray
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.max_pool_layer import PoolSpec


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


def spec_from_json(spec: dict[str, Any]) -> ConvSpec | PoolSpec:
    # a legacy conv envelope's spec; format 2 has its own (format2.layer_from_json)
    fields = {key: value for key, value in spec.items() if key != "type"}
    return PoolSpec(**fields) if spec["type"] == "pool" else ConvSpec(**fields)


class _RestorableNetwork(Protocol):
    def restore(self, snapshot: Any) -> None: ...


def load_conv_model_state[NetworkT: _RestorableNetwork](
    cls: Callable[..., NetworkT], state: dict[str, Any], hyperparameters: dict[str, Any]
) -> NetworkT:
    """
    Loads every conv network's legacy envelope, the constructor arguments and snapshot, already
    read from JSON: builds cls from them and hyperparameters (a sibling's, read from beside them),
    and restores its snapshot, which every conv network's restore() accepts in its JSON form. The
    pure-Python network's snapshot is per kernel and per node, the array networks' one (W, b) per
    layer ([] for a pool layer), so their files don't load into each other; numpy's and Rust's do.
    """
    network = cls(
        input_height=state["input_height"],
        input_width=state["input_width"],
        conv_specs=[spec_from_json(spec) for spec in state["conv_layers"]],
        dense_layer_sizes=state["dense_layer_sizes"],
        class_count=state["class_count"],
        **hyperparameters,
    )
    network.restore(state["snapshot"])
    return network
