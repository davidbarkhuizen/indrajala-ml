"""
What both network bases share of saving and loading (format2.py) and of checkpoints
(checkpoint.py): ArrayNetworkBase (numpy and Rust) and BackpropNetworkBase (pure Python) differ
only in the implementation name a file is checked against, _format2_implementation.

Backend-free, so the pure-Python networks import it too.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self, cast

from indrajala_ml.model.persistence.checkpoint import Checkpoint
from indrajala_ml.model.persistence.format2 import (
    Format2Network,
    NetworkFile,
    check_kind,
    is_format2,
    network_from_json,
    network_to_json,
    preset_init_kwargs,
    restore_file,
)
from indrajala_ml.model.persistence.model_io import load_json, save_json
from indrajala_ml.pcg64 import generator_state, set_generator_state


class Format2Persistence[W, S]:
    """
    A mixin for a network that is a format2.Format2Network: W is its snapshot() type and S its
    optimizer state's per-layer type.
    """

    format2_shape: ClassVar[str]

    @classmethod
    def _format2_implementation(cls) -> str:
        # the implementation a file must have to load into cls: "python", or the array backend's
        # name (a numpy file loads into Rust and a Rust file into numpy)
        raise NotImplementedError

    def snapshot(self) -> W:
        raise NotImplementedError

    def restore(self, snapshot: W) -> None:
        raise NotImplementedError

    def _network(self) -> Format2Network:
        # every subclass is one (both network bases)
        return cast("Format2Network", self)

    def save(self, path: str) -> None:
        # format 2 (format2.py): the specs, rule, weights, optimizer state and generator state
        save_json(path, network_to_json(self._network()))

    @classmethod
    def load(cls, path: str) -> Self:
        # a format-2 file, or the class's legacy envelope, which loads with fresh optimizer state
        state = load_json(path)
        return cls.from_format2(state) if is_format2(state) else cls._load_legacy(state)

    @classmethod
    def from_format2(cls, state: dict[str, Any]) -> Self:
        # on numpy and Rust, restore converts the file's nested lists through the backend
        file = network_from_json(state)
        check_kind(cls, cls._format2_implementation(), cls.format2_shape, file)
        network = cls._from_file(file)
        restore_file(network._network(), file)
        return network

    @classmethod
    def _from_file(cls, file: NetworkFile) -> Self:
        # a preset, from the file's preset arguments; a Sequential network builds from its specs
        preset = cast("type[Format2Network]", cls)
        return cast("Self", cast("Any", cls)(**preset_init_kwargs(preset, file)))

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        raise ValueError(f"{cls.__name__} saves in format 2 only; this file has format {state.get('format')!r}")

    def checkpoint(self) -> Checkpoint[W, S]:
        # the weights, the optimizer's state and the generator's (checkpoint.py)
        network = self._network()
        return Checkpoint(self.snapshot(), network.optimizer.state(), generator_state(network.rng))

    def restore_checkpoint(self, checkpoint: Checkpoint[W, S]) -> None:
        # as restore: on numpy and Rust, this backend's arrays or nested lists
        self.restore(checkpoint.weights)
        network = self._network()
        network.optimizer.load_state(checkpoint.optimizer)
        # in place, so the dropout layers holding the generator draw on from the checkpoint
        set_generator_state(network.rng, checkpoint.rng)
