"""
Format 2, every network's save file (the composable-layers workplan, D4 and Format 2): the layer
specs, the update rule, the weights, the optimizer's state and the generator's state, so a loaded
network resumes training by bits, its dropout masks included.

    {
      "format": 2,
      "implementation": "numpy",
      "shape": "multiclass",
      "preset": {"class": "AdamVectorizedMultiClassBackpropClassifierNetwork",
                 "arguments": {"layer_sizes": [30], "dimension": 64, "class_count": 10, "beta1": 0.9, ...}},
      "input": {"dimension": 64},
      "layers": [{"kind": "dense", "size": 30, ...}, {"kind": "dense", "size": 10, "output": true, ...}],
      "update_rule": {"rule": "adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-08},
      "weights": [[W, b], [W, b]],
      "optimizer_state": {"t": 120, "layers": [{"m_W": ..., "v_W": ..., "m_b": ..., "v_b": ...}, ...]},
      "rng": {"bit_generator": "PCG64", "state": "0x...", "inc": "0x...", "has_uint32": 0, "uinteger": 0}
    }

- implementation is "python", "numpy" or "rust", and shape "multiclass", "single_output" or
  "sequence" (a token-wise output layer, the sequence task workplan, stage 3).
- preset holds a preset's class and constructor arguments, from which its load rebuilds it. A
  Sequential network's file has none.
- input is {"dimension": d} or {"height": h, "width": w, "channels": c}, with "input_bounds" in
  pure Python. class_count is the output layer's size.
- layers holds each spec's fields under its kind: "dense", "conv", "pool", "batch_norm" (the
  batch-norm workplan, Format 2) or "residual", {"kind": "residual", "body": [...]} with the body's
  entries (the residual-connections workplan, stage 5), "patches", "position", "layer_norm",
  "attention" or "token_mean" (the layer-norm and attention workplan, stage 5), and "embedding"
  (the sequence task workplan, stage 2). An attention entry has "heads", "key_size" and "causal"
  only when not the default. A ReLU conv entry leaves out its activation, as before ConvSpec had
  one; a linear conv entry has "activation": "linear". A dense entry has "bias": true only for an
  affine layer (a residual block's, or a token-wise embedding); a token-wise dense layer's entry,
  a token-wise output layer's included, is a dense one.
- weights is the network's snapshot() as lists, one entry per layer, a residual block's flattened
  into its fork, body and add (layer_specs.expand_specs). On numpy and Rust, per layer: [W, b] (an
  affine layer's too); [W] for a linear (bias-free) layer; [gamma, beta, running_mean, running_var]
  for a batch-norm layer; [P] for a position; [E], the (vocabulary, size) table, for an embedding;
  [gamma, beta] for a layer norm; [Wq, bq, Wk, bk, Wv, bv, Wo, bo] for attention; [] for a pool
  layer, a fork, an add, patches or a token mean. In pure Python, per node or kernel: [weights,
  bias]; [weights] for a linear one; [[gamma], beta, running_mean, running_var] per batch-norm
  channel; [weights] per position row (a token's) and per embedding row (a token id's); [[gamma],
  beta] per layer-norm feature; [weights, bias] per attention row, Wq's, then Wk's, Wv's and Wo's.
- optimizer_state holds the step count t and, per layer (expanded, as weights), the rule's state or
  null: momentum's velocity, Adam's m and v. On numpy and Rust, per parameter (velocity_W,
  velocity_b; velocity_W alone for a linear layer; velocity_gamma, velocity_beta for batch norm and
  layer norm; velocity_P; velocity_E; velocity_Wq, velocity_bq, ... for attention). In pure Python,
  per node, kernel, channel or row (velocity_weights, velocity_bias; no bias entries for a linear
  one, a position row or an embedding row, and a batch-norm channel's or layer-norm feature's gamma
  is its one weight and its beta its bias).
- rng is the network's generator's state (the RNG generators workplan, D3): numpy's
  bit_generator.state, flattened, with the 128-bit state and inc as hex strings, since JSON readers
  outside Python lose precision on large integers. It's optional: a file saved before it loads with
  an OS-entropy generator (D9). All three implementations hold this state, so a numpy file's
  generator draws on in Rust, and the masks match.

A numpy file loads into Rust and a Rust file into numpy. A pure-Python file loads into pure Python
only, as its weights are per node. A network's load refuses a file whose shape, input, layer specs
or update rule aren't its own. An ensemble's file nests one format-2 file per sub-network:
{"format": 2, "implementation": ..., "shape": "ensemble", "classifiers": [...]}.

Backend-free: pure Python imports it, and arrays leave as lists through their tolist().
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from indrajala_ml.model.persistence.checkpoint import Checkpoint, OptimizerState
from indrajala_ml.model.persistence.format2_json import (
    input_shape_from_json,
    input_to_json,
    layer_from_json,
    layer_to_json,
    lists,
    optimizer_state_from_json,
    optimizer_state_to_json,
    rng_from_json,
    rng_to_json,
    rule_from_json,
    rule_to_json,
)
from indrajala_ml.model.specs.layer_specs import LayerSpec
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import UpdateRule
from indrajala_ml.pcg64 import generator_state, set_generator_state

FORMAT = 2
PYTHON = "python"
ENSEMBLE = "ensemble"
# the array backends' names (array_backend.py), which this module mustn't import
IMPLEMENTATIONS = (PYTHON, "numpy", "rust")
SHAPES = ("multiclass", "single_output", "sequence")


class _Optimizer(Protocol):
    @property
    def rule(self) -> UpdateRule: ...

    def state(self) -> OptimizerState[Any]: ...

    def load_state(self, state: OptimizerState[Any], /) -> None: ...


class Format2Network(Protocol):
    """What the writer and reader need of a network: both network bases provide it."""

    # the preset's constructor arguments beside its hyperparameters, named as the attributes that
    # hold them; None for a Sequential network, which records no preset
    preset_arguments: ClassVar[tuple[str, ...] | None]
    hyperparameters: ClassVar[tuple[str, ...]]
    format2_shape: ClassVar[str]
    input_shape: InputShape
    layer_specs: list[LayerSpec]

    @property
    def implementation(self) -> str: ...

    @property
    def optimizer(self) -> _Optimizer: ...

    @property
    def rng(self) -> Any: ...

    def snapshot(self) -> Any: ...

    def restore(self, snapshot: Any, /) -> None: ...


def is_format2(state: dict[str, Any]) -> bool:
    return state.get("format") == FORMAT


def _argument_to_json(name: str, value: Any) -> Any:
    if name == "conv_specs":
        return [layer_to_json(spec) for spec in value]
    return lists(value)


def _argument_from_json(name: str, value: Any) -> Any:
    if name == "conv_specs":
        return [layer_from_json(spec) for spec in value]
    if name == "input_bounds":
        return [tuple(bound) for bound in value]
    return value


def _input_bounds(network: Format2Network) -> list[tuple[float, float]] | None:
    # a pure-Python network's; the array networks have none
    return getattr(network, "input_bounds", None) if network.implementation == PYTHON else None


def network_to_json(network: Format2Network) -> dict[str, Any]:
    """network as a format-2 file's contents."""
    return checkpoint_to_json(
        network, Checkpoint(network.snapshot(), network.optimizer.state(), generator_state(network.rng))
    )


def checkpoint_to_json(network: Format2Network, checkpoint: Checkpoint[Any, Any]) -> dict[str, Any]:
    """
    A format-2 file's contents for network with checkpoint's weights, optimizer state and generator
    state in place of its own: a checkpoint network took earlier, as a file (run_checkpoint.py).
    """
    python = network.implementation == PYTHON
    rule = network.optimizer.rule
    state: dict[str, Any] = {"format": FORMAT, "implementation": network.implementation, "shape": network.format2_shape}
    if network.preset_arguments is not None:
        state["preset"] = {
            "class": type(network).__name__,
            "arguments": {
                name: _argument_to_json(name, getattr(network, name))
                for name in (*network.preset_arguments, *network.hyperparameters)
            },
        }
    state["input"] = input_to_json(network.input_shape, _input_bounds(network))
    state["layers"] = [layer_to_json(spec) for spec in network.layer_specs]
    state["update_rule"] = rule_to_json(rule)
    state["weights"] = lists(checkpoint.weights)
    state["optimizer_state"] = optimizer_state_to_json(rule, python, checkpoint.optimizer, network.layer_specs)
    state["rng"] = rng_to_json(checkpoint.rng)
    return state


@dataclass(frozen=True)
class NetworkFile:
    """A format-2 file's contents, read back: what the network loading it is checked against."""

    implementation: str
    shape: str
    preset: dict[str, Any] | None
    input_shape: InputShape
    input_bounds: list[tuple[float, float]] | None
    layers: list[LayerSpec]
    update_rule: UpdateRule
    weights: list[Any]
    optimizer_state: OptimizerState[Any]
    # the generator's state as numpy's bit_generator.state; None in a file saved before it had one
    rng: dict[str, Any] | None


def network_from_json(state: dict[str, Any]) -> NetworkFile:
    if not is_format2(state):
        raise ValueError(f"not a format-2 file: format is {state.get('format')!r}")
    if state["shape"] == ENSEMBLE:
        raise ValueError("an ensemble's file, not one network's: load it with an ensemble class or load_network")
    if state["shape"] not in SHAPES:
        raise ValueError(f"unknown shape {state['shape']!r}")
    if state["implementation"] not in IMPLEMENTATIONS:
        raise ValueError(f"unknown implementation {state['implementation']!r}")
    rule = rule_from_json(state["update_rule"])
    python = state["implementation"] == PYTHON
    layers = [layer_from_json(spec) for spec in state["layers"]]
    shape = state["input"]
    bounds = shape.get("input_bounds")
    return NetworkFile(
        implementation=state["implementation"],
        shape=state["shape"],
        preset=state.get("preset"),
        input_shape=input_shape_from_json(shape),
        input_bounds=None if bounds is None else _argument_from_json("input_bounds", bounds),
        layers=layers,
        update_rule=rule,
        weights=state["weights"],
        optimizer_state=optimizer_state_from_json(rule, python, state["optimizer_state"], layers),
        rng=rng_from_json(state["rng"]) if "rng" in state else None,
    )


def file_checkpoint(network: Format2Network, file: NetworkFile) -> Checkpoint[Any, Any]:
    """
    file's weights, optimizer state and generator state as a checkpoint network restores, after
    check_loadable. The weights stay lists, which restore_checkpoint converts.
    """
    check_loadable(network, file)
    if file.rng is None:
        raise ValueError("this file holds no generator state, which a checkpoint needs")
    return Checkpoint(file.weights, file.optimizer_state, file.rng)


def preset_init_kwargs(cls: type[Format2Network], file: NetworkFile) -> dict[str, Any]:
    """The constructor arguments cls, a preset, takes from file's preset."""
    if cls.preset_arguments is None or file.preset is None:
        raise ValueError(
            f"{cls.__name__} can't load a file without a preset (one a Sequential network saved); "
            "load it with load_network or a Sequential class"
        )
    arguments: dict[str, Any] = file.preset["arguments"]
    names = (*cls.preset_arguments, *cls.hyperparameters)
    missing = [name for name in names if name not in arguments]
    if missing:
        raise ValueError(
            f"{cls.__name__} can't load a file saved by {file.preset['class']}: it has no {', '.join(missing)}"
        )
    return {name: _argument_from_json(name, arguments[name]) for name in names}


def _kind_differences(implementation: str, shape: str, file: NetworkFile) -> list[str]:
    differences: list[str] = []
    if (file.implementation == PYTHON) != (implementation == PYTHON):
        differences.append(f"implementation {file.implementation}, not {implementation}")
    if file.shape != shape:
        differences.append(f"shape {file.shape}, not {shape}")
    return differences


def check_kind(cls: type[Any], implementation: str, shape: str, file: NetworkFile) -> None:
    """
    Raises ValueError unless a network of cls's implementation and shape can hold file: checked
    before cls is built from the file, which a file of another shape could fail.
    """
    differences = _kind_differences(implementation, shape, file)
    if differences:
        raise ValueError(f"{cls.__name__} can't load this file, which has " + "; ".join(differences))


def check_loadable(network: Format2Network, file: NetworkFile) -> None:
    """Raises ValueError, naming each difference, unless network is what file describes."""
    name = type(network).__name__
    differences = _kind_differences(network.implementation, network.format2_shape, file)
    if file.input_shape != network.input_shape:
        differences.append(f"input shape {file.input_shape}, not {network.input_shape}")
    if file.input_bounds != _input_bounds(network):
        differences.append(f"input bounds {file.input_bounds}, not {_input_bounds(network)}")
    if file.layers != network.layer_specs:
        differences.append(f"layers {file.layers}, not {network.layer_specs}")
    if file.update_rule != network.optimizer.rule:
        differences.append(f"update rule {file.update_rule}, not {network.optimizer.rule}")
    if differences:
        raise ValueError(f"{name} can't load this file, which has " + "; ".join(differences))


def restore_file(network: Format2Network, file: NetworkFile) -> None:
    """
    network's weights, optimizer state and generator state from file, after check_loadable. A file
    without a generator state leaves network's generator as it is.
    """
    check_loadable(network, file)
    network.restore(file.weights)
    network.optimizer.load_state(file.optimizer_state)
    if file.rng is not None:
        set_generator_state(network.rng, file.rng)


def ensemble_to_json(implementation: str, classifiers: Sequence[Format2Network]) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "implementation": implementation,
        "shape": ENSEMBLE,
        "classifiers": [network_to_json(classifier) for classifier in classifiers],
    }


def ensemble_classifiers(state: dict[str, Any]) -> list[dict[str, Any]]:
    """An ensemble's format-2 file's sub-network files."""
    if state.get("shape") != ENSEMBLE:
        raise ValueError(f"not an ensemble's file: shape is {state.get('shape')!r}")
    return state["classifiers"]
