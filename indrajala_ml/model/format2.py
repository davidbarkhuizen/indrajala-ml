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

- implementation is "python", "numpy" or "rust", and shape "multiclass" or "single_output".
- preset holds a preset's class and constructor arguments, from which its load rebuilds it. A
  Sequential network's file has none.
- input is {"dimension": d} or {"height": h, "width": w, "channels": c}, with "input_bounds" in
  pure Python. class_count is the output layer's size.
- layers holds each spec's fields under its kind: "dense", "conv", "pool", "batch_norm" (the
  batch-norm workplan, Format 2) or "residual", {"kind": "residual", "body": [...]} with the body's
  entries (the residual-connections workplan, stage 5), and "patches", "position", "layer_norm",
  "attention" or "token_mean" (the layer-norm and attention workplan, stage 5). A ReLU conv entry
  leaves out its activation, as before ConvSpec had one; a linear conv entry has "activation":
  "linear". A dense entry has "bias": true only for an affine layer (a residual block's, or a
  token-wise embedding); a token-wise dense layer's entry is a dense one.
- weights is the network's snapshot() as lists, one entry per layer, a residual block's flattened
  into its fork, body and add (layer_specs.expand_specs). On numpy and Rust, per layer: [W, b] (an
  affine layer's too); [W] for a linear (bias-free) layer; [gamma, beta, running_mean, running_var]
  for a batch-norm layer; [P] for a position; [gamma, beta] for a layer norm; [Wq, bq, Wk, bk, Wv,
  bv, Wo, bo] for attention; [] for a pool layer, a fork, an add, patches or a token mean. In pure
  Python, per node or kernel: [weights, bias]; [weights] for a linear one; [[gamma], beta,
  running_mean, running_var] per batch-norm channel; [weights] per position row (a token's); [[gamma],
  beta] per layer-norm feature; [weights, bias] per attention row, Wq's, then Wk's, Wv's and Wo's.
- optimizer_state holds the step count t and, per layer (expanded, as weights), the rule's state or
  null: momentum's velocity, Adam's m and v. On numpy and Rust, per parameter (velocity_W,
  velocity_b; velocity_W alone for a linear layer; velocity_gamma, velocity_beta for batch norm and
  layer norm; velocity_P; velocity_Wq, velocity_bq, ... for attention). In pure Python, per node,
  kernel, channel or row (velocity_weights, velocity_bias; no bias entries for a linear one or a
  position row, and a batch-norm channel's or layer-norm feature's gamma is its one weight and its
  beta its bias).
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
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Protocol, cast

from indrajala_ml.model.checkpoint import Checkpoint, OptimizerState
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.layer_specs import (
    Add,
    Attention,
    BatchNorm,
    Dense,
    ExpandedSpec,
    Fork,
    InputShape,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    TokenMean,
    expand_specs,
)
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import generator_state, set_generator_state

FORMAT = 2
PYTHON = "python"
ENSEMBLE = "ensemble"
# the array backends' names (array_backend.py), which this module mustn't import
IMPLEMENTATIONS = (PYTHON, "numpy", "rust")
SHAPES = ("multiclass", "single_output")

_RULES: dict[str, type[UpdateRule]] = {"sgd": SGD, "momentum": Momentum, "adam": Adam, "weight_decay": WeightDecay}
# the layer-norm and attention workplan's specs by kind (stage 5); a token-wise dense layer's entry
# is a "dense" one
_TOKEN_SPECS: dict[str, type[Patches | Position | LayerNorm | Attention | TokenMean]] = {
    "patches": Patches,
    "position": Position,
    "layer_norm": LayerNorm,
    "attention": Attention,
    "token_mean": TokenMean,
}
_TOKEN_KINDS = {cls: kind for kind, cls in _TOKEN_SPECS.items()}


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


def _lists(value: Any) -> Any:
    # arrays through tolist(), tuples as lists: the JSON form
    to_list = getattr(value, "tolist", None)
    if to_list is not None:
        return to_list()
    if isinstance(value, (list, tuple)):
        return [_lists(item) for item in cast("list[Any] | tuple[Any, ...]", value)]
    return value


def layer_to_json(spec: LayerSpec) -> dict[str, Any]:
    if isinstance(spec, Residual):
        return {"kind": "residual", "body": [layer_to_json(layer) for layer in spec.body]}
    if isinstance(spec, ConvSpec):
        # a ReLU conv layer's entry as before ConvSpec had an activation, so those files don't change
        fields = asdict(spec)
        if spec.activation == "relu":
            del fields["activation"]
        return {"kind": "conv", **fields}
    if isinstance(spec, BatchNorm):
        # an entry without ghost groups as before BatchNorm had a group_size, so those files don't
        # change; load_network's BatchNorm takes the default
        fields = asdict(spec)
        if spec.group_size is None:
            del fields["group_size"]
        return {"kind": "batch_norm", **fields}
    if isinstance(spec, Dense):
        # an entry without an affine bias as before Dense had one (the residual-connections
        # workplan, D4), so those files don't change; load_network's Dense takes the default
        fields = asdict(spec)
        if not spec.bias:
            del fields["bias"]
        return {"kind": "dense", **fields}
    if isinstance(spec, PoolSpec):
        return {"kind": "pool", **asdict(spec)}
    return {"kind": _TOKEN_KINDS[type(spec)], **asdict(spec)}


def layer_from_json(spec: dict[str, Any]) -> LayerSpec:
    fields = {key: value for key, value in spec.items() if key != "kind"}
    match spec["kind"]:
        case "dense":
            return Dense(**fields)
        case "conv":
            return ConvSpec(**fields)
        case "pool":
            return PoolSpec(**fields)
        case "batch_norm":
            return BatchNorm(**fields)
        case "residual":
            return Residual(tuple(layer_from_json(layer) for layer in spec["body"]))
        case kind if kind in _TOKEN_SPECS:
            return _TOKEN_SPECS[kind](**fields)
        case kind:
            raise ValueError(f"unknown layer kind {kind!r}")


def rule_to_json(rule: UpdateRule) -> dict[str, Any]:
    name = next(name for name, cls in _RULES.items() if isinstance(rule, cls))
    return {"rule": name, **asdict(rule)}


def rule_from_json(rule: dict[str, Any]) -> UpdateRule:
    name = rule["rule"]
    if name not in _RULES:
        raise ValueError(f"unknown update rule {name!r}")
    return _RULES[name](**{key: value for key, value in rule.items() if key != "rule"})


def _input_to_json(input_shape: InputShape, input_bounds: list[tuple[float, float]] | None) -> dict[str, Any]:
    if len(input_shape) == 1:
        shape: dict[str, Any] = {"dimension": input_shape[0]}
    else:
        height, width, channels = input_shape
        shape = {"height": height, "width": width, "channels": channels}
    if input_bounds is not None:
        shape["input_bounds"] = input_bounds
    return shape


def _input_shape_from_json(shape: dict[str, Any]) -> InputShape:
    if "dimension" in shape:
        return (shape["dimension"],)
    return (shape["height"], shape["width"], shape["channels"])


def _state_names(rule: UpdateRule) -> tuple[str, ...]:
    # the rule's state per parameter, in the optimizers' order
    match rule:
        case Momentum():
            return ("velocity",)
        case Adam():
            return ("m", "v")
        case SGD() | WeightDecay():
            return ()


def _parameter_names(spec: ExpandedSpec) -> tuple[str, ...]:
    # an array layer's parameters, in its parameters() order
    match spec:
        case BatchNorm() | LayerNorm():
            return ("gamma", "beta")
        case Position():
            return ("P",)
        case Attention():
            return ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo")
        case Fork() | Add() | PoolSpec() | Patches() | TokenMean():
            return ()
        case Dense() if spec.bias:
            return ("W", "b")
        case Dense() | ConvSpec() if spec.activation == "linear":
            return ("W",)
        case _:
            return ("W", "b")


def _optimizer_state_to_json(
    rule: UpdateRule, python: bool, state: OptimizerState[Any], specs: Sequence[LayerSpec]
) -> dict[str, Any]:
    names = _state_names(rule)
    expanded = expand_specs(specs)
    layers: list[Any] = [None] * len(expanded)
    for index, layer_state in state.layers.items():
        if python:
            # per weight set: (a list per name, shaped as its weights; the bias's value per name,
            # none for a weight set without a bias)
            layers[index] = [
                {
                    **{f"{name}_weights": list(weight_state[i]) for i, name in enumerate(names)},
                    **{f"{name}_bias": value for name, value in zip(names, bias_state)},
                }
                for weight_state, bias_state in layer_state
            ]
        else:
            # per parameter, an array shaped as it per name
            layers[index] = {
                f"{name}_{parameter}": _lists(layer_state[p * len(names) + i])
                for p, parameter in enumerate(_parameter_names(expanded[index]))
                for i, name in enumerate(names)
            }
    return {"t": state.t, "layers": layers}


def _optimizer_state_from_json(
    rule: UpdateRule, python: bool, state: dict[str, Any], specs: Sequence[LayerSpec]
) -> OptimizerState[Any]:
    names = _state_names(rule)
    expanded = expand_specs(specs)
    layers: dict[int, Any] = {}
    for index, layer_state in enumerate(state["layers"]):
        if layer_state is None:
            continue
        if python:
            layers[index] = [
                (
                    [weight_set[f"{name}_weights"] for name in names],
                    [weight_set[f"{name}_bias"] for name in names if f"{name}_bias" in weight_set],
                )
                for weight_set in layer_state
            ]
        else:
            layers[index] = [
                layer_state[f"{name}_{parameter}"] for parameter in _parameter_names(expanded[index]) for name in names
            ]
    return OptimizerState(state["t"], layers)


def _rng_to_json(state: dict[str, Any]) -> dict[str, Any]:
    # numpy's bit_generator.state, flattened, its 128-bit integers as hex strings
    return {
        "bit_generator": state["bit_generator"],
        "state": hex(state["state"]["state"]),
        "inc": hex(state["state"]["inc"]),
        "has_uint32": state["has_uint32"],
        "uinteger": state["uinteger"],
    }


def _rng_from_json(rng: dict[str, Any]) -> dict[str, Any]:
    return {
        "bit_generator": rng["bit_generator"],
        "state": {"state": int(rng["state"], 16), "inc": int(rng["inc"], 16)},
        "has_uint32": rng["has_uint32"],
        "uinteger": rng["uinteger"],
    }


def _argument_to_json(name: str, value: Any) -> Any:
    if name == "conv_specs":
        return [layer_to_json(spec) for spec in value]
    return _lists(value)


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
    state["input"] = _input_to_json(network.input_shape, _input_bounds(network))
    state["layers"] = [layer_to_json(spec) for spec in network.layer_specs]
    state["update_rule"] = rule_to_json(rule)
    state["weights"] = _lists(checkpoint.weights)
    state["optimizer_state"] = _optimizer_state_to_json(rule, python, checkpoint.optimizer, network.layer_specs)
    state["rng"] = _rng_to_json(checkpoint.rng)
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
        input_shape=_input_shape_from_json(shape),
        input_bounds=None if bounds is None else _argument_from_json("input_bounds", bounds),
        layers=layers,
        update_rule=rule,
        weights=state["weights"],
        optimizer_state=_optimizer_state_from_json(rule, python, state["optimizer_state"], layers),
        rng=_rng_from_json(state["rng"]) if "rng" in state else None,
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
