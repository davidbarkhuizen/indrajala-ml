"""
Format 2, every network's save file (docs/composable-layers-workplan.md, D4 and Format 2): the
layer specs, the update rule, the weights and the optimizer's state, so a loaded network resumes
training by bits.

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
      "optimizer_state": {"t": 120, "layers": [{"m_W": ..., "v_W": ..., "m_b": ..., "v_b": ...}, ...]}
    }

- implementation is "python", "numpy" or "rust", and shape "multiclass" or "single_output".
- preset holds a preset's class and constructor arguments, from which its load rebuilds it. A
  Sequential network's file has none.
- input is {"dimension": d} or {"height": h, "width": w, "channels": c}, with "input_bounds" in
  pure Python. class_count is the output layer's size.
- weights is the network's snapshot() as lists: per layer [W, b] on numpy and Rust, and a
  [weights, bias] per node or kernel in pure Python; [] for a pool layer.
- optimizer_state holds the step count t and, per layer, the rule's state or null: momentum's
  velocity, Adam's m and v, per W and b on numpy and Rust (velocity_W, velocity_b), per node or
  kernel in pure Python (velocity_weights, velocity_bias).

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

from indrajala_ml.model.checkpoint import OptimizerState
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.layer_specs import Dense, InputShape, LayerSpec
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay

FORMAT = 2
PYTHON = "python"
ENSEMBLE = "ensemble"
# the array backends' names (array_backend.py), which this module mustn't import
IMPLEMENTATIONS = (PYTHON, "numpy", "rust")
SHAPES = ("multiclass", "single_output")

_RULES: dict[str, type[UpdateRule]] = {"sgd": SGD, "momentum": Momentum, "adam": Adam, "weight_decay": WeightDecay}


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
    kind = "dense" if isinstance(spec, Dense) else "conv" if isinstance(spec, ConvSpec) else "pool"
    return {"kind": kind, **asdict(spec)}


def layer_from_json(spec: dict[str, Any]) -> LayerSpec:
    fields = {key: value for key, value in spec.items() if key != "kind"}
    match spec["kind"]:
        case "dense":
            return Dense(**fields)
        case "conv":
            return ConvSpec(**fields)
        case "pool":
            return PoolSpec(**fields)
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


def _optimizer_state_to_json(
    rule: UpdateRule, python: bool, state: OptimizerState[Any], layer_count: int
) -> dict[str, Any]:
    names = _state_names(rule)
    layers: list[Any] = [None] * layer_count
    for index, layer_state in state.layers.items():
        if python:
            # per weight set: (a list per name, shaped as its weights; the bias's value per name)
            layers[index] = [
                {
                    **{f"{name}_weights": list(weight_state[i]) for i, name in enumerate(names)},
                    **{f"{name}_bias": bias_state[i] for i, name in enumerate(names)},
                }
                for weight_state, bias_state in layer_state
            ]
        else:
            # the arrays shaped as W, one per name, then as many shaped as b
            layers[index] = {
                **{f"{name}_W": _lists(layer_state[i]) for i, name in enumerate(names)},
                **{f"{name}_b": _lists(layer_state[len(names) + i]) for i, name in enumerate(names)},
            }
    return {"t": state.t, "layers": layers}


def _optimizer_state_from_json(rule: UpdateRule, python: bool, state: dict[str, Any]) -> OptimizerState[Any]:
    names = _state_names(rule)
    layers: dict[int, Any] = {}
    for index, layer_state in enumerate(state["layers"]):
        if layer_state is None:
            continue
        if python:
            layers[index] = [
                ([weight_set[f"{name}_weights"] for name in names], [weight_set[f"{name}_bias"] for name in names])
                for weight_set in layer_state
            ]
        else:
            layers[index] = [layer_state[f"{name}_W"] for name in names] + [layer_state[f"{name}_b"] for name in names]
    return OptimizerState(state["t"], layers)


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
    state["weights"] = _lists(network.snapshot())
    state["optimizer_state"] = _optimizer_state_to_json(
        rule, python, network.optimizer.state(), len(network.layer_specs)
    )
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
    shape = state["input"]
    bounds = shape.get("input_bounds")
    return NetworkFile(
        implementation=state["implementation"],
        shape=state["shape"],
        preset=state.get("preset"),
        input_shape=_input_shape_from_json(shape),
        input_bounds=None if bounds is None else _argument_from_json("input_bounds", bounds),
        layers=[layer_from_json(spec) for spec in state["layers"]],
        update_rule=rule,
        weights=state["weights"],
        optimizer_state=_optimizer_state_from_json(rule, python, state["optimizer_state"]),
    )


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
    """network's weights and optimizer state from file, after check_loadable."""
    check_loadable(network, file)
    network.restore(file.weights)
    network.optimizer.load_state(file.optimizer_state)


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
