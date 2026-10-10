"""
The JSON codecs of format 2 (format2.py): layer specs, update rules, a network's input, the
optimizer's state and the generator's state, each to its file entry and back.

Backend-free: pure Python imports it, and arrays leave as lists through their tolist().
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, cast

from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.persistence.checkpoint import OptimizerState
from indrajala_ml.model.specs.layer_specs import (
    Add,
    Attention,
    BatchNorm,
    Dense,
    Dropout,
    Embedding,
    ExpandedSpec,
    Fork,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    TokenMean,
    expand_specs,
)
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay

_RULES: dict[str, type[UpdateRule]] = {"sgd": SGD, "momentum": Momentum, "adam": Adam, "weight_decay": WeightDecay}
# the layer-norm and attention workplan's specs by kind (stage 5), and the sequence task workplan's
# Embedding (stage 2) and the attention-dropout workplan's Dropout (D7); a token-wise dense layer's
# entry, an output layer's included, is a "dense" one
_TOKEN_SPECS: dict[str, type[Patches | Embedding | Position | LayerNorm | Attention | Dropout | TokenMean]] = {
    "patches": Patches,
    "embedding": Embedding,
    "position": Position,
    "layer_norm": LayerNorm,
    "attention": Attention,
    "dropout": Dropout,
    "token_mean": TokenMean,
}
_TOKEN_KINDS = {cls: kind for kind, cls in _TOKEN_SPECS.items()}


def lists(value: Any) -> Any:
    # arrays through tolist(), tuples as lists: the JSON form
    to_list = getattr(value, "tolist", None)
    if to_list is not None:
        return to_list()
    if isinstance(value, (list, tuple)):
        return [lists(item) for item in cast("list[Any] | tuple[Any, ...]", value)]
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
    if isinstance(spec, Attention):
        # heads, key_size, causal and dropout only when not the default (the multi-head attention
        # workplan; the sequence task workplan, D7; the attention-dropout workplan, D7), so a
        # one-head unmasked entry without dropout is as before Attention had them and older
        # checkouts load it
        default = Attention()
        fields = {key: value for key, value in asdict(spec).items() if value != getattr(default, key)}
        return {"kind": "attention", **fields}
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


def input_to_json(input_shape: InputShape, input_bounds: list[tuple[float, float]] | None) -> dict[str, Any]:
    if len(input_shape) == 1:
        shape: dict[str, Any] = {"dimension": input_shape[0]}
    else:
        height, width, channels = input_shape
        shape = {"height": height, "width": width, "channels": channels}
    if input_bounds is not None:
        shape["input_bounds"] = input_bounds
    return shape


def input_shape_from_json(shape: dict[str, Any]) -> InputShape:
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
        case Embedding():
            return ("E",)
        case Attention():
            return ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo")
        case Fork() | Add() | PoolSpec() | Patches() | Dropout() | TokenMean():
            return ()
        case Dense() if spec.bias:
            return ("W", "b")
        case Dense() | ConvSpec() if spec.activation == "linear":
            return ("W",)
        case _:
            return ("W", "b")


def optimizer_state_to_json(
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
                f"{name}_{parameter}": lists(layer_state[p * len(names) + i])
                for p, parameter in enumerate(_parameter_names(expanded[index]))
                for i, name in enumerate(names)
            }
    return {"t": state.t, "layers": layers}


def optimizer_state_from_json(
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


def rng_to_json(state: dict[str, Any]) -> dict[str, Any]:
    # numpy's bit_generator.state, flattened, its 128-bit integers as hex strings
    return {
        "bit_generator": state["bit_generator"],
        "state": hex(state["state"]["state"]),
        "inc": hex(state["state"]["inc"]),
        "has_uint32": state["has_uint32"],
        "uinteger": state["uinteger"],
    }


def rng_from_json(rng: dict[str, Any]) -> dict[str, Any]:
    return {
        "bit_generator": rng["bit_generator"],
        "state": {"state": int(rng["state"], 16), "inc": int(rng["inc"], 16)},
        "has_uint32": rng["has_uint32"],
        "uinteger": rng["uinteger"],
    }
