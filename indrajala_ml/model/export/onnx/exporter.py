"""
export_onnx (the export-and-hub workplan, D9-D14): a network's inference graph, built from its
saved description, never from the live network's objects, so a numpy network, a Rust one, a
format-2 file and a format-3 directory (a Hub download) export alike.

The graphs (D12), stages 3-5's shapes:
- multiclass and single_output: x (N, d), or an image's (N, C, H, W) (our row's channel-major
  order), to "probabilities" (N, classes), or "outputs" for a linear (given-loss) output layer;
- an ensemble (one-vs-rest): x (N, d) to each sub-network's probability, concatenated, (N, classes);
- sequence: ids (N, T), int64, to each token's "probabilities" (N, T, classes); a network whose
  first layer is an Embedding reads ids whatever its shape. Its segments come from the ids in the
  graph, as the network derives them (D12 amended: no segments input).

Precision (D10): float32 by default; float64 when asked. A window-recurrent network exports
float64 always (stage 6). Every export is checked (D13, check.py) unless check=False.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indrajala_ml.model.export.onnx.check import CheckReport, check_export
from indrajala_ml.model.export.onnx.dependencies import onnx
from indrajala_ml.model.export.onnx.graph import Dtype, Graph
from indrajala_ml.model.export.onnx.layers import Tensor, layers, onnx_dims
from indrajala_ml.model.persistence.format2 import ENSEMBLE, WINDOW_RECURRENT, description_to_json, is_format2
from indrajala_ml.model.persistence.format2_json import (
    input_shape_from_json,
    layer_from_json,
    layer_parameter_names,
    recurrence_from_json,
)
from indrajala_ml.model.persistence.format3 import float64_array, is_directory, read_directory, tensor_names
from indrajala_ml.model.persistence.model_io import load_json
from indrajala_ml.model.specs.layer_specs import Dense
from indrajala_ml.model.specs.spec_shapes import InputShape

# the shapes each stage exports, by the stage that adds them
_LATER_SHAPES = {WINDOW_RECURRENT: 6, "encoder_decoder": 7}


@dataclass(frozen=True)
class Export:
    """An export: the model (onnx.ModelProto), its dtype, and its check's report (None unchecked)."""

    model: Any
    dtype: Dtype
    report: CheckReport | None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_bytes(self.model.SerializeToString())
        return path


def document(source: Any) -> dict[str, Any]:
    """
    source's format-2 document, its weights numpy float64 arrays: a numpy or Rust network or
    ensemble, a format-2 file's path or a format-3 directory's.
    """
    if isinstance(source, str | Path):
        state = read_directory(source) if is_directory(source) else load_json(str(source))
        if not is_format2(state):
            raise ValueError(f"{source} is neither a format-2 file nor a format-3 directory")
        return state
    classifiers = getattr(source, "classifiers", None)
    if classifiers is not None:
        return {
            "format": 2,
            "implementation": classifiers[0].implementation,
            "shape": ENSEMBLE,
            "classifiers": [document(classifier) for classifier in classifiers],
        }
    state = description_to_json(source)
    state["weights"] = [[float64_array(array) for array in entry] for entry in source.snapshot()]
    layers = [layer_from_json(spec) for spec in [*state.get("encoder", ()), *state["layers"]]]
    recurrence = recurrence_from_json(state["recurrence"]) if "recurrence" in state else None
    state["optimizer_state"] = {"t": 0, "layers": [None] * len(layer_parameter_names(layers, recurrence))}
    return state


def _input_shape(state: dict[str, Any]) -> InputShape:
    if state["shape"] in _LATER_SHAPES:
        raise NotImplementedError(
            f"ONNX export of a {state['shape']} network comes in the export-and-hub workplan's stage "
            f"{_LATER_SHAPES[state['shape']]}"
        )
    return input_shape_from_json(state["input"])


def _network(g: Graph, state: dict[str, Any], x: Tensor, prefix: str) -> Tensor:
    # one flat network's graph over x, its initializers named by format 3's tensor names
    if state["implementation"] == "python":
        raise ValueError(
            "a pure-Python network holds its weights per node and doesn't export (D2): export its numpy twin"
        )
    specs = [layer_from_json(spec) for spec in state["layers"]]
    names = [[prefix + name for name in entry] for entry in tensor_names(specs, None)]
    return layers(g, specs, _input_shape(state), x, names, state["weights"])


def _output_name(state: dict[str, Any]) -> str:
    last = layer_from_json(state["layers"][-1])
    linear = isinstance(last, Dense) and last.output and last.loss == "given"
    return "outputs" if linear else "probabilities"


def export_onnx(
    source: Any, path: str | Path | None = None, *, dtype: Dtype | None = None, check: bool = True
) -> Export:
    """
    source's inference graph (D12), source a network, an ensemble, a format-2 file or a format-3
    directory, in dtype (D10: float32 unless asked), written to path if given, and checked against
    our numpy forward unless check is False (D13).
    """
    state = document(source)
    dtype = dtype or "float32"
    g = Graph(dtype)
    first = state["classifiers"][0] if state["shape"] == ENSEMBLE else state
    dims = onnx_dims(_input_shape(first))
    # a network that embeds reads token ids, int64; every other its row's floats
    ids = bool(first["layers"]) and first["layers"][0]["kind"] == "embedding"
    x = Tensor("ids" if ids else "x", tuple(dims))
    if state["shape"] == ENSEMBLE:
        classifiers = state["classifiers"]
        outputs = [_network(g, classifier, x, f"classifiers.{c}.").name for c, classifier in enumerate(classifiers)]
        output = g.rename(g.op("Concat", outputs, axis=1), "probabilities")
        output_dims: tuple[int, ...] = (len(classifiers),)
    else:
        result = _network(g, state, x, "")
        output = g.rename(result.name, _output_name(state))
        output_dims = result.dims
    model = g.model(
        [g.value_info(x.name, ["batch", *dims], onnx().TensorProto.INT64 if ids else None)],
        [g.value_info(output, ["batch", *output_dims])],
        f"indrajala-ml {state['shape']}",
        {
            "library": "indrajala-ml",
            "shape": state["shape"],
            "dtype": dtype,
            "description": json.dumps(_description(state)),
        },
    )
    report = check_export(model, state, dtype) if check else None
    export = Export(model, dtype, report)
    if path is not None:
        export.save(path)
    return export


def _description(state: dict[str, Any]) -> dict[str, Any]:
    # the document without its arrays: what the model's metadata records of the network
    if state["shape"] == ENSEMBLE:
        return {**state, "classifiers": [_description(classifier) for classifier in state["classifiers"]]}
    return {key: value for key, value in state.items() if key not in ("weights", "optimizer_state", "rng")}
