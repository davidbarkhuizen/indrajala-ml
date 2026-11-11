"""
An export's check (the export-and-hub workplan, D13): the model passes onnx's checker and strict
shape inference, and on fixed seeded inputs gives our numpy forward's outputs, within 1e-12 for
float64 (not bits: the summation order differs) and 1e-5 for float32, under onnx's reference
evaluator (the spec's own semantics) and onnxruntime's CPU provider where it is installed. A row
whose argmax differs is counted and reported, as a near tie.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from indrajala_ml.model.export.onnx.dependencies import onnx, onnx_reference, onnxruntime
from indrajala_ml.model.export.onnx.graph import Dtype
from indrajala_ml.model.persistence.format2 import ENSEMBLE

TOLERANCE: dict[Dtype, float] = {"float64": 1e-12, "float32": 1e-5}
ROWS = 16
SEED = 0


class ExportCheckError(AssertionError):
    """An export that fails its check: the checker, shape inference, or our forward's outputs."""


@dataclass(frozen=True)
class CheckReport:
    """
    Each evaluator's largest difference from our numpy forward, and its rows whose argmax differs
    from ours; the evaluators that ran (onnxruntime's only where it is installed).
    """

    tolerance: float
    rows: int
    max_difference: dict[str, float]
    argmax_disagreements: dict[str, int]


def numpy_forward(state: dict[str, Any], x: Any) -> Any:
    """Our numpy forward pass of the network state describes, over a batch x (N, d) of float64 rows."""
    from indrajala_ml.model.persistence.load_network import network_from_state

    network = network_from_state(_as_numpy(state))
    if state["shape"] == ENSEMBLE:
        return np.concatenate([np.asarray(c.forward_batch(x)) for c in network.classifiers], axis=1)
    return np.asarray(network.forward_batch(x))


def _as_numpy(state: dict[str, Any]) -> dict[str, Any]:
    # the same document on the numpy backend: a Rust network's check runs against its numpy twin
    if state["shape"] == ENSEMBLE:
        return {**state, "implementation": "numpy", "classifiers": [_as_numpy(c) for c in state["classifiers"]]}
    return {**state, "implementation": "numpy"}


def inputs(dimension: int, rows: int = ROWS, seed: int = SEED) -> Any:
    """The check's fixed inputs: rows uniform in [0, 1), float64."""
    return np.random.default_rng(seed).random((rows, dimension))


def check_export(model: Any, state: dict[str, Any], dtype: Dtype) -> CheckReport:
    """model checked against state's numpy forward (D13); raises ExportCheckError on a failure."""
    checker = onnx()
    try:
        checker.checker.check_model(model, full_check=True)
        checker.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    except Exception as error:  # onnx raises its own ValidationError and InferenceError
        raise ExportCheckError(f"the exported model fails onnx's checks: {error}") from error
    graph_input = model.graph.input[0]
    dims: list[int] = [dim.dim_value for dim in graph_input.type.tensor_type.shape.dim[1:]]
    output_dims: list[int] = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim[1:]]
    first = state["classifiers"][0] if state["shape"] == ENSEMBLE else state
    if graph_input.name == "ids":
        # token ids over the whole vocabulary, so a vocabulary with a separator and a pad has segments
        ids = np.random.default_rng(SEED).integers(0, first["layers"][0]["vocabulary"], (ROWS, *dims))
        x = ids.astype(np.float64)
        feed = {"ids": ids.astype(np.int64)}
    else:
        x = inputs(math.prod(dims))
        # our row, reshaped to the graph's input (an image's NCHW is our row's own order)
        feed = {"x": x.reshape(len(x), *dims).astype(dtype)}
    expected: np.ndarray[Any, np.dtype[np.float64]] = numpy_forward(state, x).reshape(len(x), *output_dims)
    runs: dict[str, Any] = {
        "reference": lambda: onnx_reference().ReferenceEvaluator(model).run(None, feed)[0],
    }
    runtime = onnxruntime()
    if runtime is not None:
        session = runtime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        runs["onnxruntime"] = lambda: session.run(None, feed)[0]
    tolerance = TOLERANCE[dtype]
    differences: dict[str, float] = {}
    disagreements: dict[str, int] = {}
    for name, run in runs.items():
        actual: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(run(), dtype=np.float64)
        if actual.shape != expected.shape:
            raise ExportCheckError(f"{name}: outputs of shape {actual.shape}, ours {expected.shape}")
        differences[name] = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=tolerance, atol=tolerance):
            raise ExportCheckError(
                f"{name}: outputs differ from our numpy forward's by up to {differences[name]:.3g}, "
                f"beyond {dtype}'s {tolerance:g}"
            )
        width = expected.shape[-1]
        chosen: list[int] = actual.argmax(axis=-1).tolist()
        ours: list[int] = expected.argmax(axis=-1).tolist()
        disagreements[name] = sum(a != b for a, b in zip(chosen, ours, strict=True)) if width > 1 else 0
    return CheckReport(tolerance, len(x), differences, disagreements)
