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
from indrajala_ml.model.specs.layer_specs import Embedding

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


WINDOWS = 6


def check_window(model: Any, state: dict[str, Any]) -> CheckReport:
    """
    A window-recurrent step checked over runs of windows looped by the host (D13, 4): from the
    initial_state output, each window's step fed its own next_state, against our numpy network's
    forward over the whole run. Each step's outputs and next state within 1e-12, and every factor's
    code index the same at every step; raises ExportCheckError otherwise.
    """
    from indrajala_ml.model.export.onnx.window import (
        CODES,
        INITIAL_STATE,
        NEXT_STATE,
        OUTPUTS,
        STATE,
        WINDOW,
    )
    from indrajala_ml.model.persistence.load_network import network_from_state

    checker = onnx()
    try:
        checker.checker.check_model(model, full_check=True)
        checker.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    except Exception as error:  # onnx raises its own ValidationError and InferenceError
        raise ExportCheckError(f"the exported model fails onnx's checks: {error}") from error
    network = network_from_state(_as_numpy(state))
    window_input = next(i for i in model.graph.input if i.name == WINDOW)
    dims: list[int] = [dim.dim_value for dim in window_input.type.tensor_type.shape.dim[1:]]
    columns = math.prod(dims)
    rng = np.random.default_rng(SEED)
    ids = window_input.type.tensor_type.elem_type == checker.TensorProto.INT64
    if ids:
        vocabulary = state["layers"][0]["vocabulary"]
        rows = rng.integers(0, vocabulary, (ROWS, WINDOWS * columns)).astype(np.float64)
    else:
        rows = rng.random((ROWS, WINDOWS * columns))
    outputs, states, _cells = network.window_states(rows)
    expected_outputs = np.asarray(outputs).reshape(ROWS, WINDOWS, -1)
    _outputs, _states, _kept, indices = network.per_layer_windows(rows, states[0], False, True)
    expected_codes = np.stack(
        [
            np.stack([np.full(ROWS, -1) if i is None else np.asarray(i) for i in window_indices], axis=1)
            for window_indices in indices
        ]
    )
    sessions: dict[str, Any] = {"reference": onnx_reference().ReferenceEvaluator(model)}
    runtime = onnxruntime()
    if runtime is not None:
        sessions["onnxruntime"] = runtime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
    tolerance = TOLERANCE["float64"]
    differences: dict[str, float] = {}
    for name, session in sessions.items():
        empty = np.zeros((ROWS, *dims), dtype=np.int64 if ids else np.float64)
        first = session.run([INITIAL_STATE], {STATE: np.zeros((ROWS, network.recurrence.state_size)), WINDOW: empty})[0]
        state_now = np.repeat(np.asarray(first), ROWS, axis=0)
        if not np.allclose(state_now, np.asarray(states[0]), rtol=tolerance, atol=tolerance):
            raise ExportCheckError(f"{name}: the initial state differs from our network's")
        largest = 0.0
        for w in range(WINDOWS):
            window_rows = rows[:, w * columns : (w + 1) * columns].reshape(ROWS, *dims)
            feed = {STATE: state_now, WINDOW: window_rows.astype(np.int64) if ids else window_rows}
            output, next_state, codes = session.run([OUTPUTS, NEXT_STATE, CODES], feed)
            if not np.array_equal(np.asarray(codes), expected_codes[w]):
                raise ExportCheckError(f"{name}: window {w}'s codes differ from our network's: the machine changed")
            for got, want, what in (
                (np.asarray(output).reshape(ROWS, -1), expected_outputs[:, w], "outputs"),
                (np.asarray(next_state), np.asarray(states[w + 1]), "next state"),
            ):
                difference = float(np.max(np.abs(got - want)))
                largest = max(largest, difference)
                if not np.allclose(got, want, rtol=tolerance, atol=tolerance):
                    raise ExportCheckError(
                        f"{name}: window {w}'s {what} differ from our network's by up to {difference:.3g}"
                    )
            state_now = np.asarray(next_state)
        differences[name] = largest
    return CheckReport(tolerance, ROWS * WINDOWS, differences, {name: 0 for name in sessions})


def _segments_and_positions(ids: Any, separator: int, pad: int, separator_opens: bool) -> tuple[Any, Any]:
    # a batch's segments and positions (segment_context's), the batch without either id one segment a row
    from indrajala_ml.model.layers.numpy.segments import segment_context

    context = segment_context(ids.astype(np.float64), separator, pad, separator_opens)
    if context.segments is None or context.positions is None:
        rows, tokens = ids.shape
        return np.ones((rows, tokens), dtype=np.int64), np.tile(np.arange(tokens, dtype=np.int64), (rows, 1))
    return np.asarray(context.segments, dtype=np.int64), np.asarray(context.positions, dtype=np.int64)


def check_encoder_decoder(models: dict[str, Any], state: dict[str, Any], dtype: Dtype) -> CheckReport:
    """
    An encoder-decoder's three models (D13): the full forward against our numpy network's, then
    every target token decoded through the encoder and decoder_step, teacher-forced from an empty
    cache, each step's probabilities the full forward's at that token; raises ExportCheckError.
    """
    from indrajala_ml.model.export.onnx import encoder_decoder as graphs
    from indrajala_ml.model.persistence.format2_json import (
        encoder_decoder_from_json,
        encoder_decoder_input_from_json,
    )
    from indrajala_ml.model.persistence.load_network import network_from_state

    checker = onnx()
    for name, model in models.items():
        try:
            checker.checker.check_model(model, full_check=True)
            checker.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        except Exception as error:  # onnx raises its own ValidationError and InferenceError
            raise ExportCheckError(f"the exported {name} fails onnx's checks: {error}") from error
    spec = encoder_decoder_from_json(state)
    source_tokens, target_tokens = encoder_decoder_input_from_json(state["input"])
    source_spec, target_spec = spec.encoder[0], spec.decoder[0]
    assert isinstance(source_spec, Embedding) and isinstance(target_spec, Embedding), "both stacks embed (D14)"
    vocabulary, target_vocabulary = source_spec.vocabulary, target_spec.vocabulary
    rng = np.random.default_rng(SEED)
    source = rng.integers(0, vocabulary, (ROWS, source_tokens))
    target = rng.integers(0, target_vocabulary, (ROWS, target_tokens))
    network = network_from_state(_as_numpy(state))
    expected = np.array(
        [network.predict_probabilities(tuple(float(i) for i in (*s, *t))) for s, t in zip(source, target, strict=True)]
    )
    segments, positions = _segments_and_positions(target, target_spec.separator, target_spec.pad, True)
    caches = graphs.self_attention_count(spec)
    tolerance = TOLERANCE[dtype]
    float_type = np.dtype(dtype)
    differences: dict[str, float] = {}
    for evaluator, run in _evaluators(models).items():
        full = np.asarray(run("model", {graphs.SOURCE: source, graphs.TARGET: target})[0], dtype=np.float64)
        largest = float(np.max(np.abs(full - expected)))
        if not np.allclose(full, expected, rtol=tolerance, atol=tolerance):
            raise ExportCheckError(
                f"{evaluator}: the full forward differs from our numpy forward's by up to {largest:.3g}"
            )
        memory, memory_segments = run("encoder", {graphs.SOURCE: source})
        # an empty cache: each past input at length 0
        past = _empty_cache(models["decoder_step"], float_type)
        for t in range(target_tokens):
            feed = {
                graphs.IDS: target[:, t : t + 1],
                graphs.SEGMENTS: segments[:, t : t + 1],
                graphs.POSITIONS: positions[:, t : t + 1],
                graphs.COLUMN: np.array([t], dtype=np.int64),
                graphs.MEMORY: memory,
                graphs.MEMORY_SEGMENTS: memory_segments,
                **past,
            }
            outputs = run("decoder_step", feed)
            names = [o.name for o in models["decoder_step"].graph.output]
            produced = dict(zip(names, outputs, strict=True))
            step = np.asarray(produced[graphs.PROBABILITIES], dtype=np.float64)[:, 0]
            difference = float(np.max(np.abs(step - full[:, t])))
            largest = max(largest, difference)
            if not np.allclose(step, full[:, t], rtol=tolerance, atol=tolerance):
                raise ExportCheckError(
                    f"{evaluator}: decode step {t}'s probabilities differ from the full forward's by up to {difference:.3g}"
                )
            past = {graphs.PAST_SEGMENTS: produced[graphs.PRESENT_SEGMENTS]}
            for i in range(caches):
                past[f"past_keys.{i}"] = produced[f"present_keys.{i}"]
                past[f"past_values.{i}"] = produced[f"present_values.{i}"]
        differences[evaluator] = largest
    return CheckReport(tolerance, ROWS * target_tokens, differences, {name: 0 for name in differences})


def _empty_cache(model: Any, float_type: Any) -> dict[str, Any]:
    # each past input at length 0: (N, g, 0, d_k) keys and values, (N, 0) segments
    feed: dict[str, Any] = {}
    for value in model.graph.input:
        if not value.name.startswith("past"):
            continue
        dims = [
            d.dim_value if d.dim_value else (ROWS if d.dim_param == "batch" else 0)
            for d in value.type.tensor_type.shape.dim
        ]
        integer = value.type.tensor_type.elem_type == onnx().TensorProto.INT64
        feed[value.name] = np.zeros(dims, dtype=np.int64 if integer else float_type)
    return feed


def _evaluators(models: dict[str, Any]) -> dict[str, Any]:
    # each evaluator as run(model name, feed) -> its outputs in graph order
    reference: dict[str, Any] = {name: onnx_reference().ReferenceEvaluator(model) for name, model in models.items()}
    runs: dict[str, Any] = {"reference": _runner(reference)}
    runtime = onnxruntime()
    if runtime is not None:
        sessions: dict[str, Any] = {
            name: runtime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            for name, model in models.items()
        }
        runs["onnxruntime"] = _runner(sessions)
    return runs


def _runner(sessions: dict[str, Any]) -> Any:
    # run(model name, feed): that model's session's outputs, in graph order
    def run(name: str, feed: dict[str, Any]) -> Any:
        return sessions[name].run(None, feed)

    return run
