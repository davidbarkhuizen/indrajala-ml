"""
The window-recurrent step's ONNX export (model/export/onnx/window.py, the export-and-hub workplan,
stage 6): token and array windows, one, two and three factors, crossed, continuous and linear
codes, FiLM, poolings before the state, and branches (by factor, shared, merged), on numpy and
Rust, each looped by the host over runs of windows from its own initial_state and next_state, give
our network's outputs and states and the same code index at every step (D13, 4), under both
evaluators. The export is float64 only (D10) and its metadata carries the warning.
"""

import dataclasses
from typing import Any

import numpy as np
import onnx
import pytest

from indrajala_ml.model.export.onnx.check import TOLERANCE, ExportCheckError, check_window
from indrajala_ml.model.export.onnx.exporter import FLOAT64_WARNING, document, export_onnx
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentArrayNetwork,
    WindowRecurrentRustArrayNetwork,
)
from indrajala_ml.model.specs.layer_specs import StateInput
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import WindowRecurrence
from tests.model.networks import test_window_recurrent_array_network as tokens
from tests.model.networks import test_window_recurrent_array_window as arrays
from tests.model.persistence.test_format2 import _trained

BACKENDS: dict[str, Any] = {"numpy": WindowRecurrentArrayNetwork, "rust": WindowRecurrentRustArrayNetwork}


def _linear(recurrence: WindowRecurrence) -> WindowRecurrence:
    factors = tuple(
        dataclasses.replace(
            f, codes=None if f.codes is None else dataclasses.replace(f.codes, parameterisation="linear")
        )
        for f in recurrence.factors
    )
    return dataclasses.replace(recurrence, factors=factors)


def _film(window: list[Any]) -> list[Any]:
    return [StateInput(scale=True) if isinstance(spec, StateInput) else spec for spec in window]


# (input shape, window, recurrence) by name
ARCHITECTURES: dict[str, tuple[Any, list[Any], WindowRecurrence]] = {
    **{f"tokens, {name}": ((tokens.K,), tokens.WINDOW, r) for name, r in tokens.RECURRENCES.items()},
    **{
        f"tokens, linear codes, {name}": ((tokens.K,), tokens.WINDOW, _linear(r))
        for name, r in tokens.RECURRENCES.items()
    },
    "tokens, FiLM": ((tokens.K,), _film(tokens.WINDOW), tokens.RECURRENCES["two factors"]),
    **{
        f"tokens, branches, {name}": ((tokens.K,), tokens.BRANCHED[name], tokens.BRANCHED_RECURRENCES[name])
        for name in tokens.BRANCHED
    },
    "array, one factor": (arrays.SHAPE, arrays.WINDOW, arrays.RECURRENCE),
    "array, two factors before the state": (arrays.SHAPE, arrays.WINDOW, arrays.FACTORED),
    "array, continuous, crossed": (arrays.SHAPE, arrays.WINDOW, arrays.CONTINUOUS),
}


def _network(name: str, backend: str) -> Any:
    input_shape, window, recurrence = ARCHITECTURES[name]
    return BACKENDS[backend].randomized(input_shape, window, recurrence, Adam(), seed=11)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_window_architecture_steps_as_the_network(architecture: str, backend: str):
    export = export_onnx(_network(architecture, backend))

    assert export.dtype == "float64"
    assert export.report is not None
    assert set(export.report.max_difference) == {"reference", "onnxruntime"}
    assert max(export.report.max_difference.values()) <= TOLERANCE["float64"]


@pytest.mark.parametrize(
    "name",
    [
        "WindowRecurrentArrayNetwork",
        "WindowRecurrentRustArrayNetwork",
        "ContinuousWindowWindowRecurrentArrayNetwork",
        "ContinuousWindowWindowRecurrentRustArrayNetwork",
    ],
)
def test_a_trained_window_fixture_steps_as_the_network(name: str):
    export = export_onnx(_trained(name))

    assert export.report is not None
    assert max(export.report.max_difference.values()) <= TOLERANCE["float64"]


def test_the_step_reads_a_state_and_a_window_and_gives_the_machines_parts():
    model = export_onnx(_network("tokens, two factors", "numpy")).model

    def dims(value: Any) -> list[Any]:
        return [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]

    assert {i.name: dims(i) for i in model.graph.input} == {"state": ["batch", 5], "window": ["batch", tokens.K]}
    assert {o.name: dims(o) for o in model.graph.output} == {
        "outputs": ["batch", tokens.K, 3],
        "next_state": ["batch", 5],
        "codes": ["batch", 2],
        "initial_state": [1, 5],
    }
    assert next(i for i in model.graph.input if i.name == "window").type.tensor_type.elem_type == onnx.TensorProto.INT64


def test_a_window_export_is_float64_only_and_says_so():
    network = _network("tokens, one factor", "numpy")
    with pytest.raises(ValueError, match="float64 only"):
        export_onnx(network, dtype="float32")

    model = export_onnx(network).model
    assert {p.key: p.value for p in model.metadata_props}["warning"] == FLOAT64_WARNING
    assert {i.data_type for i in model.graph.initializer if i.data_type == onnx.TensorProto.FLOAT} == set()


def test_the_check_catches_a_changed_code():
    # the step exported, then one code moved: the loop's codes no longer match, and the check says so
    network = _network("tokens, one factor", "numpy")
    model = export_onnx(network).model
    state = document(network)
    codes = next(i for i, names in enumerate(state["weights"]) if len(names) == 1 and np.shape(names[0]) == (4, 3))
    state["weights"][codes] = [np.asarray(state["weights"][codes][0])[::-1].copy()]

    with pytest.raises(ExportCheckError, match="codes differ|differs"):
        check_window(model, state)
