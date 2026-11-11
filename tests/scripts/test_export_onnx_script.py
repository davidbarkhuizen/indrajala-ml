"""scripts/export/export_onnx.py: a saved network's checked ONNX export, its report printed."""

import sys
from pathlib import Path

import onnx
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.format3 import save_directory
from indrajala_ml.model.specs.layer_specs import Dense
from indrajala_ml.model.specs.update_rules import Adam

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "export"))
import export_onnx


@pytest.mark.parametrize("form", ["file", "directory"])
def test_the_script_exports_a_saved_network(form: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    network = SequentialArrayNetwork(
        (4,), [Dense(5), Dense(3, output=True, activation="softmax", loss="cross_entropy")], Adam(), backend=NUMPY
    )
    network.randomize()
    path = tmp_path / "model.json"
    network.save(str(path))
    if form == "directory":
        path = save_directory(network, tmp_path / "model")

    assert export_onnx.main([str(path), str(tmp_path / "model.onnx"), "--dtype", "float64"]) == 0

    model = onnx.load(str(tmp_path / "model.onnx"))  # pyright: ignore[reportUnknownMemberType]
    assert [output.name for output in model.graph.output] == ["probabilities"]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert "onnxruntime: largest difference" in capsys.readouterr().err
