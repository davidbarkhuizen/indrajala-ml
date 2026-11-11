"""
The ONNX graph under construction (the export-and-hub workplan, D9, D11): nodes, initializers and
unique names, in one dtype, written as opset 23 at IR version 10 (onnxruntime 1.31 reads IR 13 at
most, and onnx 1.23 writes 14 unless told).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from indrajala_ml.model.export.onnx.dependencies import onnx

OPSET = 23
IR_VERSION = 10
Dtype = Literal["float32", "float64"]
DTYPES: tuple[Dtype, ...] = ("float32", "float64")


class Graph:
    """One graph's nodes and initializers, every float in dtype."""

    def __init__(self, dtype: Dtype) -> None:
        if dtype not in DTYPES:
            raise ValueError(f"an export's dtype is float32 or float64; got {dtype!r}")
        self.dtype: Dtype = dtype
        self.numpy_dtype = np.dtype(dtype)
        self.tensor_type = onnx().TensorProto.DOUBLE if dtype == "float64" else onnx().TensorProto.FLOAT
        self.nodes: list[Any] = []
        self.initializers: list[Any] = []
        self._names: set[str] = set()
        self._constants: dict[tuple[str, bytes, tuple[int, ...]], str] = {}

    def fresh(self, base: str) -> str:
        """A name not yet used in the graph: base, or base with a counter."""
        name, counter = base, 1
        while name in self._names:
            name, counter = f"{base}_{counter}", counter + 1
        self._names.add(name)
        return name

    def weight(self, name: str, array: Any) -> str:
        """An initializer named name (a format-3 tensor name, D2), cast to the graph's dtype."""
        values = np.asarray(array, dtype=np.float64).astype(self.numpy_dtype)
        name = self.fresh(name)
        self.initializers.append(onnx().numpy_helper.from_array(values, name))
        return name

    def constant(self, value: Any, base: str = "constant") -> str:
        """A float constant in the graph's dtype, shared by every use of the same value."""
        values = np.asarray(value, dtype=np.float64).astype(self.numpy_dtype)
        key = ("float", values.tobytes(), values.shape)
        if key not in self._constants:
            self._constants[key] = self.fresh(base)
            self.initializers.append(onnx().numpy_helper.from_array(values, self._constants[key]))
        return self._constants[key]

    def int64(self, values: Sequence[int] | int, base: str = "shape") -> str:
        """An int64 constant (a shape, axes, indices), shared as constant is."""
        array = np.asarray(values, dtype=np.int64)
        key = ("int64", array.tobytes(), array.shape)
        if key not in self._constants:
            self._constants[key] = self.fresh(base)
            self.initializers.append(onnx().numpy_helper.from_array(array, self._constants[key]))
        return self._constants[key]

    def op(self, op_type: str, inputs: Sequence[str], base: str | None = None, **attributes: Any) -> str:
        """A node of op_type over inputs, with one output, whose name it returns."""
        (output,) = self.ops(op_type, inputs, 1, base, **attributes)
        return output

    def ops(
        self, op_type: str, inputs: Sequence[str], outputs: int, base: str | None = None, **attributes: Any
    ) -> list[str]:
        """A node of op_type over inputs with outputs outputs, their names."""
        stem = base or op_type.lower()
        names = [self.fresh(stem if outputs == 1 else f"{stem}_{i}") for i in range(outputs)]
        self.nodes.append(
            onnx().helper.make_node(op_type, list(inputs), names, name=self.fresh(f"node_{stem}"), **attributes)
        )
        return names

    def value_info(self, name: str, shape: Sequence[int | str], dtype: Any = None) -> Any:
        """A graph input's or output's type: the graph's dtype unless given, a str a symbolic dimension."""
        return onnx().helper.make_tensor_value_info(name, self.tensor_type if dtype is None else dtype, list(shape))

    def rename(self, name: str, output: str) -> str:
        """name under output, a graph output's name, through an Identity."""
        self._names.add(output)
        self.nodes.append(onnx().helper.make_node("Identity", [name], [output], name=self.fresh(f"node_{output}")))
        return output

    def model(self, inputs: Sequence[Any], outputs: Sequence[Any], name: str, metadata: dict[str, str]) -> Any:
        """The model of this graph, opset 23 at IR version 10 (D11)."""
        helper = onnx().helper
        graph = helper.make_graph(self.nodes, name, list(inputs), list(outputs), self.initializers)
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", OPSET)],
            ir_version=IR_VERSION,
            producer_name="indrajala-ml",
        )
        helper.set_model_props(model, metadata)
        return model
