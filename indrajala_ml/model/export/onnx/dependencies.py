"""
The exporter's dependencies (the export-and-hub workplan, D7): onnx, the [onnx] extra's, and
onnxruntime, for the export's check only (D13). Each is imported when first used, so the package
imports without them, and a missing one is named with the extra that brings it.
"""

from __future__ import annotations

import importlib
from typing import Any

_ONNX_EXTRA = "pip install 'indrajala-ml[onnx]'"


def onnx() -> Any:
    """The onnx package, or an ImportError naming the [onnx] extra."""
    try:
        return importlib.import_module("onnx")
    except ImportError as error:
        raise ImportError(f"ONNX export needs onnx, the [onnx] extra's: {_ONNX_EXTRA}") from error


def onnx_reference() -> Any:
    """onnx's reference evaluator module (onnx.reference): the spec's own semantics, every dtype."""
    onnx()
    return importlib.import_module("onnx.reference")


def onnxruntime() -> Any | None:
    """onnxruntime, or None where it isn't installed: the check then runs the reference evaluator alone."""
    try:
        return importlib.import_module("onnxruntime")
    except ImportError:
        return None
