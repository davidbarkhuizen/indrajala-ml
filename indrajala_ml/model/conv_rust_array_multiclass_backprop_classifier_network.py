from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.array_network_shapes import ArrayConvShape
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer
from indrajala_ml.model.max_pool_rust_array_layer import MaxPoolRustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.prepared_dataset import PreparedDataset


class ConvRustArrayMultiClassBackpropClassifierNetwork(
    ArrayConvShape[pa.Array], RustArrayMultiClassBackpropClassifierNetwork
):
    """
    ArrayConvShape on the Rust backend: ConvRustArrayLayers and MaxPoolRustArrayLayers in front of
    RustArrayLayers. Only classify_rows differs from the numpy network.
    """

    conv_layer_cls = ConvRustArrayLayer
    pool_layer_cls = MaxPoolRustArrayLayer

    def classify_rows(self, prepared: PreparedDataset) -> list[int]:
        # row by row, not batched: a batched pass measured a tie for one conv layer and slower
        # with pooling or a second conv layer (docs/optimizations/rejected.md), since the batched conv
        # forward still writes cols it doesn't need
        return [self.classify_row(prepared, index) for index in range(len(prepared))]
