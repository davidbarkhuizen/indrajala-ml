from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.networks.rust.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.array_network_shapes import ArrayConvShape
from indrajala_ml.data.prepared_dataset import PreparedDataset


class ConvRustArrayMultiClassBackpropClassifierNetwork(
    ArrayConvShape[pa.Array], RustArrayMultiClassBackpropClassifierNetwork
):
    """
    ArrayConvShape on the Rust backend: ConvRustArrayLayers and MaxPoolRustArrayLayers in front of
    RustArrayLayers. Only classify_rows differs from the numpy network.
    """

    def classify_rows(self, prepared: PreparedDataset) -> list[int]:
        # row by row, not batched: a batched pass measured a tie for one conv layer and slower
        # with pooling or a second conv layer, since the batched conv
        # forward still writes cols it doesn't need
        return [self.classify_row(prepared, index) for index in range(len(prepared))]
