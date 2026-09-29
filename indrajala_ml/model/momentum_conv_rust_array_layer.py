"""
MomentumConvRustArrayLayer, kept as a name for ConvRustArrayLayer so its imports keep working: the
update lives in the network's optimizer since stage 3 of docs/composable-layers-workplan.md (the
Momentum rule, optimizers.py), and stage 6 deletes this module.
"""

from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer

MomentumConvRustArrayLayer = ConvRustArrayLayer
