"""
MomentumRustArrayLayer, kept as a name for RustArrayLayer so its imports keep working: the update
lives in the network's optimizer since stage 1 of docs/composable-layers-workplan.md (the Momentum
rule, optimizers.py), and stage 6 deletes this module.
"""

from indrajala_ml.model.rust_array_layer import RustArrayLayer

MomentumRustArrayLayer = RustArrayLayer
