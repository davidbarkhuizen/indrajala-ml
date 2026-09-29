"""
L2ArrayLayer, kept as a name for ArrayLayer so its imports keep working: the update lives in the
network's optimizer since stage 1 of docs/composable-layers-workplan.md (the WeightDecay rule,
optimizers.py), and stage 6 deletes this module.
"""

from indrajala_ml.model.array_layer import ArrayLayer

L2ArrayLayer = ArrayLayer
