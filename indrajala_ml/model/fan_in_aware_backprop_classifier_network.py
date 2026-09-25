from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.backprop_network_base import randomize_fan_in_aware


class FanInAwareBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    BackpropClassifierNetwork with randomize_fan_in_aware in place of its bounds-width scaling,
    which is tuned for 1-2D geometric problems. At MNIST's 784-pixel fan-in that scaling leaves
    83.5% of hidden activations saturated at initialization; this class alone took the full MNIST
    ensemble from 89.4% to 96.01% test accuracy at the same wall-clock cost.

    BackpropClassifierNetwork keeps its scaling for the 1-2D demos that depend on it
    (demo_backprop_circular_boundary.py, demo_backprop_linear_parity_check.py,
    demo_backprop_stripes_architecture_sweep.py, and demo_xor_backprop_convergence.py, whose
    hand-derived values a regression test pins).
    """

    def randomize(self) -> None:
        randomize_fan_in_aware(self)
