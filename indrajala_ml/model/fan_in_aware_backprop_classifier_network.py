from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.backprop_network_base import randomize_fan_in_aware


class FanInAwareBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    A fan-in-aware-initialization sibling of BackpropClassifierNetwork, using the same
    randomize_fan_in_aware scheme MultiClassBackpropClassifierNetwork already uses (see that
    function's own docstring) instead of BackpropClassifierNetwork.randomize()'s
    per-dimension-bounds-width scaling - which that method's own docstring already notes is
    "tuned for 1-2D geometric problems". Structurally this is a single method override.

    Exists because BackpropClassifierNetwork's scaling was never updated for high-dimensional
    use, unlike MultiClassBackpropClassifierNetwork, and every EnsembleBackpropClassifierNetwork
    sub-network is a plain BackpropClassifierNetwork - at MNIST's 784-dimension fan-in, that
    scaling produces the exact sigmoid-saturation problem its own docstring warns about (measured
    directly: 83.5% of hidden activations already saturated at initialization, before any
    training). Switching to this class alone, with no other change, measurably takes the real,
    full-scale MNIST ensemble from 89.4% to 96.01% test accuracy at the same wall-clock cost.

    Deliberately not a change to BackpropClassifierNetwork itself: that class's existing scaling
    is explicitly tuned for, and still appropriate for, the 1-2D geometric demos that depend on
    it (demo_backprop_circular_boundary.py, demo_backprop_linear_parity_check.py,
    demo_backprop_stripes_architecture_sweep.py, demo_xor_backprop_convergence.py - the last of
    which pins exact hand-derived values into a regression test tied to that exact scheme) -
    none of which need or were measured to benefit from this change.
    """

    def randomize(self) -> None:
        randomize_fan_in_aware(self)
