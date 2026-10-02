"""
The methods a hidden layer kind shares on either backend (the DRY rerun workplan, D3), as mixins
over the backend's array type A: the output-delta refusals, a delta that is the next layer's
downstream, and a parameter-free layer's no-op gradients. A layer lists them ahead of its other
bases, so they replace a base's sigmoid formulas; a method it writes itself replaces theirs.
"""

from __future__ import annotations

from typing import Any


class Hidden[A]:
    """A layer that is never a network's last, so it has no output delta."""

    def compute_output_delta(self, reference: A) -> None:
        raise NotImplementedError(f"a {type(self).__name__} is hidden, inside a network")

    def compute_output_delta_batch(self, reference_batch: A) -> None:
        raise NotImplementedError(f"a {type(self).__name__} is hidden, inside a network")


class DeltaIsDownstream[A]:
    """
    A layer whose delta is the next layer's downstream, with no activation derivative to multiply
    in: the identity's is 1, as is the add's on each path, and max is the identity on its winner.
    """

    delta: A
    delta_batch: A

    def compute_hidden_delta(self, next_layer: Any) -> None:
        self.delta = next_layer.downstream()

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        self.delta_batch = next_layer.downstream_batch()


class ParameterFree[A]:
    """A layer with no parameters: nothing to accumulate, and with no W the optimizer skips it."""

    def accumulate_gradient(self, input_activation: A) -> None:
        pass

    def accumulate_gradient_batch(self, input_activation_batch: A) -> None:
        pass
