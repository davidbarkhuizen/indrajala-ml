from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import Self

from indrajala_ml.model.conv_front_end import load_conv_model_json, save_conv_model_json
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.momentum_conv_layer import make_momentum_conv_layer_cls
from indrajala_ml.model.momentum_layer import make_momentum_layer_cls


class MomentumConvMultiClassBackpropClassifierNetwork(ConvMultiClassBackpropClassifierNetwork):
    """
    A momentum sibling of ConvMultiClassBackpropClassifierNetwork: momentum as Goyal et al. 2017's
    eq. (9) in every conv kernel (make_momentum_kernel_cls) and dense node
    (make_momentum_node_cls). The hooks are set as instance attributes in __init__ before the conv
    network's __init__ builds the layers, as MomentumBackpropClassifierNetwork does. Pool layers
    have no weights, so they are unchanged.

    momentum is required and saved in the envelope; the velocities are not saved, as for every
    momentum network (snapshot() covers the weights and biases).
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        conv_specs: Sequence[ConvSpec | PoolSpec],
        dense_layer_sizes: list[int],
        class_count: int,
        momentum: float,
    ) -> None:
        self.momentum = momentum
        self.conv_layer_cls = make_momentum_conv_layer_cls(momentum)
        dense_layer_cls = make_momentum_layer_cls(momentum)
        self.hidden_layer_cls = dense_layer_cls
        self.output_layer_cls = dense_layer_cls
        super().__init__(input_height, input_width, conv_specs, dense_layer_sizes, class_count)

    def save(self, path: str) -> None:
        save_conv_model_json(path, self, self.snapshot(), extra={"momentum": self.momentum})

    @classmethod
    def load(cls, path: str) -> Self:
        return load_conv_model_json(cls, path, extra_init_kwargs=lambda state: {"momentum": state["momentum"]})
