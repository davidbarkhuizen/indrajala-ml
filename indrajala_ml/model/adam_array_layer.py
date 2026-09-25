from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer


class AdamArrayLayer(ArrayLayer):
    """
    Adam (Kingma & Ba, 2014) over whole arrays: the update of adam_layer.make_adam_node_cls, with
    bias-corrected running estimates of each parameter's gradient mean (m) and uncentered variance
    (v). beta1/beta2/epsilon are required here; the defaults live on the network.
    """

    hyperparameters = ("beta1", "beta2", "epsilon")

    def __init__(self, size: int, input_size: int, beta1: float, beta2: float, epsilon: float) -> None:
        super().__init__(size, input_size)
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._m_W = np.zeros((size, input_size))
        self._v_W = np.zeros((size, input_size))
        self._m_b = np.zeros(size)
        self._v_b = np.zeros(size)
        self._t = 0

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self._t += 1
        bias_correction1 = 1 - self._beta1**self._t
        bias_correction2 = 1 - self._beta2**self._t

        g_W = self._grad_W / batch_size
        self._m_W = self._beta1 * self._m_W + (1 - self._beta1) * g_W
        self._v_W = self._beta2 * self._v_W + (1 - self._beta2) * g_W * g_W
        m_hat_W = self._m_W / bias_correction1
        v_hat_W = self._v_W / bias_correction2
        self.W -= learning_rate * m_hat_W / (np.sqrt(v_hat_W) + self._epsilon)

        g_b = self._grad_b / batch_size
        self._m_b = self._beta1 * self._m_b + (1 - self._beta1) * g_b
        self._v_b = self._beta2 * self._v_b + (1 - self._beta2) * g_b * g_b
        m_hat_b = self._m_b / bias_correction1
        v_hat_b = self._v_b / bias_correction2
        self.b -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self._epsilon)

        self._reset_gradient_accum()
