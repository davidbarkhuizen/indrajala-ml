"""
The encoder-decoder network on numpy (the cross-attention workplan, stage 4b, D2, D14):
EncoderDecoderNetwork's passes (encoder_decoder_network.py) over numpy's array operations and
layers.
"""

from __future__ import annotations

from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.networks.encoder_decoder_network import EncoderDecoderNetwork
from indrajala_ml.model.networks.numpy.numpy_array_network_base import NumpyArrayNetworkBase


class EncoderDecoderArrayNetwork(EncoderDecoderNetwork[FloatArray], NumpyArrayNetworkBase):
    """EncoderDecoderNetwork on the numpy backend."""
