"""
The encoder-decoder network on Rust (the cross-attention workplan, stage 6b, D2, D14):
EncoderDecoderNetwork's passes (encoder_decoder_network.py) over the Rust backend's array
operations and layers, the decoder's CrossAttention the crate's (CrossAttentionRustArrayLayer).
Its draws are numpy's by bits from the same seed, and it trains as numpy's within the products'
rounding.
"""

from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.networks.encoder_decoder_network import EncoderDecoderNetwork
from indrajala_ml.model.networks.rust.rust_array_network_base import RustArrayNetworkBase


class EncoderDecoderRustArrayNetwork(EncoderDecoderNetwork[pa.Array], RustArrayNetworkBase):
    """EncoderDecoderNetwork on the Rust backend."""
