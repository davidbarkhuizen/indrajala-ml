"""
The conv and pool layers' argument checks and output size, numpy-free, so the pure-Python, numpy and
Rust layers and the spec shape walk (layer_specs.spec_shapes) share one copy. The Rust layers take
their output size from the crate's pa.ConvGeometry, which agrees with output_size.
"""

from __future__ import annotations


def validate_conv_arguments(
    input_height: int, input_width: int, input_channels: int, kernel_size: int, channel_count: int, stride: int
) -> None:
    # every conv layer's constructor checks; ConvLayer adds its input_layer node count
    assert input_channels >= 1, f"input_channels must be at least 1; got {input_channels}"
    assert kernel_size >= 1, f"kernel_size must be at least 1; got {kernel_size}"
    assert channel_count >= 1, f"channel_count must be at least 1; got {channel_count}"
    assert stride >= 1, f"stride must be at least 1; got {stride}"
    assert kernel_size <= input_height and kernel_size <= input_width, (
        f"kernel_size ({kernel_size}) must fit within input_height x input_width ({input_height}x{input_width})"
    )


def validate_pool_arguments(
    input_height: int, input_width: int, input_channels: int, pool_size: int, stride: int
) -> None:
    # every max-pool layer's constructor checks; MaxPoolLayer adds its input_layer node count
    assert pool_size >= 1, f"pool_size must be at least 1; got {pool_size}"
    assert stride >= 1, f"stride must be at least 1; got {stride}"
    assert input_channels >= 1, f"input_channels must be at least 1; got {input_channels}"
    assert pool_size <= input_height and pool_size <= input_width, (
        f"pool_size ({pool_size}) must fit within input_height x input_width ({input_height}x{input_width})"
    )


def pool_stride(pool_size: int, stride: int | None) -> int:
    """A pool layer's stride: pool_size (non-overlapping windows) unless given."""
    return pool_size if stride is None else stride


def output_size(input_size: int, window: int, stride: int) -> int:
    """How many windows of window fit along input_size, stride apart: a conv or pool layer's
    out_height from input_height, or out_width from input_width."""
    return (input_size - window) // stride + 1
