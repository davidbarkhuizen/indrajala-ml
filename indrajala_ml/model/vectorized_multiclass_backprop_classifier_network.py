from __future__ import annotations

from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.array_network_shapes import ArrayMultiClassShape


class VectorizedMultiClassBackpropClassifierNetwork(ArrayMultiClassShape, ArrayNetworkBase):
    """
    A numpy-array-backed sibling of MultiClassBackpropClassifierNetwork: array-based vectorization
    replaces "one Python object, one method call, per node" with "one array, one matrix operation,
    for the whole layer", so there's no per-node compute_hidden_delta(next_layer_nodes, own_index)
    to reuse and no StateLayer/BackpropLayer involved at all.

    ArrayMultiClassShape on the numpy backend. Every numpy multiclass sibling (momentum, L2, Adam,
    ReLU, softmax, dropout, cross-entropy) subclasses this directly, the same way their per-node
    counterparts subclass MultiClassBackpropClassifierNetwork.

    Parity-checked against the pure-Python reference implementation; numpy is a vectorization
    backend here, not a permanent dependency commitment (RustArrayMultiClassBackpropClassifierNetwork
    and its Rust-backed siblings are the production path).
    """
