"""
validate_layer_specs: which lists of layer specs (layer_specs.py) every implementation builds. It
accepts only the combinations every builder maps, so every accepted list is parity-testable.
"""

from __future__ import annotations

from collections.abc import Sequence

from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    BatchNorm,
    Dense,
    Dropout,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    TokenMean,
    TokenSpec,
    TokenStart,
    expand_specs,
    token_wise_output,
)


def _check_batch_norm(spec: BatchNorm, before: LayerSpec | None) -> None:
    linear = isinstance(before, Dense | ConvSpec) and before.activation == "linear"
    assert linear, (
        f"a BatchNorm normalizes a linear layer, so one comes right before it; got {before!r} before {spec!r}"
    )
    if isinstance(before, ConvSpec):
        assert spec.activation == "relu", f"a BatchNorm after a conv layer is ReLU; got {spec!r}"
    assert spec.activation in ("sigmoid", "relu"), f"a BatchNorm is sigmoid or ReLU; got {spec!r}"
    assert spec.epsilon > 0.0, f"epsilon must be positive; got {spec!r}"
    assert 0.0 < spec.running_rate <= 1.0, f"running_rate must be in (0.0, 1.0]; got {spec!r}"
    assert spec.group_size is None or spec.group_size >= 2, f"group_size must be 2 or more; got {spec!r}"


def _check_hidden_dense(spec: Dense) -> None:
    assert not spec.output, f"only the last layer is the output layer; got {spec!r} before it"
    assert spec.size >= 1, f"a dense layer needs at least one node; got {spec!r}"
    assert spec.activation in ("sigmoid", "relu", "linear"), f"a hidden layer is sigmoid, ReLU or linear; got {spec!r}"
    assert spec.dropout is None or spec.activation == "sigmoid", (
        f"dropout is fused with the sigmoid, so it needs a sigmoid hidden layer; got {spec!r}"
    )
    assert spec.loss == "squared", f"a loss belongs to the output layer; got {spec!r}"


def _check_layer_norm(spec: LayerNorm) -> None:
    assert spec.epsilon > 0.0, f"epsilon must be positive; got {spec!r}"


def _check_attention(spec: Attention) -> None:
    assert spec.heads >= 1, f"an Attention has at least one head; got {spec!r}"
    assert spec.key_size is None or spec.key_size >= 1, f"a key_size is at least 1; got {spec!r}"
    assert 0.0 <= spec.dropout < 1.0, f"an Attention's dropout is in [0.0, 1.0); got {spec!r}"


def _check_dropout(spec: Dropout) -> None:
    assert 0.0 <= spec.p < 1.0, f"a Dropout's p is in [0.0, 1.0); got {spec!r}"


def _check_residual(spec: Residual, in_body: bool) -> None:
    assert not in_body, f"a residual block's body holds no residual block (D6); got {spec!r}"
    assert spec.body, f"a residual block's body needs at least one layer; got {spec!r}"
    assert not any(isinstance(layer, ConvSpec | PoolSpec) for layer in spec.body), (
        f"a residual block is dense only (D2): its body holds no conv or pool layer; got {spec!r}"
    )
    *layers, affine = spec.body
    assert isinstance(affine, Dense) and affine.activation == "linear" and affine.bias, (
        f'a residual block\'s body ends in an affine layer, Dense(n, "linear", bias=True) (D3, D4); got {spec!r}'
    )
    _check_hidden_dense(affine)
    _check_dense_layers(layers, affine, in_body=True)


def _check_dense_layers(dense: Sequence[LayerSpec], output: LayerSpec, in_body: bool) -> None:
    # dense hidden layers, BatchNorms, LayerNorms and residual blocks, then output after them: a
    # network's dense part before its output layer, or a block's body before its affine layer
    for i, spec in enumerate(dense):
        after = dense[i + 1] if i + 1 < len(dense) else output
        if isinstance(spec, LayerNorm):
            _check_layer_norm(spec)
            continue
        if isinstance(spec, BatchNorm):
            _check_batch_norm(spec, dense[i - 1] if i > 0 else None)
            continue
        if isinstance(spec, Residual):
            _check_residual(spec, in_body)
            continue
        assert isinstance(spec, Dense), (
            f"conv and pool layers must all come before the dense layers; got {spec!r} among them"
        )
        _check_hidden_dense(spec)
        assert not spec.bias, f"bias=True is a residual block's affine layer only, its body's last (D4); got {spec!r}"
        assert (spec.activation == "linear") == isinstance(after, BatchNorm), (
            f"a linear layer and a BatchNorm come as a pair, the linear layer first; got {spec!r} before {after!r}"
        )


def _check_token_residual(spec: Residual, in_body: bool) -> None:
    assert not in_body, f"a residual block's body holds no residual block (D6); got {spec!r}"
    assert spec.body, f"a residual block's body needs at least one layer; got {spec!r}"
    *layers, last = spec.body
    if isinstance(last, Dropout) and layers:
        # the residual dropout, after the body's last layer (the attention-dropout workplan, D2)
        _check_dropout(last)
        *layers, last = layers
    affine = isinstance(last, Dense) and last.activation == "linear" and last.bias
    assert affine or isinstance(last, Attention), (
        f'a token block\'s body ends in Attention or an affine layer, Dense(n, "linear", bias=True) '
        f"(the layer-norm and attention workplan, D1, D6), then may end in a Dropout (the attention-dropout "
        f"workplan, D2); got {spec!r}"
    )
    if isinstance(last, Dense):
        _check_hidden_dense(last)
    elif isinstance(last, Attention):
        _check_attention(last)
    _check_token_layers(layers, in_body=True)


def _check_token_layers(tokens: Sequence[LayerSpec], in_body: bool) -> None:
    # a token part's layers, between Patches and TokenMean, or a token block's body before its last
    # layer (the layer-norm and attention workplan, D1, D4, D7)
    batch_norm = next((spec for spec in expand_specs(tokens) if isinstance(spec, BatchNorm)), None)
    assert batch_norm is None, f"a BatchNorm doesn't stand among the tokens, a LayerNorm does (D4); got {batch_norm!r}"
    blocks = 0
    for i, spec in enumerate(tokens):
        if isinstance(spec, Dropout):
            assert i > 0 and isinstance(tokens[i - 1], Position), (
                "a Dropout among the tokens stands right after the Position, or ends a block's body after its "
                f"Attention or affine layer (the attention-dropout workplan, D2); got {list(tokens)!r}"
            )
            _check_dropout(spec)
        elif isinstance(spec, Position):
            assert not in_body and blocks == 0, (
                f"a Position stands in the token part, before any block (D7); got {list(tokens)!r}"
            )
        elif isinstance(spec, LayerNorm):
            _check_layer_norm(spec)
        elif isinstance(spec, Residual):
            _check_token_residual(spec, in_body)
            blocks += 1
        else:
            assert isinstance(spec, Dense), (
                "among the tokens stand a token-wise Dense, a Position, a Dropout, a LayerNorm and residual blocks, "
                f"Attention ending a block's body (D1, D4); got {spec!r}"
            )
            _check_hidden_dense(spec)
            assert spec.activation == "relu" or (spec.activation == "linear" and spec.bias), (
                f"a token-wise Dense is ReLU, or linear with bias=True, with no dropout (D4); got {spec!r}"
            )
    assert sum(isinstance(spec, Position) for spec in tokens) <= 1, f"one Position at most (D7); got {list(tokens)!r}"


def _check_no_tokens(specs: Sequence[LayerSpec], where: str) -> None:
    token = next((spec for spec in expand_specs(specs) if isinstance(spec, TokenSpec)), None)
    assert token is None, (
        f"{token!r} acts on tokens, between Patches or an Embedding, the first layer, and TokenMean or a "
        f"token-wise output layer (the layer-norm and attention workplan, D3, D8; the sequence task "
        f"workplan, D5, D6); got it {where}"
    )


def _check_token_start(spec: TokenStart) -> None:
    if isinstance(spec, Patches):
        assert spec.patch_size >= 1, f"a patch size is at least 1; got {spec!r}"
    else:
        assert spec.vocabulary >= 1, f"an Embedding's vocabulary has at least one token; got {spec!r}"
        assert spec.size >= 1, f"an Embedding's tokens have at least one feature; got {spec!r}"


def validate_layer_specs(specs: Sequence[LayerSpec]) -> None:
    """
    Rejects a spec list that some implementation can't build: a conv or pool layer after a dense
    one (a dense layer's fused hidden delta reads the next layer's W), a front end without a conv
    layer, a softmax hidden layer, dropout on anything but a sigmoid hidden layer (the dropout op
    is fused with the sigmoid), a linear layer without a BatchNorm right after it or a BatchNorm
    without one right before it (a conv one's BatchNorm is ReLU), a residual block that isn't a
    dense body ending in an affine layer (Residual), and an output layer that isn't exactly the
    last layer. A block counts as a dense layer, and a LayerNorm may stand wherever a dense hidden
    layer may. In place of a front end, a patch model has a token part (_check_token_layers), from
    Patches or an Embedding, the first layer, to TokenMean, or to the output layer, applied to each
    token, which is then softmax (the sequence task workplan, D6). Sizes are checked by spec_shapes.
    """
    assert specs, "a network needs at least one layer"
    *hidden, output = specs

    if hidden and isinstance(hidden[0], TokenStart):
        _check_token_start(hidden[0])
        if token_wise_output(specs):
            _check_token_layers(hidden[1:], in_body=False)
            _check_output(output, token_wise=True)
            return
        mean = next(i for i, spec in enumerate(hidden) if isinstance(spec, TokenMean))
        tokens, dense = hidden[1:mean], hidden[mean + 1 :]
        _check_token_layers(tokens, in_body=False)
        _check_no_tokens([*dense, output], "after TokenMean")
        assert not any(isinstance(spec, ConvSpec | PoolSpec) for spec in dense), (
            f"a patch model has no conv or pool layer (D3); got {list(specs)!r}"
        )
        _check_dense_layers(dense, output, in_body=False)
        _check_output(output)
        return
    _check_no_tokens(specs, "in a network that doesn't start with Patches or an Embedding")

    # the front end: every layer before the first dense one, LayerNorm or residual block
    front_end_length = next(
        (i for i, spec in enumerate(hidden) if isinstance(spec, Dense | LayerNorm | Residual)), len(hidden)
    )
    front_end = hidden[:front_end_length]
    assert not any(isinstance(spec, ConvSpec | PoolSpec) for spec in hidden[front_end_length:]), (
        f"conv and pool layers must all come before the dense layers; got {list(specs)!r}"
    )
    for i, spec in enumerate(front_end):
        after = hidden[i + 1] if i + 1 < len(hidden) else output
        if isinstance(spec, BatchNorm):
            _check_batch_norm(spec, front_end[i - 1] if i > 0 else None)
        elif isinstance(spec, ConvSpec):
            assert spec.activation in ("relu", "linear"), f"a conv layer is ReLU or linear; got {spec!r}"
            assert (spec.activation == "linear") == isinstance(after, BatchNorm), (
                f"a linear layer and a BatchNorm come as a pair, the linear layer first; got {spec!r} before {after!r}"
            )
    assert not front_end or any(isinstance(spec, ConvSpec) for spec in front_end), (
        "a front end of pool layers needs at least one conv layer"
    )

    _check_dense_layers(hidden[front_end_length:], output, in_body=False)
    _check_output(output)


def _check_output(output: LayerSpec, token_wise: bool = False) -> None:
    assert isinstance(output, Dense) and output.output, (
        f"the last layer must be the output layer, Dense(..., output=True); got {output!r}"
    )
    assert not token_wise or output.activation == "softmax", (
        f"a token-wise output layer is softmax, with the cross-entropy loss (the sequence task workplan, D6); "
        f"got {output!r}"
    )
    assert output.size >= 1, f"the output layer needs at least one node; got {output!r}"
    assert output.dropout is None, f"the output layer doesn't drop out; got {output!r}"
    assert output.activation in ("sigmoid", "softmax"), f"the output layer is sigmoid or softmax; got {output!r}"
    assert output.activation != "softmax" or output.loss == "cross_entropy", (
        f"a softmax output layer's delta is the cross-entropy loss's; got {output!r}"
    )
    assert not output.bias, f"bias=True is a residual block's affine layer only (D4); got {output!r}"
