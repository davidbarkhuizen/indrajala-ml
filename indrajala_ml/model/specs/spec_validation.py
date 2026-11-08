"""
validate_layer_specs: which lists of layer specs (layer_specs.py) every implementation builds. It
accepts only the combinations every builder maps, so every accepted list is parity-testable.
validate_window_recurrent: which windows and recurrences (window_recurrence.py) make a
window-recurrent network. validate_encoder_decoder: which encoders and decoders make an
encoder-decoder network (the cross-attention workplan, D14).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from indrajala_ml.model.layers.python.activations import GELUS, GLUS
from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    BatchNorm,
    Branches,
    CrossAttention,
    Dense,
    Dropout,
    Embedding,
    EncoderDecoder,
    GatedDense,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    StateInput,
    TokenMean,
    TokenSpec,
    TokenStart,
    expand_specs,
    token_wise_output,
)
from indrajala_ml.model.specs.spec_shapes import InputShape, spec_shapes
from indrajala_ml.model.specs.window_recurrence import BEFORE_STATE, OUTPUT, Factor, WindowRecurrence, array_window


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
    assert spec.activation in ("sigmoid", "relu", *GELUS, "linear"), (
        f"a hidden layer is sigmoid, ReLU, GELU or linear; got {spec!r}"
    )
    assert spec.dropout is None or spec.activation == "sigmoid", (
        f"dropout is fused with the sigmoid, so it needs a sigmoid hidden layer; got {spec!r}"
    )
    assert spec.loss is None, f"a loss belongs to the output layer (the audit-remedies workplan, D2); got {spec!r}"


def _check_layer_norm(spec: LayerNorm) -> None:
    assert spec.epsilon > 0.0, f"epsilon must be positive; got {spec!r}"


def _check_attention(spec: Attention) -> None:
    assert spec.heads >= 1, f"an Attention has at least one head; got {spec!r}"
    assert spec.key_size is None or spec.key_size >= 1, f"a key_size is at least 1; got {spec!r}"
    assert 0.0 <= spec.dropout < 1.0, f"an Attention's dropout is in [0.0, 1.0); got {spec!r}"
    assert spec.positions in ("none", "rotary"), f'an Attention\'s positions are "none" or "rotary"; got {spec!r}'
    assert spec.rotary_base > 0.0, f"a rotary_base is positive; got {spec!r}"
    assert spec.positions == "rotary" or spec.rotary_base == Attention().rotary_base, (
        f'a rotary_base is a rotary Attention\'s, positions="rotary" (the rotary-positions workplan, D2); got {spec!r}'
    )
    g = spec.key_value_heads
    assert g is None or (g >= 1 and spec.heads % g == 0), (
        f"an Attention's key_value_heads is at least 1 and divides its heads (the grouped-query attention workplan, "
        f"D9); got {spec!r}"
    )
    assert g != spec.heads, (
        f"an Attention whose every query head has its own key/value head says key_value_heads=None (the grouped-query "
        f"attention workplan, D9); got {spec!r}"
    )


def _check_cross_attention(spec: CrossAttention) -> None:
    assert spec.heads >= 1, f"a CrossAttention has at least one head; got {spec!r}"
    assert spec.key_size is None or spec.key_size >= 1, f"a key_size is at least 1; got {spec!r}"
    assert 0.0 <= spec.dropout < 1.0, f"a CrossAttention's dropout is in [0.0, 1.0); got {spec!r}"
    g = spec.key_value_heads
    assert g is None or (g >= 1 and spec.heads % g == 0), (
        f"a CrossAttention's key_value_heads is at least 1 and divides its heads; got {spec!r}"
    )
    assert g != spec.heads, (
        f"a CrossAttention whose every query head has its own key/value head says key_value_heads=None; got {spec!r}"
    )


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


def _check_token_residual(spec: Residual, in_body: bool, window: bool) -> None:
    assert not in_body, f"a residual block's body holds no residual block (D6); got {spec!r}"
    assert spec.body, f"a residual block's body needs at least one layer; got {spec!r}"
    *layers, last = spec.body
    if isinstance(last, Dropout) and layers:
        # the residual dropout, after the body's last layer (the attention-dropout workplan, D2)
        _check_dropout(last)
        *layers, last = layers
    affine = isinstance(last, Dense) and last.activation == "linear" and last.bias
    assert affine or isinstance(last, Attention | CrossAttention), (
        f"a token block's body ends in Attention, CrossAttention (a decoder's, the cross-attention workplan, D14) "
        f'or an affine layer, Dense(n, "linear", bias=True) (the layer-norm and attention workplan, D1, D6), then '
        f"may end in a Dropout (the attention-dropout workplan, D2); got {spec!r}"
    )
    if isinstance(last, Dense):
        _check_hidden_dense(last)
        if layers and isinstance(layers[-1], GatedDense):
            # the gated FFN, right before its affine layer (the FFN activations workplan, D3, D5)
            _check_gated_dense(layers[-1])
            layers = layers[:-1]
    elif isinstance(last, Attention):
        _check_attention(last)
    elif isinstance(last, CrossAttention):
        _check_cross_attention(last)
    _check_token_layers(layers, in_body=True, window=window)


def _check_gated_dense(spec: GatedDense) -> None:
    assert spec.size >= 1, f"a GatedDense needs at least one unit; got {spec!r}"
    assert spec.activation in GLUS, f'a GatedDense\'s activation is "silu", "gelu" or "gelu_tanh"; got {spec!r}'


def _check_token_layers(tokens: Sequence[LayerSpec], in_body: bool, window: bool) -> None:
    # a token part's layers, between Patches and TokenMean, or a token block's body before its last
    # layer (the layer-norm and attention workplan, D1, D4, D7); a window's may hold StateInputs
    # (the window-recurrent adder workplan, W18)
    batch_norm = next((spec for spec in expand_specs(tokens) if isinstance(spec, BatchNorm)), None)
    assert batch_norm is None, f"a BatchNorm doesn't stand among the tokens, a LayerNorm does (D4); got {batch_norm!r}"
    blocks = 0
    for i, spec in enumerate(tokens):
        assert not isinstance(spec, GatedDense), (
            'a GatedDense stands in a token block\'s body, right before its affine layer, Dense(n, "linear", '
            f"bias=True) (the FFN activations workplan, D3, D5); got {spec!r} in {list(tokens)!r}"
        )
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
        elif isinstance(spec, StateInput):
            assert window, (
                "a StateInput is a window-recurrent network's, among its window's tokens (the window-recurrent adder "
                f"workplan, W18): check it with validate_window_recurrent; got {list(tokens)!r}"
            )
            assert not in_body and i + 1 < len(tokens) and isinstance(tokens[i + 1], Residual), (
                "a StateInput stands among a window's tokens right before a residual block, which reads the state "
                f"(the window-recurrent adder workplan, W1b-ii, W18); got {list(tokens)!r}"
            )
        elif isinstance(spec, Residual):
            _check_token_residual(spec, in_body, window)
            blocks += 1
        elif isinstance(spec, Branches):
            assert window and not in_body and (i + 1 == len(tokens) or spec.merge), (
                "Branches are a window-recurrent network's, its window's last token layer, right before the output "
                f"layer (the window branches workplan, D1, D6), unless merged; got {list(tokens)!r}"
            )
            after = tokens[i + 1 :]
            assert not spec.merge or (after and isinstance(after[-1], LayerNorm)), (
                "merged Branches are followed by the recombining layers, ending in a LayerNorm before the output "
                f"layer (the post-paper workplan, D7); got {list(tokens)!r}"
            )
            assert not any(isinstance(s, Branches) for s in after), (
                f"a window holds one Branches (the window branches workplan, D1); got {list(tokens)!r}"
            )
            assert len(spec.branches) >= 2, f"Branches are two or more branches; got {spec!r}"
            assert i > 0 and isinstance(tokens[i - 1], Residual), (
                "Branches follow a residual block, the shared trunk's last (the window branches workplan, D2; an add "
                f"reads the split's downstream as it reads a fork's); got {list(tokens)!r}"
            )
            for branch in spec.branches:
                assert branch and isinstance(branch[-1], LayerNorm), (
                    "each branch ends in a LayerNorm, as a window before its output layer, which its pooling or the "
                    f"output layer reads (the window branches workplan, D2); got {spec!r}"
                )
                inner = next((s for s in branch if isinstance(s, Position | Branches | Dropout)), None)
                assert inner is None, (
                    f"a branch holds a StateInput, residual blocks, LayerNorms and token-wise Dense layers, not "
                    f"{inner!r} (the window branches workplan, D2); got {spec!r}"
                )
                _check_token_layers(branch, in_body=False, window=window)
        else:
            assert isinstance(spec, Dense), (
                "among the tokens stand a token-wise Dense, a Position, a Dropout, a LayerNorm and residual blocks, "
                f"Attention ending a block's body (D1, D4); got {spec!r}"
            )
            _check_hidden_dense(spec)
            assert spec.activation in ("relu", *GELUS) or (spec.activation == "linear" and spec.bias), (
                f"a token-wise Dense is ReLU or GELU, or linear with bias=True, with no dropout (D4; the FFN "
                f"activations workplan, D5); got {spec!r}"
            )
    assert sum(isinstance(spec, Position) for spec in tokens) <= 1, f"one Position at most (D7); got {list(tokens)!r}"
    rotary = next((s for s in expand_specs(tokens) if isinstance(s, Attention) and s.positions == "rotary"), None)
    assert rotary is None or not any(isinstance(spec, Position) for spec in tokens), (
        f"a token part's positions are a Position or rotary attention, not both (the rotary-positions workplan, D6); "
        f"got {rotary!r} after a Position"
    )


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
    A StateInput is refused: it is a window's (validate_window_recurrent).
    """
    _validate_layer_specs(specs, window=False)


def _validate_layer_specs(specs: Sequence[LayerSpec], window: bool, memory: bool = False) -> None:
    # memory: the specs are an encoder-decoder's decoder, whose CrossAttention reads the encoder's output
    assert specs, "a network needs at least one layer"
    cross = next((spec for spec in expand_specs(specs) if isinstance(spec, CrossAttention)), None)
    assert memory or cross is None, (
        f"a CrossAttention reads an encoder's output, so it stands only in an EncoderDecoder's decoder (the "
        f"cross-attention workplan, D14); got {cross!r}"
    )
    *hidden, output = specs

    if hidden and isinstance(hidden[0], TokenStart):
        _check_token_start(hidden[0])
        if token_wise_output(specs):
            _check_token_layers(hidden[1:], in_body=False, window=window)
            _check_output(output, token_wise=True)
            return
        mean = next(i for i, spec in enumerate(hidden) if isinstance(spec, TokenMean))
        tokens, dense = hidden[1:mean], hidden[mean + 1 :]
        _check_token_layers(tokens, in_body=False, window=window)
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
    assert output.loss is not None, (
        f'the output layer names its loss, loss="squared", "cross_entropy" or "given" (the audit-remedies '
        f"workplan, D2); got {output!r}"
    )
    assert output.size >= 1, f"the output layer needs at least one node; got {output!r}"
    assert output.dropout is None, f"the output layer doesn't drop out; got {output!r}"
    assert output.activation in ("sigmoid", "softmax", "linear"), (
        f"the output layer is sigmoid, softmax or linear; got {output!r}"
    )
    assert (output.activation == "linear") == (output.loss == "given"), (
        f'a linear output layer, and only one, has a given loss (loss="given"): its gradient comes from the '
        f"caller, learn_batch_output_gradient; got {output!r}"
    )
    assert output.activation != "softmax" or output.loss == "cross_entropy", (
        f"a softmax output layer's delta is the cross-entropy loss's; got {output!r}"
    )
    assert not output.bias, f"bias=True is a residual block's affine layer only (D4); got {output!r}"
    assert output.init == "fan_in", (
        f"He initialisation is a hidden layer's, for its rectifier (the audit-remedies workplan, D8); got {output!r}"
    )


def validate_window_recurrent(
    specs: Sequence[LayerSpec], recurrence: WindowRecurrence, input_shape: InputShape
) -> None:
    """
    Rejects a window-recurrent network (the window-recurrent adder workplan) that isn't one: a
    window that validate_window_specs rejects, and a recurrence without factors (a tuple of
    Factors), a factor without a state, with fewer than two codes or a negative commitment, a
    cross-term with one factor (the factored-state workplan, D1), or an update bias that isn't
    finite. A continuous factor (no codes) has neither codes nor a commitment to check.
    """
    validate_window_specs(specs, input_shape)
    factors = recurrence.factors
    assert isinstance(factors, tuple) and factors and all(isinstance(f, Factor) for f in factors), (
        f"a recurrence's factors are a tuple of one or more Factors; got {recurrence!r}"
    )
    for factor in factors:
        assert factor.state_size >= 1, f"a factor's state has at least one feature; got {recurrence!r}"
        if factor.codes is not None:
            assert factor.codes.count >= 2, (
                f"a factor of one code carries nothing: two codes at least (W4-i); got {recurrence!r}"
            )
            assert factor.codes.parameterisation in ("direct", "linear"), (
                f'a factor\'s codes are "direct" or "linear" (the audit-remedies workplan, D6); got {recurrence!r}'
            )
            commitment = factor.codes.commitment
            assert math.isfinite(commitment) and commitment >= 0.0, (
                f"a commitment weight is finite and not negative (W14); got {recurrence!r}"
            )
    assert not recurrence.cross or len(factors) >= 2, (
        f"a cross-term reads the other factors' states: two factors at least; got {recurrence!r}"
    )
    assert math.isfinite(recurrence.cell.update_bias), f"an update bias is finite (W13); got {recurrence!r}"
    assert recurrence.pooling.reads in (OUTPUT, BEFORE_STATE), (
        f"a pooling reads the output layer's tokens or those before the first StateInput (D10); got {recurrence!r}"
    )
    branches = next((spec for spec in specs if isinstance(spec, Branches)), None)
    if branches is not None and branches.merge:
        if branches.reads is None:
            assert len(branches.branches) == len(factors), (
                f"merged Branches are one a factor, none for the answers (the post-paper workplan, D7): "
                f"{len(factors)} for {len(factors)} factors; got {len(branches.branches)}"
            )
        else:
            assert len(branches.reads) == len(factors) and set(branches.reads) <= set(range(len(branches.branches))), (
                f"merged Branches' reads name each factor's branch (the post-paper workplan, D7): {len(factors)} among "
                f"branches 0-{len(branches.branches) - 1}; got {branches.reads!r}"
            )
    elif branches is not None:
        if branches.reads is None:
            assert len(branches.branches) == len(factors) + 1, (
                f"a window's Branches are one a factor and one for the answers (the window branches workplan, D1): "
                f"{len(factors) + 1} for {len(factors)} factors; got {len(branches.branches)}"
            )
        else:
            read = len(branches.branches) - 1  # the factors' branches; the last is the answers'
            assert len(branches.reads) == len(factors) and set(branches.reads) == set(range(read)), (
                f"Branches' reads name each factor's branch, every branch but the answers' (the last) read by one "
                f"or more (the one-code-one-state workplan, R1): {len(factors)} among branches 0-{read - 1}; got "
                f"{branches.reads!r}"
            )
    if branches is not None:
        assert recurrence.pooling.reads == OUTPUT, (
            f"with Branches each factor's pooling reads its branch's end, AttentionPooling(OUTPUT) (the window branches "
            f"workplan, D1); got {recurrence!r}"
        )


def validate_window_specs(specs: Sequence[LayerSpec], input_shape: InputShape) -> None:
    """
    Rejects a window, specs over input_shape, that isn't a sequence model reading its k columns'
    token ids through an Embedding and ending in a token-wise output layer, its sum digits (W9,
    W17), or that has no StateInput, by which the state enters it (W1b-ii, W18). Otherwise the
    window is checked as validate_layer_specs checks a sequence model, StateInputs allowed, then by
    the shape walk. A window of array layers (array_window) is checked by _validate_array_window
    instead. What a builder of a window checks (validate_window_recurrent checks the rest).
    """
    if array_window(specs):
        _validate_array_window(specs, input_shape)
        return
    assert specs and isinstance(specs[0], Embedding), (
        f"a window starts with an Embedding of its columns' token ids (the window-recurrent adder workplan, The "
        f"network), or a Conv or Dense over each step's observation (the array window workplan, D1); got "
        f"{list(specs)!r}"
    )
    assert len(input_shape) == 1 and input_shape[0] >= 1, (
        f"a window's input is its k columns' token ids, k at least 1; got the input shape {input_shape}"
    )
    assert token_wise_output(specs), (
        "a window ends in a token-wise output layer, a digit per column (the window-recurrent adder workplan, The "
        f"network); got {list(specs)!r}"
    )
    _validate_layer_specs(specs, window=True)
    gelu = next(
        (
            spec
            for spec in expand_specs(specs)
            if isinstance(spec, GatedDense) or (isinstance(spec, Dense) and spec.activation in GELUS)
        ),
        None,
    )
    assert gelu is None, (
        f"a window's token-wise layers are ReLU: the window program's ops are (the FFN activations workplan, D5, "
        f"D8); got {gelu!r}"
    )
    grouped = next(
        (spec for spec in expand_specs(specs) if isinstance(spec, Attention) and spec.key_value_heads is not None),
        None,
    )
    assert grouped is None, (
        f"a window's attention gives each query head its own key/value head: the window program's does (the "
        f"grouped-query attention workplan, D10); got {grouped!r}"
    )
    assert any(isinstance(spec, StateInput) for spec in specs), (
        "the state enters a window through at least one StateInput, right before a residual block (the "
        f"window-recurrent adder workplan, W1b-ii, W18); got {list(specs)!r}"
    )
    spec_shapes(specs, input_shape)


def _validate_array_window(specs: Sequence[LayerSpec], input_shape: InputShape) -> None:
    # a window of array layers (the array window workplan, D1-D3): a conv or dense network over one
    # step's observation, its StateInputs aside, each adding the state to the shape it stands in as
    # one token (D2), ending in a given loss's output layer (D3)
    state_inputs = [i for i, spec in enumerate(specs) if isinstance(spec, StateInput)]
    assert state_inputs, (
        "the state enters a window through at least one StateInput (the array window workplan, D2); got "
        f"{list(specs)!r}"
    )
    assert all(0 < i < len(specs) - 1 for i in state_inputs), (
        "an array window's StateInput stands after its first layer and before its output layer (the array window "
        f"workplan, D2); got {list(specs)!r}"
    )
    output = specs[-1]
    assert isinstance(output, Dense) and output.output and output.loss == "given", (
        'an array window\'s output layer has a given loss, Dense(n, "linear", output=True, loss="given"): its '
        f"gradient comes from the caller at every step (the array window workplan, D3); got {output!r}"
    )
    norm = next((spec for spec in expand_specs(specs) if isinstance(spec, BatchNorm)), None)
    assert norm is None, (
        "an array window holds no BatchNorm: a training step re-runs each step's forward pass, which would move "
        f"its running statistics twice (the array window workplan); got {norm!r}"
    )
    # the rest as a network of array layers: no token layer but the StateInputs
    _validate_layer_specs([spec for spec in specs if not isinstance(spec, StateInput)], window=False)
    spec_shapes(specs, input_shape)


def validate_encoder_decoder(spec: EncoderDecoder) -> None:
    """
    Rejects an encoder-decoder some implementation can't build (the cross-attention workplan, D14).
    The encoder: an Embedding with segments, step 11's rule (a separator ends its pair), then token
    layers as a sequence model's, whose every Attention is not causal (the encoder reads its whole
    source), ending in a LayerNorm, with no output layer and no CrossAttention. The decoder: a
    sequence model (validate_layer_specs's rules) from an Embedding with segments and
    separator_opens (D13), whose every Attention is causal, with at least one CrossAttention ending
    a residual block's body, and a token-wise softmax output layer. Sizes are checked by
    encoder_decoder_shapes.
    """
    encoder, decoder = spec.encoder, spec.decoder
    assert encoder and decoder, f"an encoder-decoder has an encoder and a decoder; got {spec!r}"
    start = encoder[0]
    assert isinstance(start, Embedding) and start.segments and not start.separator_opens, (
        f"an encoder starts with an Embedding with segments, a separator ending each source pair (D13); got {start!r}"
    )
    _check_token_start(start)
    assert len(encoder) >= 2 and isinstance(encoder[-1], LayerNorm), (
        f"an encoder ends in a LayerNorm, its output the decoder's memory (D14); got {list(encoder)!r}"
    )
    output = next((s for s in expand_specs(encoder) if isinstance(s, Dense) and s.output), None)
    assert output is None, f"an encoder has no output layer: its output is the memory (D14); got {output!r}"
    cross = next((s for s in expand_specs(encoder) if isinstance(s, CrossAttention)), None)
    assert cross is None, f"a CrossAttention stands in the decoder, not the encoder (D14); got {cross!r}"
    causal = next((s for s in expand_specs(encoder) if isinstance(s, Attention) and s.causal), None)
    assert causal is None, f"an encoder reads its whole source: its attention isn't causal (D14); got {causal!r}"
    _check_token_layers(encoder[1:], in_body=False, window=False)

    start = decoder[0]
    assert isinstance(start, Embedding) and start.segments and start.separator_opens, (
        "a decoder starts with an Embedding with segments and separator_opens, a separator opening each target "
        f"pair (D13); got {start!r}"
    )
    assert token_wise_output(decoder), (
        f"a decoder ends in a token-wise output layer, with no TokenMean (D14); got {list(decoder)!r}"
    )
    _validate_layer_specs(decoder, window=False, memory=True)
    attention = [s for s in expand_specs(decoder) if isinstance(s, Attention)]
    assert all(s.causal for s in attention), (
        f"a decoder's self-attention is causal (D14); got {next(s for s in attention if not s.causal)!r}"
    )
    assert any(isinstance(s, CrossAttention) for s in expand_specs(decoder)), (
        f"a decoder reads the encoder through at least one CrossAttention (D14); got {list(decoder)!r}"
    )
