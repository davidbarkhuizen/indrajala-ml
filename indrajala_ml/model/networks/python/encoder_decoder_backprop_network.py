"""
The encoder-decoder network in pure Python (the cross-attention workplan, stage 7b, D2, D14), the
counterpart of encoder_decoder_network.py's: an EncoderDecoder's two stacks, the encoder over the
source's ids and the decoder over the target's, the decoder's CrossAttention layers reading the
encoder's output, its memory.

One example's state is its source's S ids, then its target input's T ids, and its label the
target's T labels, a left pad's label the pad itself, which is no class and doesn't count: the
array network's. Each stack reads its own input layer (a StateLayer of S, and one of T).

Every pass runs both stacks through the layer-major batch path (layer_major.py), the encoder's
over the sources, then the decoder's over the targets, with each example's encoder output as its
memory in the decoder's context. So the dropout layers draw numpy's masks layer by layer in the
array network's order, and the network draws the array network's weights from a seed by bits. The
backward pass runs the decoder's layers back from the output layer; then each example's memory
gradient, each CrossAttention's summed in layer order (a left fold, D2), runs back through the
encoder from its last layer; then every layer's gradient accumulates and the optimizer steps them
all, the encoder's then the decoder's. A one-example step is a batch of one. Its sums are folds and
its products Python floats, so it trains as the array networks do within their products' rounding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ClassVar, Self

from indrajala_ml.generation.decoder import Decoder, SpecialIds
from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.python.cross_attention_layer import CrossAttentionLayer
from indrajala_ml.model.layers.python.layer_major import LayerMajorBatch
from indrajala_ml.model.layers.python.python_layer_builder import build_python_encoder_decoder_layers
from indrajala_ml.model.layers.python.segments import segment_rows
from indrajala_ml.model.layers.python.state_layer import StateLayer
from indrajala_ml.model.layers.python.token_layer import TokenSoftmaxLayer
from indrajala_ml.model.networks.python.backprop_network_base import BackpropNetworkBase, randomize_fan_in_aware
from indrajala_ml.model.persistence.format2 import ENCODER_DECODER, NetworkFile
from indrajala_ml.model.protocols.classification import argmax_first_occurrence
from indrajala_ml.model.protocols.layer_protocols import TrainableLayer
from indrajala_ml.model.specs.bounds import validate_batch, validate_class_count
from indrajala_ml.model.specs.layer_specs import Dense, Embedding, EncoderDecoder, LayerSpec, Position, expand_specs
from indrajala_ml.model.specs.spec_hooks import GivenRule
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import UpdateRule

Batch = Sequence[tuple[tuple[float, ...], Sequence[int]]]


class _Memory:
    """The encoder's last layer's next layer, as its hidden deltas read it: one example's memory gradient."""

    def __init__(self, gradient: Sequence[float]) -> None:
        self.gradient = gradient

    def downstream_sum(self, own_index: int) -> float:
        return self.gradient[own_index]


class EncoderDecoderBackpropNetwork(GivenRule, BackpropNetworkBase[TrainableLayer, TokenSoftmaxLayer]):
    """
    An EncoderDecoder over source_tokens source ids and target_tokens target ids, trained by
    update_rule: a softmax over class_count classes for each target token, its loss the mean of the
    counted tokens' cross-entropies. Its trainable_layers are the encoder's then the decoder's, so
    randomize, snapshot, restore, the optimizer and format 2 walk them in the array network's order.
    """

    format2_shape: ClassVar[str] = ENCODER_DECODER
    preset_arguments: ClassVar[tuple[str, ...] | None] = None

    def __init__(
        self,
        spec: EncoderDecoder,
        source_tokens: int,
        target_tokens: int,
        update_rule: UpdateRule,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        output = spec.decoder[-1]
        assert isinstance(output, Dense), f"a decoder ends in its output layer; got {output!r}"
        validate_class_count(output.size)
        source, target = spec.encoder[0], spec.decoder[0]
        assert isinstance(source, Embedding) and isinstance(target, Embedding), (
            f"an encoder and a decoder start with an Embedding (D14); got {source!r} and {target!r}"
        )
        # read by _build_layers, which the base calls
        self.spec = spec
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.tokens = target_tokens
        self.class_count = output.size
        self.update_rule = update_rule
        self._source = source
        self._target = target
        dimension = source_tokens + target_tokens
        bounds = [(0.0, 1.0)] * dimension if input_bounds is None else input_bounds
        super().__init__(spec.layers(), (dimension,), bounds)

    def _build_layers(self, specs: Sequence[LayerSpec], input_shape: InputShape) -> list[TrainableLayer]:
        S, T = self.source_tokens, self.target_tokens
        self.source_input = StateLayer(S, [(0.0, 1.0)] * S)
        self.target_input = StateLayer(T, [(0.0, 1.0)] * T)
        encoder, decoder = build_python_encoder_decoder_layers(self.spec, S, T, self.source_input, self.target_input)
        self.encoder_layers = encoder
        self.decoder_layers = decoder
        self._crosses = [i for i, layer in enumerate(decoder) if isinstance(layer, CrossAttentionLayer)]
        return [*encoder, *decoder]

    # the passes

    def _context(self, rows: Sequence[Sequence[float]], embedding: Embedding, opens: bool) -> ForwardContext:
        return segment_rows(rows, embedding.separator, embedding.pad, opens) if embedding.segments else NO_CONTEXT

    def _passes(
        self, states: Sequence[Sequence[float]]
    ) -> tuple[LayerMajorBatch, LayerMajorBatch, ForwardContext, ForwardContext]:
        # both stacks' batch passes, forward, and their contexts
        S = self.source_tokens
        for state in states:
            assert len(state) == S + self.target_tokens, (
                f"an encoder-decoder's state is its {S} source ids then its {self.target_tokens} target ids; got "
                f"{len(state)} values"
            )
        sources = [tuple(state[:S]) for state in states]
        targets = [tuple(state[S:]) for state in states]
        encoder_context = self._context(sources, self._source, False)
        encoder = LayerMajorBatch(self.source_input, self.encoder_layers, sources)
        encoder.forward(context=encoder_context)
        decoder_context = replace(
            self._context(targets, self._target, True),
            memory=encoder.outputs(),
            memory_segments=encoder_context.segments,
        )
        decoder = LayerMajorBatch(self.target_input, self.decoder_layers, targets)
        decoder.forward(context=decoder_context)
        return encoder, decoder, encoder_context, decoder_context

    def _forward(self, state: tuple[float, ...], *, context: ForwardContext | None = None) -> list[float]:
        return self._forward_batch_outputs([state])[0]

    def _forward_batch_outputs(self, states: Sequence[tuple[float, ...]]) -> list[list[float]]:
        return self._passes(states)[1].outputs()

    def predict_probabilities(self, state: tuple[float, ...]) -> list[list[float]]:
        """Each target token's class probabilities."""
        values = self._forward(state)
        width = self.class_count
        return [values[i : i + width] for i in range(0, len(values), width)]

    def classify_state(self, state: tuple[float, ...]) -> tuple[int, ...]:
        return tuple(argmax_first_occurrence(token) for token in self.predict_probabilities(state))

    # decoding (the cross-attention workplan, D7, D15, D16)

    def decoder(self, sources: Sequence[Sequence[int]]) -> Decoder:
        """
        EncoderDecoderNetwork.decoder's: a Decoder over the decoder's layers with sources, one a row,
        left-padded to the source's window, encoded once, the memory in every decode pass's context.
        """
        S, source = self.source_tokens, self._source
        assert sources and all(1 <= len(row) <= S for row in sources), (
            f"each source has 1 to {S} ids, the source's window; got lengths {[len(row) for row in sources]}"
        )
        rows = [tuple(float(i) for i in [source.pad] * (S - len(row)) + list(row)) for row in sources]
        encoder_context = self._context(rows, source, False)
        encoder = LayerMajorBatch(self.source_input, self.encoder_layers, rows)
        encoder.forward(context=encoder_context)
        memory = encoder.outputs()
        count = len(rows)

        def context(segments: Any, positions: Any, examples: int, column: int) -> ForwardContext:
            assert examples == count, f"a decode of {count} sources decodes {count} rows; got {examples}"
            return ForwardContext(
                segments=segments,
                positions=positions,
                examples=examples,
                column=column,
                memory=memory,
                memory_segments=encoder_context.segments,
            )

        def full_pass(targets: Sequence[Sequence[int]]) -> list[list[float]]:
            states = [
                (*source_row, *(float(i) for i in target)) for source_row, target in zip(rows, targets, strict=True)
            ]
            return [row[-self.class_count :] for row in self._forward_batch_outputs(states)]

        target = self._target
        return Decoder(
            self.decoder_layers,
            self.target_tokens,
            ids_input=lambda ids: [float(i) for i in ids],
            rows=lambda out: [list(row) for row in out],
            full_pass=full_pass,
            context=context,
            learned_positions=any(isinstance(spec, Position) for spec in expand_specs(self.spec.decoder)),
            special=SpecialIds(target.separator, target.pad, separator_opens=True),
        )

    # training

    def learn(self, learning_rate: float, state: tuple[float, ...], category: Any) -> None:
        # a one-example step is a batch of one: the encoder-decoder has no single-example path
        self.learn_batch(learning_rate, [(state, category)])

    def learn_batch(self, learning_rate: float, batch: Batch, clip_norm: float | None = None) -> None:
        validate_batch(batch)
        self._set_training_mode(True)
        try:
            encoder, decoder, encoder_context, decoder_context = self._passes([state for state, _ in batch])
            decoder.backward(
                lambda e: self._output_deltas(batch[e][1], decoder_context.example(e)), context=decoder_context
            )
            memory = [self._memory_gradient(decoder, e) for e in range(len(batch))]
            last = self.encoder_layers[-1]
            encoder.backward(
                lambda e: last.compute_hidden_deltas(_Memory(memory[e]), context=encoder_context.example(e)),  # pyright: ignore[reportArgumentType]
                context=encoder_context,
            )
            encoder.accumulate_gradients(context=encoder_context)
            decoder.accumulate_gradients(context=decoder_context)
            self.optimizer.step(self.trainable_layers, learning_rate, len(batch), clip_norm)
        finally:
            self._set_training_mode(False)

    def _memory_gradient(self, decoder: LayerMajorBatch, example: int) -> list[float]:
        # the example's memory gradient: each CrossAttention's, a left fold in layer order (D2)
        gradients: list[list[float]] = [decoder.lanes[i][example][0]["_dM"] for i in self._crosses]
        total = list(gradients[0])
        for gradient in gradients[1:]:
            total = [a + b for a, b in zip(total, gradient, strict=True)]
        return total

    def _output_deltas(self, category: Sequence[int], context: ForwardContext = NO_CONTEXT) -> None:
        # one-hot per token, a pad's label no class: node t * class_count + k against 1.0 where k is
        # token t's label
        assert len(category) == self.tokens, (
            f"an encoder-decoder's label is one class per target token, {self.tokens}; got {len(category)}"
        )
        pad, width = self._target.pad, self.class_count
        for label in category:
            assert label == pad or 0 <= label < width, f"a target label is a class or the pad; got {label}"
        for i, node in enumerate(self.output_layer.nodes):
            t, k = divmod(i, width)
            target = 1.0 if category[t] != pad and k == category[t] else 0.0
            if context.counted is None:
                node.compute_output_delta(target)
            else:
                # the mean over the batch's counted targets (the segments workplan, D6): (p - y) * B
                # / C, zero at a target that doesn't count; context is the example's view
                node.delta = (node.value() - target) * context.examples / context.count * context.counted[0][t]

    def _backward(self, target: Any, /, *, context: ForwardContext = NO_CONTEXT) -> None:
        raise NotImplementedError("an encoder-decoder's backward pass is learn_batch's, over both stacks")

    def randomize(self) -> None:
        randomize_fan_in_aware(self)

    # format 2's file (D14)

    @classmethod
    def from_file(cls, file: NetworkFile) -> Self:
        """The network a format-2 file describes, unweighted: restore_file then fills it."""
        assert file.encoder_length is not None and file.source_tokens is not None, "an encoder-decoder's file"
        n = file.encoder_length
        spec = EncoderDecoder(tuple(file.layers[:n]), tuple(file.layers[n:]))
        return cls(spec, file.source_tokens, file.input_shape[0] - file.source_tokens, file.update_rule)

    @classmethod
    def _from_file(cls, file: NetworkFile) -> Self:
        return cls.from_file(file)

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        raise ValueError(f"{cls.__name__} saves in format 2 only; this file has format {state.get('format')!r}")
