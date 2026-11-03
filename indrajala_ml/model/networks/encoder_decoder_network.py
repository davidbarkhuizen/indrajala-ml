# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, M, which strict mode takes for constants)
"""
An encoder-decoder network on either array backend (the cross-attention workplan, D2, D14; Vaswani
et al. 2017, §3.1): EncoderDecoderArrayNetwork on numpy (stage 4b) and
EncoderDecoderRustArrayNetwork on Rust (stage 6b), the same passes over the backend's array
operations (array_backend.py) and layer classes. An EncoderDecoder's two stacks, the encoder over the source's ids and the decoder over the
target's, the decoder's CrossAttention layers reading the encoder's output, its memory.

One example's state is its source's S ids, then its target input's T ids, so the trainers, their
examples and prepared datasets carry it as any sequence network's state; its label is the target's
T labels, one class per token, as a sequence network's (a packed pair window's decoder_input and
labels, parallel_data.py), a left pad's label the pad itself, which is no class and doesn't count.
The network's input is S + T ids wide.

The forward pass runs the encoder over the source, in its own context (its segments: step 11's
rule, a separator ending its pair), then the decoder over the target, in its context (its
segments by D13's rule, a separator opening its pair) with the encoder's output as its memory and
the source's segments as the memory's. The backward pass runs the decoder's layers back from the
output layer, as a sequence network's; then the memory's gradient, each CrossAttention's summed in
layer order (a left fold, D2), runs back through the encoder from its last layer. Every layer's
gradient is then accumulated, the encoder's then the decoder's, and the optimizer steps them all.

Its layers are the encoder's then the decoder's, one list (EncoderDecoder.layers(), expanded), so
randomize, snapshot, restore and checkpoint work over them as over any network's. A one-example
learn is a batch of one. It saves to format 2's "encoder_decoder" file (D14) and loads from it,
resuming by bits; decoding is stage 8.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

from indrajala_ml.data.prepared_dataset import CLASSIFY_CHUNK_ROWS, PreparedDataset
from indrajala_ml.model.layers.array.array_layer_builder import build_encoder_decoder_layers
from indrajala_ml.model.layers.forward_context import ForwardContext
from indrajala_ml.model.persistence.format2 import ENCODER_DECODER, NetworkFile
from indrajala_ml.model.protocols.array_protocols import ArrayNetworkLayer, BackendArray
from indrajala_ml.model.specs.array_network_shapes import TokenWiseOutput
from indrajala_ml.model.specs.bounds import validate_class_count
from indrajala_ml.model.specs.layer_specs import Dense, Embedding, EncoderDecoder, LayerSpec
from indrajala_ml.model.specs.spec_hooks import GivenRule
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import UpdateRule

type Passes[A] = tuple[list[A], list[A], ForwardContext, ForwardContext]


class _Memory[A]:
    """The encoder's last layer's next layer, as its hidden delta reads it: the memory's gradient."""

    def __init__(self, delta: A) -> None:
        self.delta = delta

    def downstream_batch(self, *, context: ForwardContext) -> A:
        return self.delta


@runtime_checkable
class _ReadsMemory[A](Protocol):
    """A decoder's CrossAttention, on either backend: it gives the memory's gradient, (N * S, d_m) rows."""

    def memory_downstream_batch(self) -> A: ...


class EncoderDecoderNetwork[A: BackendArray](GivenRule, TokenWiseOutput[A]):
    """
    An EncoderDecoder over source_tokens source ids and target_tokens target ids, trained by
    update_rule: a softmax over class_count classes for each target token, its loss the mean of
    the counted tokens' cross-entropies (the decoder's token-wise output layer's).
    """

    format2_shape: ClassVar[str] = ENCODER_DECODER
    preset_arguments: ClassVar[tuple[str, ...] | None] = None

    def __init__(self, spec: EncoderDecoder, source_tokens: int, target_tokens: int, update_rule: UpdateRule) -> None:
        output = spec.decoder[-1]
        assert isinstance(output, Dense), f"a decoder ends in its output layer; got {output!r}"
        validate_class_count(output.size)
        source, target = spec.encoder[0], spec.decoder[0]
        assert isinstance(source, Embedding) and isinstance(target, Embedding), (
            f"an encoder and a decoder start with an Embedding (D14); got {source!r} and {target!r}"
        )
        self.spec = spec
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.tokens = target_tokens
        self.class_count = output.size
        self.dimension = source_tokens + target_tokens
        self.update_rule = update_rule
        self._source = source
        self._target = target
        super().__init__(spec.layers(), (self.dimension,))

    def _build_layers(self, specs: Sequence[LayerSpec], input_shape: InputShape) -> list[ArrayNetworkLayer[A]]:
        name = self.backend.name
        encoder, decoder = build_encoder_decoder_layers(self.spec, self.source_tokens, self.target_tokens, name)
        self.encoder_layers: list[ArrayNetworkLayer[A]] = encoder
        self.decoder_layers: list[ArrayNetworkLayer[A]] = decoder
        self._crosses: list[_ReadsMemory[A]] = [layer for layer in decoder if isinstance(layer, _ReadsMemory)]
        return [*encoder, *decoder]

    # the passes

    def _passes(self, X: A) -> Passes[A]:
        # both stacks' activations, each list from its input, and their contexts
        assert len(X.shape) == 2 and X.shape[1] == self.dimension, (
            f"an encoder-decoder's state is its {self.source_tokens} source ids then its {self.target_tokens} target "
            f"ids; got rows of {X.shape[1:]}"
        )
        source = self.backend.columns(X, 0, self.source_tokens)
        target = self.backend.columns(X, self.source_tokens, self.dimension)
        encoder_context = self.backend.segment_context(source, self._source.separator, self._source.pad)
        encoded = [source]
        for layer in self.encoder_layers:
            encoded.append(layer.forward_batch(encoded[-1], context=encoder_context))
        decoder_context = replace(
            self.backend.segment_context(target, self._target.separator, self._target.pad, True),
            memory=encoded[-1],
            memory_segments=encoder_context.segments,
        )
        decoded = [target]
        for layer in self.decoder_layers:
            decoded.append(layer.forward_batch(decoded[-1], context=decoder_context))
        return encoded, decoded, encoder_context, decoder_context

    def _forward_batch(self, X: A) -> A:
        return self._passes(X)[1][-1]

    def _one_row(self, x: A) -> A:
        # one example's state as a batch of one
        return self.backend.matrix([x.tolist()])

    def _forward_input(self, x: A) -> A:
        return self.backend.row(self._forward_batch(self._one_row(x)), 0)

    def forward_rows(self, prepared: PreparedDataset) -> Iterator[A]:
        states = self._prepared_states(prepared)
        for start in range(0, len(prepared), CLASSIFY_CHUNK_ROWS):
            yield self._forward_batch(
                self.backend.row_range(states, start, min(start + CLASSIFY_CHUNK_ROWS, len(prepared)))
            )

    def _learn_input(self, learning_rate: float, x: A, category: Any) -> None:
        # a one-example step is a batch of one: the encoder-decoder has no single-example path
        self._learn_batch_input(learning_rate, self._one_row(x), [category])

    def _learn_batch_input(
        self, learning_rate: float, X: A, categories: Sequence[Any], clip_norm: float | None = None
    ) -> None:
        self._set_training_mode(True)
        try:
            encoded, decoded, encoder_context, decoder_context = self._passes(X)
        finally:
            self._set_training_mode(False)

        decoder, encoder = self.decoder_layers, self.encoder_layers
        decoder[-1].compute_output_delta_batch(self._target_batch_array(categories), context=decoder_context)
        for i in reversed(range(len(decoder) - 1)):
            decoder[i].compute_hidden_delta_batch(decoder[i + 1], context=decoder_context)

        # the memory's gradient: each CrossAttention's, a left fold in layer order (D2)
        memory_delta: A = self._crosses[0].memory_downstream_batch()
        for cross in self._crosses[1:]:
            total: A = memory_delta + cross.memory_downstream_batch()
            memory_delta = total
        encoder[-1].compute_hidden_delta_batch(
            _Memory(self.backend.reshape(memory_delta, *encoded[-1].shape)), context=encoder_context
        )
        for i in reversed(range(len(encoder) - 1)):
            encoder[i].compute_hidden_delta_batch(encoder[i + 1], context=encoder_context)

        for layer, activations in zip(encoder, encoded, strict=False):
            layer.accumulate_gradient_batch(activations, context=encoder_context)
        for layer, activations in zip(decoder, decoded, strict=False):
            layer.accumulate_gradient_batch(activations, context=decoder_context)
        self.optimizer.step(self.layers, learning_rate, len(categories), clip_norm)

    # the targets: one class per target token, as a sequence network's

    def _labels(self, category: Sequence[int]) -> Sequence[int]:
        assert len(category) == self.tokens, (
            f"an encoder-decoder's label is one class per target token, {self.tokens}; got {len(category)}"
        )
        return category

    def _target_array(self, category: Sequence[int]) -> A:
        return self.backend.row(self._target_batch_array([category]), 0)

    def _target_batch_array(self, categories: Sequence[Sequence[int]]) -> A:
        # a pad's label (a left pad's, before the window's first separator) is no class: its row stays
        # zero, and its target doesn't count, its input being a pad (the segments workplan, D6)
        pad = self._target.pad
        rows: list[list[float]] = []
        for category in categories:
            row = [0.0] * (self.tokens * self.class_count)
            for t, label in enumerate(self._labels(category)):
                if label == pad:
                    continue
                assert 0 <= label < self.class_count, f"a target label is a class or the pad; got {label}"
                row[t * self.class_count + label] = 1.0
            rows.append(row)
        return self.backend.matrix(rows)

    # format 2's file (D14): its "encoder" and "layers" entries, and its source's and target's ids

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
