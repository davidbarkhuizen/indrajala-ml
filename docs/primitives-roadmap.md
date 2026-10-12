# Roadmap: the next ML primitives

**Status: a proposed order, not a workplan. Each step gets its own workplan before any code.**

Which primitive to add next, and in what order: normalization, residual connections, attention,
transformers.

## What exists

Dense backprop, the perceptron and MADALINE, ReLU, softmax with cross-entropy, conv and max
pooling, momentum, Adam, L2, dropout, ensembles, linear warmup, batch norm with ghost groups,
residual blocks, layer norm and multi-head self-attention over patch tokens, stacked into transformer
layers, and a causal transformer over token ids for next-token prediction, with a per-token output
and loss, and dropout on its attention weights and among its tokens (README, Models, Batch
normalization, Residual connections, Layer norm and attention, The sequence task and Dropout among
the tokens), built from composable layer specs and update rules. Each is in all three
implementations (pure Python, numpy, Rust). There is no padding mask, no text generation and no
recurrence.

## The order

1. **Composable layers and optimizers** (a refactor, before the next primitive): done
   (2026-09-30); its open items are in [next-steps.md](next-steps.md).
2. **Batch normalization**: done (2026-09-30), and the conv batch-size study rerun with it
   (2026-10-01).
3. **Residual connections**: done (2026-10-01), dense blocks with identity shortcuts.
4. **Layer normalization and single-head self-attention**: done (2026-10-02), as a patch model
   on MNIST.
5. **Multi-head attention and a full transformer block**: done (2026-10-09), as deeper multi-head
   patch models on MNIST.
6. **A sequence task with causal masking**: done (2026-10-10), as next-character prediction on
   four text corpora.

Before step 7, a fix, not a primitive: pure Python's random draws in numpy's order, so that all
three implementations start, and drop out, alike from one seed; done (2026-10-10, #649-#652;
[rng-audit.md](rng-audit.md)). Step 7's attention dropout then tests its masks in all three by
bits.

7. **Dropout in attention**: done (2026-10-11), on the attention weights and token-wise, for the
   embedding and each block's output.
8. **Rotary position embedding**: positions inside attention, so that a cached key stays valid
   as a window slides; planned (2026-10-10,
   [rotary-positions-workplan.md](rotary-positions-workplan.md)).
9. **Generation**: sampling text from a trained sequence model, with a key/value cache; planned
   (2026-10-10, [generation-workplan.md](generation-workplan.md)).
10. **The training recipe**: learning-rate schedules as data, AdamW and gradient clipping, with
    longer training; planned (2026-10-10, [training-recipe-workplan.md](training-recipe-workplan.md)).
11. **Segments: packing, document masks and padding**: one segment id per token, for packed
    training windows and padded inference batches; planned (2026-10-10,
    [segments-workplan.md](segments-workplan.md)).
12. **Context length**: windows of 256 and 1,024 characters, with tiled attention; planned
    (2026-10-10, [context-length-workplan.md](context-length-workplan.md)).
13. **FFN activations: GELU and SwiGLU**; planned (2026-10-10,
    [ffn-activations-workplan.md](ffn-activations-workplan.md)).
14. **Grouped- and multi-query attention**; planned (2026-10-10,
    [grouped-query-attention-workplan.md](grouped-query-attention-workplan.md)).
15. **Cross-attention**, with an encoder-decoder model on Greek-English translation; planned
    (2026-10-10, [cross-attention-workplan.md](cross-attention-workplan.md)).

Steps 7 to 13 were ordered by the owner on 2026-10-10 from the candidates the multi-head attention
and sequence task workplans left (next-steps.md). The same day the owner chose a key/value cache
for generation and put rotary positions before it (step 8), then set the goal the later steps are
planned to: real-world production, the long-form corpora the targets and no toy reference deciding
a design. That reframed padding as segments (step 11), added context length (step 12), widened GELU
to the FFN activations production uses (step 13) and gave cross-attention a real translation task
(step 15). The case for each, and for the order, is below.

## 1 to 7. Done

Composable layers and optimizers, then batch normalization, both done (2026-09-30); their retired
workplans and the work they left open are in [next-steps.md](next-steps.md). The case made for
each is in this file's history: `git show 058087a:docs/primitives-roadmap.md`.

The conv batch-size study was rerun with batch norm, plain and with ghost groups of 32 (Goyal et
al. 2017's full setup), on 2026-10-01. Batch norm adds about a point of accuracy, but the linear
rule still fails at B = 512 at both momenta (`batch_size_scaling.py`'s docstring; open questions
in next-steps.md).

Residual connections, dense blocks `out = x + F(x)` with identity shortcuts, were done on
2026-10-01; the retired workplan and its open work (conv blocks, projection shortcuts) are in
next-steps.md. The depth study (`scripts/residual_depth_study.py`) found what He et al. 2016
predict: plain networks get worse with depth and residual ones don't.

Layer norm (over tokens and in flat dense networks) and single-head self-attention were done on
2026-10-02, as a small vision transformer on MNIST: `Patches`, a token-wise `Dense` embedding,
learned `Position`s, pre-LN attention and FFN blocks, and `TokenMean`. The retired workplan and its
open work are in next-steps.md. The patch-attention study (`scripts/patch_attention_study.py`)
found the expected result, weakly: attention then FFN beats FFN alone by less than a standard
deviation, and the conv network beats every patch model at this data size.

Multi-head attention was done on 2026-10-09: `Attention(heads, key_size)` in all three
implementations and the crate, each restructured into the same three blocks (project, attend,
combine) with named places for masking, dropout and grouped heads; a transformer layer is the
attention and FFN blocks step 4 built, written again for depth. The retired workplan and its open
work are in next-steps.md. The multi-head attention study (`scripts/multi_head_attention_study.py`)
found that more heads change nothing at this size (16 tokens of 32), and depth helps: two layers of
4 heads reach 97.0%, a point under the conv network. The case made for step 5 is in this file's
history: `git show d70cf0f:docs/primitives-roadmap.md`.

The sequence task was done on 2026-10-10: `Embedding(vocabulary, size)`, a token part ending in
a token-wise softmax output with the mean per-token cross-entropy (the `"sequence"` network shape),
and `Attention(causal=True)`, the mask in attend before the max shift, in all three
implementations and the crate; four character-level corpora (Tiny Shakespeare, Herodotus, the
*Muqaddimah*, Euclid's *Elements*), each in its own dataset repository. The retired workplan and
its open work are in next-steps.md. The sequence study (`scripts/sequence_study.py`) found what
the literature predicts: a per-token model only reaches the bigram floor, one causal attention
layer takes 0.8 to 1.5 bits per character under it, a second layer 0.09 to 0.17 more (2-layer:
Euclid 1.50, Herodotus 1.95, Tiny Shakespeare 2.48, the *Muqaddimah* 2.79), and without the mask
the model copies the next input, 0.04 to 0.06 bits per character on held-out text. The case made
for step 6 is in this file's history: `git show f6e843b:docs/primitives-roadmap.md`.

Dropout in attention was done on 2026-10-11: `Attention(dropout=p)`, inverted dropout on the
attention weights before the weighted sum of the values, and a token-wise `Dropout(p)` after
`Position` and ending a block's body (GPT's `attn_pdrop`, `embd_pdrop` and `resid_pdrop`), in all
three implementations and the crate, their masks by bits in all three. The retired workplan and its
open work are in next-steps.md. The attention-dropout study (`scripts/attention_dropout_study.py`)
found the sequence study's train/held-out gaps mostly the text's shift (Euclid's 0.49 bits per
character falls to 0.04 on a split from across the text), and dropout narrowing them by fitting
the training text worse, a cost of 0.01 to 0.35 bits held out at 10 epochs. The case made for step
7 is in this file's history: `git show 1ea3fb5:docs/primitives-roadmap.md`.

## 8 to 15. The order after the sequence task

Ordered by what each can measure on the models that exist, and by cost. Each step's open questions
are its workplan's to settle.

| step | pros | cons |
| --- | --- | --- |
| 8. Rotary position embedding | parameter-free positions inside attention (Su et al. 2021), the scheme of most decoders since; scores depend on token distance only, so step 9's cached keys stay valid as its window slides (with learned positions every cached key goes stale once the window moves); a `positions` field on `Attention` admits other schemes later | a new attention option in three implementations and the crate; `cos` and `sin` aren't correctly rounded, so one table must feed all three; models retrain with it |
| 9. Generation | the most persuasive demo of a language model; a key/value cache makes a character one token's pass (a full 64-token pass is 1.5 ms in Rust, 2.5 ms in numpy, on `pyramidon`), and incremental decoding is structure later sequence work reuses | a second forward path, a decode step, through every token layer in three implementations and the crate; a 10-epoch model at 2.48 bits per character on Tiny Shakespeare gives mostly garbled text |
| 10. The training recipe, longer training | the study's clearest open question: neither transformer had converged at 10 epochs; production trains with all three of a warmup-cosine schedule, AdamW and global-norm clipping (Brown et al. 2020, appendix B), so the step brings the recipe, not the schedule alone; every task gains | a training feature, not a primitive; lifts composable layers' "schedulers beyond today's `lr_schedule.py`" out of scope (the owner, 2026-10-10); its payoff is a longer study (10 epochs took about 2.5 h on `jebel`) |
| 11. Segments: packing, document masks, padding | one segment id per token is production's single mechanism for packed training windows (GPT-3 packs documents; Llama 3 masks attention between them) and padded inference batches (batched decoding, encoders, cross-attention's sources); the long-form corpora's paragraphs (median 729 to 1,266 characters) are the units it packs; generalizes the causal mask | a forward context carrying per-batch information to every layer, a refactor of every layer's signature before the feature; special tokens in the tokenizer; the loss over counted tokens only |
| 12. Context length | the long-form corpora's paragraphs run 729 to 1,266 characters (medians), so a 64-character window sees under a tenth of the paragraph it predicts within; every production model's first scaling axis after width and depth; tiled attention (an online softmax, Dao et al. 2022) makes memory linear in `T` and skips the causal half | attention's third rewrite, its arithmetic changed at every `T`, so every attention golden entry is re-recorded (owner approval, measurement.md §8); windows re-cut every epoch need an epoch-dependent dataset |
| 13. FFN activations: GELU and SwiGLU | GELU is BERT's, GPT-2's and ViT's; SwiGLU (Shazeer 2020) is PaLM's and LLaMA's, production decoders' FFN today | `erf`, which stable Rust and numpy lack, implemented once (fdlibm's) for the same bits in all three; a gated FFN layer beside the activation |
| 14. Grouped- and multi-query attention | a named place; fewer key and value parameters; a smaller key/value cache per generated token (step 9) | an inference-memory optimization, and nothing here is memory-bound: the study compares loss at equal parameters and step 9's decode speed |
| 15. Cross-attention | completes the original transformer; opens tasks with a second input (translation, conditioning) | the largest: a second input to a layer, a network that isn't a list of layers, an encoder-decoder task and its dataset; it needs step 11 |

Dropout led because its question was measured (done, above); rotary positions come next because generation's
cache needs them; generation then shows what steps 7, 8 and 10 buy; the training recipe comes
before the larger steps so that their studies train to convergence. Segments come before context
length, whose long windows pack paragraphs, and before cross-attention, which needs their padding.
The FFN activations and grouped heads refine the model steps 7 to 12 make, grouped heads paying in
step 9's cache at step 12's lengths; cross-attention is last as the largest, and needs everything
before it.

## Out of scope for now

- GPU backends, for the moment (the owner, 2026-10-12). Nothing else is excluded: work with no
  step yet is listed in next-steps as "not planned yet".
