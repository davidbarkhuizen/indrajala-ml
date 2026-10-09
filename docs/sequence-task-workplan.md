# Workplan: a sequence task with causal masking (roadmap step 6)

**Status: decisions D1-D12 settled by the owner (2026-10-09), each as recommended. Stage 1 done
(#621, #622, #623); stage 2 done: the specs (#625); stage 3 done: numpy. Stages 4-9 not started.**

Roadmap step 6 ([primitives-roadmap.md](primitives-roadmap.md)): next-token prediction on a small
text corpus, a causal transformer. It brings the network's first per-token output and loss and
attention's first mask, in all three implementations and the crate.

## Why

Every network so far reads one example and gives one label; a patch model pools its tokens with
`TokenMean` before a single output. A language model predicts every token of a window from the
tokens before it: `T` predictions and `T` losses per example, and a causal mask so that token `t`
attends only to tokens `<= t` (Vaswani et al. 2017, section 3.2.3; Radford et al. 2018). Without the
mask, position `t` reads the token it is asked to predict (the input at `t + 1` is the target at
`t`), and the loss falls toward zero on held-out text as well as on training text: a model that
learned to copy, not to predict.

Step 5 placed the mask (next-steps, From multi-head attention, Extension points): in attend, an
additive mask on the scores `S[i]` before the max shift, with the backward pass unchanged, since a
masked weight `P_ij` is exactly zero and so is its `dS_ij`.

## What exists, and what changes

| piece | today | this plan |
| --- | --- | --- |
| data | MNIST (pinned parquet, `scripts/fetch_datasets.py`), UCI digits, iris | a text corpus and its loader (D2, D3), windows of `T + 1` tokens (D4) |
| labels | one per example, `Example[int]` | one per token, `Example[tuple[int, ...]]`: the trainer is already generic over the label |
| input | `Patches` from an image | one-hot tokens or token ids (D5) |
| token part | `Patches` ... `TokenMean`, then a dense part | may end in a token-wise output layer, no `TokenMean` (D6) |
| output and loss | one softmax and cross-entropy per example | one per token, combined per example (D6) |
| network shapes | `multiclass`, `single_output` | a third, `sequence` (D6): targets, `classify_state`, save and load |
| attention | no mask | `Attention(causal=True)` (D7), in attend in every implementation and the crate |
| evaluation | accuracy, confusion matrix | per-token cross-entropy in bits per token, and per-token accuracy (D8) |

The three-block structure (project, attend, combine) doesn't change; the mask is an attend option.

## Decisions (settled 2026-10-09)

Each lists the options considered, with pros and cons, and the choice.

- **D1. Scope. Settled: (a).**
  - (a) *Chosen.* A character-level corpus and its loader; one-hot or embedded input (D5); a
    token-wise softmax output with a per-token loss as a `sequence` network shape; `causal` on
    `Attention`; all three implementations and the crate; fixtures, checkpoints and golden
    entries; a study (D9). Pros: the smallest scope that trains a causal language model end to end
    and is as complete as steps 4 and 5 (every implementation, the save format, the golden run).
    Cons: the largest step yet (a new task family touches the trainer, the evaluation and the
    save format, not just a layer).
  - (b) (a) plus padding masks and variable-length windows. Pros: the general mask. Cons:
    fixed-length windows of a corpus never pad; nothing would exercise it.
  - (c) (a) without pure Python. Pros: pure Python at `T = 64` with a 65-symbol output is slow even
    for parity tests. Cons: breaks every step's rule that each primitive is in all three
    implementations; the parity tests can use a small `T` and vocabulary.
- **D2. The corpus. Settled: (a).**
  - (a) *Chosen.* Tiny Shakespeare (Karpathy 2015, char-rnn): 1,115,394 characters of 65
    symbols. Pros: small (1.1 MB); the standard character-level toy, with published losses to
    compare against (char-rnn, nanoGPT); a 65-way output is cheap; public-domain text. Cons: one
    author's style, so small models plateau on spelling and short-range structure; held-out text is
    the same plays' later lines, not new text.
  - (b) A synthetic sequence task from a seed (copying, reversal, or a small formal grammar).
    Pros: no download; known optimal loss; a task built so that only attention over earlier tokens
    solves it tests the mask directly. Cons: not text; says nothing about the framework on real
    data, which the roadmap asked for.
  - (c) Both: (a) for the study, (b) for tests. Pros: tests get a task whose answer is known. Cons:
    two loaders; the mask test (a future token can't change an earlier output) needs no task at
    all.
  - (d) A word-level corpus (WikiText-2, Penn Treebank). Pros: closer to real language modelling.
    Cons: a vocabulary of 10,000 to 33,000 makes the output layer the whole cost; PTB's licence.
  - *Added by the owner (2026-10-09)*: two more corpora beside Tiny Shakespeare, each in its own
    repository like it (D3), cleaned by a script there that rebuilds the text byte for byte from
    a pinned source:
    - Herodotus' *Histories* in George Rawlinson's translation (`indrajala-datasets-herodotus-rawlinson`,
      from Wikisource's 1910 text): 1,496,601 characters, ASCII, 76 symbols; public domain.
    - Ibn Khaldun's *Muqaddimah* in Arabic (`indrajala-datasets-muqaddimah`, from OpenITI's text
      of the Dar al-Qalam edition, 1984): 1,012,838 characters, unvocalized, 40 symbols. The text
      is CC BY-NC-SA 4.0, OpenITI's licence: for study only, fetched and never redistributed here.

    The loader reads any of the three (`text_data.CORPORA`). Tiny Shakespeare stays the study's
    corpus (D9); whether the study also covers the other two is settled with stage 8.
- **D3. Hosting the corpus. Settled: (a).**
  - (a) *Chosen.* A new `indrajala-datasets-tinyshakespeare` repository, tagged, fetched and
    checksum-verified by `scripts/fetch_datasets.py` as MNIST is. Pros: the existing pattern; the
    checkout stays free of data; provenance in one place. Cons: a new public repository (the
    owner's to create or approve); a network fetch on setup.
  - (b) Commit the file in `data/`, as UCI digits' CSV is. Pros: no fetch, no new repository.
    Cons: 1.1 MB in the history; breaks the pattern MNIST set for anything not tiny.
  - (c) Fetch from Karpathy's char-rnn URL, pinned by checksum. Pros: no new repository. Cons: a
    third party's URL, which can move; the fetch fails rather than silently changes, but fails.
- **D4. Examples. Settled: (a).**
  - (a) *Chosen.* Fixed windows of `T + 1` characters, non-overlapping, the first 90% of the
    text for training and the last 10% held out (nanoGPT's split): input the first `T`, targets the
    last `T`. `T = 64`. Pros: a fixed dataset like MNIST's, so the trainer, prepared datasets,
    shuffling and checkpoints work unchanged; about 15,400 training windows. Cons: each character
    is a target at one position only per epoch; a model sees each window's start without context.
  - (b) Windows at random offsets, drawn each epoch. Pros: every position of every character;
    the usual language-model sampler. Cons: a new sampler in the trainer (an epoch is no longer a
    permutation of fixed rows), new checkpoint state, and per-epoch results that depend on it.
  - (c) (a) with overlapping windows at stride `T / 2`. Pros: twice the windows, every
    character a target twice at different context lengths. Cons: correlated examples; still not
    every position.
- **D5. The input. Settled: (b).**
  - (a) One-hot tokens: a window as a `(1, T, V)` image, `Patches(1)` giving `T` tokens of `V`
    values, then the token-wise `Dense(d, "linear", bias=True)` already used as an embedding. Pros:
    no new layer; a one-hot row times `W` is exactly an embedding lookup. Cons: `T * V` floats per
    example (4,160 at `T = 64`, `V = 65`; about 64 M for the training set, more than MNIST's 47 M);
    the embedding's forward and gradient are dense products over mostly zeros; `Patches` on a
    `(1, T, V)` "image" is a misnomer.
  - (b) *Chosen.* `Embedding(vocabulary, d)`: reads `T` token ids, gives `(T, d)`; its gradient
    a scatter-add into the rows it read. Pros: `T` values per example; the standard layer; the
    lookup and scatter-add are cheap; it replaces both `Patches` and the embedding `Dense`. Cons: a
    new layer in three implementations, the crate, the optimizer accessors and format 2; token ids
    travel as floats in `State` (exact up to 2^53), or `State` widens.
  - (c) (a) now, (b) later. Pros: the mask and the per-token loss land first, measured. Cons:
    the study measures the slower path; the input is redone.
- **D6. The output and the loss. Settled: (a).**
  - (a) *Chosen.* A token part may end in a token-wise `Dense(V, "softmax", output=True,
    loss="cross_entropy")`: a softmax over each token's `V` outputs, the loss the mean of the `T`
    tokens' cross-entropies (the output delta `(P - Y) / T`), a `sequence` network shape beside
    `multiclass` and `single_output` (one-hot targets per token, `classify_state` the per-token
    argmax), and format 2 and the legacy-free load path for it. Pros: one spec, the existing
    output layer applied per token as the token-wise dense layers are; the mean keeps a step's
    size independent of `T`, as GPT-style training does. Cons: validation, the builders, the
    shapes, the save format and the trainer's targets all learn a third shape.
  - (b) As (a) with the loss summed over the tokens. Pros: matches the per-example losses of the
    other shapes exactly (each token a full example's weight). Cons: the learning rate then scales
    with `T`; studies at different `T` aren't comparable.
  - (c) A separate `TokenOutput(V)` spec. Pros: explicit. Cons: a second spelling of a softmax
    output layer; `Dense` already means "this layer, per token" inside a token part.
- **D7. The mask. Settled: (a).**
  - (a) *Chosen.* `Attention(causal=False)`, a field with a default, written to format 2 only
    when true. In attend, `S_ij` for `j > i` is set to `-inf` before the max shift: the row max
    comes from the unmasked entries (the diagonal is never masked), `exp(-inf) = 0` exactly, so
    `P_ij = 0` exactly and the backward pass needs no change. The crate's four attention ops take a
    `causal` keyword into `AttentionOptions`. Pros: step 5's named place; per layer, so an encoder
    block and a causal block could mix later; exact zeros keep parity by bits where it held. Cons:
    a new keyword through the crate's API (a crate PR first, then a bump).
  - (b) A network-level flag applied to every attention layer. Pros: one switch. Cons: a property
    of a layer stored on the network; blocks a mixed model later.
  - (c) A general additive mask input. Pros: padding and other masks for free. Cons: a second
    input to a layer, which `Sequential` doesn't have; nothing here needs it (D1 (b)).
- **D8. Evaluation. Settled: (a).**
  - (a) *Chosen.* Mean per-token cross-entropy on the held-out windows, in nats and in bits
    per character, and per-token accuracy. Pros: the corpus's standard measure; accuracy keeps the
    reports' existing form. Cons: a second evaluation function beside `multiclass_evaluate`.
  - (b) Per-token accuracy only. Pros: no new function. Cons: accuracy hides most of a language
    model's progress (the top choice is often right long before the distribution is).
- **D9. The study. Settled: (a).** `scripts/sequence_study.py`, numpy, Adam, one rate per arm
  from a short `tune`, 3 or 5 seeds, timed first and the grid sized from it, on the patch studies'
  protocol where it fits.
  - (a) *Chosen.* Arms: a unigram and a bigram model computed from counts (no training: the
    floors); an FFN-only model, each token blind to the others apart from positions; 1 and 2
    layers of a causal transformer at `d = 64`, 4 heads; and the 2-layer model without the mask,
    the leak: its loss should fall far below every causal arm's, held-out included. Pros: each arm
    answers one question (does attention over the past beat a per-token model; does depth help;
    does the mask do its job). Cons: the leak arm spends a cell on a known result.
  - (b) (a) without the leak arm, which becomes a test instead. Pros: a cheaper grid. Cons: a test
    on a toy corpus shows the leak exists, not its size on real text.
- **D10. Pure Python. Settled: (a).**
  - (a) *Chosen.* All three implementations, the pure-Python one parity-only at small `T` and
    vocabulary, as step 5. Pros: the project's rule; the mask is a few lines there. Cons: the
    sequence shape and `Embedding` (if D5 (b)) in a third implementation.
  - (b) numpy and Rust only (D1 (c)).
- **D11. Generation. Settled: (a).**
  - (a) *Chosen.* Out of scope. Pros: the study measures loss; sampling needs a decode loop
    that the fixed-`T` network doesn't have (a prefix shorter than `T`, or a sliding window). Cons:
    no text to read, which is the most persuasive demo of a language model.
  - (b) A small `scripts/sample_text.py`: a sliding window of the last `T` characters, sampling
    from the output at the last position. Pros: a readable demo, cheap once the model trains.
    Cons: `T` forward passes per sampled window's worth of text; a new script to keep working.
- **D12. Golden run. Settled: (a).**
  - (a) *Chosen.* New entries for a causal sequence model in each implementation, as #612
    did for multi-head (the old entries bit-identical, the file re-recorded with the new entries
    on both machines and archived as `new-functionality`). Pros: the established form. Cons: none
    beyond the re-record.

## Stages

1. **The corpus**: the dataset repository (D3), `scripts/fetch_datasets.py`'s entry, a loader in
   `indrajala_ml/data/` (vocabulary, windows, split) and its tests.
2. **Specs**: `Embedding`, `Attention(causal)`, a token part ending in a token-wise output, the
   `sequence` shape in `spec_shapes` and validation, format 2's entries. No layer yet. Done:
   `Embedding(vocabulary, size)` starts a token part as `Patches` does, over a flat input of `T`
   ids; a token part without `TokenMean` ends in the output layer, applied to each token, which is
   softmax (`token_wise_output`, the `sequence` shape's specs); `Attention(causal=False)`. Format 2
   has an `"embedding"` entry, its table `E`, and writes `"causal"` only when true. Until their
   stages the builders refuse all three with "not yet" (`refuse_sequence_specs_until`): numpy at 3,
   Rust at 5, pure Python at 6. Format 2's `"sequence"` network shape comes with the network, at 3.
3. **numpy**: the mask in attend, `Embedding`, the token-wise softmax output and loss, the
   `sequence` shape, per-token targets in the trainer, the evaluation (D8). Tier 1 A/B of the
   attention case (its unmasked path must not move). Done: a causal `AttentionArrayLayer` sets
   `S_ij`, `j > i`, to `-inf` before the max shift (an unmasked one computes what it did);
   `EmbeddingArrayLayer` reads ids (refusing any that isn't a whole number in range), its gradient
   a scatter-add in row order (`np.add.at`), its `E` drawn as a weight matrix of fan-in `size`
   and not decayed, as `P` isn't; `TokenSoftmaxArrayLayer`, its delta `(P - Y) / T`, its sum a
   left fold; the `sequence` shape (`SequentialSequenceShape`, format 2's `"sequence"`,
   `load_network`), which the multiclass and single-output shapes refuse; the trainer's accuracy
   per token (a window counts the fraction of its tokens right); `sequence_evaluate` (D8).
4. **The crate** (`indrajala-math-rust`): `causal` in `AttentionOptions` and the four ops,
   `embedding_*` ops; then a "Bump rust/" PR here. Done (indrajala-math-rust #53): the two
   forward ops take `causal` (keyword, default false), masking as numpy does; the two backward
   ops take none, as they read `P`, whose masked weights are exactly 0. An unmasked pass keeps
   its bits. `embedding_forward(x, table)` and `embedding_accumulate_gradient(delta, x,
   grad_table)`, the scatter-add in row order (`np.add.at`'s bits), refuse ids as numpy does. The
   token-wise softmax output needs no new op: `array_softmax`'s row sum already was a left fold,
   now written out and pinned by bits against `np.cumsum`, so stage 5 compares it to numpy by
   bits (with numpy's `exp` replaced, as for attention).
5. **Rust layers**: the Rust `Embedding` and the mask through the ops; parity with numpy.
6. **Pure Python**: the mask, `Embedding`, the sequence shape; parity.
7. **Fixtures, checkpoints and golden entries** (D12).
8. **The study** (D9).
9. **Docs**: README, roadmap, next-steps; the workplan retired.
