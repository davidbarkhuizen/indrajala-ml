# Workplan: ONNX export and the Hugging Face Hub

**Opened 2026-11-11.** The owner: today a model is saved and loaded only in our own format, as a
local file. In future it must also:
1. export to a more universal format, ONNX;
2. save to and load from Hugging Face, presumably through a custom extension of Hugging Face's own
   format, where that is pluggable.

This plan settles how, in stages; nothing is built yet. Its decisions are D1-D14 and the owner's calls
O1-O3.

## The short answers

- **Hugging Face has no single universal model format to extend.** A Hub model is a git repository
  of files, with three conventions:
  - `config.json`, the model's description;
  - weights in **safetensors**, a flat file of named tensors, with no pickle and no code;
  - `README.md`, a model card with YAML metadata, among it `library_name`.
- **transformers' format is not open to us.** It means `PreTrainedModel` subclasses in PyTorch, and
  custom ones load with `trust_remote_code`, as PyTorch code. Ours is a numpy and Rust library with
  no torch, so that route would mean rewriting the models.
- **The pluggable point is `huggingface_hub`'s `ModelHubMixin`.** Its docstring says "a generic
  mixin to integrate ANY machine learning framework with the Hub". It takes a library's own
  `_save_pretrained` and `_from_pretrained` and adds `save_pretrained`, `from_pretrained`,
  `push_to_hub` and a model card. This is what timm and sentence-transformers do: their own
  `config.json` and safetensors under their own `library_name`.
- **Our extension:** our model description (format 2's document, without its arrays) as
  `config.json`, the arrays in `model.safetensors`, `library_name: indrajala-ml`, and optionally a
  `model.onnx` beside them. That last file makes the repository usable without our library.
- **ONNX:** an exporter that writes each network's *inference* graph from its saved description,
  checked against our numpy forward. The window-recurrent network exports as its transition, one
  step a call: (state, a window) → (outputs, the next state). That is the finite-state transducer's
  step, the form the research proves things about.

## What we have (2026-11-11)

**Format 2** (`indrajala_ml/model/persistence/format2.py`):
- **One JSON document a network:**
  - its `format`, `implementation` (`python`, `numpy`, `rust`) and `shape`;
  - its preset (class and arguments), if it has one;
  - its `input`, the layer specs (`layers`), and `recurrence` or `encoder` for those shapes;
  - its `update_rule`;
  - its `weights` as nested float64 lists, its `optimizer_state` (Adam's `m` and `v` per
    parameter) and its generator's state (`rng`).
- **It resumes training by bits.** A numpy file loads into Rust and back. A pure-Python file holds
  its weights per node and loads into pure Python only.
- **Loading builds from the specs, never from a class named in the file** (`load_network.py`). That
  matters on a hub: a downloaded file can't name code to run.
- **Shapes:** `multiclass`, `single_output`, `sequence`, `window_recurrent`, `encoder_decoder`,
  and `ensemble`, one nested document a one-vs-rest sub-network.
- **Layer kinds:**
  - dense (sigmoid, ReLU, softmax and linear), conv (valid, ReLU or linear), max pool, batch norm;
  - residual (fork, body, add), dropout, patches, token mean;
  - embedding, position, rotary positions, layer norm;
  - attention (heads, grouped key/value heads, causal, segments), cross-attention, gated dense;
  - state input (additive or FiLM), branches, split and merge.
- **The window-recurrent recurrence:** factors, each a GRU cell; codes, nearest of `m`
  straight-through, with linear (SimVQ) codes `E = Q W`; attention pooling; a learned initial
  state.
- **The tokenizer** is a separate file beside the model, `<model>.tokenizer.json`
  (`generation/tokenizer.py`), kind `characters`.
- **Cost:** about 20-30 bytes a parameter with the optimizer's state. The NES checkpoints are
  8-17 MB, and a 10M-parameter model would be about 300 MB of JSON. The load parses every float
  through Python.
- **The arithmetic is float64 throughout,** on numpy and on the crate.

**The packages** (PyPI, 2026-11-11):

| package | version | wheels for 3.14 |
|---|---|---|
| `onnx` | 1.23.2 (opset 28, IR 14) | abi3, x86_64 and aarch64 |
| `onnxruntime` | 1.31.1 | cp314, x86_64 and aarch64 |
| `safetensors` | 0.8.0 | abi3, aarch64 included; numpy API, no torch |
| `huggingface_hub` | 2.4.0 | pure Python; `ModelHubMixin` still there |

So `litshe` can run the aarch64 parity checks.

**A probe of onnxruntime 1.31's CPU provider** (scratch, 2026-11-11; one tiny graph an op):
- **The IR version must be pinned.** onnxruntime reads IR 13 at most, and onnx 1.23 writes 14 by
  default, so every graph failed to load until the probe set `ir_version=10`.
- **float64 runs** for MatMul, Gemm, Relu, Sigmoid, Tanh, Exp, Softmax, LayerNormalization,
  MaxPool, BatchNormalization, Gather, ReduceMean and opset 23's `Attention` (causal).
- **float64 doesn't run** for Conv (no kernel), AveragePool (no kernel), or opset 23's
  `RotaryEmbedding` (its schema allows no double).
- **float32 runs all of them.**

**What `ModelHubMixin` does to a class that inherits it:**
- It overrides `__new__`, recording every constructor's arguments.
- It writes `config.json` from them.

We have dozens of network classes, Sequential networks built from spec lists, and Rust-backed
twins, and our description is the format-2 document, not the constructor's arguments. Mixing it into
the network bases would fight both (D5).

## Decisions

I settled these myself (the owner delegates workplan decisions); the owner's calls are O1-O3. Each
lists every option it weighed.

### D1: one directory form, locally and on the Hub

- **(a) The Hub gets its own layout, and format 2 stays the only local one.**
  - Pros:
    - Nothing local changes.
  - Cons:
    - Two writers and two readers, for the same thing.
    - A model downloaded from the Hub isn't a file the rest of the code reads.
- **(b) Format 3: a directory, with format 2's document as `config.json` and its arrays in
  safetensors. One reader and writer, locally and on the Hub. Format 2 stays as it is, read and
  written.** **Chosen.**
  - Pros:
    - One code path.
    - A Hub download is a local model.
    - The load needs no float parsing: safetensors maps the file.
    - Its size is 8 bytes a float64 instead of about 20.
  - Cons:
    - A third format to keep (format 2 and the legacy envelopes stay).
- **(c) Replace format 2.**
  - Pros:
    - One format.
  - Cons:
    - The golden run's files, every study's saved networks and the datapacks are format 2.
    - Paper 1's datapack must stay readable.

Format 3's `config.json` is format 2's document with `"format": 3`:
- Its `weights` are replaced by `"tensors": "model.safetensors"`.
- Its `optimizer_state` and `rng` move to `training_state.json` and `optimizer.safetensors`,
  present only in a resumable save (D4).
- Every other field is format 2's, written by the same code (`format2_json.py`).

### D2: the tensors' names

- **(a) Positional:** `0`, `1`, ... in snapshot order.
  - Pros:
    - Trivial.
  - Cons:
    - Unreadable on the Hub's file viewer.
    - A layer added in the middle renames everything after it.
- **(b) By expanded layer and parameter:** `layers.3.Wq`, `layers.3.bq`, `layers.0.E`;
  `recurrence.0.cell.W`, `recurrence.0.codes.E` (or `.Q`, `.W` for linear codes),
  `recurrence.0.pooling.q`, `recurrence.0.initial`; the optimizer's as `layers.3.m_Wq`. An
  encoder-decoder's layers are numbered over `expand_specs(encoder + decoder)`, as format 2's
  weights are; an ensemble's are `classifiers.4.layers...`. **Chosen.**
  - Pros:
    - Readable.
    - They are format 2's own names (its docstring's per-layer lists), so the mapping is
      checkable in both directions.
  - Cons:
    - Each layer kind's parameter names become part of the format.
    - A rename is a format change.
- **(c) torch-style names,** to look like transformers' checkpoints.
  - Pros:
    - Familiar to people who know transformers.
  - Cons:
    - Our layers aren't torch's; the names would promise a correspondence that isn't there.

**The pure-Python implementation stays format 2.** It holds its weights per node; it is for
correctness and parity, and its networks are small. A converter to the array layout is possible
later, as the numpy twins compute the same.

### D3: the dtype on disk

- **(a) float64, always.** **Chosen.**
  - Pros:
    - Exact.
    - A format-3 round trip trains on by bits, as format 2's does, so the golden discipline holds.
  - Cons:
    - Twice float32's size. At our sizes, megabytes.
- **(b) float32 for the Hub.**
  - Pros:
    - Smaller.
  - Cons:
    - It isn't the trained network.
    - A window-recurrent network's codes can snap differently at a near tie (D10).
    - A smaller inference copy is ONNX's job (D10).

### D4: training state, optional and apart

- **(a) Weights only.**
  - Pros:
    - What an inference user wants.
  - Cons:
    - No resuming from the Hub, where the format-2 file resumes by bits.
- **(b) Weights always, and with `training_state=True`, the optimizer's state and the generator's
  in their own files** (`optimizer.safetensors`, `training_state.json`). **Chosen.**
  - Pros:
    - A download for inference skips them (`allow_patterns`).
    - Resuming is still exact.
  - Cons:
    - Two layouts of one directory, told apart by the files present (the reader checks).
- **(c) Always everything.**
  - Pros:
    - One layout.
  - Cons:
    - Adam's moments triple every download.

### D5: how we plug into the Hub

- **(a) `ModelHubMixin` on the network bases.**
  - Pros:
    - The standard method names on every network.
  - Cons:
    - It overrides `__new__` on every network class, Rust-backed ones included.
    - It writes `config.json` from constructor arguments, which aren't our description (a
      Sequential network's are spec objects), so it needs a coder for each spec type.
    - A loaded class decides what is built, against `load_network`'s rule.
- **(b) Free functions over `huggingface_hub`'s plain API:** `save_pretrained(network, directory)`,
  `push_to_hub(network, repo_id, ...)` (`HfApi.create_repo`, `upload_folder`), and
  `from_pretrained(repo_id_or_directory, revision=...)` (`snapshot_download`, then format 3's
  reader). The model card is written with `ModelCard` and `ModelCardData`, `library_name:
  indrajala-ml`. **Chosen.**
  - Pros:
    - No metaclass behaviour on our classes.
    - Loading stays "built from the specs".
    - The same names users know, as functions.
    - Easy to test offline: everything but the upload and download runs on a directory.
  - Cons:
    - No `Network.from_pretrained` classmethod. `from_pretrained` returns whatever the files
      describe, as `load_network` does.
- **(c) A small wrapper class, `HubModel(ModelHubMixin)`, holding a network.**
  - Pros:
    - The mixin's card generation and its API.
  - Cons:
    - A wrapper type around every result.
    - Its `config.json` handling still has to be bypassed.

**Pinning:** `from_pretrained` takes a `revision`, a commit, so a study can cite the exact weights
as it cites a Zenodo DOI. The card records the ml and crate commits the network was trained at.

### D6: the tokenizer's file name

- **(a) `tokenizer.json`.**
  - Pros:
    - The usual name.
  - Cons:
    - It is the `tokenizers` library's own format. transformers and the Hub's widgets would try
      to read ours as theirs and fail.
- **(b) `indrajala_tokenizer.json`,** our format unchanged. **Chosen.**
  - Pros:
    - No clash.
    - The same file as `<model>.tokenizer.json`, renamed.
  - Cons:
    - Tools that look for a tokenizer won't find ours. That is right, as they can't read it.
- **Later, not now:** a `tokenizers`-format export of a character tokenizer. Every character is
  one token, so it can be expressed as `tokenizers`' word-level model over a per-character
  pre-tokenizer.

### D7: dependencies

- **(a) Core dependencies.**
  - Pros:
    - Always there.
  - Cons:
    - `onnx`, `onnxruntime` and `huggingface_hub` for everyone who trains a perceptron.
- **(b) Extras: `indrajala-ml[hub]` (`safetensors`, `huggingface_hub`) and `indrajala-ml[onnx]`
  (`onnx`). `onnxruntime` is for tests and checks only (the dev dependencies, and CI).** Format 3's
  reader imports `safetensors` lazily, with a message naming the extra. **Chosen.**
  - Pros:
    - Small core.
    - Clear failures.
  - Cons:
    - CI must install the extras, so its jobs grow by about 30 MB of wheels.

### D8: the default save stays format 2

`network.save(path)` keeps writing format 2. Format 3 is `save_pretrained(network, directory)`, and
`load_network(path)` reads both (a directory with `config.json` is format 3).
- **The alternative,** format 3 by default: it changes every study's output, for no gain until a
  model is shared.
- **Revisit:** when a model passes about 10M parameters, or a study saves many.

### D9: how the ONNX graph is built

- **(a) `onnx.helper`, our own emitter a layer kind,** walking the specs as `spec_shapes` does.
  **Chosen.**
  - Pros:
    - One dependency.
    - Every node is ours to check.
    - It mirrors the builders, one function a kind.
  - Cons:
    - More code than a tracer.
- **(b) `onnxscript`.**
  - Pros:
    - Graphs written as Python functions.
  - Cons:
    - A second dependency and a younger API, for graphs that are mostly MatMul and Add.
- **(c) Through torch** (rebuild in torch, then `torch.onnx.export`).
  - Pros:
    - None that holds.
  - Cons:
    - A second implementation of every layer, and torch.

**The exporter reads the description, not the live network.** Its input is a `NetworkFile`
(format 2) or a format-3 directory. So numpy and Rust networks export alike, and so does a Hub
download. It is backend-free, as `format2.py` is: `indrajala_ml/model/export/onnx/`.

### D10: precision (revised 2026-11-11, with the owner)

**The owner:** portability first, but a machine that no longer snaps as trained makes the
exercise pointless. So the default is float32 everywhere, and a window-recurrent network's float32
export must earn it.

**Why the shapes differ:**
- **A continuous network** (the classifiers, the sequence models): float32 moves each number by
  about 1e-7 relative, and nothing feeds a discrete choice back. An answer changes only at a
  genuine near-tie between two outputs, and the error doesn't grow.
- **A window-recurrent network with codes:** each step snaps the state to the nearest code, and
  that code is the next step's input. A flip changes every later step: the machine itself changes.
- **But snapping corrects error.** Where the winning code beats the runner-up by more than
  float32's error, the snap lands on the same code, the state is that code vector again, and the
  error resets to zero each step. So float32 computes the same machine exactly when, at every
  reachable (state, window) pair, the next code's margin and the output's argmax margin exceed
  float32's error.
- **For a token window, that is a finite check:** finitely many reachable states, finitely many
  distinct windows (the family's 11 tokens, 3 a column: at most 1,331).
- **A continuous factor** (codes None) has no snapping: float32's error never resets and drifts
  over long sequences.

**The rule:**

1. **float32 is the default for every shape.** It is what runtimes, accelerators and converters
   expect. `dtype="float64"` is always available.
2. **A window-recurrent network's float32 export runs the exhaustive check** (`check_machine`,
   D13 step 4) on the export's own float32 graph, through onnxruntime:
   - Its input is a set of states. The library verifies the set; it doesn't find it. The
     studies' certificate code supplies it: a certified network's reachable states.
   - **It starts** from the float32 graph's own initial state, snapped.
   - **For every state in the set and every window** (the vocabulary to the window's width, its
     padding included), it checks that the next code equals float64's, the next state lies in the
     set, and the output's argmax equals float64's.
   - **Passing proves** that the float32 graph computes the same transitions from the same initial
     state, so the same machine at every length (the certificate's induction).
   - **Its scope:** this runtime. A float32 runtime that sums in another order could differ.
3. **The smallest margins found go into the export's metadata and the model card:** the next
   code's (the runner-up's distance less the winner's, relative to the winner's) and the output
   argmax's.
   - A margin far above float32's error (1e-3 against about 1e-7) is strong evidence that any
     faithful float32 runtime agrees. It also says whether float16 would hold.
   - **Evidence, not proof:** a rigorous bound on float32's error through attention, layer norm
     and softmax is not planned. The card says which runtime the proof covers.
4. **The exporter refuses float32 and says why** when:
   - the check fails, a margin is under `margin_floor` (default 1e-4), or no state set is given
     for a coded network;
   - or the network has a continuous factor.
   The user then exports in float64, or, for a coded network, gives the state set.
5. **Array windows** (continuous inputs: the NES observations) can't be enumerated:
   - Their check runs on given sequences, and the card says "sampled".
   - Their float32 export is allowed, never called faithful to the machine.

**Where float64 has no kernel** (onnxruntime: Conv, AveragePool, `RotaryEmbedding`), the emitter
decomposes: Conv as im2col and MatMul, rotary as constant cos and sin tables. This is the same
graph in both dtypes, so a float64 export of any shape runs on onnxruntime's CPU provider.

**On the Hub,** `model.onnx` is the float32 graph where it passed, with `model_fp64.onnx` beside
it for a window-recurrent network. Otherwise `model.onnx` is float64.

**The cost of float64, for the record** (from the runtimes' documentation as known, not tested
here except onnxruntime's CPU provider):
- onnxruntime runs a float64 graph wherever it runs, falling back to CPU for nodes an
  accelerator lacks.
- TensorRT, CoreML, DirectML, NNAPI, QNN and WebGPU have no float64.
- OpenVINO and the TFLite converters downcast to float32 silently, the risk this rule closes.

**The options weighed:**
- **(a) float32 only.**
  - Pros:
    - Portable.
  - Cons:
    - A coded network's machine can change unseen.
- **(b) float64 only.**
  - Pros:
    - Faithful.
  - Cons:
    - It loses accelerators.
    - A downcasting converter reopens the risk with no check at all.
- **(c) float64 for window-recurrent networks, float32 for the rest** (this plan's first choice).
  - Pros:
    - Safe.
  - Cons:
    - It gives up portability where the check shows float32 holds.
- **(d) float32 by default, a coded network's export gated by the exhaustive check and its
  margins.** **Chosen.**
  - Pros:
    - Portable where it is proved safe.
    - The margin tells users of other runtimes how safe.
  - Cons:
    - The check needs the reachable states from the studies.
    - The proof covers one runtime.
    - Array windows get evidence only.

### D11: opset and IR

**Opset 23** (2025: `Attention` with grouped key/value heads and an additive mask, and
`RotaryEmbedding`). **IR version 10**, pinned explicitly: onnx writes IR 14 by default, which
onnxruntime 1.31 refuses (its maximum is 13).
- **The alternatives:**
  - An older opset, decomposing attention everywhere: runs on old runtimes, loses the fused kernel.
  - The newest, 28: buys nothing we use.
- **Revisit** when onnxruntime's maximum IR passes 10 and a newer op is wanted.

**Attention uses the `Attention` op** (fused in runtimes), with the segments' mask as its additive
mask. **Rotary is always decomposed** (D10). A unit test checks that the op's scaling and masking
match ours on a case with grouped heads, causal and segments.

### D12: what graphs each shape exports

| shape | graph(s) | inputs → outputs |
|---|---|---|
| `multiclass`, `single_output` | one | `x` (N, d), or (N, h, w, c) in our channels-last layout, with a Transpose inside where Conv needs NCHW → probabilities (or outputs) |
| ensemble (one-vs-rest) | one | each sub-network's probability, concatenated, (N, classes) |
| `sequence` | one | ids (N, T), optional segments (N, T) → per-token probabilities (N, T, V) |
| `window_recurrent` | **the step** | state (N, Σ s_g), a window's tokens (N, k) or array → the window's outputs, the next state, and each factor's code index; the initial state as a graph constant output (`initial_state`) |
| `encoder_decoder` | three | the full forward (source, target) → probabilities; the encoder; and a decoder step with its key/value cache in and out, the form optimum's `decoder_with_past` uses |

- **The window-recurrent network exports as its step.** That is the transducer's transition and
  output: the host loops it over a sequence, and a user can read the machine off it. A whole-sequence
  graph (ONNX `Scan`) is a later option, once someone needs one call a sequence.
- **Training is not exported.** ONNX has training artifacts, but our training is the library's.
  Not planned, not excluded.
- **Dropout is the identity, and batch norm uses its running statistics,** as our inference does.

### D13: how an export is checked

Every export, in its tests and at export time (`check=True` by default), passes:
1. `onnx.checker.check_model` and `onnx.shape_inference` (strict).
2. **The same outputs as our numpy forward** on fixed seeded inputs:
   - float64: within 1e-12 relative. Not bits: the summation order differs.
   - float32: within 1e-5.
   - Under both `onnx.reference.ReferenceEvaluator` (pure numpy, every dtype, the spec's own
     semantics) and onnxruntime.
3. **Classes:** the same argmax on every input; a disagreement is reported, as at a near tie.
4. **Window-recurrent networks:**
   - Over a run of windows looped by the host, the same code index at every step (float64).
   - For float32, `check_machine`, D10's exhaustive check over a given state set and every
     window. It is a proof for token windows on the runtime it ran on, sampled for array
     windows, and it records the smallest margins.
5. **aarch64:** the same on `litshe` once a stage's emitters are whole (the wheels exist).

### D14: where it lives, and the user's entry points

- `indrajala_ml/model/persistence/format3.py`: the directory format (D1-D4), reusing
  `format2_json.py`.
- `indrajala_ml/model/hub.py`: `save_pretrained`, `push_to_hub`, `from_pretrained` and the model
  card (D5).
- `indrajala_ml/model/export/onnx/`: `export_onnx(network_or_path, path, dtype=..., check=True)`,
  one emitter module a family of layer kinds (D9-D13).
- A script, `scripts/export/export_onnx.py PATH OUT [--dtype]`, for saved networks.

## The owner's calls

- **O1: the Hub namespace and the token.** Under your user name or an organisation
  (`indrajala`?). The login is yours: `! hf auth login` in this session, or a token in the
  environment. **Every push is outward-facing:** the first push, and each model made public, needs
  your OK, as Zenodo does. Repositories are created private by default.
- **O2: which models to publish, and their licence.** MIT, as the code, is the obvious default.
  The candidates are the certified window-recurrent networks behind paper 1 and the certificate
  note: their datapack is on Zenodo already, so a Hub copy is a second home. Each card carries the
  AI declaration (model and harness, use at every stage, no AI author).
- **O3: registering `indrajala-ml` as a Hub library** (a PR to `huggingface.js`'s
  `model-libraries.ts`). It gives a "Use this model" snippet and download counts by our own rule.
  Only after the package is on PyPI (next-steps' infrastructure 52); not needed before.

## Stages

Each stage is one PR, merged before the next, with tests in every implementation it touches.
**Tier 0 for timing** (no timed path changes), but stages 1-2 change the package's save and load,
so they pass the golden run.

1. **Format 3** (D1-D4, D6-D8):
   - The directory writer and reader for numpy and Rust networks, ensembles included, with the
     tokenizer beside them.
   - `load_network` reads both formats.
   - Tests:
     - a round trip that trains on by bits, numpy into Rust and back;
     - inference-only and resumable directories;
     - every layer kind's tensor names against format 2's parameter lists;
     - a refusal naming a missing extra.
   - The `[hub]` extra.
2. **The Hub functions** (D5): `save_pretrained`, `push_to_hub`, `from_pretrained(revision=...)`,
   and the model card.
   - Offline tests run on a directory, with the Hub's upload and download replaced.
   - One live test is skipped unless `HF_TOKEN` is set.
   - **Then, with O1:** one tiny network pushed to a private repository and pulled back by
     revision, bits checked. This is the stage's acceptance.
3. **The ONNX exporter's core** (D9-D13): dense (every activation), dropout, batch norm, residual,
   and the one-output, multiclass and ensemble shapes; float32 and float64; both evaluators. The
   `[onnx]` extra.
4. **Conv, max pool, patches, token mean;** float64 conv as im2col.
5. **Sequence models:** embedding, position, rotary (decomposed), layer norm, attention (heads,
   grouped heads, causal, segments, through the `Attention` op), gated dense, token-wise outputs.
6. **The window-recurrent step:**
   - state inputs (additive and FiLM), branches, split and merge;
   - the poolings, the GRU cells, the codes (ArgMin and Gather; linear codes' `E = Q W` computed at
     export), continuous factors, the initial state;
   - array windows;
   - the code-sequence check (D13, 4) and `check_machine` (D10), tested against a certified
     division network's states from the studies' data.
7. **Encoder-decoder:** the full forward, the encoder, and the decoder step with its cache.
8. **Together, and on aarch64:**
   - `push_to_hub(..., onnx=True)` adds `model.onnx`, which tags the repository `onnx`.
   - The checks re-run on `litshe`.
   - The README's section on saving, sharing and exporting.
   - Retire the plan.

**Order and gates:** 1 before 2; 3 before 4-7; 4-7 in any order (5 before 7, which reuses its
attention). Stages 3-7 don't need 1-2: the exporter reads format 2 too. So the ONNX line can start
first if the Hub waits on O1.

**Not in this plan, each when asked for:**
- the pure-Python networks in format 3 or ONNX (D2);
- a whole-sequence window-recurrent graph (`Scan`);
- ONNX training artifacts;
- a `tokenizers`-format tokenizer (D6);
- quantised exports;
- the rl repository's policies (an rl wrapper over these functions).
