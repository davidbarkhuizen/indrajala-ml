# Next steps: work to be done

**The goal all of it serves:** [machine-discovery.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/machine-discovery.md), discovering finite-state
machines from behaviour and known structure, each discovery proved; its route in stages A-F.

This file lists work that no open workplan holds: what a retired workplan left open (its "After this
plan" list, and what its "Out of scope" left unbuilt), with enough of the reasoning to take it up.
When an item is done, it is deleted here; the record of what was built is the README, the study
scripts' docstrings, and the retired plan itself ([workplans/retired.md](workplans/retired.md) says
how to read each from git history). Nothing listed here is excluded: "not planned yet" means only
that no workplan has taken it up.

The open workplans keep their own lists: this repository's are indexed in
[workplans/README.md](workplans/README.md) (the parked `jebel` plan holds the crate's remaining speed
work), the research plans in indrajala-research's
[workplans/README.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/workplans/README.md).
Of the research plans, the general algebras, the factorisation machine and the adders of any length
have been paused since 2026-10-20; each waits for a decision to resume or retire (the programme
audit's F29). The candidate next ML primitives are gathered in [primitives-roadmap.md](primitives-roadmap.md).

Sections are named for the plan that left the work ("From ..."), so citations of the form
"next-steps, From X" find them.

## From the array window (retired 2026-11-08)

The window-recurrent network's window of array layers, learning from a given gradient at every
step, from stored states with resets (`git show 32d4d23:docs/workplans/library/array-window-workplan.md`;
the machine-policy plan's M1, indrajala-research). Not planned yet:

- **A supervised per-step output on an array window** (its D3 (b)): a softmax output layer with a
  label per step. Only the given loss is accepted now; a task that needs labels gets the target
  layout and its loss weighting then.
- **The pure-Python implementation has no array window** (D8): `WindowRecurrentBackpropNetwork`
  refuses one by name; a Python conv front end has no user.
- **The program's array ops are the policy's** (D7): conv, max pool, a StateInput, a flat ReLU dense
  layer and the given output. An array window holding a residual block, a LayerNorm, a sigmoid dense
  layer, a dense layer with dropout or a FiLM StateInput runs the per-layer path (correct, slower).
  Each gets an op, checked by bits against the per-layer path, when a window needs it.
- **No golden entries for an array window:** the golden run holds token windows only; the array
  window is pinned by its tests (central differences on numpy, numpy against Rust within rounding,
  the program against the per-layer path by bits). Entries when its bits must be watched across
  refactors (M2's recipe, if it settles on one).
- **The state enters after the conv stack only as the policy places it** (D2 (a)); D2 (b) (only
  before the output layer) and (c) (as extra planes into the first conv) wait for M5's probes.
- **A candidate +1.2% in the program's inference forward on token windows** (S4's guard, klip:
  `window_recurrent_timing`'s `forward family p7 f32 n1024`, consistent through ONNONO, 5.761 ->
  5.832 s; within noise on stone, whose control moved). No mechanism found: the forward's
  per-window work is unchanged but for a restarts `Option` check and the temperatures argument.
  Re-check on `jebel` when it is back (an A/B of crate f73d652 against 2c4415b, then
  `kernel_code_diff.py` on `program::forward` if it holds), with the parked plan's runs.

## From the crate performance workplan (retired 2026-11-05)

Its `jebel`-dependent work is **[parked](workplans/infrastructure/jebel-parked-workplan.md)** (the
owner's call, as `jebel` resets itself). Not in the parked plan:

- **K17:** SVE or AVX-512 kernels (`exp`, the transpose, the products), in the canonical order
  (the aarch64 plan's D5), when a machine with them is in hand. Not on the Pi 5's Cortex-A76.
- **K18:** a bits-changing kernel (fused or reassociated arithmetic, in the window program too:
  the window program plan's D3 (b)); it needs the owner's approval and a re-recorded golden run
  (measurement.md, §8).
- **K19:** musl's golden run. CI builds and tests the crate on musl (Alpine, both architectures),
  not the committed reference; a check to add if musl becomes a target.
- **K20:** numpy's expressions (the reference path's own speed).
- **K22:** checkpointing the window's forward (the window program plan's D5 (c)): keep every
  `r`-th window's forward pass and recompute the rest, if a study trains windows long enough to
  pass the memory budget (2.9 MiB at E2b''s cell, 21.7 MiB at the fork probe's, against 2 GiB).
- **K13:** a batched decode step: Rust's batched decode runs `attention_step` once a row, each
  row's cache its current paragraph's keys (the segments workplan, D8 (b)); a crate step over an
  `(N, c)` cache with the keys' segments, numpy's form (D8 (a)), would batch it and give the two
  one cache form. No epoch reaches it: on `jebel` (2026-11-04) Rust reads 20,173 characters a
  second at batch 8 and 20,806 at 32, flat, against numpy's 7,770 and 13,652.
- **The next crates.io and PyPI release is 0.2.0** (below, From the release).

## From cross-attention (retired 2026-11-03)

The encoder-decoder stays the translation model, decided from the literature (README,
Cross-attention and the encoder-decoder, "When to use it"). Left:

- **An encoder-decoder that reads its source.** The translation probe (#1242, 2026-11-04;
  `scripts/studies/translation_probe.py`'s docstring) was abandoned by the owner: d 128, 2 + 2
  blocks, the Lives, 23 epochs on stone; English bits per character fell from 3.33 to 1.81, but the
  source ablation's gap (another window's Greek against the window's own) reached only 0.027 bits,
  growing about 0.001 an epoch, and the greedy outputs ignore the Greek (chrF 20.6). No translation
  claim rests on the model until a run learns to use its source.
- **A decoder-only translation model (D18 (c)):** the Greek as a prefix, the loss on the English
  only (counted targets by position, not only "not a pad"), when a question needs the comparison.
- **D10's out-of-scope items:** pretraining the encoder on Greek alone, a BPE tokenizer over both
  scripts, beam search.
- **Batched cross-attention steps:** each step attends from the cached keys and values in one call
  a pass (crate `cross_attention_attend_batch`), but the decoder's self-attention steps row by row
  on Rust and in pure Python, as step 9's do.
- **No benchmark times cross-attention:** `attention_timing` times self-attention only; a change to
  the `cross_attention_*` ops has nothing to A/B against until one is added (measurement.md).

## From the audit remedies (retired 2026-11-03)

- **The studies (D11):** each codebook primitive the plan built (k-means codes, re-seeding dead
  codes, the codebook's rate, SimVQ's codes, the selection set, He initialisation, the FiLM state
  input, MRI) is a candidate in indrajala-research's state-formation workplan (research #139),
  judged only by a registered study there, on fresh-seed certified rates against the failure kind
  it targets. Nothing here adopts one.
- **`window_study_timing.py`'s default is the long test alone:** a crate change to the window
  program also needs `window_recurrent_timing.py`'s learn and forward cases (#1212's second A/B).
  Adding them to the default would make one A/B enough. In #1212's first, the numpy `matmul 256`
  control moved -22.9%, which nothing in the crate reaches (unexplained).

## From grouped-query attention (retired 2026-11-03)

Multi-head stays the default, decided from the literature (README, Layer norm and attention, "When
to use it"; the options weighed are the plan's D14). Left:

- **Not measured here:** GQA's quality at this library's sizes, an uptrained model against one
  trained from scratch (the study's U1; Ainslie et al. didn't compare them either), and the decode
  rate at each `g` (`generation_timing.py` measures it in minutes when a question needs it). The
  registered study's code (`indrajala_ml/studies/gqa_study.py`, its script and tests) stays, never
  run.
- **Out of scope (D8):** multi-head latent attention (DeepSeek-V2: above multi-head at 14% and 4%
  of its cache, its Table 9) and cache quantization, each its own step if a cache ever binds.
- **Reinvesting the saved parameters** (MobileLLM widened the embedding: +0.4 and +0.7 points at
  125M and 350M) is a model-design choice, not planned.

## From FFN activations (retired 2026-11-03)

The study (D7, D11) was abandoned by the owner after 24 of its 80 runs, so ReLU stays the sequence
models' FFN (`scripts/studies/ffn_activation_study.py`'s docstring). Not planned:

- **The study itself,** if a later model needs the FFN decided by measurement. Its 24 partial runs
  (Tiny Shakespeare's 20 and 4 of Herodotus's contiguous GELU, from main `dcedf28`, crate `070e61d`)
  are in pyramidon's `data/studies/ffn-activations/sweep/` (gitignored, with a manifest), kept but
  not analysed: no conclusions from a registered study's partial data. A budget for it is sized on
  the machine's throughput at the worker count it will run (working-notes, `stone`'s throughput).

## From context length (retired 2026-11-02)

Later sequence work on the long-form corpora uses windows of 256, the document mask the packed
default (`scripts/studies/context_length_study.py`'s docstring). Left:

- **Per-segment (varlen) tiling** if bitwise position independence is ever needed (D3's
  refinement: global blocks agree within rounding across a segment's placements).
- **Out of scope (D10):** mixed precision, sequence parallelism, sparse or sliding-window
  attention: each its own step when a context past 1,024 or a larger model needs it.
- **The study's readings, not planned** (the owner, 2026-11-02: no follow-ups): C3 holds only at
  this model (2 layers, `d` 64), whose last positions in a 1,024 window predict worse than at 256;
  at the token budget the 1,024 arm trains on 2 windows a step; Euclid's contiguous split overfits
  by epoch 30 in every arm.

## From the training recipe (retired 2026-11-01)

- **A run file doesn't record `clip_norm`**, so a resumed run under another clip isn't refused as
  one under another schedule is. Record it beside the schedule (run file version 3) when a resumed
  clipped run needs it.
- **The cosine's peak** stayed the constant rate's tuned one (D8 (a)); a decay usually tolerates a
  higher peak, untested here.
- **Which half of the recipe gains, not planned.** C2 compared cosine against cosine with AdamW 0.1
  and clipping at 1.0 together; an arm with each alone would say which gives the 0.021 bits. The
  owner (2026-11-03) took it from the literature instead: most likely AdamW's (Loshchilov & Hutter
  2019; clipping guards against loss spikes, Pascanu et al. 2013, which rarely show at this
  scale). Both halves stay in the recipe.
- **Corpora to add** (the owner, 2026-10-31: both in the long run): Newton's *Principia* (Motte's
  1729 English translation, public domain), long uniform prose like Euclid's; and Whitehead and
  Russell's *Principia Mathematica* (public domain in the US), whose notation survives plain text
  only if transcribed, so a source with its symbols is needed before it can be a character corpus.
  Check each source's licence for the exact edition held (the literature rule).

## From the research separation (retired 2026-10-31)

- **The runner's hosts keep ml's venvs** (`/home/david/code/indrajala-ml/.venv`, stone's
  `wt-branches/.venv`): they hold the same dependencies; a studies-only dependency would need them
  rebuilt from indrajala-studies.

## From the release (retired 2026-10-31)

Later releases follow the crate README's Releasing section.

- **0.2.0, the next release,** is a breaking one: the crate structure's renames with no aliases (its
  D4; the map is in the crate's `pypi-pre-release-notes.md`), the ReLU ops gone into the activation
  family (`layer_relu_*`, `array_relu_mask`), and the crate performance work.
- **A rehearsal tag is never moved.** A failed rehearsal is re-run when its fix is outside the
  workflow; a fix to `release.yml` itself needs a new tag, and the verify job only accepts
  `testpypi-v<version>`. Accept a suffix (`testpypi-v0.1.1-2`) if that recurs.
- **crates.io's trusted publishing** is first exercised by the next release.
- **Free-threaded wheels** (3.14t): `gil_used = false` and extra wheels (abi3 doesn't cover them),
  after a thread-safety review of `Array`'s interior mutability.
- **ndarray interop** in the core (`From` conversions behind a feature), if a Rust user asks.
- **indrajala-ml on PyPI** (the release plan's D1); the studies-repo move it waited on is done.
- `cargo semver-checks` in CI once a second release exists.
- A benchmark wheel matrix on each platform's runner, only for a clear question.

## From the crate kernels (retired 2026-10-29)

- **A transpose for x86 that doesn't lose to 4K aliasing** (stage 1, closed): the AVX2 kernel lost
  up to 52% where the output's rows lie a multiple of 4 KB apart (512 rows). One that gathers a
  block into a contiguous buffer before writing it out, never interleaving loads with stores to
  aliasing rows, is the candidate; the measured kernels are on the crate's `avx2-transpose` branch.
- **x86 placement still moves a row through the uop cache** (the release workplan's stage 2b,
  2026-10-30; the multi-head attention plan's stage 5 saw the same, up to 14% on `jebel` with the
  same instructions): moving conv's forward into the core left its instructions as they were and
  made it 3-4.5% slower on jebel, with uops from the legacy decoder +12% and uops not delivered
  +42%. D7's JCC padding keeps branches off 32-byte boundaries but not a hot loop's place in the
  uop cache's 32-byte windows. The candidate layout (loops aligned to 32 bytes) is the parked
  plan's P8; until it holds, a crate A/B's few-percent rows on the i7 may be placement.

## From aarch64 (retired 2026-10-29)

- Other aarch64 machines (Graviton, Apple silicon) are checked by the committed reference alone.
- numpy's cross-machine behaviour stays unclaimed (D1); if ever wanted, a question about OpenBLAS.

## From the batch runner and the snap wall (retired 2026-10-29)

The runner's known limits (its hosts, ledger and gotchas are in [working-notes.md](working-notes.md),
The batch runner):

- **Find mode's verifier** (the programme audit's F7): restarts are optimal for Las Vegas runs,
  right whenever they stop. The certificate makes a run so only with the target known. In discovery
  the stop is answers-only under a state bound, which gave false discoveries at a slack bound (1.1
  times the target's size: 25 false stops, 1 false discovery); find mode there needs an exact
  answers-only check (Chow's test set under the bound, with queries) before Luby's cost model applies.
- Restarts assume attempts independent and identically distributed: true of fresh seeds, not of
  attempts on different backdrops (each has its own `t*`).
- A cut attempt keeps its last step's networks, but a certificate that would have come after the
  cutoff is lost by design (the reheat's late recoveries are the case).
- A ledger from before stage 5 (the stage-4 checks) holds summaries without `epochs`; `runs_of`
  leaves them out.

**The snap wall,** paused by the owner (2026-10-16), then overtaken by the state formation plan (P1
refuted C1: a wall on the snap's margin lost the proof in 8 of 10 replays, 9 of 10 without). To
resume from: the candidate read after the data (the collapsing seed snaps its 3 minimised states to
8 codes, the others to 3-5; spare codes beside each other could swap or merge under an update),
which a registered probe on networks with many codes in use would test. The wall acts only on states
training visits; the certificate stays the proof.

## From the programme audit (2026-10-23; the workplan retired the same day)

Four dated reviews in indrajala-research (`docs/reviews/review-2026-10-23-*.md`): the programme's
evolution, its reasoning, 33 findings (F1-F33) and the earlier reviews re-assessed. Left, the
owner's calls (S1/S2):

- **F1:** the fresh-seed rate's (D64) registered verdict (7 of 12) against the deadline's
  completion (D67) standing in for its cut runs.
- **F2, F3, F20, F22, F23, F27:** the papers' wording (the guard in training; "because it is one";
  E2b''s reused arm; stopping at the proof; the 0 of 160 baseline) and whether they wait for stage A.
- **F5:** retire the panel's gate (D1, D12, D40) or run its affordable form as the runner's first batch.
- **F6, F32:** the slack bound read offline on the D54-D64 step networks; Chow's test set built.
- **F7:** the batch runner's find mode needs a stated verifier (above).
- **F9, F10, F24:** the protocol's sidedness and mid-run moves; a table of the method's constants.
- **F13-F18:** contribution.md brought up to the fresh-seed evidence; the class line narrowed; the
  120-against-121 confound; the method's scope; an approximation beside S3 (D39 (i)).
- **The reviews' lesson:** when a gate is dropped, re-home every recommendation that waited on it.

## From the commitment, one-code-one-state, consolidation and population plans (retired 2026-10-20)

The owner abandoned these plans' remaining claims on 2026-10-20 for
[state-formation-workplan.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/workplans/state-formation-workplan.md).
Their options stay built (`--commitment-from`, `--thermostat`, `--anneal`, `--keep-machine`,
`--counterexamples`, `--answer-branch`). Left open, none planned:

- **C2 at 6 seeds with R3/R4 on every backdrop**, and R3/R4 against R4 alone (which did the work).
- **The population plan's level 2** (selection among runs by their machines).
- **Division `p` 17 at width 32**, for whether width's cure holds as the machine grows.
- **The guard's softening** (arming only on a step that passes both verdicts; below, Beyond
  arithmetic).

## From the window branches (2026-10-19; the workplan retired the same day)

The branches (`--branches`) were built and not adopted; they stay an option. Left open, none planned:

- **A stop-gradient between the answer branch and the trunk** (the plan's D4): the trunk is still
  shaped by the answers' loss.
- **Two heads a branch** (D2's alternative), and **a deeper trunk** before the branches: a single
  head a factor at width 16 starved the transitions on S4.
- **The branches at the base width (32) on the algebras' `n` 4 cells**, for no harm: the side
  probe at 32 on S5 (0 of 2, pairs per code unchanged) gave no reason to run it.

## From the window program (2026-10-17)

- Threads in the program (D8 (b)) are the parked plan's P4.
- **The attention window gains least** from the program (14% forward, 31% training): its crate
  share is larger, so the next gain is in the attention kernels, not the loop.
- Checkpointing (D5 (c)) and a bits-changing kernel (D3 (b)) are K22 and K18 (From the crate
  performance workplan).

## From the window-recurrent claims (2026-10-17)

- **E3's dropped cells (D34):** `m` 32 and every `k` 1 cell; the curriculum against all at once at
  more cells, if the curriculum's role is asked again (C4 holds, all at once leaning higher on far
  fewer examples).
- **The transformer against the linear window** at `k` 1 `s` 16 `m` 16, trained to the bound with
  every step certified (E2b snapped's cell with only the window changed): a candidate, not
  registered.

## From the division work (2026-10-15; the division, modular-family and factored-state workplans retired 2026-10-18)

- **The collapse fork's leftovers** (the collapse-fork workplan, `git show
  89cf3a3:docs/collapse-fork-workplan.md`; its probe worked as registered): **the fork in the
  harness** (`window_recurrent_study`; D5 built it in the probe tool only), and **the cause** of a
  collapse. Candidates, cheapest first: log the loss and the gradient's norm per batch around a
  collapse (a probe, to confirm the cause); clip the gradient's global norm; a rate that warms up
  at each curriculum step and decays; rollback and cool (restore the step's best network and halve
  the rate on a sharp fall of the training answers); an exponential moving average of the weights,
  certified in place of the raw weights; then the codebook's own remedies (code restarts, an EMA
  codebook) and, last, the commitment controls (`git show
  df1dcac:docs/commitment-control-workplan.md`). Each probed on replays of `p` 7 seeds 0, 1 and 3.
  **Division stage 5's runs** (the probe script's findings) sharpen it: large drops are common (24
  in 14 of 40 runs, in successes too) and 17 of 24 recovered, so a failure is a drop not recovered
  from; the curriculum moves on after a plateau whatever the accuracy (10 epochs without beating
  the step's peak), so an unrecovered drop meets longer lengths broken; and the fork's trigger (the
  mean screen score below 0.6 of the peak) misses about half the drops, seed 101's among them. Two
  candidates from this: **a step ends only when its peak is recovered** (or the step's best network
  is restored before moving on), and **a drop threshold on the worst answer**. Most of the room
  above the 65% at `p` 5 looks like retention, not search.
- **The fork with the certificate guard:** the guard joins the fork, not replaces it, tested as the
  fork alone against the fork plus the guard (below, Beyond arithmetic).
- **A second composite task: the running affine map mod `p`** (`x' = a x + b`, `a != 0`; the
  group AGL(1, p), `p (p - 1)` states when the output is the composed map). Its Krohn-Rhodes
  decomposition is a cascade, not a direct product: the multipliers' running product is its one
  approximation (a homomorphic image), the translations the dependent part, so no cover by
  approximations exists: the test of whether one approximation helps where no cover does. Later,
  `a = 0` adds resets.
- **The approximation claim's open questions** (A1-A3 hold: `git show
  c201622:docs/approximation-claim-workplan.md`): whether the sum helps *because* it is an
  approximation needs a learnable non-approximation control (the programme audit's F3); whether
  any approximation helps (one control, the product, at one machine), and at larger `p`; the affine
  map above tests one approximation where no cover exists.
- **"Characteristic sample"** is used only with a named algorithm (de la Higuera 1997, Definition
  1, after Gold: relative to the algorithm, RPNI's or OSTIA's); Oncina & García 1992 (RPNI) is
  still to read.
- **Factored readouts** (the division workplan's stage 6, its D9; the modular family's stage 2):
  every certified division network spreads its state over both codebooks, no factor a residue or
  any `xA + yB`. To make factors carry residues again, the factored-state workplan's D10 reopens:
  (a) the poolings over the output layer's tokens with independence stated only at the cell, or (c)
  one window pass per factor, its `StateInput`s reading that factor alone. Its (b), reading before
  the state, blocked learning even one counter (the D10 check, 0 of 2 against 2 of 2).
- **Staggered factors, a next design** (the factored-state workplan's C9'', 0 certified: each later
  factor's cell collapsed to one output in its turn): start a factor at its turn re-drawn and
  re-seeded, its commitment off until it splits, a turn kept only if the answer improved; or train
  jointly until each factor splits, then stagger.
- **The factored state's untried variations** (that plan's "Not in this plan"): the cross-term on,
  more factors; a state loss (its D3 (b)-(d)) only if the owner chooses it.
- **The commitment controls** (option 1, the usage thermostat) stay deferred in
  `git show df1dcac:docs/commitment-control-workplan.md`, for a probe that must train
  past a solution or shows factors collapsing.

## Beyond arithmetic: other finite-state targets (the owner and Claude, 2026-10-17)

Every target so far is an arithmetic transducer (the carry automaton, modular sums, division), the
weakness indrajala-research's `contribution.md` names under "Breadth": the next targets should test
whether the method needs that structure. Every one below is an explicit machine, so the certificate
applies unchanged; only the tasks are new. Each cell keeps the feasibility rules (codes at least the
minimal states; training to the exclusion bound `m + q - 1`), which for random machines with many
states means long training lengths: the bound decides which sizes are worth running.

Item 1 and the two items before it were taken up by
[general-algebras-workplan.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/workplans/general-algebras-workplan.md)
(S_n, move-to-front and random machines at 6, 24 and 120 states, then the trigger and the guard
tuned across them; its stages 5-7), **paused since 2026-10-20**. Their detail stays here until that
plan resumes:

- **The certificate guard, softened and tested across these machine classes** (the guard
  workplan, retired 2026-10-17: `git show ed0e378:docs/guard-training-workplan.md`; deferred by
  the owner with the trigger, "fine tune when we can use genuinely divergent tasks"). Built and
  kept: `certificate_guard.py` (the read by levels, `Guard`), `train_epoch(..., guard)`, the
  family's `--guard off|on|both` with its stop rule for a frozen run, and `guard-epoch` (one epoch
  from a saved network, unguarded and guarded). Its registered test, G1 (division at `p` 11, the
  fork alone against the fork plus the guard), was never run: the plan's budget probe found that
  from 7 right `p` 11 networks one further epoch lost the right machine in 7 of 7, and the guard
  held each only by undoing 66-100% of the updates (98-100% where not yet certified in both
  verdicts), freezing learning. Before a test, softenings to compare with `guard-epoch`: judge a
  flip only after N batches (stage 2c's flips often repaired themselves); scale a flipping update
  down instead of undoing it; restore the last right machine once an epoch (the fork with a
  correctness trigger); undo only updates that add wrong pairs; guard only the states training
  visits; calm the optimizer once right (cap the update, reset Adam's moments); shrink the codebook
  to the minimised machine once right. First the cause probe: whether the undone flips land on
  rarely visited codes (`p` 11 reaches 210-456 codes for 121 states; stage 2d's network 8 for 3).
  **Seen again in the consolidation plan's P1 (2026-10-19):** armed on a perfect screen, the guard
  froze two move-to-front 4 networks that were right at snap 1.0 but had 6-8 near ties at 1.05,
  undoing 96-100% of the updates until the stall rule, where unguarded they certified within 1-3
  steps. The first softening to try is arming only on a step that passes both verdicts.
- **The fork's trigger, tuned across these machine classes**, not on division alone (the
  p11-trigger workplan's stage 4, deferred: `git show 28894cf:docs/p11-trigger-workplan.md`; the
  owner, 2026-10-17: the trigger "would need to be either target machine agnostic, or we need a way
  to tune it in general"). The tools exist: `collapse.Trigger` (signal, rule, threshold, floor;
  today's, pooled below 0.6 of the peak, the default), every screen's per-kind shares in the logs,
  and `modular_family_probe.py drops`, which scores a trigger offline against any logs. The plan's
  candidate from division's logs: the mean over answer kinds more than 0.2 below the step's peak,
  floor 0.8 (the worst kind, 0.3 below, floor 0.9, had no false alarms there). The test as the plan
  set it: a trigger chosen with `drops` over every class's logs, registered, then paired replays
  (the same seeds, only the trigger changed); **better** if it certifies more seeds, or as many
  with fewer epochs, losing none today's trigger certified.

The targets, in the order recommended:

1. **Machines with no arithmetic reading, split by their algebra (first; the general algebras
   plan).** Random Mealy machines over a range of state counts and alphabets are the plain
   generality test. Splitting them by Krohn-Rhodes makes it a study rather than a demonstration:
   pure permutation (group) machines, the word problems of S3, A5 and S5, against aperiodic (reset,
   flip-flop) machines, the two extremes of the decomposition. Does the snapped recurrence learn one
   kind more easily? It extends the cover result: P2's terms are already Krohn-Rhodes's (Egri-Nagy's
   covering lemma: approximation, cover, coordinates), and the general claim is that **training on
   the coordinates of a machine's holonomy decomposition helps**, not only for division; that would
   make P2's training signal a method. It meets the literature: Liu et al. 2023 ("Transformers learn
   shortcuts to automata") and Merrill's circuit-complexity results make S5 the hard case for
   transformers, so a certified recurrence exact on S5 at every length is a sharp contrast. The
   running affine map (above) is the cascade case within arithmetic, a bridge to this.
2. **The standard regular-language benchmarks:** the Tomita grammars, parity, bounded-depth Dyck.
   The extraction literature uses them (Weiss et al. 2018; Wang et al. 2019), so results are
   directly comparable; cheap, but they fill tables rather than change the story.
3. **Real-world transducers**, for practical relevance: a UTF-8 decoder or validator (about 9
   states, Hoehrmann's DFA); finite-state phonology and morphology, the field whose native formalism
   transducers are (Kaplan and Kay's rewrite rules compiled to transducers, a fragment of an HFST
   morphological analyser); protocol and controller machines (TCP's connection states, a lexer, a
   vending-machine or traffic-light controller); sequential circuits (Gray and BCD converters,
   LFSRs and CRCs, the last linear over GF(2): arithmetic again, in another algebra).
4. **Later, beyond finite state:** counters and pushdown languages (unbounded Dyck). Nothing excludes
   them; the certificate as it stands wouldn't apply.

**A measurement gap: the components' kind (2026-10-21).** `approximations` reads a learned machine's
decomposition two ways: a parallel cover (the factors' approximations' meet separates every state) and
a cascade (a factor's approximation nontrivial, the joint codes determining the state: that quotient
on top, the rest below reading it). It doesn't read whether the components are irreducible (Krohn and
Rhodes: primes, the simple groups, and units, the flip-flop's divisors): each component's own
transformation monoid, and for the lower one its action within each block above (its holonomy, in
Egri-Nagy and Nehaniv's terms). So it tells S3's network cascade (a point's place on top, the other
two items' order below) from Krohn and Rhodes's (the sign on top, a counter mod 3 below) only by the
named congruence on top, not by the parts' algebra. Wanted before a discovery study reports a
decomposition: a reading of each component's monoid, its groups and its prime loops.

## From composable layers

- **New combinations.** The Sequential networks build every spec list `validate_layer_specs`
  accepts under every rule, in all three implementations. Every one-change combination has a
  preset in all three (the presets workplan; README, Presets). These have none:
  - two or more changes at once: ReLU, dropout, cross-entropy or softmax under `Momentum`, `Adam`
    or `WeightDecay`, ReLU with dropout, dropout with a softmax output, and so on, dense and conv
    (the presets workplan rejected one preset per combination: the class count grows
    multiplicatively, and the Sequential networks are the API for them);
  - batch norm, dense and conv, under every rule (README, Batch normalization).

  One of them, conv then pool, ReLU, dropout and a softmax output under `Adam`, is trained against
  its pure-Python reference (`tests/model/networks/test_sequential_array_network.py`). Giving any other one a
  preset, or using it in a demo, is a behaviour change, not refactoring: its own PR, with
  hand-computed tests and parity tests in all three implementations.
- **Still refused by the specs:** dropout on a ReLU layer (the dropout op is fused with the
  sigmoid), a conv or pool layer after a dense one (the fused hidden delta reads the next layer's
  `W`), and one rule per layer.
- **Weight decay with momentum** needs a published form chosen and cited first (README, Update
  rules); for Adam it is AdamW (roadmap step 10).
- **Ensembles over the new one-output presets** (the presets workplan). The ensembles fix their
  sub-network class (`classifier_cls`); a choice of sub-network is its own change.
- **Demos for the new conv presets** (the presets workplan's D4), each with its own tuned,
  measured PR: a demo makes an accuracy claim.

Not planned yet (none of it is excluded; any later workplan may take it up):

- Separate activation layers, which would add a crossing per layer on Rust. Activations stay fused.
- Parameter groups and per-layer learning rates (the optimizer's `learning_rate_scales` covers the
  codebook's rate only).
- Deleting or renaming any network class (the workplan's D1).

## From batch norm

- **Batch norm under dropout or after pool**, if a use appears. Neither passes
  `validate_layer_specs` today.
- **Folding batch norm into the preceding weights** for inference: a speed change, measured.

Not planned yet:

- Group norm and instance norm. They normalize within an example, a different layer (layer
  norm, the third, is built: README, Layer norm and attention).
- Synchronized statistics across workers or processes.

## From the conv batch-size study with batch norm

The rule still fails at B = 512 with batch norm, plain or in ghost groups of 32
(`batch_size_scaling.py`'s docstring). Open questions it leaves, each a scratch probe or a short
sweep with the existing script:

- **A gentler base rate.** At momentum 0.0 every batch-32 rate from 0.5 to 4 is within the seeds'
  spread, and the study's rule picked 2. Scaled from 0.5, B = 512 runs at 8, inside the stable
  range, where scaled from 2 it runs at 32, past the rate that fails at B = 32.
- **Capped rates at B = 512,** as the old study probed without batch norm (4, 8, 16, 24 with a
  1-epoch warmup): how far batch norm raises the stable rate at B = 512.
- **Longer runs at B = 512.** The unscaled control at momentum 0.9 ends 0.3 points below the band
  after 3 epochs (354 steps) and is still climbing; the scaled rate 32 at 0.0 reaches the band's
  neighbourhood in epoch 2 and then falls. More epochs would show whether either settles.
- **The dense momentum 0.9 rerun.** The dense study's momentum 0.9 cells with warmup, including
  its finding that the rule holds to B = 512, were measured with eq. (10) momentum without the
  momentum correction, and are pending a rerun on eq. (9) (`batch_size_scaling.py`'s docstring).
  The sweep script runs it as it is: `scaling --lr32 0.9=0.25` at the dense study's grid.

The old conv plan's timing and demo stages were for a rule that holds, and stay unplanned.

## From the A/B harness

- **Count the savings.** Count the agent turns and characters read per A/B from the session
  transcripts, against the numbers before the harness: 20-30 checks and 40-130K characters of raw
  output per A/B. Record the result here.
- **Waiting on CI** (`gh pr checks --watch` loops) is worth a similar treatment.
- **A `perf_region.py` adapter.** Its counters need `sudo` and a driver marking regions, so the
  harness has none yet.

Not planned yet:

- Non-timing gates. The golden run, tests and lint are each already one command.
- Significance tests (Mann-Whitney and the like). The protocol judges by ranges and agreement
  between passes, and the report adds no p-values.
- Other machines. The profile check refuses them; comparing machines is a different question.

## From the DRY audit

- **Rerun the audit after each new primitive**, so new copies are caught while they are small:
  `.venv/bin/python scripts/measurement/duplicates.py` (the clone scan, over the package, the
  scripts, both test suites and `cli`, at 8 lines, and `--lines 10`), a search for function names
  defined in more than one module, and the twin classes read side by side. The last pass (the
  third, 2026-10-30) left, at 8 lines: package 2.2%, scripts 1.8%, tests 3.1%, crate tests 2.0%;
  at 10: 1.3%, 1.4%, 1.6%, 1.2%: the baseline a rerun compares against.
- **Shared homes for the next layer kind.** A new layer kind extends these rather than adding a
  copy. The conv and pool argument checks and output size are in `model/specs/window_geometry.py`,
  one shape walk (`spec_shapes.spec_shapes`) feeds both builders, the optimizers share
  `OptimizerBase` and `ArrayOptimizerBase`, and a layer's optimizer accessors are `WeightAndBias`,
  `GammaAndBeta` or `AttentionProjections` (`array_parameters.py`). A hidden layer takes its
  output-delta refusals, its downstream-as-delta methods and its no-op gradient accumulation from
  `model/specs/hidden_layers.py` (`Hidden`, `DeltaIsDownstream`, `ParameterFree`, and `BatchOnly`
  for a layer that trains on batches only); a network its spec and rule hooks from
  `model/specs/spec_hooks.py` (`SpecHooks`; a rule preset `RulePreset`, which derives its
  hyperparameters from the rule's fields; a network given its rule `GivenRule`); a token-wise
  output its predictions from `TokenWiseOutput` and a conv preset its argument checks from
  `ConvArguments` (`specs/array_network_shapes.py`); a recurrence layer its schedule's setters from
  `RecurrenceSchedule` (`specs/window_recurrence.py`); a pure-Python gamma-and-beta row from
  `GammaAsWeights`; a token shape from `spec_shapes.token_shape`. Tests share `bits`, `split`,
  `max_relative_gap`, `patching`, `randomized`, `learn_in_step` and the `assert_*` scenarios
  (`tests/helpers.py`, the checkpoint resume among them), the crate's op tests `bits`, `to_numpy`
  and `to_array` (`rust/tests/array_helpers.py`), and a scenario every implementation runs takes
  conftest's `implementation` fixture
  (`tests/model/networks/test_{attention,residual,layer_norm,sequence}_network.py`). A study takes
  its cached MNIST loading, mean ± sd cells and Markdown tables from `indrajala_ml/studies/common.py`.
- **The builders' wiring loop** (the rerun's D6). The fork, add and body-first loop is the same in
  `python_layer_builder.py` and `array_layer_builder.py`; sharing it needs callbacks for the fork
  and the add, more indirection than its 12 lines are worth. Revisit if another layer kind that
  links layers arrives.
- **The third pass's leftovers** (its rerun at stage 8, the 8-line scan):
  - *The presets' typed `__init__`s* (its D3): each preset keeps its own signature, which differs
    by implementation (the numpy one-output presets take `input_bounds` optional and their
    hyperparameter keyword-only; the pure-Python ones `input_bounds` required). One `__init__` per
    rule and topology would change public signatures: an API decision, for the 1.0 release.
  - *Inside single test files:* `rust/tests/test_window_program.py` (two 22-line set-ups),
    `rust/tests/test_attention_ops.py`, `tests/array_network_contract.py`,
    `tests/model/persistence/test_format2.py`, `tests/measurement/test_machine_profile_capture.py`,
    and the pure-Python and Rust sequence files' embedding tests (13 lines). Each a helper inside its
    file, when it is next touched.
  - *The pure-Python recurrence layer's set-up* (14 lines, the numpy one's first lines): shared
    when the pure-Python layer gains the schedule's setters.
  - *Not copies:* `tests/test_pcg64.py` against `rust/tests/test_random_pcg64_parity.py` (each
    suite's own, standalone); `ab.py`'s and the archive's `_git` (different errors, which their
    callers read).

Not planned yet:

- The crate (the workplan's D1), and scripts and demos (D5): they get their own audit if wanted.
- The numpy and Rust layer twins' numerics (D3): they share identical methods only; their method
  bodies (a numpy expression against one fused Rust call) stay separate by design.
- The Add twins (the rerun's D2): `AddArrayLayer` and `AddRustArrayLayer` have the same method
  bodies, but each backend keeps its own named, typed class, which the tests check.

## From residual connections

- **Conv residual blocks** (D2). A conv block must keep its shape, and conv is 'valid' padding
  only (`conv_layer.py`), so it needs 'same' padding (new geometry, im2col and crate ops in all
  three implementations) and a conv layer with a bias and no activation: a workplan of its own.
- **Projection shortcuts** (D5): `x W_s + F(x)` when a block changes size (He et al. 2015's
  option B), a trained shortcut layer with a second gradient sum and its own parity tests.
- **Zero-initialized block ends** (D7): the body's last layer starting at zero, so each block
  starts as the identity (Goyal et al. 2017's zero-γ; Zhang et al. 2019, Fixup), as an option.
- **The depth study's open question**: with batch norm, the plain network's first-layer gradient
  at initialization grows tenfold from 2 to 16 hidden layers (the residual one's 1.8x), and at 16
  its seeds' standard deviation is 3 points in the first two epochs. Unexplained.

Not planned yet:

- Nested blocks (D6). The design nests without change; only validation refuses it.
- An activation after the add, `relu(x + F(x))` (D3), and a standalone activation layer.

## From the RNG generators workplan

- **Run checkpoints beyond one network's mini-batch run** (`indrajala_ml/training/run_checkpoint.py`):
  resuming mid-epoch, resuming `train_linear_classifier_network` (it returns no run checkpoint),
  and resuming an ensemble's run, whose sub-networks train as separate jobs.

Not planned yet:

- Normal draws, integer draws and broadcast `low`/`high`. Nothing uses them.
- Other bit generators (Philox, SFC64, PCG64DXSM).
- Changing the crate's legacy MT19937 module functions, which mirror `np.random` (the workplan's
  D7).

## From the RNG draw order

- **The bounds-width networks' draws** (D1 (c)): `BackpropClassifierNetwork`'s presets and
  `LinearClassifierNetwork` scale each node's weights by the input bounds, which no array network
  draws, so there is no order to match. Out of scope unless they gain an array twin.
- **A step-by-step gradient check of the batch-norm conv networks** would need the attention
  tests' `_scales` to give a linear conv layer before batch norm batch norm's scale, as it gives a
  `LinearLayer` (found in stage 2, #649). Batch norm's own gamma and beta gradients can also cancel
  to about 1e-6 there ("conv and dense pairs", seed 19). Nothing checks those networks step by step
  today.
- **Docstring figures measured with the old draw order**, such as `randomize_fan_in_aware`'s
  99.5% training and 96.9% test accuracy on UCI digits, are historical: re-measure one only when it
  is cited as current.

## From layer norm and attention

- **Layer norm against batch norm on the dense networks** (the workplan's D11 (b)): the residual
  depth study's batch-norm cells rerun with a flat `LayerNorm` in place of `BatchNorm`.
- **Parity under Adam is per step.** Under `SGD`, `Momentum` and `WeightDecay` the pure-Python
  and Rust patch models are compared with numpy after 50 steps; under `Adam` they are compared
  step by step, each step's gradients from the same parameters
  (`tests/model/networks/test_attention_python_network.py`, `tests/model/networks/test_attention_rust_network.py`). Adam's
  step is steepest where `|g|` is at or under its `ε`, which turns `bk`'s rounding-noise gradient
  into whole steps, so the trajectory itself is sensitive: numpy against numpy with one weight
  nudged by one ulp drifts 2e-8 to 3e-5 in 50 steps, as much as the implementations differ, and
  with `bk`'s gradient zeroed on both sides the gap the bias alone explains falls to 2.4e-12. The
  owner hasn't settled this standard; a 50-step Adam comparison with `bk` held out is the
  alternative.
- **A flat `LayerNorm` after a conv front end** is accepted and normalizes the whole `(h, w, c)`
  output as one token, keeping its shape. Per position (each pixel's channels, as a transformer's
  token) or per channel (group norm) are the other readings; none is used yet.
- **Inert parameters.** `bk` (no effect on `P`) and `bv` (the same as `bo`) are kept to match the
  reference models (D6). A flat `LayerNorm`'s `beta` right before a linear layer and its
  `BatchNorm` is inert too: the batch norm cancels the constant shift, so its gradient is noise.
- **A preset** (D12), if a patch model configuration is worth naming, under From composable
  layers' rule for presets.

Not planned yet:

- A class token (D8 (b)), fixed sin-cos or drawn positions (D7 (b), (c)).
- Conv-then-tokens hybrids.

## From multi-head attention

- **The extension points left** (the workplan's D6, D7), each a field with a default, added by the
  work that uses it:

  | later work | where it goes | what it adds |
  | --- | --- | --- |
  | padding masks beyond segments (causal is `Attention(causal=True)`; packed and padded segments are `segments`) | attend, forward: an additive mask on `S[i]` before the max shift, as the causal mask's `-inf` | a mask input; backward unchanged (`P_ij = 0` zeroes `dS_ij`) |
  | a separate value width | project and combine: `Wv` `(h·d_v, d)`, `Wo` `(d, h·d_v)` | a `value_size` field defaulting to `key_size` |

- **The numpy attention backward's 10% from stage 3.** Rerun on 2026-10-09 with the fixed script
  (`ab.py run --script-from`, #616; report on #616), stage 3 made `hidden_delta_batch` 10.2% slower
  (243 to 268 us per call, the stage-1 attention case at batch 32, consistent over 6 passes);
  `forward_batch` and `accumulate_gradient_batch` stayed within noise. The cost came with the
  three-block structure (D10), likely in the backward's (N, h, T, d_k) views or its stacked
  `np.matmul`s. Still worth a look: a fix must keep the blocks and their bits.

Not planned yet:

- A `TransformerLayer` spec (D5 (b)): a transformer layer is the attention and FFN blocks written
  again; a spec would freeze pre-LN against post-LN, the activation and where dropout goes.
- A class token, relative position biases, and head pruning or per-head analysis (Michel et al.
  2019, "Are Sixteen Heads Really Better than One?").
- A faster multi-head kernel: a crate-tuning item, with its own A/B.

## From the sequence task

- **A synthetic sequence task** (D2 (b)), copying or reversal from a seed: a known optimal loss,
  and a task only attention over earlier tokens solves. The study's leak arm and the mask's own
  test (a future token can't change an earlier output) already cover what it would test.

Not planned yet:

- A summed per-token loss (D6 (b)): with the mean, a step's size doesn't depend on `T`.
- A separate `TokenOutput` spec (D6 (c)): inside a token part `Dense` already means "per token".
- A network-level causal flag (D7 (b)): the mask is per layer, so an encoder and a causal block
  can mix later.
- Word-level corpora (D2 (d)): a vocabulary of 10,000 or more makes the output layer the whole cost.

## From attention dropout

- **Two crate shifts for the next baseline to measure against.** After crate #54 (#656's tier 1
  A/B, 6 passes), Rust `hidden_delta_batch` at batch 32 was 3.5% slower with dropout off, though its
  products are unchanged, while in the same run the unchanged `accumulate_gradient_batch` at batch
  32 moved 7.6% faster: read as a codegen or layout shift, as #627's (the sequence task's crate
  bump, +1.5% to +5.5% on some batch ops). On `jebel` such rows may be code placement (From the
  crate kernels).
- **`Dropout(0.0)` draws its mask; `Attention(dropout=0.0)` draws nothing.** The token layer
  follows the dense dropout layer, which draws at 0 too; the attention keeps a model without
  dropout on its old bits. A `Dropout(0.0)` could be refused by validation, or skipped at 0, which
  would change its draws (a golden entry has none at 0 today).
- **Drawing only the unmasked entries under a causal mask** (D4 (b)): about half an attention
  mask's draws fall on weights already 0. Skipping them changes the order every implementation
  must walk and every dropout golden entry's bits; worth it only if the draws show up in a profile.
- **The study's open questions.** A rate tuned per arm (every arm ran at the control's 0.002;
  dropout may prefer a higher one); longer runs (no arm had converged at 10 epochs, the dropout
  arms least: 0.044 to 0.081 bits per character in the last three epochs, against 0.014 to 0.047);
  a larger model, which overfits more and leaves dropout something to remove; and whether
  attention dropout's gain across Euclid's shift is real or a 10-epoch accident.

Not planned yet:

- Dropout fields on the layers that end a body (D2 (b)): `Dropout(p)` is the one token-wise form;
  a linear `Dense`'s dropout would need a linear variant of every fused `layer_dropout_*` op.
- Renormalizing each row of `P` after dropping (D3 (b)): no reference uses it.

## From generation

- **numpy's cached step is overhead-bound:** 5.1x uncached against Rust's 12.1x (jebel, the
  2-layer model). A step is a dozen small numpy calls per layer; a fused step would make it
  arithmetic-bound, but nothing needs numpy's decode fast yet.
- **A batched crate step** is K13 (From the crate performance workplan).
- **A prompt longer than `T`** is prefilled up to `T` and then stepped token by token. A
  band-masked prefill in chunks of `T` would be one pass per chunk, and is exact.
- **A learned `Position` past `T`** falls back to a full pass per token (D3 (a)), which is exact
  but uncached. Rotary models, the default, slide.
- **Not planned yet:** beam search (on `prefill_batch` and `step_batch`), repetition penalties,
  stop sequences, and a BPE tokenizer (the tokenizer file's `kind`).

## From rotary positions

- **#678's unexplained Rust gain.** The tier 1 A/B of the crate bump (rotation off) came out 7 to
  18% faster on five Rust rows, unexplained and not claimed: a codegen or layout shift, as #627's,
  for the next baseline to measure against.
- **The study's open questions.** `rotary_base` wasn't tuned (10,000 everywhere); every arm ran at
  the learned arm's tuned rate, and rotary may prefer another; nothing had converged at 10 epochs.

Not planned yet:

- ALiBi or T5's relative biases, beside rotary (D1 (c)): `positions` is a string so either can join
  without a new spec.
- A `Position()` and rotary attention in one token part (D6): refused today, one scheme a model.
- Rotating only a fraction of each head's features (GPT-NeoX's `rotary_pct`), or the interleaved
  pairing (D3 (b)), the same model up to a permutation.

## From the benchmark machine baseline

- **Rerunning the baseline** after a change that could move every number (kernel, BIOS,
  numpy/OpenBLAS, rustc, the frequency policy): rerun the workplan's stages 4-5 at the same
  tags, then at `main`, and replace [machine_profiles/i7-9700k.md](machine_profiles/i7-9700k.md)'s
  tables, with the reason, in one PR. The parked plan's closing baseline is one.
- **The laptop's columns** need rerunning only after a software change on the laptop (numpy,
  OpenBLAS, rustc, its kernel). The two machines run the same stack today, and their golden runs
  are byte-identical (D5). A crate change for the i7 is A/B'd on the laptop too, preferably on one
  thread: its default-threading A/As are noisy (a 32% per-pass spread at the median row).

Not planned yet:

- Claiming a cross-machine speedup from the laptop columns: they're context, and only their
  Rust/numpy ratios compare.
- GPU work, BIOS settings beyond cooling and power limits, and kernel boot parameters
  (`isolcpus`, `nohz_full`): the i7's noise didn't need them.

## From the benchmark archive

- **Results from other people's machines,** if the project is used widely: submissions to the
  archive by PR, with a recorded profile, a clean tree and a named commit. The record format
  allows it; reviewing untrusted submissions needs its own plan.
- **A run queue on `jebel`** (several A/Bs queued from `pyramidon`, run back to back), if one run
  at a time under the run lock turns out to be the bottleneck.

Not planned yet:

- Changing the timing protocol (the verdict rule, the noise rules per machine) except from data,
  as tier 1's pass count was.
- Hosting results anywhere but the archive repository: no dashboards, no release assets.
- Benchmarks in CI: GitHub's runners are shared and noisy, so no timing is ever taken there.

## From studies sharing `jebel`

The study-runs workplan built the machine lock that studies and A/Bs share, `ab.py`'s load
refusal and studies' result files with provenance (measurement.md §2). The owner trimmed it to
those three (#673); the rest waits here until a planned piece of work needs it.

- **Draining a study for an A/B** (its D4 (b) and D5): `ab.py run --drain` stops a resumable study
  at its next gap, runs the A/B and leaves the study to be resumed. The addition and
  window-recurrent harnesses resume already; `run_parameter_sweep` (the patch, sequence and other
  `*_study.py` sweeps) would need a resume of its own first. Today an A/B is refused and waits for
  the study to end (D4 (a)). A variant has `ab.py` resume the study itself; it would start commands
  read from the lock's record.
- **A harness's end-time estimate** (D6): `status` with the rate over the last hour and an
  end-time range, for the addition and window-recurrent harnesses. A sweep's average rate
  overstates its later, slower ladders (the addition sweep: 2.5 runs a minute on average, 1.1 at
  07:30), so until then an end time is quoted as a range from a measured epoch's cost.
- **The addition harness's timing fields** (D7): its `train_seconds` includes the whole
  catalogue on a passing run (16 s at `n = 4` for a column transformer, 213 s for the string
  format), so seconds an epoch read 4-8 times the true 2.5-3 s. The window-recurrent harness
  records training, screening, the check and each test length apart; the addition harness would
  take the same split (`fit_seconds`, `screen_seconds`, `check_seconds`). Until then, cost its
  budgets from failing runs' seconds an epoch.
- **Archiving studies' raw results off the machine** (D8 (b)): in `indrajala-benchmarks`, which
  would widen the archive past timing results (an owner's call). Today each file has one copy, recorded by path, host and sha256 in its
  study's docstring and copied to `pyramidon` by hand; the addition sweep's results went to the
  model archive (`indrajala-models`) with its threshold models.
- **Stopping a study stops its workers.** A study's forkserver workers don't hold the lock
  (`machine_lock`'s docstring): a SIGTERM to the study's process alone frees the lock while each
  worker trains on to the end of the run it holds, and only then fails to report it. Today the
  rule is to kill the whole process group. A fix: the harness traps SIGTERM and terminates its
  pool before it exits (or each worker asks for `PR_SET_PDEATHSIG`), with a test that the lock
  and the workers end together.

## From length generalisation (terminated)

The owner terminated the plan (2026-10-12) for the window-recurrent design; its literature section
stands (`git show c3d619b:docs/length-generalisation-workplan.md`, What the literature found).

- **Length-free token networks (its P1) are parked.** Stage 3 (P1a: `(None,)` token ids in the
  shape walk, `AnyInputShape`, format 2's `{"dimension": null}`) was built and passed (tests and
  the golden run) but never opened as a PR: the local branch `length-free-shapes` (`bd13b16`).
  A window has a fixed `T`, so the window-recurrent plan doesn't need it; a later length study may.
- **For any later fixed-depth length study:** rotary and no positions fail at long range by
  construction (resolution and dilution); an additive distance bias (ALiBi, P2) is the scheme that
  doesn't, and D12's start token is what a no-positions control needs.

## From the demo split

- **At 1.0, indrajala-demos depends on the published package** (the plan's D2): until then a
  "Bump indrajala-ml/" PR there carries each library change to the demos, and its CI runs their
  tests against it.

## From the addition study

The workplan's leftovers: its decisions' unchosen options and its Excluded for missing
infrastructure (`git show 8990f74:docs/addition-study-workplan.md`; the sweep's findings are in
indrajala-studies' `scripts/studies/addition_study.py`'s docstring).

- **The adders' threshold as a check for each new primitive.** The column transformers sit on a
  sharp edge (at `n = 4`, d 4 fails at every candidate and shape, d 6 passes from 2 layers), and a
  run near it takes a few epochs. Whether a change moves that edge is a cheap, quantitative check:
  one harness config per question, no new study design. Not yet run for the primitives built since:
  - step 8's rotary positions, and no positions, against the learned `Position`;
  - step 10's recipe (schedule, AdamW, clipping), including whether it smooths the non-monotone
    ladders, where a size succeeds and the next one up doesn't (column-unordered, 1 layer, 2
    heads at `n = 4`), which suggests the "too small" verdicts near the edge are partly the
    optimiser's;
  - steps 13 and 14 (FFN activations, grouped heads) at the sizes where each component matters
    most.
- **Milestone A's baseline** ([arbitrary-length-addition-workplan.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/workplans/arbitrary-length-addition-workplan.md),
  H2). The saved models use a learned `Position` tied to their `T`, so they can't be run at
  another length. Milestone A retrains the column transformers with rotary or no positions at the
  sweep's threshold sizes, rates and budget: the sweep tells it what to train, not what to load.
- **Probes of the passing adders.** Probing the smallest passing 2-layer networks against
  near-misses one width below (d 4 at `n = 4`) could locate the carry lookahead: which layer
  finds the deciding column below, and which head propagates the carry. This needs the accessor
  for a layer's activations on a batch (below). The probes don't need the sweep's exact files:
  when this is planned, it retrains the passing and near-miss networks it needs (minutes for a few
  seeds), on the code of the day. Only the passing models at each threshold and the width above are
  archived (the model archive, after the sweep).
- **Thresholds for candidates 4-8** (D12): the sweep searched only the column transformers, the
  calibration's passes. The string format (5), the FFN-only control (4), the dense networks per
  sum digit (6) and the regressions (7, 8) failed at one middle size and the budget of 200 epochs,
  patience 30 (4 must fail by construction). A sweep of the others wants a reason to expect a pass
  first: step 10's recipe, step 11's answer-only loss for 5, or a larger budget for 7 and 8, whose
  mean best screens reached 0.999 and 0.998, two seeds of each still rising when the budget stopped
  them (D7 (f): a budget per candidate, as its own config).
- **A third width** (D3 (c)): `n = 8` gives a curve of threshold against `n` (two points today:
  `d` 6 from 2 layers at 4, from 3 at 6). Its `7^8` carry patterns (5.8 M) are sampled, not
  exhaustive, and its runs are the costliest.
- **An FFN-width knob** (D5): every ladder held the FFN at `2d`. Wanted if a threshold turns out to
  be the FFN's rather than attention's; the probes would say.
- **Sample efficiency** (D7 (b)): one fixed training set instead of a fresh one each epoch, for a
  curve of examples against passing. It mixes overfitting into the size question, so it is its own
  study.
- **Candidates the framework couldn't build then** (the workplan's Excluded for missing
  infrastructure), each a new candidate entry and a catalogue run:
  - now buildable, not yet run as candidates: the string format's loss on answer tokens only
    (step 11's counted targets) and variable-length answers decoded autoregressively (step 9);
    schedules, AdamW and clipping, in case a "too small" verdict is the optimiser's (step 10);
  - length generalisation (train at `n`, test at `n + k`): rotary positions exist (step 8), but
    loading trained weights into a network of another `T` doesn't (the parked `length-free-shapes`
    branch, From length generalisation);
  - not planned yet: a weight-tied (looped) transformer, whose depth grows with `n` by iterating
    one block (blocks sharing weights, a variable iteration count); one dense network for every
    sum digit (several independent softmaxes on one flat network); supervised carries (auxiliary
    output heads and losses for each column's carry); an accessor for a layer's activations on a
    batch, for the probes above (it may be small); and a 1-D convolution along the tokens in the
    `sequence` shape, a local, ripple-like bias against attention's.
