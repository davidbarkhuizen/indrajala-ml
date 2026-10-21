# Next steps: work left over from completed workplans

**The goal all of it serves:** [machine-discovery.md](machine-discovery.md), discovering finite-state
machines from behaviour and known structure, each discovery proved; its route in stages A-F.

A workplan is deleted once its last stage merges, and never before: only when every stage and
decision is resolved and only future work is left. Whatever it left open (its "After this plan"
list, and what its "Out of scope" left unbuilt) moves here. Nothing listed here is excluded:
"not planned yet" means only that no workplan has taken it up. A workplan still
in progress keeps its own list: [general-algebras-workplan.md](general-algebras-workplan.md),
[pypi-release-workplan.md](pypi-release-workplan.md),
[generation-workplan.md](generation-workplan.md),
[training-recipe-workplan.md](training-recipe-workplan.md),
[segments-workplan.md](segments-workplan.md),
[context-length-workplan.md](context-length-workplan.md),
[ffn-activations-workplan.md](ffn-activations-workplan.md),
[grouped-query-attention-workplan.md](grouped-query-attention-workplan.md),
[cross-attention-workplan.md](cross-attention-workplan.md),
[arbitrary-length-addition-workplan.md](arbitrary-length-addition-workplan.md),
[state-formation-workplan.md](state-formation-workplan.md),
[aarch64-workplan.md](aarch64-workplan.md),
[factorisation-machine-workplan.md](factorisation-machine-workplan.md), and the paused
[snap-wall-workplan.md](snap-wall-workplan.md). The order
of the next ML
primitives is in [primitives-roadmap.md](primitives-roadmap.md).

## Retired workplans

Each one is still in git history, with its decisions (D1, D2, ...), design and stages. Code and
docs cite them by section:

| Workplan | Plan PR | Stages | Read it with |
| --- | --- | --- | --- |
| Composable layers and optimizers (roadmap step 1) | #475 | #476-#485 | `git show 3a5d179:docs/composable-layers-workplan.md` |
| The A/B harness and a stand-alone measurement guide | #487 | #488-#491 | `git show 3a5d179:docs/ab-harness-workplan.md` |
| Batch normalization (roadmap step 2) | #486 | #493-#503 | `git show 189921c:docs/batch-norm-workplan.md` |
| Removing duplicated code (DRY) | #506 | #507-#514 | `git show e066333:docs/dry-workplan.md` |
| The conv batch-size study with batch norm | #516 | #517-#519 | `git show acbc49c:docs/conv-batch-norm-scaling-workplan.md` |
| Explicit generator objects (RNG generators) | #521 | crate #46, #522-#529 | `git show aaf8bb2:docs/rng-generators-workplan.md` |
| Residual connections (roadmap step 3) | #531 | crate #47, #532-#538 | `git show 365a173:docs/residual-connections-workplan.md` |
| Layer norm and single-head attention (roadmap step 4) | #540 | crate #48, #542-#548 | `git show 7d3bd0a:docs/layer-norm-attention-workplan.md` |
| Removing duplicated code, second pass (DRY rerun) | #550 | #551-#557 | `git show 88e1815:docs/dry-rerun-workplan.md` |
| Presets for the one-change combinations | #559 | #560-#564 | `git show 2a799bf:docs/presets-workplan.md` |
| The source layout | #566 | #567-#573 | `git show b362621:docs/source-layout-workplan.md` |
| A baseline on the new benchmark machine (i7-9700K) | #575 | #576-#590, #592 | `git show 480164d:docs/benchmark-machine-workplan.md` |
| A benchmark archive, tiered benchmarking, and a remote benchmark machine | #584 | #593-#598, archive #1-#11 | `git show 056b808:docs/benchmark-archive-workplan.md` |
| Multi-head attention and a transformer block (roadmap step 5) | #602 | crate #51-#52, #603-#613, archive #12-#13 | `git show d70cf0f:docs/multi-head-attention-workplan.md` |
| A sequence task with causal masking (roadmap step 6) | #620 | crate #53, #621-#634, archive #14-#17 | `git show f6e843b:docs/sequence-task-workplan.md` |
| Pure Python's random draws in numpy's order (RNG draw order) | #637 | #649-#652, archive #18-#19 | `git show 7a80858:docs/rng-draw-order-workplan.md` |
| Dropout in attention (roadmap step 7) | #638 | crate #54, #654-#660, #666, archive #20-#21 | `git show 8d2019e:docs/attention-dropout-workplan.md` |
| Length generalisation for the column adders (terminated by the owner before its existence study) | #683 | #684 | `git show c3d619b:docs/length-generalisation-workplan.md` |
| Rotary position embedding (roadmap step 8) | #639 | crate #55, #668-#669, #678-#682, #688, archive #22-#23 | `git show 9b27175:docs/rotary-positions-workplan.md` |
| Studies and timing runs sharing `jebel` (study-runs) | #670 | #673-#676, #695 | `git show 868838e:docs/study-runs-workplan.md` |
| The smallest network that adds (the addition study) | #661 | #662-#665, #672, #677, #696, models `7748416` | `git show 8990f74:docs/addition-study-workplan.md` |
| The demos in their own repository (the demo split) | #720 | #722-#724, indrajala-demos #1-#2 | `git show 9c50fe6:docs/demos-split-workplan.md` |
| The window loop in the crate (the window program) | #792 | #799-#802, crate #58-#59, archive #28-#30, #804 | `git show ad651d8:docs/window-program-workplan.md` |
| Substantiating the window-recurrent adder's findings (the claims workplan, C0-C8, E1-E8) | #701 | #703-#808 (findings #712-#808), models #1; report [window-recurrent-report.md](window-recurrent-report.md) | `git show fea9de1:docs/window-recurrent-claims-workplan.md` |
| The window-recurrent adder (milestone A; its claims in the row above) | #685 | #686-#702, #809 | `git show e45944e:docs/window-recurrent-adder-workplan.md` |
| A factored state for the window-recurrent network (C9, C9', C9'') | #737 | #738-#764, crate #57 | `git show e45944e:docs/factored-state-workplan.md` |
| A family of modular operations on one state (option 2) | #765 | #766 | `git show e45944e:docs/modular-family-workplan.md` |
| Modular division with the family (`p` 5 and 7, the registered claim) | #766 | #767-#782 | `git show e45944e:docs/division-workplan.md` |
| Where division fails in `p` (the division ladder, L1-L3) | #817 | #824, #856, #859, #861 | `git show f9101d8:docs/division-ladder-workplan.md` |
| A path through the window for each factor and for the answers (window branches; W1, W2, B1-B3) | #863 | #864-#868, crate #66, #884 | `git show e7f1ff6:docs/window-branches-workplan.md` |
| Consolidating codes (A) and longer words (B) for extra codes (P1, P2) | #822 | #829, #832-#836, #845, #849, #854, #857 | `git show df1dcac:docs/code-consolidation-workplan.md` |
| Controlling the codes' commitment (the diagnostic, the thermostat, annealing) | #765 | #851, #852, #871, #876, #881, #885, crate #67, #886 | `git show df1dcac:docs/commitment-control-workplan.md` |
| One code, one state (R1-R4, C0, C2) | #873 | #874, #875, #878, #879, #883 | `git show df1dcac:docs/one-code-one-state-workplan.md` |
| A population of runs selected by their machines (level 2, not built) | #882 | - | `git show df1dcac:docs/population-workplan.md` |

The optimization docs (the Rust-against-numpy baseline, and the implemented, rejected and
candidate optimizations) were retired the same way: `git show 0a04977:docs/optimizations.md` and
`git show 0a04977:docs/optimizations/<name>.md`.

What they built is documented in the README (Models, Saving and loading, Update rules, Batch
normalization, Residual connections, Layer norm and attention, The sequence task, Dropout among the
tokens, Rotary positions, The addition study), [measurement.md](measurement.md)
and [rng-audit.md](rng-audit.md). The batch-size studies' findings are in
`indrajala_ml/studies/batch_size_scaling.py`'s docstring, the depth study's in
`scripts/residual_depth_study.py`'s, the patch-attention study's in
`scripts/patch_attention_study.py`'s, the multi-head attention study's in
`scripts/multi_head_attention_study.py`'s, the sequence study's in `scripts/sequence_study.py`'s,
the attention dropout study's in `scripts/attention_dropout_study.py`'s, the rotary study's in
`scripts/rotary_study.py`'s, the addition study's in `scripts/addition_study.py`'s.

## From the commitment, one-code-one-state, consolidation and population plans (retired 2026-10-20)

The owner abandoned these plans' remaining claims on 2026-10-20 and set
[state-formation-workplan.md](state-formation-workplan.md) in their place (the night's results are in
the probe script's findings). Built and kept as options: `--commitment-from`, `--thermostat`
(retired as a remedy), `--anneal` (as built it merges codes: the new plan's option 1 rebuilds its
start), `--keep-machine` and `--counterexamples` (R3/R4: C2 worked, 5 of 12), `--answer-branch` (R1,
shelved: harm on division at 32). Left open, none planned:

- **C2 at 6 seeds with R3/R4 on every backdrop**, and R3/R4 against R4 alone (which did the work).
- **The population plan's level 2** (selection among runs by their machines).
- **Division `p` 17 at width 32**, for whether width's cure holds as the machine grows.
- **The guard's softening** (arming only on a step that passes both verdicts).

## From the window branches (2026-10-19; the workplan retired the same day)

The branches (`--branches`: a two-block trunk, a single-head block a factor and one for the answers)
were built and not adopted: B1 and B2 worked in one seed a cell, W2 (width alone, at B1's size)
did better on division, and B3 showed harm on S4 (the probe script's findings, stage 6). They stay
an option. The answers' own branch with the factors on a shared state branch is R1, in
`git show df1dcac:docs/one-code-one-state-workplan.md`. Left open, none planned:

- **A stop-gradient between the answer branch and the trunk** (the plan's D4): the trunk is still
  shaped by the answers' loss.
- **Two heads a branch** (D2's alternative), and **a deeper trunk** before the branches: a single
  head a factor at width 16 starved the transitions on S4.
- **The branches at the base width (32) on the algebras' `n` 4 cells**, for no harm: the side
  probe at 32 on S5 (0 of 2, pairs per code unchanged) gave no reason to run it.

## From the window program (2026-10-17)

The window-recurrent network's window loop runs in the crate on Rust, inference and training, by
bits the per-layer path's (`per_layer_windows`, `per_layer_gradients`, `per_layer_path`;
`scripts/window_program_check.py` checks saved networks). Before and after on `jebel` (stage 6,
from before stage 1 to the end, tier 2): a real study run (a `p` 5 family probe run, 40 epochs) 32.9%
faster (139 s to 93 s), the family long test 46.6% (21.1 s to 11.3 s), the training steps 43.9% (E2b''s
linear window) and 34.1% (the `p` 7 family). Left:

- **Threads in the program (D8 (b)):** split the examples across threads inside one call; rows are
  independent, so the bits hold. Measure the crate's per-call thread spawn first; it competes with
  the studies' and long tests' own processes.
- **Checkpointing (D5 (c)):** keep every `r`-th window's forward pass and recompute the rest, if a
  study trains windows long enough to pass the memory budget (2.9 MiB at E2b''s cell, 21.7 MiB at
  the fork probe's, against 2 GiB).
- **A bits-changing kernel (D3 (b)):** fused or reassociated arithmetic in the program; needs the
  owner's approval and a re-recorded golden run (measurement.md, section 8).
- **The attention window gains least** (14% forward, 31% training): its crate share is larger, so the
  next gain is in the attention kernels, not the loop.

## From the window-recurrent claims (2026-10-17)

Every claim decided ([window-recurrent-report.md](window-recurrent-report.md)): C0, C2, C2b, C3
(working, not certified), C4 (after Holm) and C8 hold; C1 refuted for the sampled "works" and
restated on the certificate; C6 not supported; C7 and C7' refuted (division learned later, with a
factored state and a cover). Left:

- **E3's dropped cells (D34):** `m` 32 and every `k` 1 cell; the curriculum against all at once at
  more cells, if the curriculum's role is asked again (C4 holds, all at once leaning higher on far
  fewer examples).
- **The transformer against the linear window** at `k` 1 `s` 16 `m` 16, trained to the bound with
  every step certified (E2b snapped's cell with only the window changed): a candidate, not
  registered.

## From the division work (2026-10-15; the division, modular-family and factored-state workplans retired 2026-10-18)

Decided with the owner, each to become a workplan or a stage:

- **The collapse fix first** (before larger `p`): the collapse is abrupt (the probe script's
  findings, "the collapse, traced"). **Done: snapshot and fork on collapse** (the owner's
  choice; the collapse-fork workplan, retired 2026-10-16: `git show 89cf3a3:docs/collapse-fork-workplan.md`).
  **Its probe works as registered** (`p` 7, two factors of 32, 4 seeds an arm, the probe script's
  findings): 3 of 4 certified with the fork against 2 of 4 without, no fork run ending collapsed;
  the arms are paired (the same run until the first fork), and the one seed the fork rescued (3)
  lost everything without it. A screen, not a rate. Left open: **the fork in the harness**
  (`window_recurrent_study`; D5 built it in the probe tool only); **the cause** (the loss and the
  gradient's norm per batch around a collapse, the probe below); and the rest stay candidates. Candidates, cheapest first: log the loss and the gradient's
  norm per batch around a collapse (a probe, to confirm the cause); clip the gradient's global
  norm (none today); a rate that warms up at each curriculum step and decays; rollback and cool
  (restore the step's best network and halve the rate on a sharp fall of the training answers);
  an exponential moving average of the weights, certified in place of the raw weights; then the
  codebook's own remedies (code restarts, an EMA codebook) and, last, the commitment controls
  (`git show df1dcac:docs/commitment-control-workplan.md`). Each probed on replays of
  `p` 7 seeds 0, 1 and 3. **Division stage 5's runs** (the probe script's findings) sharpen it:
  large drops are common (24 in 14 of 40 runs, in successes too) and 17 of 24 recovered, so a
  failure is a drop not recovered from; the curriculum moves on after a plateau whatever the
  accuracy (10 epochs without beating the step's peak), so an unrecovered drop meets longer lengths
  broken; and the fork's trigger (the mean screen score below 0.6 of the peak) misses about half the
  drops, seed 101's among them. Two candidates from this: **a step ends only when its peak is
  recovered** (or the step's best network is restored before moving on), and **a drop threshold on
  the worst answer**. Most of the room above the 65% at `p` 5 looks like retention, not search.
- **Division at `p` 11 (121 states)**, **done with the collapse trigger in the p11-trigger
  workplan** (retired 2026-10-17: `git show 28894cf:docs/p11-trigger-workplan.md`): the largest run for `jebel`, **the fork on** (its probe's
  registered next step): `p` 7 with two factors of 32 is done (the fork probe: 5 certified 49-state
  networks in 8 runs, each passing every long-test case), and **`p` 11 works** (2026-10-16: 4 of 6
  certified 121-state networks at 32 a factor, each passing every long-test case; the probe's
  findings). `k` stays 1: the certificate's cost is codes x 9 tokens there. The plan's stage 4 (the
  alternate trigger) waits for the beyond-arithmetic targets below, where a trigger can be tuned
  across machine classes, not to division; today's trigger stays the default (its stage 5's
  decision).
  The certificate guard joins the fork, not replaces it, tested as the fork alone against the
  fork plus the guard.
- **A second composite task: the running affine map mod `p`** (`x' = a x + b`, `a != 0`; the
  group AGL(1, p), `p (p - 1)` states when the output is the composed map). Its Krohn-Rhodes
  decomposition is a cascade, not a direct product: the multipliers' running product is its one
  approximation (a homomorphic image), the translations the dependent part, so no cover by
  approximations exists: the test of whether one approximation helps where no cover does. Later,
  `a = 0` adds resets.
- **The approximation claim: done, A1-A3 hold** (2026-10-17; the workplan retired:
  `git show c201622:docs/approximation-claim-workplan.md`; findings in `scripts/modular_family_probe.py`'s docstring).
  The sum alone certifies 10 of 20 against the quotient alone's 1 and the product's 0, so one
  approximation helps because it is one. Open: whether any approximation helps (one control,
  the product, at one machine), and at larger `p`; the affine map above tests one
  approximation where no cover exists.
- **Terms** (settled with the owner, the division paper): target machine and its output function;
  approximation (homomorphic image); cover (emulation); coordinates; k-distinguishable (Chow). Not
  "family" (the code's arm name only), "basis", "ideal" (a semigroup term), "model". "Characteristic
  sample" (grammatical inference) is unverified: no open copy of Oncina & García 1992 or de la
  Higuera 1997 was found.
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
- **Division's edge between `p` 11 and 13: explained by W1** (the window branches workplan,
  2026-10-19): neither the budget (L2) nor the factors' width (L3), but the window's: at 32 wide
  instead of 16, 5 of 6 certified at 169 states against 0 of 6. The round-2 algebras' failures at
  120 states are other failure kinds (formation or conversion, extra codes): the kinds are mapped in
  [failure-kinds.md](failure-kinds.md), each with its plan and next step.

## Beyond arithmetic: other finite-state targets (the owner and Claude, 2026-10-17)

The owner asked which domains to examine once the papers' work is wrapped up. Every target so far
is an arithmetic transducer (the carry automaton, modular sums, division), the weakness
`~/code/indrajala-papers/contribution.md` names under "Breadth": the next targets should test
whether the method needs that structure. Every one below is an explicit machine, so the
certificate applies unchanged; only the tasks are new. Each cell keeps the feasibility rules
(codes at least the minimal states; training to the exclusion bound `m + q - 1`), which for
random machines with many states means long training lengths: the bound decides which sizes are
worth running. In the order recommended, after one item that runs across them:

- **The certificate guard, softened and tested across these machine classes** (the guard
  workplan, retired 2026-10-17: `git show ed0e378:docs/guard-training-workplan.md`; deferred by
  the owner with the trigger, "fine tune when we can use genuinely divergent tasks"). Built and kept: `certificate_guard.py` (the read by levels,
  `Guard`), `train_epoch(..., guard)`, the family's `--guard off|on|both` with its stop rule for a
  frozen run, and `guard-epoch` (one epoch from a saved network, unguarded and guarded). Its
  registered test, G1 (division at `p` 11, the fork alone against the fork plus the guard), was
  never run: the plan's budget probe found that from 7 right `p` 11 networks one further epoch
  lost the right machine in 7 of 7, and the guard held each only by undoing 66-100% of the updates
  (98-100% where not yet certified in both verdicts), freezing learning. Before a test, softenings
  to compare with `guard-epoch`: judge a flip only after N batches (stage 2c's flips often repaired
  themselves); scale a flipping update down instead of undoing it; restore the last right machine
  once an epoch (the fork with a correctness trigger); undo only updates that add wrong pairs;
  guard only the states training visits; calm the optimizer once right (cap the update, reset
  Adam's moments); shrink the codebook to the minimised machine once right. First the cause
  probe: whether the undone flips land on rarely visited codes (`p` 11 reaches 210-456 codes for
  121 states; stage 2d's network 8 for 3). **Seen again in the consolidation plan's P1 (2026-10-19):** armed on a
  perfect screen, the guard froze two move-to-front 4 networks that were right at snap 1.0 but had
  6-8 near ties at 1.05, undoing 96-100% of the updates until the stall rule, where unguarded
  they certified within 1-3 steps. The first softening to try is arming only on a step that passes
  both verdicts.
- **The fork's trigger, tuned across these machine classes**, not on division alone (the
  p11-trigger workplan's stage 4, deferred; the owner, 2026-10-17: the trigger "would need to be
  either target machine agnostic, or we need a way to tune it in general"). The tools exist:
  `collapse.Trigger` (signal, rule, threshold, floor; today's, pooled below 0.6 of the peak, the
  default), every screen's per-kind shares in the logs, and `modular_family_probe.py drops`,
  which scores a trigger offline against any logs. The plan's candidate from division's logs: the
  mean over answer kinds more than 0.2 below the step's peak, floor 0.8 (the worst kind, 0.3
  below, floor 0.9, had no false alarms there). The test as the plan set it: a trigger chosen with
  `drops` over every class's logs, registered, then paired replays (the same seeds, only the
  trigger changed); **better** if it certifies more seeds, or as many with fewer epochs, losing
  none today's trigger certified.


1. **Machines with no arithmetic reading, split by their algebra (first).** Random Mealy machines
   over a range of state counts and alphabets are the plain generality test. Splitting them by
   Krohn-Rhodes makes it a study rather than a demonstration: pure permutation (group) machines,
   the word problems of S3, A5 and S5, against aperiodic (reset, flip-flop) machines, the two
   extremes of the decomposition. Does the snapped recurrence learn one kind more easily? It
   extends the cover result: P2's terms are already Krohn-Rhodes's (Egri-Nagy's covering lemma:
   approximation, cover, coordinates), and the general claim is that **training on the coordinates
   of a machine's holonomy decomposition helps**, not only for division; that would make P2's
   training signal a method. It meets the literature: Liu et al. 2023 ("Transformers learn
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

**Recommended first:** item 1, a random-machine study split into group and aperiodic machines,
with the S5 word problem: it turns "arithmetic only" into a finding and tests whether the cover
result generalises. When it is taken up: a workplan with a feasibility table per cell (states,
alphabet, codes, the exclusion bound against the curriculum's cap).
**Taken up (2026-10-17)** by [general-algebras-workplan.md](general-algebras-workplan.md): S_n, move-to-front
and random machines over the same states at 6, 24 and 120, then the trigger and the guard tuned
across them (the two items above).

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
- **Weight decay with momentum or Adam** needs a published form chosen and cited first (README,
  Update rules). For Adam it is chosen: AdamW (Loshchilov & Hutter 2019), roadmap step 10.
- **Ensembles over the new one-output presets** (the presets workplan). The ensembles fix their
  sub-network class (`classifier_cls`); a choice of sub-network is its own change.
- **Demos for the new conv presets** (the presets workplan's D4), each with its own tuned,
  measured PR: a demo makes an accuracy claim.

Not planned yet (none of it is excluded; any later workplan may take it up):

- Separate activation layers, which would add a crossing per layer on Rust. Activations stay fused.
- Parameter groups and per-layer learning rates. (Schedulers beyond today's `lr_schedule.py` were
  lifted out of scope on 2026-10-10: roadmap step 10.)
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

- **Count the savings.** After the first two real A/Bs through `scripts/ab.py`, count the agent
  turns and characters read per A/B from the session transcripts, against the numbers before the
  harness: 20-30 checks and 40-130K characters of raw output per A/B. Record the result here.
- **Waiting on CI** (`gh pr checks --watch` loops) is worth a similar treatment.
- **A `perf_region.py` adapter.** Its counters need `sudo` and a driver marking regions, so the
  harness has none yet.
- **`ab.py run --old-crate X --new-crate X` installs neither wheel:** given the same crate on both
  sides, it times the venv's crate on both (found in the consolidation plan's stage 3, #834, timing
  the crate alone). It should build and install that crate, or refuse.

Not planned yet:

- Non-timing gates. The golden run, tests and lint are each already one command.
- Significance tests (Mann-Whitney and the like). The protocol judges by ranges and agreement
  between passes, and the report adds no p-values.
- Other machines. The profile check refuses them; comparing machines is a different question.

## From the DRY audit

- **The per-rule presets** (the workplan's D2). The preset classes that set a rule each repeat a
  hyperparameter `__init__` and a two-line `_update_rule`. There were 12 at the audit (Adam,
  Momentum, conv Momentum and L2, each in pure Python, numpy and Rust); the presets workplan
  brought them to 27 (`Momentum`, `Adam` and `WeightDecay`, each one-output, multiclass and conv,
  in all three implementations). D2 waited for more presets before choosing a shared form; that
  point has come, and the form is the next DRY pass's to choose.
- **Rerun the audit after each new primitive**, so new copies are caught while they are small:
  `symilar` (pylint's duplicate finder, with `-i --ignore-docstrings --ignore-imports
  --ignore-signatures`) over `indrajala_ml/`, `scripts/` and `tests/`, a search for function names
  defined in more than one module, and the twin classes read side by side. The 2026-10-01 audit
  found 0.29% duplicated at 8 lines or more in the package, and 0.91% at 10 lines or more in the
  tests.
  The rerun of 2026-10-02 (the DRY rerun workplan, #550-#557) found 0.20% at 8 lines or more in
  the package and 0.16% at 10 lines or more in the tests, the copies left having come in with
  residual connections, layer norm and attention.
  The rerun of 2026-10-09, after multi-head attention, found 0.16% at 8 lines or more in the
  package, 0.14% at 10 lines or more in the tests and 0.00% in the scripts, and removed what it
  found in one PR: `AttentionProjections` took the attention layers' shared setup (`_set_up`), the
  gradients' order (`_set_gradients`) and `reset_gradient_accum`, and `tests/helpers.py` took
  `batches` and `learn_in_step` (the 50-step parity loop, 11 copies), `assert_snapshots_close` (5)
  and `assert_learn_and_a_batch_of_one_agree` (3), `test_checkpoint` the restored-checkpoint
  scenario (3) and `test_layer_specs` the README's patch model (2).
- **Shared homes for the next layer kind.** The conv and pool argument checks and output size are
  in `model/specs/window_geometry.py`, one shape walk (`spec_shapes.spec_shapes`) feeds both builders,
  the optimizers share `OptimizerBase` and `ArrayOptimizerBase`, and a layer's optimizer accessors
  are `WeightAndBias`, `GammaAndBeta` or `AttentionProjections` (`array_parameters.py`). A hidden
  layer takes its output-delta refusals, its downstream-as-delta methods and its no-op gradient
  accumulation from `model/specs/hidden_layers.py` (`Hidden`, `DeltaIsDownstream`, `ParameterFree`); a
  pure-Python gamma-and-beta row from `GammaAsWeights`; a token shape from
  `spec_shapes.token_shape`. Tests share `bits`, `split`, `max_relative_gap`, `patching`,
  `randomized`, `learn_in_step` and the `assert_*` scenarios (`tests/helpers.py`), and a scenario
  every implementation runs takes conftest's `implementation` fixture (`tests/model/networks/test_{attention,residual,layer_norm}_network.py`). A new layer
  kind extends these rather than adding a copy. A study takes its cached MNIST
  loading, mean ± sd cells and Markdown tables from `indrajala_ml/studies/common.py`.
- **The linear, conv and batch-norm layers' own output-delta refusals** (the rerun's stage 1).
  They are hand-written, with their own messages, rather than taken from `Hidden`; they were not
  in the rerun's findings. Move them onto the mixin when one of those layers is next touched.
- **The builders' wiring loop** (the rerun's D6). The fork, add and body-first loop is the same in
  `python_layer_builder.py` and `array_layer_builder.py`; sharing it needs callbacks for the fork
  and the add, more indirection than its 12 lines are worth. Revisit if another layer kind that
  links layers arrives.

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

From one seed, every pure-Python network with an array twin now builds numpy's weights and draws
numpy's dropout masks by bits ([rng-audit.md](rng-audit.md)), and the parity tests seed both
sides. Left over:

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

Multi-head attention and a key size were roadmap step 5, now done; masking and dropout in attention
moved to From multi-head attention. Besides those:

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

The workplan settled `Attention(heads, key_size)`, the parameters packed with heads as row blocks
(D2), and every implementation and the crate in three blocks per pass: project, attend, combine
(D4). The patch study's open margin is answered: at 5 seeds attention then FFN beats FFN alone by
0.40 +- 0.30 points, on 4 of 5 seeds (`scripts/multi_head_attention_study.py`).

- **The extension points** (the workplan's D6, D7), each a field with a default, added by the work
  that uses it:

  | later work | where it goes | what it adds |
  | --- | --- | --- |
  | padding masks (causal is done: `Attention(causal=True)`, the sequence task) | attend, forward: an additive mask on `S[i]` before the max shift, as the causal mask's `-inf` | a mask input; backward unchanged (`P_ij = 0` zeroes `dS_ij`) |
  | dropout on the weights (done: `Attention(dropout=p)`, roadmap step 7) | attend: an inverted mask on `P[i]` before `H[i] = P[i] V[i]`, kept for backward | a `dropout` field, the mask drawn by the layer from the network's generator |
  | dropout after the projection (done: `Dropout(p)`, roadmap step 7) | after combine | a token-wise dropout layer, outside attention |
  | a separate value width | project and combine: `Wv` `(h·d_v, d)`, `Wo` `(d, h·d_v)` | a `value_size` field defaulting to `key_size` |
  | grouped- or multi-query attention | project: `Wk`, `Wv` with `g` row blocks; attend: head `i` reads key/value block `i // (h / g)` | a `key_value_heads` field defaulting to `heads` |
  | cross-attention | project: `K`, `V` from a second input | a second input to the layer, which `Sequential` doesn't have: its own design |

- **GELU** (D5 (c)), ViT's FFN activation, an activation for `Dense`: it needs `erf`, which stable
  Rust lacks (the `tanh` form is a different function), in all three implementations and the crate.
  Planned with SwiGLU as roadmap step 13 ([ffn-activations-workplan.md](ffn-activations-workplan.md)).
- **The study's other arms** (D8 (b), (c)): run on 2026-10-09 (`scripts/patch_geometry_study.py`,
  findings in its docstring). Patch 4 (`T = 49`) costs every arm 0.4 to 1.2 points at 5 epochs;
  the low-rank bound still doesn't measurably bind (+0.42 +- 0.83 for `key_size=32`); heads start
  to help with more or wider tokens (+0.6 over one head at `T = 49` and at `d = 64`); and `d = 64`
  adds 0.13 points to the best model at 3.7 times the parameters. Patch 7 and `d = 32` stay the
  MNIST patch models' trade; the gap to conv (0.9 points) is for the sequence task to revisit, not
  more MNIST geometry.
- **The numpy attention backward's 10% from stage 3.** #606's tier 1 A/B reported the numpy rows
  within noise, but before #610 both sides imported the new tree's package. Rerun on 2026-10-09
  with the fixed script (`ab.py run --script-from`, #616; report on #616), stage 3 made
  `hidden_delta_batch` 10.2% slower (243 to 268 us per call, the stage-1 attention case at batch
  32, consistent over 6 passes); `forward_batch` and `accumulate_gradient_batch` stayed within
  noise. Main's numpy backward is still stage 3's. The cost came with the three-block structure
  (D10), likely in the backward's (N, h, T, d_k) views or its stacked `np.matmul`s. The sequence
  task's mask touched the numpy layer's forward only and left this as it was. Still worth a look:
  a fix must keep the blocks and their bits.

Not planned yet:

- A `TransformerLayer` spec (D5 (b)): a transformer layer is the attention and FFN blocks written
  again; a spec would freeze pre-LN against post-LN, the activation and where dropout goes.
- A class token, relative position biases, and head pruning or per-head analysis (Michel et al.
  2019, "Are Sixteen Heads Really Better than One?").
- A faster multi-head kernel: a crate-tuning item (From the benchmark machine baseline), with its
  own A/B.

## From the sequence task

The workplan settled `Embedding(vocabulary, size)` over token ids (D5), a token-wise softmax output
with the mean of the `T` cross-entropies as the `"sequence"` network shape (D6), and
`Attention(causal=True)`, `-inf` on the future scores before the max shift (D7), in all three
implementations and the crate; four corpora, each in its own dataset repository (D2, D3); fixed
non-overlapping windows of 65 characters, the last 10% held out (D4). The study
(`scripts/sequence_study.py`, D9) found every arm where D9 expected it (README, The sequence task).

- **Generation** (D11 (b)): `scripts/sample_text.py`, a sliding window of the last `T` characters,
  sampling from the output at the last position, `T` forward passes per window's worth of text. The
  fixed-`T` network has no shorter prefix; a sliding window needs none.
- **Longer training.** Neither transformer had converged at 10 epochs: their held-out losses
  still fell by 0.01 to 0.05 bits per character over the last three. More epochs, with a decaying
  rate once the curve flattens (`lr_schedule.py` has linear warmup only), would show where the
  2-layer models stop; the 5-seed sweep took about 2.5 h on `jebel` at 6 workers.
- **Euclid's train/held-out gap.** On Euclid the 2-layer model is 0.49 bits per character lower on
  training windows than held out (1.01 against 1.50), the widest of the four corpora (Herodotus:
  none), and the gap grows with the model (ffn 0.17, 1-layer 0.44). The held-out part is the
  text's last 10%, from Book XII's similar pyramids (XII.8) through Book XIII, and even the
  counted bigram is 0.16 bits worse there, so the gap mixes overfitting with a shift in the text.
  Answered by the attention-dropout study (#666): on a split from across the text (each of 100
  blocks' last tenth held out) the 2-layer model's gap is 0.04 (1.05 against 1.02), so about 93%
  of it is the shift and little is overfitting; Tiny Shakespeare's 0.30 falls to 0.07 the same way.
- **Windows at random offsets** (D4 (b)): every position of every character a target, the usual
  language-model sampler. It needs a sampler in the trainer (an epoch no longer a permutation of
  fixed rows) and its state in run checkpoints. Overlapping fixed windows at stride `T / 2`
  (D4 (c)) are the cheaper middle step, with correlated examples.
- **Padding masks and variable-length windows** (D1 (b)): nothing pads while windows are fixed;
  the mask's place is From multi-head attention's table.
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

The workplan settled both of the literature's mechanisms (D1): `Attention(dropout=p)`, inverted
dropout on `P` (D3), one `(N, h, T, T)` mask a batch drawn row-major (D4), and a token-wise
`Dropout(p)` after `Position` and ending a block's body (D2); the crate's attention ops take
`dropout`, `training` and `rng` keywords (D5); all three implementations, their masks by bits (D6);
format 2 entries and three golden entries (D7, D9). The study (D8, #666;
`scripts/attention_dropout_study.py`'s docstring) found the sequence study's gaps mostly the text's
shift (on a split from across the text Euclid's 0.49 bits per character falls to 0.04, Tiny
Shakespeare's 0.30 to 0.07), and dropout narrowing every gap by raising the training loss, at a
cost of 0.01 to 0.35 bits held out; the one gain is attention dropout 0.1 on Euclid's contiguous
split (0.023 lower, every seed), which the spread split doesn't repeat.

- **Two crate shifts for the next baseline to measure against.** After crate #54 (#656's tier 1
  A/B, 6 passes), Rust `hidden_delta_batch` at batch 32 was 3.5% slower with dropout off, though its
  products are unchanged, while in the same run the unchanged `accumulate_gradient_batch` at batch
  32 moved 7.6% faster: read as a codegen or layout shift, as #627's (the sequence task's crate
  bump, +1.5% to +5.5% on some batch ops). On `jebel` such rows may be code placement (From the
  benchmark machine baseline). Beside the numpy backward's 10% (From multi-head attention).
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

## From rotary positions

The workplan built rotary position embedding as an attention option, `Attention(positions="rotary",
rotary_base=10000.0)` (D2), rotating half (feature `j` with `j + d_k / 2`, D3), its angles one
table per layer from Python's `math` so all three implementations rotate by the same bits (D4), the
crate's attention ops taking `cos` and `sin` (D5), and one position scheme per token part (D6).
The study (D8; `scripts/rotary_study.py`'s docstring) found rotary lower held out than learned
positions on every corpus and arm, 0.024 to 0.062 bits per character at epoch 10 (39 of 40 seeds),
at +3.5% an epoch alone; so new sequence models default to rotary (D9, the README's sequence
example).

- **D9's other half: step 9's `train` builds rotary models by default.** The generation workplan
  owns `train`; its stage that adds it takes the default.
- **#678's unexplained Rust gain.** The tier 1 A/B of the crate bump (rotation off) came out 7 to
  18% faster on five Rust rows, unexplained and not claimed: a codegen or layout shift, as #627's,
  for the next baseline to measure against.
- **The study's open questions.** `rotary_base` wasn't tuned (10,000 everywhere); every arm ran
  at the learned arm's tuned rate, and rotary may prefer another; nothing had converged at 10 epochs; and rotary's gain at a longer
  context, where generation's sliding window will run, is untested (the context-length workplan).

Not planned yet:

- ALiBi or T5's relative biases, beside rotary (D1 (c)): `positions` is a string so either can join
  without a new spec.
- A `Position()` and rotary attention in one token part (D6): refused today, one scheme a model.
- Rotating only a fraction of each head's features (GPT-NeoX's `rotary_pct`), or the interleaved
  pairing (D3 (b)), the same model up to a permutation.

## From the benchmark machine baseline

- **Crate tuning for the gaps it found.** Where Rust lags numpy on the i7, or lost ground from
  the laptop, and what to investigate for each, ordered by payoff:
  [i7-9700k-rust-optimization.md](i7-9700k-rust-optimization.md). Each item gets its own crate
  workplan with an A/B on both machines (measurement.md §9).
- **Code placement on the i7** (multi-head attention, stage 5, #609). The multi-head crate's
  attention ops ran up to 14% slower per call on `jebel` with the same instructions per call
  (within 0.3%): in the new build, forward's µops from the legacy decoder (`idq.mite_uops`) tripled
  as uop-cache delivery fell, a placement effect on Coffee Lake (the JCC erratum's 32-byte
  windows), and ops whose code didn't change moved by up to 13% between builds. `pyramidon` showed
  no change. To investigate: `perf record` the hot loop's addresses in both builds, and whether
  `-C llvm-args=-x86-branches-within-32B-boundaries` (or function and loop alignment) makes crate
  A/Bs on the i7 stable; until then, a crate A/B's few-percent rows there may be placement.
- **Rerunning the baseline** after a change that could move every number (kernel, BIOS,
  numpy/OpenBLAS, rustc, the frequency policy): rerun the workplan's stages 4-5 at the same
  tags, then at `main`, and replace [machine_profiles/i7-9700k.md](machine_profiles/i7-9700k.md)'s
  tables, with the reason, in one PR.
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
- **Crate tuning on CPU attributes** ([i7-9700k-rust-optimization.md](i7-9700k-rust-optimization.md))
  uses the archive's profiles and runs from both machines.

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

The owner terminated the plan (2026-10-12) after its reading (stage 2), for the window-recurrent
design (`git show e45944e:docs/window-recurrent-adder-workplan.md`, Why this design):
a fixed-depth network that finds a carry's deciding column in one long hop needs position codes that
stay distinguishable at any distance, which bounded codes can't give.

- **Its literature section stands** (`git show c3d619b:docs/length-generalisation-workplan.md`,
  What the literature found), and the window-recurrent plan builds on it.
- **Length-free token networks (its P1) are parked.** Stage 3 (P1a: `(None,)` token ids in the
  shape walk, `AnyInputShape`, format 2's `{"dimension": null}`) was built and passed (tests and
  the golden run) but never opened as a PR: the local branch `length-free-shapes` (`bd13b16`).
  A window has a fixed `T`, so the window-recurrent plan doesn't need it; generation (step 9) or
  a later length study may.
- **For any later fixed-depth length study:** rotary and no positions fail at long range by
  construction (resolution and dilution); an additive distance bias (ALiBi, P2) is the scheme that
  doesn't, and D12's start token is what a no-positions control needs.
- **The result's `crate` field** hashed `indrajala_math_rust/__init__.py`, not the extension
  module, so the addition sweep's results all show `639a07d4bc7d`; fixed by the window-recurrent
  plan's stage 8 (`studies.common.crate_hash`), for later results only.

## From the demo split

The demos, the matplotlib plotting and the tkinter capture are in
[indrajala-demos](https://github.com/davidbarkhuizen/indrajala-demos), which uses this repository
as a submodule; the measurement tools they lived beside are in `indrajala_ml/measurement/` and
`indrajala_ml/studies/` (`git show 9c50fe6:docs/demos-split-workplan.md`).

- **At 1.0, indrajala-demos depends on the published package** (the plan's D2): until then a
  "Bump indrajala-ml/" PR there carries each library change to the demos, and its CI runs their
  tests against it.

## From the addition study

The catalogue, the eight candidates and the unattended harness are in the README (The addition
study); the sweep's findings are in `scripts/addition_study.py`'s docstring. The first three entries
were recorded while the sweep ran (2026-10-11); the rest are the workplan's leftovers: its
decisions' unchosen options and its Excluded for missing infrastructure (`git show 8990f74:docs/addition-study-workplan.md`).

- **The adders' threshold as a check for each new primitive.** The column transformers sit on a
  sharp edge (at `n = 4`, d 4 fails at every candidate and shape, d 6 passes from 2 layers), and a
  run near it takes a few epochs. Whether a change moves that edge is a cheap, quantitative check:
  one harness config per question, no new study design. First uses:
  - step 8's rotary positions, and no positions, against the learned `Position`;
  - step 10's recipe (schedule, AdamW, clipping), including whether it smooths the non-monotone
    ladders, where a size succeeds and the next one up doesn't (column-unordered, 1 layer, 2
    heads at `n = 4`), which suggests the "too small" verdicts near the edge are partly the
    optimiser's;
  - steps 13 and 14 (FFN activations, grouped heads) at the sizes where each component matters
    most.
- **Milestone A's baseline** ([arbitrary-length-addition-workplan.md](arbitrary-length-addition-workplan.md),
  H2). The saved models use a learned `Position` tied to their `T`, so they can't be run at
  another length. Milestone A retrains the column transformers with rotary or no positions at the
  sweep's threshold sizes, rates and budget: the sweep tells it what to train, not what to load.
- **Probes of the passing adders.** Probing the smallest passing 2-layer networks against
  near-misses one width below (d 4 at `n = 4`) could locate the carry lookahead: which layer
  finds the deciding column below, and which head propagates the carry. This needs the accessor
  for a layer's activations on a batch (below). The
  probes don't need the sweep's exact files. When this is planned, it retrains the passing and
  near-miss networks it needs (minutes for a few seeds), on the code of the day. Only the passing
  models at each threshold and the width above are archived (the model archive, after the sweep).

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
  be the FFN's rather than attention's; the probes below would say.
- **Sample efficiency** (D7 (b)): one fixed training set instead of a fresh one each epoch, for a
  curve of examples against passing. It mixes overfitting into the size question, so it is its own
  study.
- **Candidates the framework couldn't build** (the workplan's Excluded for missing
  infrastructure), each a new candidate entry and a catalogue run once its piece exists:
  - a recurrent adder (one column a step, the carry as state): the window-recurrent adder,
    `git show e45944e:docs/window-recurrent-adder-workplan.md`;
  - length generalisation (train at `n`, test at `n + k`): rotary positions exist (step 8), but
    loading trained weights into a network of another `T` doesn't (the parked `length-free-shapes`
    branch, From length generalisation); the window-recurrent adder sidesteps it with a fixed
    window;
  - the string format's loss on answer tokens only, and variable-length answers decoded
    autoregressively: steps 11 (counted targets) and 9 (generation);
  - schedules, AdamW and clipping, in case a "too small" verdict is the optimiser's: step 10;
  - not planned yet: a weight-tied (looped) transformer, whose depth grows with `n` by iterating
    one block (blocks sharing weights, a variable iteration count); one dense network for every
    sum digit (several independent softmaxes on one flat network); supervised carries (auxiliary
    output heads and losses for each column's carry); an accessor for a layer's activations on a
    batch, for the probes above (it may be small); and a 1-D convolution along the tokens in the
    `sequence` shape, a local, ripple-like bias against attention's.
