# Workplan: a factored state for the window-recurrent network (modular division)

**Status: written 2026-10-14, at the owner's request (the claims workplan's D32: "division resumes
once the recurrence has a factored state, a package change with its own plan"). Decisions D1-D9
settled with the owner in dialog, each with its options' pros and cons; D10 settled in stage 4,
when its test showed the factors weren't independent as first described; D11 in stage 8; D12 (the
owner) in stage 9, from the probe; D13 (the owner) in stage 9, from E8c's codes used. Stages 1-4
and 6-8 done; stage 9: C9 (the 6-code fallback, D12) ran from 15:46, stopped for **C9'** (16 codes
a factor, a 4 h budget, D13); stage 5 (the migration) unblocked, after stage 7's analysis of the
claims workplan reads the old files. C9' found nothing by step 10 and its diagnosis showed why (the
factors learn only "is it zero": D14); **C9''** registered (stage 9b, D14-D17, the owner,
2026-10-14 evening): a spare third factor and the factors' learning staggered, on both machines.**

## Why

E7 and E7' (the claims workplan) failed to learn modular division at 0 of 80 each. E7' gave the
networks codes for every state, yet 79 of its 80 counterexamples were a wrong next state: **one
codebook must learn division's 25- or 49-state table flat, and fitting it, not representing it,
was the limit.** Division's machine (`window_tasks.modular_machine`) has more structure than a
flat table:

- its state is two residues `(A, B)`, `A' = (3A + a) mod p` and `B' = (3B + b) mod p`;
- **each residue changes from its own digit only**; only the output, the running quotient
  `A / B mod p`, reads both.

So the state is two independent counters, each a modular-sum-like automaton (E8's rung), and an
output over their pair. This plan gives the network a state of that shape: **G factors, each with
its own pooling, cell and codebook**, the machine's state a tuple of codes, and the
infrastructure around it (the crate's kernels, persistence, the certificate, the harness), then
one existence test on division.

The claim, registered here before any run:

> **C9.** A window-recurrent network with a factored state (two independent factors, unsupervised)
> learns modular division at `p` 5: some network, trained by the curriculum at `k` 1 with 8 codes
> per factor, is certified against division's machine (every length), within the budget (D7, D8).

C9 is an **existence** claim (D7): no rate, no arms, no controls. A certified network proves it;
running out of budget records "not found within the budget", not a refutation.

Registered in stage 9 (D13), before its data:

> **C9'.** As C9, with E8c's counter for each factor: some network with two independent factors
> of 16 codes each, trained by the curriculum at `k` 1, is certified against division's machine
> (every length) within a 4 h budget on `jebel`.

C9' is an existence claim too; its refutation, as C9's, is none: "not found within the budget".

Registered in stage 9b (D14-D17), before its data:

> **C9''.** As C9', with three independent factors of 16 codes each and their learning staggered
> (D14): one factor learning at a time at the rate, those that have had a turn at a tenth of it,
> the rest frozen, the turn passing to the next factor at each plateau. Some such network, trained
> by the curriculum at `k` 1, is certified against division's machine (every length) within a 4 h
> budget on `jebel` (seeds 0-5) and `pyramidon` (seeds 6-8).

C9'' is an existence claim; its refutation, as for C9', is none: "not found within the budget".

## The factored state

A factor `g` has a state `c_g` of width `s_g` and, if snapped, a codebook of `m_g` codes. Per
window:

- **pooling** `g`: its own learned query over the window's tokens, giving `u_g` (D2: a factor
  attends to the tokens it needs, `A`'s to `a`, `B`'s to `b`): the tokens the output layer reads,
  or those before the first `StateInput` (D10);
- **cell** `g`: a GRU, `h_g = GRU_g(c_g, u_g)`, or with the **cross-term** on (D1),
  `GRU_g(c_g, [u_g, c_other])`, the other factors' states in its input;
- **codes** `g`: `h_g` snapped to the nearest of its `m_g` codes, straight-through (W4), the next
  window's `c_g`; none for a continuous factor;
- **initial state** `g`: learned, snapped by its own codes (W8).

The window's `StateInput`s read the whole state, the factors' concatenation `[c_1, ..., c_G]`, so
the outputs see every factor. With the cross-term off (the default) and the poolings reading the
tokens before the first `StateInput` (D10), **factor `g`'s next state is a function of its own
state and the window's columns only**: the independence division has. (Pooling the tokens the
output layer reads, the default, a factor sees the others through the `StateInput`s; stage 4's
test found this.) With G = 1 the network is today's.

The spec:

```python
WindowRecurrence(
    factors=(Factor(4, Codes(8)), Factor(4, Codes(8))), cross=False, pooling=AttentionPooling(BEFORE_STATE)
)
WindowRecurrence(factors=(Factor(4, Codes(8)),))  # today's network
```

The network's machine has a state `(i_1, ..., i_G)`, one code per factor, up to `m_1 ... m_G`
states.

## Decisions

Settled with the owner, 2026-10-14, each option's pros and cons as put:

- **D1. Where the factoring lives. Settled: (d).**
  - (a) Product quantisation of the codes only: one GRU, its output split into slices, each snapped
    to its own codebook (Jégou et al. 2011). Pros: smallest change. Cons: the GRU mixes the
    slices, so the factoring isn't enforced; the network can learn the tangled table again.
  - (b) A cell and a codebook per factor, independent. Pros: enforces division's structure; each
    factor's job is a counter. Cons: factors that genuinely interact can't be expressed; adding
    that later changes the spec and format again.
  - (c) A learned projection per factor, then snapped. Cons: independence not enforced; extra
    parameters; what feeds the next window is ambiguous.
  - (d) *Chosen.* (b) with an **optional cross-term** (a factor's cell may read the others'
    states), off by default. Pros: covers (b) exactly and coupled states too; ablations are one
    flag; one structure. Cons: a little more to build and test now; a run with the cross-term on
    doesn't enforce independence, so the flag is in every result key and certificate. The
    cross-term is built and parity-tested in stage 4 but not used in C9's study.
- **D2. Generality. Settled: (b).**
  - (a) Exactly two factors. Pros: least to build. Cons: three or more (the factorisation
    workplan's registers) need a new spec and format; two code paths.
  - (b) *Chosen.* **G factors, each its own size** (`Factor(state_size, codes)`, codes or None),
    the single-factor case being today's network, and **a pooling per factor**. Pros: one path,
    no special case at G = 1; unequal factors (`mod 5` beside `mod 7`); a continuous factor
    beside snapped ones for free. Cons: every reader of `recurrence.codes`, `state_size` and the
    parts' order changes.
  - (c) G equal factors. Cons: unequal residues pay the larger codebook; a later change to (b)
    changes the spec and format again.
- **D3. Supervision of the factors. Settled: (a).**
  - (a) *Chosen.* **Unsupervised only**: the outputs' and the codes' losses, nothing telling a
    factor what to hold. Pros: the strongest claim; nothing extra to build; no factor-target
    interface on the tasks. Cons: a failure doesn't separate "can't fit" from "wasn't found".
  - (b) A state loss always on (a head per factor predicting its residue). Cons: trace
    supervision; no discovery claim.
  - (c) An optional state loss, unsupervised first. Cons: a long run before the diagnostic.
  - (d) An optional state loss, supervised first for existence. Pros: separates fitting from
    discovery. Cons: two runs before the headline; the loss in three backends.
- **D4. The machine's state in the certificate and the harness. Settled: (b).**
  - (a) Flatten to one mixed-radix integer. Pros: the certificate's search unchanged. Cons:
    reports lose which factor holds what.
  - (b) *Chosen.* **Tuples everywhere**: the state type is `tuple[int, ...]` in the certificate,
    the harness and every JSON, G = 1 included. Pros: readable; per-factor analysis direct. Cons:
    every reader changes, and the old files (D6, D9's migration).
  - (c) Flat in the certificate's core, tuples at the edges. Cons: two representations and a
    conversion.

  Near ties (the certificate's snap margin) are per factor: a state branches over every
  combination of its factors' near-tied codes.
- **D5. The backends. Settled: (b).**
  - (a) The existing parts per factor, looped in Python. Pros: no crate change. Cons: about 3 G part
    calls per window; at `k` 1 the overhead shows in seconds per epoch.
  - (b) *Chosen.* **Grouped ops in the crate**: the codes over G codebooks, the G cells (block
    diagonal, the cross-term an input), the G poolings, each one call per window, with numpy
    mirrors. **Fused ops preferred** (the owner): where it pays, one op for a window's whole
    recurrence step (every factor's pooling, cell and codes, forward and backward) over separate
    grouped ops. Pros: fastest; reusable for any multi-factor machine. Cons: a crate stage and a
    bump before the package; a fused op's operation order may differ from today's (D6).
  - (c) (a) now, measured; (b) if it matters. Cons: a possible second round of changes.
- **D6. Persistence and the golden run. Settled: (c), widened.** The owner: "we can't shackle our
  kernels' futures - having them fast and extensible is one of our main assets", and "we should
  have some good reason for abandoning bit-parity, but we should be able to do it".
  - (a) G = 1 bit-identical, always: parts factor by factor, the grouped ops reducing to today's
    arithmetic at G = 1. Pros: the golden run checks the refactor. Cons: shackles the kernels'
    layout and operation order.
  - (b) The kernels' layout free from the start; the window entries re-recorded regardless. Pros:
    no constraint. Cons: bit-parity, the refactor's strongest check, given up without a reason.
  - (c) *Chosen.* **Bit-parity at G = 1 is the default; the grouped ops may abandon it for a good
    reason**: a measured speed-up, or a layout that later work (more factors, the cross-term, other
    multi-factor machines) needs. Stage 2's PR states the reason, with its measurement if it's
    speed, and the window entries are then re-recorded under measurement.md §8 and archived.
    Pros: the check kept when it costs nothing; the kernels never shackled. Cons: a judgement per
    kernel. When parity goes, stages 4-5 carry the check instead: the numpy and Rust networks in
    parity, and every migrated certified network (E1, E8) certifying again with the same verdicts
    (stage 5).
- **D7. The bound and the codes. Settled: (a) of the existence design.**
  - The exclusion bound `L >= m + q - 1` (experiment-design.md, section 1) counts the network's
    states: with factors, **`m = m_1 m_2`**. Independent factors don't lower it: two counters
    `mod m_1` and `mod m_2` (coprime), ignoring the input, under an output of 1 only when both are
    at 0, agree with an all-0 target on every word shorter than `m_1 m_2`. So an additive bound
    is false, and a surplus costs `r^2`: at `p` 5, `r` 2 (10 codes a factor) is a bound of 124,
    `r` 3 is 249.
  - The owner: no experimental plan of multi-day runs; and the aim is existence, not the phase
    space. An existence claim needs no rates, arms or controls, and **the certificate is the
    proof** (a certified network is division's machine at every length, however trained), so
    the bound's role is to make success likely, not valid.
  - (a) *Chosen.* **The bound is the curriculum's cap**: each run is certified after every
    curriculum step and stops when it certifies; a run fails only at the cap. Pros: the rule
    kept (a failed run trained past where memorising fits); a success costs only what it needs,
    so a surplus is cheap when it works. Cons: failed runs pay the whole cap.
  - (b) Drop the bound for existence tests, a cheap fixed cap (30 columns). Pros: more seeds a day.
    Cons: E8 showed short training mostly memorises; against the owner's rule.
  - The codes: **8 per factor** (`r` 1.6, the most the day's limit allows), cap 88; **if the
    probe puts a batch over a day, 6 per factor**, cap 60 (registered here, so nothing changes
    after the data).
- **D8. The cell. Settled.** **`p` 5** (25 states, the smallest division not degenerate: at `p` 3
  the residue is the last digit, at `p` 2 the quotient ignores `B`; `p` 7 only after a `p` 5
  network exists), **`k` 1** (E7's near misses; the certificate fast enough to run every step;
  `k` 4 never fitted in E7 and certifies in about 140 s), the curriculum arm only, **6 seeds in
  parallel on `jebel`, all stopped at the first certified network**.
- **D9. What each factor holds. Settled: (b); old files migrated.**
  - Any linear form `alpha A + beta B (mod p)` updates as a counter does (`x' = 3x + alpha a + beta
    b`), so an exact network may hold `A + B` and `A - B` rather than `A` and `B`.
  - (a) No readout. Cons: a working network without its explanation.
  - (b) *Chosen.* **In the certificate**: for each factor, its codes against the reachable target
    states: whether the factor is a function of the target state, and its partition named (`A`,
    `B`, `alpha A + beta B`, coarser, or none; the bare partition for tasks without names). Pros:
    cheap (the certificate's map is already built); every certified network carries it.
  - (c) A separate readout script. Cons: a second pass that can drift.
  - **Old files are migrated, not shimmed** (the owner): results, certificates and saved networks
    of E1-E8 rewritten to the tuple and factored layouts by a script that records each file's old
    and new sha256; the study docstrings' recorded hashes updated, the old ones kept beside them.
- **D10. Where the poolings read. Settled in stage 4: (b).** Stage 4's test showed that with the
  poolings over the tokens the output layer reads (the section above, as first written), a
  factor's next state moves when another factor's state does, the cross-term off: every
  `StateInput` adds the whole state to the tokens before the poolings read them. D1's enforced
  independence didn't hold.
  - (a) Keep the output layer's tokens; independence stated only at the cell. Pros: nothing to
    build. Cons: D1's reason for (b)/(d) void; C9 tests a network that can tangle its factors
    again through the window, the failure E7' showed.
  - (b) *Chosen.* **An option on the pooling, `AttentionPooling(reads=...)`: `OUTPUT` (the
    default, today's network) or `BEFORE_STATE`, the tokens the first `StateInput` reads**, so a
    factor's next state is a function of its own state and the window's columns. Pros: D1's
    independence enforced; G = 1 and every saved file unchanged (format 2 writes `reads` only when
    not the default, as attention's fields); one more reading point, not a second network. Cons:
    a factor's cell sees the window's columns before any block, so a window's computation beyond
    a pooling of its embeddings must happen in the cell (division's counters need only the digit,
    `x' = 3x + a`).
  - (c) Run the window once per factor, its `StateInput`s reading that factor's state alone. Pros:
    each factor sees the blocks' computation. Cons: G window passes a step; the outputs no longer
    see every factor at once.
  - (d) Pool before the first `StateInput` always, G = 1 included. Cons: changes today's network
    and every result's; the golden run re-recorded for no gain at G = 1.

  **C9's cell reads `BEFORE_STATE`** (registered here, before any data).
- **D11. Each factor's state width. Settled in stage 8: (b).** D8 named the codes, not the width
  `s_g`.
  - (a) 8 a factor, 16 in all. Pros: E7's and E8's total width. Cons: each factor half the width
    E8 gave a single counter; C9's factors not comparable with C8's cell.
  - (b) *Chosen.* **16 a factor**: each factor is E8b's counter cell (`s` 16), with 8 codes for
    its 5 residues. Pros: a C9 failure can't be laid on narrower factors; C9 and C8 read side by
    side. Cons: twice (a)'s state; the cell is a small part of a step, and the probe measures it.
  - (c) 4 a factor (stage 7's example). Pros: cheapest. Cons: narrower than any counter trained
    so far, a variable no run has tested.

  The rest of C9's settings are E8b's (rate, epoch examples, batch, patience, checks), with steps
  88 (the cap, `k` 1), 6 seeds, `certify_steps` and `stop_at_certified`
  (`scripts/window_recurrent_factored_p5.json`; the fallback,
  `scripts/window_recurrent_factored_p5_fallback.json`, 6 codes a factor and steps 60).

- **D12. C9 launches as the 6-code fallback (the owner, 2026-10-14).** The probe (Cells and
  budget) puts the registered cell's batch at 38-84 h.
  - (a) *Chosen.* **The fallback, as D7 registered it**: 6 codes a factor, cap 60
    (`scripts/window_recurrent_factored_p5_fallback.json`); 16-27 h if no run certifies, far less
    if one does (the study stops at the first; E8c's counters certified by step 5). Pros: the
    registration unchanged; the cap is still the bound, so a failure means trained past memorising.
    Cons: the worst case may pass a day; 6 codes for 5 residues is a thin surplus (1.2; E3 trained
    well from 1.33).
  - (b) 8 codes, cap 40, a new cap below the bound. Pros: 6.5-8.5 h; the better surplus; the
    certificate is the proof at any training length. Cons: departs from D7's cap = bound; a failure
    says only "not found by 40 columns".
  - (c) 8 codes, cap 30. Pros: about 4 h. Cons: further below the bound; a failure says least.
  - (d) The fallback's own probe first. Pros: measured, not modelled. Cons: codes barely move the
    cost; it would confirm the model.
- **D13. C9 stops for C9': each factor E8c's counter, a wall-clock budget (the owner,
  2026-10-14: "existence first, then refine"; "don't add too many codes").** E8c's curriculum
  counters at `m` 16 certified 6 of 6 at `q` 5 and 3 of 6 at `q` 4, every one at step 4 or 5, in
  about 5 minutes a run; **their certified networks reached 7-13 codes (median 10 at `q` 5) and
  minimised to the 5 or 4 states**. Learning a counter used about twice its states in codes. C9's
  6 codes a factor leave one spare, and at 4.3 h its runs were at steps 17-19, none certified:
  a run predicted to yield nothing (the claims workplan's D33).
  - (a) *Chosen.* **16 codes a factor**, 6 seeds, the cap at the product bound (`16 x 16 + 25 - 1
    = 280`), and a **4 h budget**: the study stops at its first certified run or at the deadline,
    "not found within the budget". Pros: each factor is the one cell with direct evidence (6 of
    6); every network E8c certified fits its codes; the bound kept, so the cell is feasible; a step
    costs about the same with more codes (D12 (d)), so 4 h reaches about steps 15-20, three or four
    times the step where E8c's counters certified. Cons: 16 is 3.2 times the states, the owner's
    floor (experiment-design.md, section 1), not a minimum: fewer codes are the refinement, after a
    network exists; the bound (280) is never reached, so a failure says only "not found by the
    budget's step".
  - (b) 12 codes a factor (2.4 times), bound 168. Pros: fewer codes. Cons: untested; below the
    owner's 3-5 times; one E8c network reached 13.
  - (c) 8 codes a factor (1.6 times), bound 88, the registered C9. Pros: the registered cell.
    Cons: can't hold the 9-13 codes half E8c's `q` 5 networks reached; 38-84 h to its cap.
  - (d) Let C9 finish (16-27 h). Pros: no compute discarded; the registration kept. Cons: the
    evidence above predicts nothing found, and `jebel` held for a day.

  C9 is stopped through its stop file (its text the reason), its runs ending at their next epoch;
  its results are its findings ("not found; stopped at steps ..."). *Built for it:* the
  `budget_hours` setting (the deadline set at the study's first launch and kept on a resume;
  `status` shows it) and the stop file's reason.

- **D14. How the factors learn: staggered (the owner, 2026-10-14 evening, from the diagnosis of C9').**
  C9' networks plateaued from step 2: "0" and "undefined" 74-89% right, the nonzero quotients
  27-44% (25% is a guess among them); each factor's code carried at most 0.41 of `A` or `B` but
  0.48-0.67 of "is it zero" (`diagnose`). The reason is in the task: alone, a residue tells the
  quotient only whether it is 0 (`A`) or undefined (`B`): I(q; A) = 0.58 bits = I(q; A = 0),
  I(q; B) = 0.72 = I(q; B = 0), of H(q) = 2.58; the other 1.28 bits (50%) only the pair gives
  (23% at `p` 3, 63% at 7). Trained together, the factors settle on the zero tests. Yet predicting
  "undefined" exactly needs all of `B` (whether the next `B` is 0 depends on all of it), so one
  factor given the whole gradient can learn its residue; then the other's residue is worth 1.86
  bits, a strong signal.
  - (a) *Chosen.* **One factor learns at a time.** The active factor at the rate, the factors that
    have had a turn in the background at a fraction of it (D16; the owner: "keep the first factor
    in the background at a low frequency ... to keep it memorised"), the rest frozen (rate 0: their
    weights unchanged, their gradients cleared). **The turn passes at a plateau** (the harness's,
    on the answer's screen), to the next factor in order, round and round, and the step is tried
    again; a step ends at its check passing, at its epoch budget, or at a plateau once every
    factor's turn there has plateaued. The turns carry from step to step and are recorded in each
    (`turns`, `active`, `started`). Built in the harness, not the network: the optimizer is
    wrapped while an epoch trains, scaling each factor's parts' rate (`FactorRates`); with every
    factor at the rate it trains bit for bit as the network alone (a test). Pros: the active
    factor has the whole gradient; nothing of the decomposition given (which factor learns what
    is the gradient's choice; the schedule only says when); the pattern extends to any `G`; no
    timed path changed. Cons: a schedule's parameters (D16); a frozen factor feeds the output
    fixed features of the input until its turn; retried steps cost epochs.
  - (b) Joint training, C9'. Cons: the diagnosis above.
  - (c) The background by frequency, a factor stepped every `N`th batch. Pros: the owner's word
    literally. Cons: Adam's moments go stale between steps; on average the same as a fraction of
    the rate, which Adam doesn't undo (it normalises the gradient, not the rate).
  - (d) The turn passed on the diagnosis's shares. Cons: they read the target's `A` and `B`: the
    decomposition handed over. Never; `diagnose` only reads the runs.
  - (e) A fixed schedule (a turn every `N` epochs). Pros: simplest. Cons: arbitrary; cuts a factor
    off while it learns, or idles it after it plateaus.
- **D15. Three factors, one spare (the owner, 2026-10-14 evening: "one more than we need ... to
  give the learning process flexibility").**
  - (a) *Chosen.* **Three factors of 16 codes**, E8c's counter each. Pros: if a factor's turn ends
    in a useless mixture, the rotation has a fresh codebook, not a dead end; any two can pair.
    Cons: the cap at the product bound is `16^3 + 25 - 1 = 4,120` columns, never reached in the
    budget, so a failure says only "not found within the budget" (as for C9'); a third more of the
    state's compute; the certificate over 3-tuples (its map is many-to-one, a spare factor
    allowed).
  - (b) Two factors. Pros: C9' cell, the smallest change. Cons: no spare.
  - (c) Three factors of 10 codes (the bound 1,024). Pros: a smaller bound. Cons: under E8c's
    counters (7-13 codes reached at `q` 5) and the owner's floor (3 times the states).
- **D16. The background's fraction: 0.1.** The rest as C9' (E8b's settings: patience 30, at most
  200 epochs a step, so three turns of at least 31 epochs fit a step).
  - (a) *Chosen.* **0.1.** Pros: a learned factor follows the others' changes without being
    retrained by them. Cons: a guess; untested.
  - (b) 0.01. Pros: nearly frozen. Cons: can't adapt to the factor now learning.
  - (c) 0.3. Pros: adapts. Cons: closer to joint training, the trap.
- **D17. Both machines (the owner: "as much as is useful of our available resources on both
  machines").**
  - (a) *Chosen.* **Seeds 0-5 on `jebel` (6 workers), 6-8 on `pyramidon` (3 workers)**, one config
    each (`window_recurrent_factored_p5_stagger.json`, `_stagger_pyramidon.json`; `first_seed`
    numbers the second's seeds), the same directory name, each machine's study stopping at its own
    first certified run or its 4 h; results copied together after, with manifests. Pros: 9 seeds
    in the budget; a certified network on either proves C9''. Cons: `pyramidon`'s runs are slower
    (Ryzen 7 3700U) and memory-bound (5.3 GB, one memory-heavy job at a time); each machine stops
    only its own runs.
  - (b) `jebel` only. Cons: 6 seeds.
  **The stop rule** (registered here, before the data): at 2 h from launch, each machine's study
  continues only if one of its runs has, in `diagnose`, a factor sharing at least 0.6 of `A` or of
  `B` and rising over its last two steps, or nonzero quotients at least 0.55; otherwise it is
  stopped through its stop file: "not found; stopped at 2 h by the registered rule". `diagnose
  --follow` runs beside each study.

## Stages

1. **Done (#737).** **This workplan** (docs).
2. **Done (indrajala-math-rust #57).** **The crate's grouped ops** (`indrajala-math-rust`), fused where it pays (D5): the codes'
   forward and backward over G codebooks of their own sizes and widths; the G GRU cells, the
   cross-term an optional input; the G attention poolings; a fused recurrence step over them;
   numpy mirrors; parity tests, stubtest. Bit-parity at G = 1 unless the PR gives a good reason to
   drop it (D6); a fusion that measurably speeds the step is one. *Done (#57):* the fused step
   (`recurrence_step_*`) and the codes over G codebooks (`factored_codes_*`); no separate grouped
   pooling or cell ops, which nothing would call. Each factor runs the single-factor ops, so G = 1
   is today's step by bits. The fusion measured 0-10% over per-factor calls (the kernels dominate).
3. **Done (#738).** **Bump `rust/`.**
4. **Done (#739).** **The package**: `Factor` and `WindowRecurrence(factors, cross)` with their validation; the
   network on numpy, Rust and pure Python over the grouped ops; format 2's `factors` and parts
   order; the seeded initialisation; the registry walks and fixtures; the cross-term tested.
   The golden run passes, or (if stage 2 dropped parity, D6) its window entries are re-recorded
   (§8) and archived on both machines; a tier 2
   A/B of a window network's `learn` (G = 1, old against new), since the grouped ops claim speed.
   *Built:* each backend's recurrence (`RecurrenceArrayLayer`, `RecurrenceRustArrayLayer`, the
   crate's fused step, and pure Python's `RecurrenceLayer`), the poolings' reading point (D10), and
   `scripts/migrate_factored_state.py`, which migrates the saved-model fixtures here and stage 5's
   files later. The certificate and the harness read one-factor states (asserted) until stages 6-7.
5. **Code done (this PR); the run after stage 7's analysis.** **The migration** (after the claims workplan's stage 7 and E8b finish): saved networks, results
   and certificates rewritten (D9); the hash table committed with it; the docstrings' hashes
   updated. Check: every migrated E1 and E8 certified network certifies again, the same
   verdicts.
   *Built (the code; the run waits for stage 7's analysis):* `migrate_factored_state.py tree ROOT...
   --table FILE` migrates every network, step record, result and certificate under the roots
   (networks first: a certificate's `sha256` becomes its migrated network's), then each manifest
   listing a migrated file (the old one kept as `<name>.before-factored-state.sha256`), and writes
   the hash table; a second run changes nothing. A certificate's per-factor readout is computed
   from its own reachable states and `phi`, as `certify` computes it; E1's keep their names. An
   old result gains what today's harness writes for a run without per-step certificates
   (`certified` false, `certified_step` null, `stopped` false). The check is stronger than the
   verdicts: `recertify CERTIFICATE...` certifies each network again and compares the whole
   certificate but its seconds. On copies of E1 (with the control and matrix networks) and E8 `q`
   4, 1,032 files migrated, every manifest verified, and all 110 `k` 1 certificates came out the
   same as certificates made now, field by field.
6. **Done (#742).** **The certificate**: tuple states (D4), per-factor near ties, the per-factor readout (D9); tests
   on a hand-built factored machine for division `p` 5 (certifies, its factors read `A` and `B`),
   one holding `A + B` and `A - B` (certifies, named so), and faulty ones (a wrong next code in one
   factor, named in the counterexample). *Built:* `State` tuples through the search (no flat
   encoding: `np.unique` over each window's next-state rows), `next_states` branching every
   combination of the factors' near-tied codes, `FactorReadout` per factor, and
   `window_tasks.Coordinates` naming the target's states (division's `A`, `B`; the carry; the sum's
   total). The JSON: `codes` a list per factor, a state a list of codes, `phi` a list of
   `[state, target]` pairs (stage 5 migrates the old certificates to it). A `k` 4 certificate takes
   no longer (49 s against 56 s, one network).
7. **Done (#743; the report's fix for stopped runs, #747).** **The harness**: factored cells in configs (`factors`, `cross` in a run's key), tuples in
   `codes_used` and `state_codes`, **certification after every curriculum step with a stop on
   passing** (D7) and **the pool stopped at the first certified run** (D8). *Built:* a config cell
   `{"k": [1], "factors": [[4, 8], [4, 8]], "reads": "before_state"}` (and `"cross"`), its key
   naming each factor (`...-k1-s4-m8-s4-m8-before-state-seed0`; one factor's keys unchanged);
   `factors`, `cross`, `reads`, `certified`, `certified_step` and `stopped` in every result; codes
   as tuples, G = 1 included (stage 5 migrates the old files); Settings `certify_steps` (each step's
   certificate, `step-<j>.certificate.json`, its summary in the step) and `stop_at_certified` (a
   `models/stop` file: running workers end their step "stopped" at the next epoch, queued runs
   are cancelled, a later `run` does nothing).
8. **Done (#744; the probe in stage 9).** **C9 registered**: the config (`scripts/window_recurrent_factored_p5.json`, its poolings
   `BEFORE_STATE`, D10), a test checking it
   against D7-D8 (cells, cap = `m_1 m_2 + p^2 - 1`), and the probe: two epochs at 88 columns on
   `jebel`, its epoch time giving the batch's estimate. Over a day: the 6-code fallback, its probe.
   *Built:* both configs (D11: `s` 16 a factor; the fallback registered beside it), and the
   configs' feasibility tests counting a factored cell's states as the product of its codes (they
   read only `m` before, so skipped factored cells); C9's own test pins the cell, seeds, cap and
   E8b's settings. The probe waits for `jebel` (E8b's curriculum arm holds it); its figures go
   into stage 9's launch.
9. **Budget and tools done (this PR); the run next.** **C9's run on `jebel`** (6 workers, memory measured first), then the findings: the certified
   network's factors (D9), its long tests (U1, S1), or "not found within the budget".
   **Launch (D12):** the fallback config, 6 workers (592 MB peak each, 3.6 GB of `jebel`'s 31 GB).
   *Built for it:* `probe` (the budget's figures, `--json` keeps them) and `peak_mb` in every
   result (its worker's peak memory, for sizing the next study's workers; measurement.md, section 2).
   **C9' (D13):** `scripts/window_recurrent_factored_p5_m16.json`, 6 workers on `jebel`, C9 stopped
   first; then its findings, as C9's.
9b. **C9'' (D14-D17): built (this PR); the run next.** *Built:* a factored cell's `stagger` (in
   its run's key and label, `-stagger0.1`), `Turns` and `FactorRates` in the harness, the turn at
   each plateau and the step's retry, the turns recorded in each step and carried on a resume;
   `first_seed`; the curriculum's steps as ranges (a cap of 4,120 steps costs nothing). Tests: the
   turns, a frozen factor's weights bit for bit, every factor at the rate bit for bit the network
   alone, a step's rotation, a run's turns resumed, the configs against the cell. *Next:* the
   launch on both machines with `diagnose --follow`, the stop rule at 2 h, then C9, C9' and C9''
   findings together.
10. **Retire this workplan**; leftovers to next-steps.md.

## Cells and budget

| cell | `p` | `k` | codes per factor | network states `m_1 m_2` | division's states `q` | cap `m_1 m_2 + q - 1` |
| --- | --- | --- | --- | --- | --- | --- |
| C9 | 5 | 1 | 8 | 64 | 25 | 88 |
| C9, fallback | 5 | 1 | 6 | 36 | 25 | 60 |
| C9' | 5 | 1 | 16 | 256 | 25 | 280 (a 4 h budget) |
| C9'' | 5 | 1 | 16, 3 factors | 4,096 | 25 | 4,120 (a 4 h budget) |

Feasibility: each factor holds a residue (5 states) in 8 or 6 codes, so the cell can represent
division (A in factor 1, B in factor 2); the cap is the product bound, past which no network of
these codes can fit the training lengths without being division's machine. C9'' needs two of its
three factors for the residues, one spare (D15); its codes multiply to 4,096.

Budget: one day on `jebel`, one batch of 6 seeds. Scaled from E8b's estimate (1-3 h a curriculum
run at 20 columns), a run to the cap is 4-13 h at 88 and 4-9 h at 60, before the grouped ops'
speed; a run that certifies stops early. The probe (stage 8) sets the figure; nothing launches
whose estimate exceeds a day.

**The probe (2026-10-14, `jebel`, `713ff5f`; `scripts/window_recurrent_study.py probe` reproduces it):**
C9's seed 0, two epochs at step 8 (windows 1-8) and at step 88 (1-88). Training 6.8 s an epoch at
step 8 and 53.4 s at 88; the screen 0.6 s and 61 s (190 s in a step's first epoch, which builds
its screen cases); the whole check at 88 columns 215 s (a passing step checks every length 1..j);
peak memory 592 MB (the check). Training grows with the step, the screen and the checks with its
square. Modelled to the cap: **88 is 38 h** for a run that plateaus every step (31 epochs a step,
E8b's), 61 h at 50 epochs, 84 h for one passing every step in 3 epochs; **60 is 16 h, 25 h and
27 h**. So the batch at 88 exceeds a day, and D7's fallback applies (D12).

**The probe of C9' (2026-10-14, `pyramidon`, beside E8c's long tests; `probe
scripts/window_recurrent_factored_p5_m16.json --steps 4 12`):** seed 0, two epochs at steps 4 and
12. Training 7 s an epoch at step 4 and 15 s at 12; the screen 0.3-0.5 s and 2-15 s; the whole
check at 12 columns 39 s; peak memory 240 MB (6 workers: 1.5 GB of `jebel`'s 31 GB). A step that
plateaus (31 epochs) is about 9-10 minutes at step 12; one that certifies at step 4 or 5, as
E8c's, a few minutes. The budget (D13) bounds the batch at 4 h whatever the steps reach.

**The probe of C9'' (2026-10-14, `pyramidon`, beside the recertify check; `probe
scripts/window_recurrent_factored_p5_stagger.json --steps 4 12`):** seed 0, two epochs at steps 4
and 12. Training 12.6 s an epoch at step 4 and 28-31 s at 12 (about twice C9' figures, a third factor
and the recertify check beside it); the screen 0.4-0.5 s and 3-21 s; the whole check at 12 columns
72 s; peak memory 238 MB (`jebel`'s 6 workers 1.4 GB of 31 GB; `pyramidon`'s 3, 0.7 GB). A step
whose three turns plateau (at least 93 epochs) is about 45 minutes at step 12 on `pyramidon`, less
on `jebel`; a run that certifies early, as E8c's at steps 4-5, stops in under an hour. The budget
(4 h) bounds each machine whatever the steps reach.

## Not in this plan

- Stage 8 of the claims workplan and its compute: a separate decision (trimming its cells to
  existence tests, as D33 and D34 did).
- `p` 7, the cross-term on, more factors: after a `p` 5 network exists, each a small follow-up.
- A state loss (D3 (b)-(d)): only if C9 isn't found and the owner chooses it.

## Timing and checks

Stage 2 adds crate functions (parity tests, no A/B); stage 4 changes `learn` on the window
network: a tier 2 A/B, and the golden run passed or (D6) re-recorded and archived. Stages 5-7 are tier 0 but
change saved data and readers: every migrated certified network re-certified. Studies on `jebel`
under the machine lock, workers sized by memory, PL1 at 65 W.
