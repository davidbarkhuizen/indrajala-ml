# Workplan: a factored state for the window-recurrent network (modular division)

**Status: written 2026-10-14, at the owner's request (the claims workplan's D32: "division resumes
once the recurrence has a factored state, a package change with its own plan"). Decisions D1-D9
settled with the owner in dialog, each with its options' pros and cons; D10 settled in stage 4,
when its test showed the factors weren't independent as first described. Stages 2 and 3 done
(indrajala-math-rust #57, #738); stage 4 is this PR.** The
infrastructure stages (2-7) are development on `pyramidon` and can start now; the migration
(stage 5) waits for stage 7 and E8b of the claims workplan to finish writing their files; the
study (stages 8-9) waits for `jebel`.

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

## Stages

1. **This workplan** (docs).
2. **The crate's grouped ops** (`indrajala-math-rust`), fused where it pays (D5): the codes'
   forward and backward over G codebooks of their own sizes and widths; the G GRU cells, the
   cross-term an optional input; the G attention poolings; a fused recurrence step over them;
   numpy mirrors; parity tests, stubtest. Bit-parity at G = 1 unless the PR gives a good reason to
   drop it (D6); a fusion that measurably speeds the step is one. *Done (#57):* the fused step
   (`recurrence_step_*`) and the codes over G codebooks (`factored_codes_*`); no separate grouped
   pooling or cell ops, which nothing would call. Each factor runs the single-factor ops, so G = 1
   is today's step by bits. The fusion measured 0-10% over per-factor calls (the kernels dominate).
3. **Bump `rust/`.**
4. **The package**: `Factor` and `WindowRecurrence(factors, cross)` with their validation; the
   network on numpy, Rust and pure Python over the grouped ops; format 2's `factors` and parts
   order; the seeded initialisation; the registry walks and fixtures; the cross-term tested.
   The golden run passes, or (if stage 2 dropped parity, D6) its window entries are re-recorded
   (§8) and archived on both machines; a tier 2
   A/B of a window network's `learn` (G = 1, old against new), since the grouped ops claim speed.
   *Built:* each backend's recurrence (`RecurrenceArrayLayer`, `RecurrenceRustArrayLayer`, the
   crate's fused step, and pure Python's `RecurrenceLayer`), the poolings' reading point (D10), and
   `scripts/migrate_factored_state.py`, which migrates the saved-model fixtures here and stage 5's
   files later. The certificate and the harness read one-factor states (asserted) until stages 6-7.
5. **The migration** (after the claims workplan's stage 7 and E8b finish): saved networks, results
   and certificates rewritten (D9); the hash table committed with it; the docstrings' hashes
   updated. Check: every migrated E1 and E8 certified network certifies again, the same
   verdicts.
6. **The certificate**: tuple states (D4), per-factor near ties, the per-factor readout (D9); tests
   on a hand-built factored machine for division `p` 5 (certifies, its factors read `A` and `B`),
   one holding `A + B` and `A - B` (certifies, named so), and faulty ones (a wrong next code in one
   factor, named in the counterexample). *Built:* `State` tuples through the search (no flat
   encoding: `np.unique` over each window's next-state rows), `next_states` branching every
   combination of the factors' near-tied codes, `FactorReadout` per factor, and
   `window_tasks.Coordinates` naming the target's states (division's `A`, `B`; the carry; the sum's
   total). The JSON: `codes` a list per factor, a state a list of codes, `phi` a list of
   `[state, target]` pairs (stage 5 migrates the old certificates to it). A `k` 4 certificate takes
   no longer (49 s against 56 s, one network).
7. **The harness**: factored cells in configs (`factors`, `cross` in a run's key), tuples in
   `codes_used` and `state_codes`, **certification after every curriculum step with a stop on
   passing** (D7) and **the pool stopped at the first certified run** (D8).
8. **C9 registered**: the config (`scripts/window_recurrent_factored_p5.json`, its poolings
   `BEFORE_STATE`, D10), a test checking it
   against D7-D8 (cells, cap = `m_1 m_2 + p^2 - 1`), and the probe: two epochs at 88 columns on
   `jebel`, its epoch time giving the batch's estimate. Over a day: the 6-code fallback, its probe.
9. **C9's run on `jebel`** (6 workers, memory measured first), then the findings: the certified
   network's factors (D9), its long tests (U1, S1), or "not found within the budget".
10. **Retire this workplan**; leftovers to next-steps.md.

## Cells and budget

| cell | `p` | `k` | codes per factor | network states `m_1 m_2` | division's states `q` | cap `m_1 m_2 + q - 1` |
| --- | --- | --- | --- | --- | --- | --- |
| C9 | 5 | 1 | 8 | 64 | 25 | 88 |
| C9, fallback | 5 | 1 | 6 | 36 | 25 | 60 |

Feasibility: each factor holds a residue (5 states) in 8 or 6 codes, so the cell can represent
division (A in factor 1, B in factor 2); the cap is the product bound, past which no network of
these codes can fit the training lengths without being division's machine.

Budget: one day on `jebel`, one batch of 6 seeds. Scaled from E8b's estimate (1-3 h a curriculum
run at 20 columns), a run to the cap is 4-13 h at 88 and 4-9 h at 60, before the grouped ops'
speed; a run that certifies stops early. The probe (stage 8) sets the figure; nothing launches
whose estimate exceeds a day.

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
