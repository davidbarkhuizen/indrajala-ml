# Workplan: a family of modular operations on one state (division's dense signals)

**Status: written 2026-10-15 at the owner's request (option 2 after C9''; option 1, the commitment
controls, is deferred behind it: [commitment-control-workplan.md](commitment-control-workplan.md)).
Decisions D1-D6 settled (the owner delegates a workplan's decisions; each with its options' pros
and cons). Probe-driven (the owner, 2026-10-15): short probes, the next chosen from each result; no
long run without a stated reason. Stages 1-2 done (the probe: none works with D10's poolings; the
poolings reading the window's output, the family certified 2 of 2 against division's machine, the
quotient alone 0 of 2; the state not factored by residue). The follow-up is
[division-workplan.md](division-workplan.md); this plan's stage 3 is done there (the certificate's
adapter was built in stage 2).**

## Why

Modular division at `p` 5 hasn't been learned by a factored state (the factored-state workplan,
stages 9 and 9b): C9, C9' and C9'' certified nothing. Two failures, each diagnosed:

- **The zero-test trap (C9', D14).** Alone, a residue tells the quotient only whether it is 0
  (`A`) or undefined (`B`): I(q; A) = I(q; A = 0), I(q; B) = I(q; B = 0). Trained together, the
  factors settle on the zero tests (each factor at most 0.41 of `A` or `B`, the nonzero quotients
  31-44%, a guess 25%); more of a residue pays only through the pair, which neither factor holds.
- **Collapse (C9'').** A staggered factor's cell collapsed to one output in its turn. The factor
  restart probe (`data/window-recurrent/probe-factor-restart/`, manifest
  `probe-factor-restart.sha256` 488174f1...1cedbb32, both machines) re-drew such a factor and
  kept it alive with the commitment loss off for 5 epochs (14-16 codes, against 2-8 with it on),
  but **the live factor then learned its zero test and stopped** (seed 6: 0.41 of `B = 0`, 0.27 of
  `B`; "undefined" 75% to 84% right, the nonzero quotients unchanged).

So the barrier is the signal: a residue's only direct signal is its zero test, sparse (about 1
window in 5) and binary. E8c's counters, whose answer *is* their residue (the modular sum, a dense
signal every window), certified 6 of 6 at `q` 5 within about 5 minutes.

**The idea.** Every modular task here reads the same columns (a token `3a + b`) and every answer
is a readout of one state, `(A, B) mod p`:

| operation | answer | its state alone | its signal |
| --- | --- | --- | --- |
| sum | `(A + B) mod p` | one counter `S' = (3S + a + b) mod p` | dense: the counter is the answer |
| difference | `(A - B) mod p` | one counter `D' = (3D + a - b) mod p` | dense |
| quotient | `A / B mod p`, or undefined | the pair | the zero tests alone |

For odd `p` the sum and difference determine the pair (`A = (S + D) / 2`, `B = (S - D) / 2`), so
one network answering all three at every window has one shared state that serves them all, the
25-state pair, and two of its answers give a factor the dense signal E8c's counters learned from.
Nothing is labelled but answers (the owner's "learn the method, not imitate"); choosing the family
is curriculum design, as the modular-sum ladder before division was, and it is named as such.

## Decisions

- **D1. How a window gives several answers: query tokens.**
  - (a) *Chosen.* **Each window is `[column, Q_sum, Q_diff]`**: the column token's output is the
    quotient (as now); each query token's output is its operation's answer; one output softmax
    (6 classes: residues 0-4 and undefined). Pros: the network unchanged (tokens and labels only);
    the readouts factored by query, not flattened; the poolings get more than one token to choose
    between (at `k` 1 today their queries never moved). Cons: 3 tokens a window (about 3 times an
    epoch's cost); the certificate and `diagnose` read windows of the machine's alphabet only, so
    the probe reads its own diagnosis until a certificate is needed.
  - (b) An output head per operation. Pros: no extra tokens. Cons: a multi-label loss, a network
    change in all three implementations and the crate.
  - (c) One class for the triple (150 classes). Cons: flat (the owner: factored, not flat); the
    label is then the state itself.
  - (d) One operation per example, named by a task token. Cons: a third of the signal per example;
    the state would carry the token.
- **D2. The operations: sum, difference, quotient.**
  - (a) *Chosen.* Pros: the least that spans the pair with dense signals; the quotient the target.
    Cons: two factors could both learn one counter (the difference's answer argues against it).
  - (b) Add the product `A B mod p`. Pros: another pair readout. Cons: one more sparse signal, a
    fourth token; a variation if (a) half works.
  - (c) The residues `A` and `B` as answers. Cons: the decomposition handed over; never.
  - (d) Sum only, with the quotient. Cons: one counter `S` doesn't give the pair.
- **D3. The cell: C9''s.** Two factors of 16 codes (E8c's counter), `s` 16 each, trained jointly,
  the poolings reading before the first `StateInput` (D10), the transformer window (W11), `k` 1
  column. Pros: the nearest evidence (C9' is the control's cell; E8c's counters); one change from
  C9' (the family). Cons: two factors only, no spare (C9'' added one; a variation).
- **D4. The probe's training (probe-sized, not the harness's).** The curriculum as the harness
  runs it (step `j` trains lengths `1..j`; a step ends at its check passing, at a plateau of 10
  epochs, or at 60 epochs), 2^15 examples an epoch, batch 64, rate 0.003, uniform digits; the cap
  at the exclusion bound (D5) and a **wall-clock budget of 1 h**. Pros: about 3 times the harness's
  steps per hour; the same learning rule. Cons: not the harness's settings (patience 30, 200
  epochs, 2^16 examples, its case mixture), so the control (D6) runs under the same settings.
- **D5. Feasibility.** The target: the pair machine, 25 states (`p^2`; division's machine is the
  same 25-state table, E7). The
  model: two factors of 16 codes, product 256 states, each factor at least `p` = 5 (its counter):
  can represent it, 3.2 times a counter's states each (the owner's floor). The exclusion bound
  `m + q - 1 = 256 + 25 - 1 = 280` columns: the cap at step 280, never reached in an hour (as C9'),
  so a probe claims nothing about memorising: a promising network is tested at 24 and 48 columns
  in the probe and certified before any claim.
- **D6. The control: the quotient alone, the same probe.** Windows `[column]`, the same cell,
  settings and seeds. Pros: the family is the one difference; it re-measures C9' under the
  probe's settings. Cons: a third of the family's cost per epoch, so it reaches further steps in
  the hour (reported).

## The probe (stage 2), registered before its data

**Question:** does training on the family give the factored state its residues where the quotient
alone doesn't?

**Arms:** `family` and `quotient` (D6), seeds 0 and 1 each: 4 runs, one process each, on `jebel`
(about 1 h). Each step's end and every 5 epochs: each operation's accuracy at the step's lengths
(the quotient by class: 0, undefined, nonzero), each factor's codes used and its share
(`window_diagnostics.share`) of `A`, `B`, `S`, `D`, `A = 0`, `B = 0`; at a step's end the
quotient's accuracy at 24 and 48 columns.

**Works:** a `family` run with some factor at least 0.9 of `A`, `B`, `S` or `D` and the quotient's
nonzero classes at least 0.55 at its training lengths, within the hour. **Fails:** no `family` run
does; the factors end on the zero tests (each factor's share of a zero test above its share of
the residue) or collapsed. The control is expected to fail as C9' did; if it works under these
settings, the comparison says so.

**Next, by outcome** (decided now, probe by probe after):
- works: the same configuration's seeds 2-5 (a short follow-up), the quotient's readout
  certified (stage 3); then the long tests;
- partial (a factor holds a counter, the quotient below 0.55): variations, one at a time:
  three factors, the product added (D2 (b)), a longer budget;
- fails: option 1 ([commitment-control-workplan.md](commitment-control-workplan.md)).

## Stages

1. **This plan.**
2. **Done (this PR). The probe tool and its run:** `scripts/modular_family_probe.py` and
   `indrajala_ml/studies/modular_family.py` (the family's examples with the harness's final zero
   column, the query-token encoding, the probe's curriculum, its diagnosis, `--reads`, each step's
   quotient certified and the run stopped at the first certified, and `QueryMachine`, the
   certificate's adapter: a column run with its queries, one slot read), tests against
   `modular_data`'s machines; findings in the script's docstring.
   *Result:* **as registered (D10's poolings), no `family` run works, nor the control**: the
   nonzero quotients 0.33-0.37, every factor's share at most 0.18. **A check the probe prompted
   (the harness, E8c's modular sum):** one factor reading before the state plateaus from step 3, 0
   of 2 certified; reading the window's output, 2 of 2, as E8c. So D10 blocks even one counter.
   **Reading the output (an unregistered variation, the same probe otherwise): the family certified
   2 of 2 (seeds 0 and 1, the quotient and the sum, both verdicts, division's 25 states), the
   quotient alone 0 of 2** (nonzero quotients at most 0.64). The registered success's factor clause
   is not met: the certified states are spread over both codebooks, no factor a residue or any
   `xA + yB`. Every run that got far collapsed when trained on past its solution; the tool now stops
   at the first certified step. The outcome the plan didn't name (solved, not factored) leads to
   [division-workplan.md](division-workplan.md).
3. **The certificate for the encoding:** built in stage 2 (`QueryMachine`); its use (long tests,
   more seeds) moves to [division-workplan.md](division-workplan.md).
4. **Retire this plan;** leftovers to next-steps.md.

Tier 0 throughout unless stage 3 touches the certificate's code path (a test-covered pure-Python
change; no timed path).
