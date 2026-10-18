# Workplan: controlling the codes' commitment (a factor that collapses)

**Status: written 2026-10-15 at the owner's request; DEFERRED behind option 2
(`git show e45944e:docs/modular-family-workplan.md`): it starts only if option 2's probe
fails, or if a later probe shows factors collapsing. Decisions D1-D5 settled (the owner delegates
a workplan's decisions; each with its options' pros and cons). Probe-driven, as option 2.**

**Since (2026-10-15):** option 2's probe certified division (2 of 2, the poolings reading the
output), so this plan stays deferred. Collapse was seen again, but **after** a solution: every run
that solved or nearly solved collapsed when trained on (codes to 2-8, the answer to chance), none
while learning. If a later probe needs a run to keep training past a solution, the thermostat
(D1 (a)) is the first thing to try; otherwise stopping at the first certified step suffices.

**Later (2026-10-15 evening): traced, the division runs' collapse is abrupt** (within 5 epochs,
mostly mid-step, no slow fall of the codes in use before it; `scripts/modular_family_probe.py`'s
findings, "the collapse, traced"). That fits an optimiser instability better than the commitment's
pull, so this plan stays deferred: the fix for the division runs is planned apart (the owner: not
the commitment, unless it is shown to be the cause), its candidates in
[next-steps.md](next-steps.md), "From the division work".

## Why

Each code layer's loss, per window, is `weight * (||sg[h] - e||^2 + beta ||h - sg[e]||^2)`
(`CodesArrayLayer`, W4, after van den Oord et al. 2017; `beta` the commitment, 0.25): the codebook
term pulls a code to its outputs, the commitment term pulls an output `h` to its code.

- **C9''s collapse.** A staggered factor's cell came out of its turn giving one output for every
  input (seed 6's factor 1: a feature's spread across inputs 0.018 at initialisation, under 0.001
  after; its update gate's bias 1.0 to 2.5), on one code. A state of one code passes nothing on;
  the commitment pull is the one consistent force left, and the cheapest way to meet it is a
  constant.
- **The restart probe** (`data/window-recurrent/probe-factor-restart/`, manifest
  `probe-factor-restart.sha256` 488174f1...1cedbb32, both machines; its script beside the data):
  a re-drawn factor with the commitment on fell from 11-12 codes to 2-8 within 5 epochs; with it
  off for those 5 epochs, it kept 14-16. Two networks, so suggestive, not shown.
- **What the commitment buys.** With the state snapped in the forward pass whatever `beta` is, the
  commitment shapes learning, not the machine: it keeps `h` near its code, so the straight-through
  gradient (computed at the code, applied to `h`) is accurate and **the snap margins are wide:
  the certificate's snap ratio** (E8c's certified counters 1.02-7.91; a margin under 1.05 is a
  near tie, branched). Off for good: outputs spread to their regions' edges, near ties multiply,
  codes flip between batches.

So the commitment should be off while a factor is starting or collapsing, and on whenever it is
working and before it is certified.

## Decisions

- **D1. What switches it: (a) a usage thermostat, and (d) a soft-to-hard quantiser as its
  alternative, both built and probed** (the owner, 2026-10-15: "both a and d").
  - (a) *Chosen.* **The usage thermostat.** Per factor, the entropy of its codes' use over an
    epoch's training windows (the network's own statistic, never the target): below a floor,
    `beta` 0; above a higher restore level, `beta` ramped back to 0.25 (D2). Pros: reads only the
    network; catches a collapse at any time, a restart included; per factor; harness-level (a
    wrapper as `FactorRates`), the crate already takes a commitment per factor. Cons: two levels
    to set; without the gap between them it oscillates.
  - (d) *Chosen as the alternative.* **A soft-to-hard quantiser:** in training, a factor's next
    state the softmax-weighted mix of its codes (weights from `-||h - e||^2 / tau`), `tau`
    annealed to a hard snap (D3). Pros: every code gets a gradient, so no dead codes and no pull
    to one; the principled fix (soft-to-hard vector quantisation, Agustsson et al. 2017). Cons: a
    change to the codes' forward and backward in all three implementations and the crate (the
    crate first, its own PR); the state is a blend, not a machine, until the anneal ends, so a
    step is certified only after it.
  - (b) A phase schedule (off for a turn's first epochs). Pros: the probe's setting. Cons: blind to
    a collapse later; a special case of (a).
  - (c) A per-window gate (the commitment only where `h` is already near its code). Cons: a crate
    change for a second margin; a variation of (a) if (a) half works.
  - (e) A learned `beta`. Cons: its gradient drives it to 0; never.
- **D2. The thermostat's levels.** Off below 1.58 bits (fewer than about 3 codes in effective
  use, `log2 3`), restored above 2.32 bits (`log2 5`, a counter's states at `p` 5), ramped from 0
  to 0.25 over 5 epochs. Pros: the probe's survivors used 14-16 codes, collapsed factors 1-2: the
  levels sit between; the ramp avoids a jump in the loss. Cons: chosen from one probe.
- **D3. The quantiser's anneal.** `tau` from 1.0, halved every 5 epochs, a hard snap once a
  window's largest weight is above 0.99 on 99% of an epoch's windows; then the commitment on.
  Pros: ends hard by a measured test, not a fixed epoch. Cons: a schedule to set; a probe first.
- **D4. Certification only with the commitment on and the state hard:** a step is certified only
  after 5 epochs with `beta` at 0.25 (and, for (d), hard). Pros: the margins the certificate
  needs. Cons: a certified network may come a few epochs later.
- **D5. Probes, not runs.** Each mechanism probed on (i) the collapsed C9'' networks (seeds 0 and
  6, the restart probe's arms as controls) and (ii) a fresh C9' cell, 2 seeds, about 1 h on
  `jebel`; success registered with each probe (a factor that keeps at least 3 codes through 40
  epochs and gains share of a residue from its post-restart baseline).

## Stages (deferred)

1. **This plan.**
2. **The thermostat** in the harness, with tests (the levels, the hysteresis, the ramp, a frozen
   `beta` bit for bit the network as now); the restart probe turned into a repo tool beside it;
   its probes.
3. **The quantiser:** the crate's codes forward and backward with `tau` (`indrajala-math-rust`
   first), then the package's three implementations, parity tests; its probes.
4. **Retire this plan;** leftovers to next-steps.md.

Stage 2 is tier 0 (harness, pure Python). Stage 3 changes a timed path (the crate's recurrence):
tier 1, with the golden run.
