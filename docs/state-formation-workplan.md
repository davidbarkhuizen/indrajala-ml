# Workplan: state formation from the literature (one option at a time)

**Status: written 2026-10-20 at the owner's direction: "abandon all our current planned work,
completing only processing of existing data ... work systematically forward based on the insights
from the literature review ... work systematically through the new options one by one, first
understanding, testing then either accepting or rejecting each in turn", builds permitted; and
"using an annealing temperature profile on the initial snapping - if possible". D1-D12 settled under
the owner's delegation, each with its options' pros and cons. Probe-driven. Supersedes the
commitment-control and one-code-one-state plans' remaining claims (closed at the same direction).**

The goal is [machine-discovery.md](machine-discovery.md)'s stage A: one general architecture and
training method that forms a state, keeps it and certifies on every backdrop
([bottlenecks.md](bottlenecks.md)). The options are the literature review's
([window-recurrent-literature.md](window-recurrent-literature.md), extension 2026-10-20).

## Why: what the night of 2026-10-19 showed, and what the literature says it is

- **The commitment hurts before a state forms and helps after.** With it never on, 4 of round 1's 6
  failure-A runs certified (against at most 1 of 6 for every earlier remedy); with it never on, S4
  and division seeds that certify with it (C0, W1) didn't (S4 seeds 0, 1 at steps 22 and 30; division
  at 73-108 pairs per code), and R3/R4 lost move-to-front-5 seed 1. Every state formed with the
  commitment off and then met it at full weight eroded within 1-2 steps (the diagnostic, both
  thermostat probes). **Zhao et al. 2026:** quantising before a direction is learned freezes it, through
  the commitment term and the straight-through gradient; fix, an unquantised warm-up.
- **The snap alone still collapses some seeds:** S3 seeds 1 and 4 fell to 1-4 codes with the
  commitment never on. Our snap is on from step 1, so Zhao et al.'s mechanism predicts this too.
- **Annealing as built merged the codes:** at `tau` 1.0, 23-27 of a factor's 32 codes were identical
  by step 2-3. **Rose 1998, Theorem 1:** soft assignments at temperature `T` hold every code at the
  data's mean above `T_c = 2 lambda_max` (the outputs' covariance's largest eigenvalue); identical
  codes split only under a perturbation. The codes start uniform in `±1/sqrt(size)` (squared distances
  about 0.67 at size 16) and had merged by step 1's end, so `tau` 1.0 was above `T_c` from the start.
- **Certified networks have thin margins** (near ties in every commitment-off certificate and in some
  of C0's and C2's); **codes standing for several states** (bottleneck 2) moved only with R3/R4.

## The options, in order (D1)

1. **The annealed initial snap** (the review's options 1 and 2 joined, at the owner's suggestion): the
   state continuous for a warm-up, then the codes initialised from the states seen and cooled from a
   `tau_0` set by the outputs' scale to hard, the commitment off until hard and ramped back after.
2. **Noise in the cell's pre-activation** (Oliva & Lago-Fernández 2020): saturated, attracting states
   and wider margins.
3. **The rotation trick** (Fifty et al. 2025): a gradient that can separate windows on one code
   (bottleneck 2).
4. **FSQ-style fixed-grid factors** (Mentzer et al. 2024): no codebook to learn, no commitment.

Each option in turn: **understand** (a measurement on existing networks that tests the literature's
mechanism here, before any build), **build** (tests, parity, the golden run), **test** (the panel,
D11), **accept or reject** (D12), recorded in this plan's findings before the next option starts. An
accepted option becomes part of the base the next one is tested on.

## Decisions

- **D1. One option at a time, in the order above.**
  - (a) *Chosen.* Pros: the owner's direction; each result read against one change; the order is the
    review's, by fit to the bottleneck seen most (formation) and by cost (harness, then a layer, then
    the crate's gradient, then a new component). Cons: slower than running several at once; an
    interaction (e.g. noise helping only with the warm-up) is seen only if the earlier option is
    accepted.
  - (b) Several options in parallel probes. Cons: what made the night's results hard to compare.
- **D2. Option 1 joins the review's options 1 and 2** (the owner: "an annealing temperature profile on
  the initial snapping").
  - (a) *Chosen.* Pros: the warm-up gives the codes something to be initialised from, and the anneal
    turns the continuous state into codes without the abrupt snap that freezes unlearned directions;
    one build, one probe. Cons: if it fails, which half failed is read from the logs, not separate
    arms (the understanding step and the per-step readings below are there for that).
  - (b) Option 1 alone (the built annealing with `tau_0` fixed). Cons: the codes would still start
    random, with nothing learned to initialise them from; S3 seeds 1 and 4 collapse under the snap
    whatever the commitment.
  - (c) The warm-up with an abrupt snap at its end. Cons: the snap is the step that freezes directions
    (Zhao et al.); annealing it is the point.
- **D3. Understanding first: two measurements on existing networks, before the build.**
  - Rose's prediction: on annealing arm 2's saved networks (`commitment-control/anneal-probe/stone/`),
    compute each factor's `h` covariance on a fixed sample and `T_c = 2 lambda_max`; **expected: `tau`
    above `T_c` wherever codes merged, below it where they split** (seed 0 hardened at `tau` 1/64).
  - Zhao et al.'s: the effective dimension `d_eff` of each factor's `h` (principal components for 99% of
    the variance) at each step, on the never probe's runs that certified against S3 seeds 1 and 4, and
    on round 1's failures; **expected: lower `d_eff` in the collapsing runs from early steps.**
  - Pros: tests the mechanism here at a few minutes' cost; both become durable measurements
    (`machine_reading`-style subcommands of the probe script). Cons: offline only; a mismatch means the
    literature's picture doesn't hold for a recurrent state, and option 1 is redesigned before building.
- **D4. The warm-up's state: continuous, the codes bypassed** (the cell's output passed on as the
  next state; the crate's recurrence step gains a per-factor continuous mode).
  - (a) *Chosen.* Pros: Zhao et al.'s warm-up exactly; the network's continuous form already exists
    (factors without codes) and is tested (`test_a_continuous_state_is_the_snapped_networks_with_identity_codes_by_bits`). Cons: a crate change (tier 1); a continuous state can
    drift (Zeng et al. 1993) and solve short words without discrete states, so the warm-up is kept short.
  - (b) Soft codes at a high `tau`. Cons: that is the merge (Rose): every code at the mean, so the
    state carries nothing.
  - (c) Noise as the warm-up. Cons: option 2's question; one change at a time.
- **D5. The warm-up's end: the first step whose screen reaches 0.95, capped at 8 steps.**
  - (a) *Chosen.* Pros: answers-only (the fork's and D2'''s signal); the diagnostic's states formed
    where the answers came right (steps 4-5); general across tasks. Cons: a continuous state may reach
    0.95 on short words without discrete structure, ending the warm-up early; the cap bounds the drift.
  - (b) Zhao et al.'s adaptive criterion (`d_eff` stops growing). Pros: their theory's. Cons: unmeasured
    for a recurrence until D3; logged at every step, adopted later if D3 supports it.
  - (c) A fixed number of steps. Cons: blind to the run.
- **D6. The codes at the warm-up's end: k-means of each factor's `h` on the fixed sample** (k-means++
  seeding from the run's own generator, `m_g` centres, a fixed number of Lloyd iterations).
  - (a) *Chosen.* Pros: Agustsson et al.'s initialisation; the codes start where the states are, spread
    as they are. Cons: fewer distinct `h` than codes leaves duplicates (perturbed apart by a small
    amount, as deterministic annealing does, and logged).
  - (b) Random draws of `h` (Łańcucki et al.'s restarts). Cons: duplicates likelier; no cluster structure.
  - (c) Keep the random codes. Cons: they merge (the probe; Rose).
- **D7. `tau_0` from the clusters: twice the largest eigenvalue of the pooled within-cluster
  covariance.**
  - (a) *Chosen.* Pros: Rose's `T_c` for a cluster: soft within each cluster (every code near a window
    gets a gradient), below the critical temperature of any union of clusters (whose variance includes
    the between-centre spread), so codes don't merge. Cons: an estimate on a sample; logged with the
    global `T_c` so a merge can be read.
  - (b) A fraction of the global `T_c`. Cons: a fraction to choose; the clusters' scale is the relevant one.
  - (c) Absolute (D9 of the commitment-control plan). Cons: what failed.
- **D8. The schedule and the hardness test as built** (`tau` halved every 10 epochs; hard for good once
  99% of the sample's windows weight one code above 0.99), from `tau_0`. Pros: built and tested;
  one change. Cons: halving every 10 may be slow from a lower `tau_0`; the logs show epochs to hard.
- **D9. The commitment: 0 during the warm-up and while soft, ramped back over 20 epochs once hard**
  (the built annealing's D7).
  - (a) *Chosen.* Pros: the night's harm data says the commitment is needed once a state exists; Zhao
    et al.'s warm-up keeps it for after; the ramp is against the erosion seen at a full return. Cons:
    the return eroded states in the thermostat probes (there, before the codes meant anything).
  - (b) Never on. Cons: the harm check (S4, division).
- **D10. Readings during the warm-up: screens and the fork on the continuous network; no certificate,
  no machine readings, R3 off until the first hard step.** Pros: a continuous network has no finite
  machine to certify or read; the fork still guards training. Cons: steps passed during the warm-up
  pass on a continuous network's screen (the hard machine is read from the first hard step on).
- **D11. The panel: 12 runs at width 32** (the base), round 1's and 2's cell (two factors,
  `--codes-ratio 32/11`, reading the output, the fork, 600 epochs; R3/R4 off unless accepted):
  - failure A: word-problem-s3 seeds 0, 1, 4 (0 certified with the commitment off; 1 and 4 never
    formed); move-to-front-4 seed 3; move-to-front-5 seeds 0, 1;
  - harm: word-problem-s4 seeds 0, 1 (C0 certified both), move-to-front-4 seed 0 (C0), division `p` 13
    seeds 0, 1 (W1, `--factor 32 38` twice);
  - bottleneck 2: word-problem-s5 seed 1.
  Controls by the by-bits rule where they exist (C0, W1); S3 at 32 has none, so S3 seeds 0, 1, 4 run
  without the option beside the panel once (the base control). Feasibility: the cells' registered
  bounds (S3 q 6, 8 codes, m 64, cap 69; mtf-4 24, 15, 225, 248; S4 the same; S5 and mtf-5 120, 32,
  1,024, 1,143; division 169, 38, 1,444, 1,612), each width 32 certified before.
  - Pros: every bottleneck and both kinds of harm in one batch, about 3 h on `stone` at night. Cons:
    one or two seeds a cell: existence, not rates.
- **D12. Accept or reject.** An option is **accepted** when, on the panel, it certifies more failure-A
  runs than their controls, loses none of the harm runs' certificates, and its certificates pass both
  verdicts; **rejected** when it loses a harm certificate or moves no failure-A run; in between, one
  follow-up probe chosen from its readings, then a decision. An accepted option joins the base. Adoption
  as the general method stays the owner's, on 6 seeds per backdrop.

## Findings: stage 2, option 1's understanding (2026-10-20)

`state-spread` (`scripts/modular_family_probe.py`, `mf.state_spread`): each factor's `h` sampled at a
step's screen lengths (100 cases a length), on saved step networks. Data in
data/window-recurrent/state-formation/spread/ (`anneal-arm2.jsonl`, `s3-never-round1.jsonl`), manifest
`spread.sha256` (run from state-formation/).

- **Rose, refined: the codes merge within each cluster, not globally.** On annealing arm 2 (move-to-front-5
  at 32), the global critical temperature `2 lambda_max(h)` was 2.3-6.5, **above** `tau` 1.0, so D3's
  expectation as written (merging only where `tau > T_c`) is refuted for the global `T_c`. But `h` sat in
  only 4-9 clusters (`effective_dimension` 1-5 of 16), and each cluster's critical temperature (twice the
  within-cluster top eigenvalue, 0.0004-0.36) was far **below** `tau`: the surviving groups (6-9 at steps
  2-3) match the clusters in use (5-9), as Rose's theorem gives below the global `T_c`. So the merge is
  deterministic annealing between the clusters' and the whole's critical temperatures, and **D7's `tau_0`
  (twice the pooled within-cluster eigenvalue) is the right scale.**
- **Zhao et al. confirmed for the commitment: it collapses `h`'s dimension.** On round 1's S3 failures
  (the commitment on), `h`'s effective dimension stayed 1-5 of 16 through steps 1-6, on 1-5 codes in use;
  with the commitment never on, it rose to 10-14 from step 2 on 6-8 codes, in the runs that certified
  (seeds 0, 3, 5).
- **But S3 seeds 1 and 4 aren't dimensional collapse:** with the commitment never on, their `h` reached
  11-15 dimensions and 5-8 codes in use (seed 1 dipped to 3 at step 4) and still formed no state. Their
  failure is the codes not aligning with the states, a different mechanism; **the warm-up isn't
  expected to rescue them** (still on the panel, to read).
- **So option 1's case rests on keeping the commitment's benefit after formation** (the harm data)
  without its collapse before it: the warm-up (continuous, no commitment) gives the dimensions, the
  anneal from D7's `tau_0` turns them into codes without merging, the commitment returns once hard. Its
  panel expectation: failure-A runs at least as the never probe's (S3 0, mtf-4 3 certify), and **no harm**
  (S4 0, 1; division 0, 1; mtf-4 0 certify, as with the commitment on).

## Stages

1. **This plan** (docs). Before it, the night's findings PR (the existing data, at the owner's direction).
2. **Option 1, understanding:** D3's two measurements, as subcommands of `scripts/modular_family_probe.py`
   (`critical-temperature`, `effective-dimension`), with tests; findings here.
3. **Option 1, build:** the crate's per-factor continuous mode (`indrajala-math-rust` first, then a
   "Bump rust/" PR); the harness's `--warm-up` (D4, D5, D10), k-means codes (D6), `tau_0` (D7), on the
   built `--anneal`; parity tests, the golden run unchanged without the flags. Tier 1: the A/B once
   `jebel` is back (about 2026-10-22); the probes run from the branch meanwhile.
4. **Option 1, test and decision:** the panel (D11), its expectation written in its script first;
   accept or reject (D12).
5. **Option 2 (noise):** understanding (the certified networks' snap margins against the failures'),
   build (`--cell-noise`, numpy and crate), the panel, decision.
6. **Option 3 (the rotation trick):** understanding (on bottleneck-2 failures, whether windows of
   different target states share codes, from `breaks`), build (the crate's straight-through gradient),
   the panel, decision.
7. **Option 4 (FSQ factors):** understanding, design (a fixed grid a factor, its levels from the
   codes ratio), build, the panel, decision.
8. **Findings** in [bottlenecks.md](bottlenecks.md) and the probe script; this plan retired, leftovers
   to [next-steps.md](next-steps.md).

## Machines

`stone`: about half its 12 cores by day, all at night (the panel's 12 runs at night). `pyramidon`:
development, up to about 4 small workers. `jebel`: down until about 2026-10-22, then timing.
