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
- **D7' (2026-10-20, replacing D7 at the build, before any run): `tau_0` a quarter of the median
  squared distance from a code to its nearest (after k-means).** The build's test showed D7's `tau_0`
  (the clusters' own critical temperature) leaves every window hard at once: with codes at the cluster
  centres, the gap between neighbouring codes dwarfs it, so the "anneal" is an abrupt snap. Rose's
  theory leaves a soft phase only between the clusters' critical temperature and the temperature at
  which neighbouring codes merge (about `Delta^2 / 2` for codes `Delta` apart): D7' starts at half the
  latter for a typical neighbour.
  - (a) *Chosen.* Pros: genuinely soft (a typical neighbour weighted `e^-4` relative), no merge but of
    codes much closer than typical (k-means' splits the anneal finds not real); checked on saved
    networks before any run: S3 (the never probe, step 4) `tau_0` 1.4-2.4 against the clusters' 0.9-1.9
    and the whole's 5-8, 6-33% of windows hard at `tau_0`, 92-98% at `tau_0 / 8`; move-to-front-5 (arm 3,
    step 8) 1.2-1.7 against 0.45-0.61 and 9-14, 8-18% hard, 82-95% at `tau_0 / 8` (about 30 epochs).
    Cons: a median over codes; logged with both critical temperatures.
  - (b) D7 as written. Cons: an abrupt snap with good codes, not the owner's annealed snap.
  - (c) The closest pair's merge temperature. Cons: k-means' near-duplicates (perturbed apart) would
    make it tiny, hard at once.
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

### Option 1 rebuilt: heat, hold, cool, crystallise (2026-10-20, after the void panel)

The owner, after the first panel's flaw (findings below): "we definitely want retry annealing until we
get it right and get a proper verdict either way"; and the profile, from crystallisation: "heat to hot
quite quickly, hold it there, then let it cool long and slow ... smoothly ramp up to a high entropy
state, leave it there long enough for the activation energy of the various states to be overcome, then
slowly cool to accumulate structure ... and allow the state to crystallize over time". D13-D20 replace
D5, D7' and D9 for option 1; D4, D6, D8's hardness test, D10-D12 stand.

- **D13. The first panel is void, not a verdict:** its anneal never ran (every factor hard within one
  epoch of the warm-up's end), so D12 isn't applied to it; option 1 is retried until its profile runs as
  designed, then decided by D12.
  - (a) *Chosen.* Pros: the panel tested an abrupt snap with the commitment's ramp, not option 1; the
    owner's direction. Cons: a second panel's cost.
  - (b) Apply D12 (a lost harm certificate: rejected). Cons: rejects an option that didn't run.
- **D14 (replaces D5). The warm-up ends at the first step whose best screen reaches 0.95 and where
  every snapped factor's states seen (distinct cell outputs on the step's screen, to 6 decimals) number
  at least 4 times its codes; capped at 8 steps.**
  - (a) *Chosen.* Pros: the void panel ended every warm-up at step 1, on 5-10 distinct states for 8-38
    codes, so k-means left duplicates and `tau_0` fell to 1e-6; 4 a code gives k-means real clusters
    (step 2 on every backdrop: S3's 43 words of length up to 2 against 32, S4's 601 against 60, division's
    against 152). Cons: a count, not structure; the cap still ends a warm-up that never gets there.
  - (b) Zhao et al.'s `d_eff` criterion. Cons: still unmeasured as a stopping rule here.
  - (c) A fixed later step. Cons: blind to the backdrop.
- **D15 (the owner's profile; replaces D7' and the halving's start). Each snapped factor, from the
  warm-up's end: heated from `T_hot / 64`, doubling each epoch (6 epochs), held at `T_hot` (`hold` 10
  epochs), then cooled exponentially (halved every `halving` 15 epochs) with no floor.** The hardness
  test (D8's) is read only once cooling has begun.
  - (a) *Chosen.* Pros: Rose's deterministic annealing as it is meant to run, from above the transitions
    down through each; the heat is quick and smooth (doubling), the hold lets the codes reach the melt's
    mean, the cool is slower than the built anneal's (15 against 10 epochs a halving) and exponential, so
    its tail is long. `T_hot` is set from the whole's critical temperature (D16), measured reliably at the
    warm-up's end (0.13-0.57 on the void panel), not from the codes' gaps. Cons: about 6 + 10 + 150
    epochs to hard (a factor of about 1,000 cooled); the melt erases the codes' meaning while it holds.
  - (b) Cool only, from the clusters' temperature (D7, D7'). Cons: what ran: no soft phase.
  - (c) A linear cool. Cons: fast through the low temperatures where the fine splits happen.
- **D16. `T_hot`: a multiple of the whole's critical temperature `2 lambda_max(h)`, settled by the
  mechanics probe (stage 4a) between 2 (a full melt: every code at the mean, Rose's Theorem 1) and 0.5
  (a partial melt: the codes merge into the top-level groups only).**
  - (a) *Chosen.* Pros: the full melt is the owner's "high entropy state" and Rose's start; the partial
    melt keeps the coarse structure the warm-up learned, if the full melt's erasure costs more than it
    frees. One probe reads which splits back to the target's states. Cons: a probe's cost (6 short runs).
  - (b) Full melt without a probe. Cons: annealing arm 2 merged and never split back (D17's reason).
- **D17. Thermal noise (the owner: "the noise/perturbation to also be a function of the temperature -
  so that the crystallization analogy is more complete"): each soft epoch, every code of a soft factor
  moved by Gaussian noise of variance `kappa T` in all (per coordinate `kappa T / size`), from the run's
  seed, step, epoch and forks; `kappa` settled by the mechanics probe between `10^-2` (agitation: a
  displacement about a tenth of `sqrt(T)` an epoch, which can carry a code across a shallow barrier) and
  `10^-4` (only Rose's symmetry breaking).**
  - (a) *Chosen.* Pros: variance proportional to `T` is thermal noise's own scaling (Langevin dynamics,
    equipartition), so it is strong in the melt, falls with the cool and vanishes as the codes freeze;
    Rose 1998: identical codes split at a transition only under a perturbation, and annealing arm 2's
    stayed identical (23-27 of 32) and never split back. Cons: noise in the codes changes training's
    bits (the flag's runs only); too much keeps the codes from settling (the probe reads it).
  - (b) Noise in the cell's pre-activation (option 2). Cons: option 2's own question; one option at a time.
  - (c) Perturb only codes found merged. Cons: a merge threshold to choose; Rose's own variant, kept in
    reserve if (a)'s noise is too much.
  - Thermal noise on the states themselves (the cell's pre-activation, the particles rather than the
    lattice) at the same `T` is option 2's build, the natural combination once option 1 is decided.
- **D18. No fork with the warm-up (the owner: "drop the forking - it is not compatible with
  crystallization"); during the heat and the hold the step's best follows the screen, so no plateau
  ends a step on a melted network.**
  - (a) *Chosen.* Pros: the fork restores an earlier network whenever the screen falls, undoing the melt
    and every disorder a crystal passes through on its way; the harness refuses `--warm-up` with
    `--fork`. Cons: no guard against a collapse; the panel's option runs differ from their controls (C0,
    W1, the S3 controls, all with the fork) by the fork too, so D12 compares the method, not the anneal
    alone.
  - (b) The fork held off during the heat and the hold only. Cons: it still undoes the cool's disorder.
- **D19 (the owner's "crystallise over time"; replaces D9). After a factor first passes the hardness
  test it stays soft and keeps cooling for `tail` 40 epochs more; then hard, its commitment ramped back
  over `ramp` 60 epochs.** The commitment is 0 from the warm-up's start to the tail's end.
  - (a) *Chosen.* Pros: the hard machine is read and certified from the snapped network at every step's
    end throughout (soft codes are training's only), so a crystal certified while soft stops the run as
    any; the tail lets structure settle before the snap; both void panels' runs broke at a commitment of
    about 0.225 of 0.25 on the 20-epoch ramp, so 60 is gentler. Cons: about 100 epochs more before the
    commitment's benefit (the harm runs').
  - (b) Hard at the test, the 20-epoch ramp (D9). Cons: what eroded the void panel's states.
  - (c) Never hard, never committed. Cons: the harm data says the commitment is needed once a state exists.
- **D21 (the owner: "grossly characterise the material (measure its melting point)"). The states'
  melting curve, measured at the warm-up's end on the run's own states and logged:** Rose's
  deterministic annealing of the factor's codes on its sampled outputs held fixed, from twice the
  whole's critical temperature down 20 octaves (4 temperatures an octave, 30 updates each), the codes'
  groups at each temperature and the transitions where they change; the first is the melting point.
  Also a subcommand for saved networks (`melting-curve`).
  - (a) *Chosen.* Pros: the material is the run's own, measured where the profile starts (a separate
    earlier run's states would be another material: they move with training and with the step's
    lengths); seconds a run (15 s for two factors of 15 codes); it shows where the profile must hold and
    how far it must cool, and the mechanics probe reads training's splits (D20) against it. Cons: on
    fixed states, while training moves them; a split is seen up to an octave late (critical slowing
    down: the codes part slowly just below a transition).
  - (b) A separate characterisation run per task first. Cons: another material; a run's cost.
  - (c) The critical temperature alone (built). Cons: the melt's top only, not where the fine
    structure forms.
  - **First reading** (word-problem-s4 seed 1's step-4 network from the void panel, its best: codes
    12,14, share 0.57): `T_c` 4.1 and 2.5, the codes leaving the melt at 2.9 and 1.8, 4-6 groups by 0.6,
    and the last splits (to 12-14) at `10^-4` to `10^-5`: five decades, so the cool must run about 17
    octaves (about 250 epochs at D15's halving). The built anneal's start (1.0) sat inside the coarse
    transitions and D7''s `10^-6` below every one.
- **D22 (after 4a; the owner: "the annealing is working correctly, but it is not compatible with our
  current curriculum ... all features of the target machine have to be visible at all time ... The
  entropy of the information the system sees needs to be a maximum at the maximum temperature ... i
  don't think we could build an ideal curriculum without already knowing too much about the target
  machine"; "i don't [know] if we should have a curriculum at all"). No curriculum with option 1
  (`--whole`): every epoch's examples drawn over every length 1 to the cap (the exclusion bound) in
  turn, from the first epoch, warm-up included; every step's screen over a fixed spread of lengths
  (1-4 and 10 geometric to the cap, `whole_lengths`); a step is a block of epochs ending in its
  certificate, the material unchanged.**
  - (a) *Chosen.* Pros: annealing assumes one material; the curriculum changed it through the cool
    (4a: the steps ran 8 to 14 while the codes were soft, the screen falling), so every transition
    measured at the warm-up's end was stale within steps; the whole distribution is the maximum
    entropy view of the task and assumes nothing of the machine beyond the bound the cell must hold
    anyway. Cons: credit assignment through long words from the start (the curriculum's reason; the
    old all-at-once arm nearly worked at `m` 64, failed elsewhere); a continuous warm-up may drift on
    long words; epochs cost the mean length, half the cap (S3 35, S4 124, against about 20 late in 4a).
  - (b) The curriculum held during the anneal (the proposal before the owner's). Cons: still a
    material that changes, between anneals; a length order is generic, but not the anneal's material.
  - (c) If (a)'s risks hit: uniform over a shorter range, never an ordered curriculum.
- **D23. The commitment's return as an arm (`--anneal-never-commit`):** 4a's losses came in the late
  cool and the ramp alike, so the follow-up probe runs the commitment's return (D19) against its never
  returning, both with D22. Pros: separates the two suspects left. Cons: two arms a task.
- **D24 (the owner: "we may need more time, and a very slow cool - and to possibly experiment with
  different shapes of cooling curve ... i don't think we should see time as the issue, only whether
  our network can actually learn generically"; "test a concave curve and a convex curve ... those
  represent the two limiting cases"; "and also maybe a s-curve"). The cool's shape on a
  log-temperature scale (`--melt-shape`), each falling `octaves` 20 in `cool` 600 epochs from
  `T_hot` (about 17 octaves span the melting curves, 4a), then halved every `halving` 30 epochs:**
  concave (octaves fallen as `u^2`, `u` the share of the cool passed: it stays near the melt, then falls
  fast), convex (as `1 - (1 - u)^2`: it falls fast, then a long slow tail where the fine splits are), and
  s (as `3u^2 - 2u^3`: slow at the melt and in the tail, fast between). The exponential (a straight
  line) lies between the two limits. The hardness test is read only once a shaped cool has run, so each
  shape's whole curve trains soft (read from the cool's start, the convex limit would reach 4a's hard
  temperature, about 0.05, by epoch 115 of 600, the concave by 355: the probe would compare times to the
  snap, not shapes). The hold 20 epochs (4a's 10 melted to 4-5 groups, not 1); `kappa` 1e-2 (agitation,
  the owner's thermal picture, readable now D20's distance sits above the noise); the commitment never
  back (D23 (b): it certified the failure-A seeds before, and its return is the next question once a
  crystal forms, not this one).
  - (a) *Chosen.* Pros: the limits bracket the shapes, the s-curve lingers at both ends where the
    transitions sit; a result says which side to explore; time isn't the constraint (the owner;
    multi-day runs allowed where justified). Cons: three arms a task; a whole-distribution epoch costs
    18.6 s on S3, 61 s on S4 (on `stone`, two at once), so about 1,200 epochs is 6-9 h and 20-25 h.
  - (b) The exponential alone, slower. Cons: no reading of where the shape matters.
- **D25 (the owner: "is it possible to have a gradual transition from continuous to snapped ?";
  "would that be linked to temperature"). The gradual handover (`--handover`, the crate's
  `couplings`): each soft factor's next state `(1 - l) h + l q`, its snapped path's gradient `l dq`
  (its codebook and commitment terms whole), the cell's output `(1 - l) dq`; `l` 0 through the heat
  and the hold, then in the cool log-linear in T from 0 at the factor's melting point to 1 at its last
  transition (D21's curve, measured at the warm-up's end), 1 once hard.**
  - (a) *Chosen.* Pros: the separation readings (findings below) put the loss at the snap itself, not
    the melt: the warm-up's states carry the target's (joint nearest neighbour 0.98-1.00), its k-means
    codes snapped with no training drop that to 0.82 and 0.25 (lengths 4, 16); a gradual coupling makes
    the network shape each factor's states into clusters the codes can hold while it still has its
    information; at 0 the codes form on the liquid (their codebook loss) without touching the forward
    path, so the melt no longer erases the network's memory (4a's holds took the screen to 0.5); the
    material sets the window (S4's transitions 2.5 to 0.0015, S3's 2.5 to 0.2). Cons: a crate change
    (`couplings` per call, indrajala-math-rust #69); a second schedule, tied to the first.
  - (b) `l` from fixed fractions of `T_hot`. Cons: blind to where the transitions are.
  - (c) `l` from the codes' hardness. Cons: the forward path coupled to a noisy reading of itself.
- **D20 amended: the groups' distance is `T max(1/100, 10 kappa)` squared,** above the thermal noise
  (4a's `full` arm unreadable at `T / 100`).
- **D20. Each factor's profile logged each epoch** (its phase, temperature, hardness, commitment and its
  codes' groups closer than `T / 100` squared), and the warm-up's end row with each factor's distinct
  states, critical temperatures and `T_hot`. Pros: the melt, each split and the crystal are read from the
  log. Cons: a little more per epoch (the codes only).

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

## Findings: stage 4, option 1's first panel, void (2026-10-20)

The panel (`--anneal --warm-up` at #891's c875b1a, D11) ran from 06:07 on `stone` and `pyramidon` and was
stopped at 07:40 at the owner's word, once read: its anneal never ran (D13). Data in
data/window-recurrent/state-formation/option1/ (`stone/`, `pyramidon/`).

- **Every warm-up ended at step 1** (12 of 12 factors): the continuous network screened 1.0 on step 1's
  words, D5's early-end risk. The states seen were 5-10 distinct points (the clusters' own temperature
  about 1e-31), fewer than the codes (8-38), so k-means left duplicates, perturbed apart by 1e-3 of the
  spread.
- **So `tau_0` (D7': a quarter of the median nearest-code gap) was 1e-6 to 0.02,** 30 to 500,000 times
  below the whole's critical temperature (0.13-0.57): D7''s option (c)'s failure, reached through the
  duplicates. Every factor was hard by the next step's first or second epoch: an abrupt snap.
- **The commitment's ramp then eroded what formed:** S3 seeds 0, 1 broke at a commitment of about 0.225
  (codes 6,5 to 3,4; 6,7 to 2,3, a collapse); S4 seed 1 reached 12,14 codes and share 0.57 at step 4,
  then eroded to 4,8.
- **Readings, not a verdict:** S3 seeds 0, 1 uncertified at steps 21-26 (their controls collapsed to 1-2
  codes by step 2: the warm-up's codes held 3-4 and share about 0.19 against 0.04 and 0); S4 seeds 0, 1
  uncertified at 23-24 (C0 certified them at 11, 12); move-to-front-4 seed 0 at codes 3,2 from step 3;
  division seeds 0, 1 uncertified at step 9 (seed 1 at share 0.965, every answer 0.94-0.96, rising);
  word-problem-s5 seed 1 at share 0.81. S3 seed 4, move-to-front-4 seed 3 and move-to-front-5 seeds 0, 1
  never started.

## Findings: stage 4a, the mechanics probe (2026-10-20)

`probe-4a.sh` (its expectations in its header, before the data) at b8c3799: word-problem-s3 seed 0 on
`pyramidon`, word-problem-s4 seed 0 on `stone`, arms `full` (`T_hot` 2 `T_c`, `kappa` 1e-2), `quiet`
(2 `T_c`, 1e-4), `partial` (0.5 `T_c`, 1e-2); 600 epochs each, all spent. Data in
data/window-recurrent/state-formation/option1/probe-4a/ (`s3/`, `s4/`), manifest `option1.sha256` (run
from state-formation/, with the void panel's).

- **The warm-up's end (D14):** S4 at step 7 (70 distinct states a factor against 60 needed, 15 of 15
  codes in use, `d_eff` 9-11); S3 at its cap, step 8 (the continuous states never passed 24 distinct
  against 32: the continuous network passed steps 1-8 on 16-24 states, saturated). Expectation met on
  S4, not on S3.
- **The melting curves (D21):** melting points 2.5-2.6 on S4 (`T_c` 3.5-3.7), the full 15 groups by
  0.07-0.0015; on S3 2.5 and 2.05 (`T_c` 3.5, 2.9), 7 groups by 0.2-0.26.
- **The melt and the splits, as Rose's theory says (the `quiet` arm, readable):** at the hold the codes
  merged (S4 15 to 5 and 4 groups in 10 epochs; S3 8 to 2-3); in the cool they split back from about
  the curve's melting point (S4 from 3.2 to 14-15 groups by 1.6; S3 factor 0 from about 3.5, factor 1
  held at 2 groups until 2.3, its melting point 2.05). So the curve measured on fixed states predicts
  training's transitions within about a fifth of an octave, early side.
- **`full`'s groups are unreadable (a measurement flaw):** at `kappa` 1e-2 the noise moves each code
  by about `T / 100` squared an epoch, D20's grouping distance, so the count stayed near all codes
  whatever the melt; its screen did fall in the hold (S4 0.80 to 0.45). The threshold must sit above
  the noise.
- **`partial` never melted:** 0.5 `T_c` is below the melting point (about 0.7 `T_c` on both tasks),
  so D16's "partial melt" was a hold in the solid; its codes stayed apart.
- **No state formed, in any arm; nothing certified.** Best state shares 0.44-0.46 on S4 (steps 9-13,
  soft), 0.23-0.32 on S3 (steps 9-11); then every arm lost its codes: S4 `full` and `partial` to 1-2
  a factor, `quiet` to 5,3; S3 to 1-3. The losses came in the late cool and in the commitment's ramp
  (S3 `full` lost its codes at step 12, soft, the commitment 0; S3 `quiet` at steps 14-15, early in
  the ramp; S4 within 1-2 steps of the commitment's full weight). S3 seed 0 certified with the
  commitment never on (round 1, width 16); here it didn't in any arm.
- **The curriculum ran ahead of the crystal:** the plateau rule advanced the steps through the cool
  (S4 8 to 14, S3 9 to 14, in about 130 epochs), the screen falling (S4 0.80 to 0.40-0.50) while the
  codes were soft: new lengths arrived faster than structure settled.
- **D16, D17 by the header's rule:** at hard, S4 favours `full` (codes 11,10, share 0.45), S3 `partial`
  (6,5, 0.23); no consistent winner, and the rule's tie goes to `full`. But the mechanics are right
  and the outcome isn't: the bottleneck is after the splits, not the melt's settings.

## Findings: the state separation check (2026-10-20)

`state-separation` (`scripts/modular_family_probe.py`, `mf.state_separation`): whether a network's cell
outputs `h` determine the target machine's state, the target read only to measure: leave-one-out
nearest neighbour's and nearest target centroid's accuracy on the joint `h` and each factor's, against
chance (the commonest state's share); a warm-up's network read live (`--live`, its continuous factors
unsnapped). Readings in data/window-recurrent/state-formation/option1/separation/separation.txt.

- **The warm-up's continuous states carry the target's:** 4a's S4 seed 0 at its warm-up's end (step 7,
  curriculum to length 7) joint 0.98-1.00 to length 16 (chance 0.06-0.24), 0.85 at 32, 0.61 at 64; S3
  (step 8) 1.00 to 32, 0.94 at 64. Centroid accuracy nearly as high. A certified S3 network (snapped):
  1.00 at every length, and 1.00 in each factor alone; the warm-up's factors alone weaker (S4 centroid
  0.77 at length 16): the warm-up encodes the state jointly, not factor by factor.
- **4a lost it at once and never recovered:** S4 `quiet` joint 0.74 and 0.25 (lengths 4, 16) at step 8
  (`partial`, never melted, 0.78 and 0.34), 0.31-0.55 and 0.12-0.20 through steps 11-25.
- **The snap itself loses it:** the step-7 network, its codes set by k-means as the warm-up's end sets
  them, read snapped with no training: 0.82 and 0.25. So per-factor quantisation of the jointly
  encoded state cuts distinctions that live in the combination, and in a recurrence each wrong snap
  carries forward. Neither the melt nor the cool caused 4a's loss; the abrupt handover did (D25).

## Stages

1. **This plan** (docs). Before it, the night's findings PR (the existing data, at the owner's direction).
2. **Option 1, understanding:** D3's two measurements, as subcommands of `scripts/modular_family_probe.py`
   (`critical-temperature`, `effective-dimension`), with tests; findings here.
3. **Option 1, build:** the crate's per-factor continuous mode (`indrajala-math-rust` first, then a
   "Bump rust/" PR); the harness's `--warm-up` (D4, D5, D10), k-means codes (D6), `tau_0` (D7), on the
   built `--anneal`; parity tests, the golden run unchanged without the flags. Tier 1: the A/B once
   `jebel` is back (about 2026-10-22); the probes run from the branch meanwhile.
4. **Option 1, test and decision:** the panel (D11), its expectation written in its script first;
   accept or reject (D12). The first panel void (D13); rebuilt to D14-D20 on #891's branch, then:
   - **4a. The mechanics probe:** word-problem-s3 seed 0 and word-problem-s4 seed 0 in three arms, a full
     melt with `kappa` `10^-2`, a full melt with `10^-4`, a partial melt with `10^-2` (D16, D17); 6 runs,
     about 2 h; read whether the codes melt at the hold, split back as they cool (D20's groups) and
     harden, and the states formed; it settles D16 and D17. Every cell's bounds as D11's.
   - **4a'. The shapes probe** (D22-D24): word-problem-s3 seed 0 and move-to-front-4 seed 3 (the
     failure-A seeds that certified with the commitment never on; not S4 seed 0, a harm seed that needs
     the commitment), no curriculum, each shape; 6 runs, up to 1,600 epochs (S3 on `pyramidon`, mtf-4 on
     `stone`, up to about a day). Existence only: one seed a shape ranks nothing; every shape forming a
     state goes forward.
   - **4a''. The handover probe** (D25): its mechanics first, separation read through the anneal
     (`state-separation` at each step); registered before its data.
   - **4b. The panel** (D11) at 4a's `T_hot`, registered before its data; D12.
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
