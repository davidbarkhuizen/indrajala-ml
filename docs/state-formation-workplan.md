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
([failure-kinds.md](failure-kinds.md)). The options are the literature review's
([window-recurrent-literature.md](window-recurrent-literature.md), extension 2026-10-20).

## The IDs, in time order

Every ID this plan uses, with the short label to write before it (memory `label-decision-ids`: "the
quasi-static cool (D29)", never a bare ID). A primed ID replaces the one it names. IDs from earlier
plans follow, each with its plan (`next-steps.md`'s table gives the `git show` that reads it).

| ID | When | Short label | What it is |
|---|---|---|---|
| options 1-4 | 2026-10-20 | the literature's options | the annealed initial snap, cell noise, the rotation trick, FSQ factors (D1's order) |
| D1 | 2026-10-20 | one option at a time | the options in turn, each understood, built, tested, decided |
| D2 | 2026-10-20 | the annealed initial snap | option 1 joins a continuous warm-up and an anneal of the snap |
| D3 | 2026-10-20 | understanding first | Rose's critical temperature and Zhao et al.'s effective dimension, on saved networks |
| D4 | 2026-10-20 | the continuous warm-up | the codes bypassed, the cell's output the next state |
| D5 | 2026-10-20 | the warm-up's end (first) | the first step screening 0.95, capped at 8 steps; replaced by D14 |
| D6 | 2026-10-20 | k-means codes | each factor's codes from k-means of its outputs at the warm-up's end |
| D7 | 2026-10-20 | `tau_0` from the clusters | twice the pooled within-cluster eigenvalue; replaced by D7' |
| D8 | 2026-10-20 | the halving and hardness test | `tau` halved every 10 epochs; hard at 99% of windows above 0.99 |
| D9 | 2026-10-20 | the commitment's ramp (first) | 0 while soft, ramped back over 20 epochs; replaced by D19 |
| D10 | 2026-10-20 | readings in the warm-up | screens and the fork only; no certificate until the first hard step |
| D11 | 2026-10-20 | the panel | 12 runs at width 32 over every failure kind and both harms |
| D12 | 2026-10-20 | accept or reject | the panel's rule for an option |
| stages 1-8 | 2026-10-20 | the plan's stages | the plan, option 1's understanding, build, test; options 2-4; findings |
| stage 2 | 2026-10-20 | option 1's understanding | `state-spread`: the merge within clusters, the commitment's collapse |
| D7' | 2026-10-20, at the build | `tau_0` from the codes' gaps | a quarter of the median nearest-code squared distance |
| stage 4 | 2026-10-20, 06:07-07:40 | the void panel | the first panel; its anneal never ran |
| D13 | 2026-10-20, after the void panel | the void panel | not a verdict; option 1 retried until its profile runs |
| D14 | 2026-10-20 | the warm-up's end | screen 0.95 and 4 distinct states a code, capped at 8 steps |
| D15 | 2026-10-20 | heat, hold, cool | the owner's profile: doubling to `T_hot`, a hold, an exponential cool |
| D16 | 2026-10-20 | the melt's height | `T_hot` a multiple of the critical temperature (2, the full melt) |
| D17 | 2026-10-20 | thermal noise | each soft code moved by noise of variance `kappa T` (`kappa` 1e-2) |
| D18 | 2026-10-20 | no fork | the fork dropped with the warm-up |
| D19 | 2026-10-20 | the tail and the slow ramp | soft 40 epochs past hard, then the commitment over 60 |
| D20 | 2026-10-20 | the profile's log | each factor's phase, temperature, hardness, commitment and groups an epoch |
| 4a | 2026-10-20 | the mechanics probe | full, quiet and partial melts on S3 and S4 seed 0 (`probe-4a.sh`) |
| D20 amended | 2026-10-20, after 4a | the groups above the noise | the grouping distance `T max(1/100, 10 kappa)` squared |
| D21 | 2026-10-20 | the melting curve | the run's own states annealed at the warm-up's end; the melting point and last transition |
| D22 | 2026-10-20, after 4a | no curriculum | `--whole`: every length to the cap from the first epoch |
| D23 | 2026-10-20 | the commitment's return as an arm | `--anneal-never-commit` against D19's return |
| D24 | 2026-10-20 | the cool's shapes | concave, convex and s on log temperature, 20 octaves in 600 epochs |
| 4a' | 2026-10-20, 12:15-12:53 | the shapes probe | D22-D24 with the abrupt snap; stopped once its warm-ups were read |
| D25 | 2026-10-20 | the gradual handover | the coupling `l`, log-linear in `T` from the melting point to the last transition |
| 4a'' | 2026-10-20, from 13:00 | the handover probe | D25 on S3 seed 0 and mtf-4 seed 3, each shape; mtf-4 convex certified (16:00) |
| D26 | 2026-10-20, evening | seeded crystallisation | the crucible: the curriculum's warm-up held through the profile, then growth |
| D27 | 2026-10-20, evening | a population of crucibles | S3 seeds 0-7, the certificate the selector |
| 4a''' | 2026-10-20 18:05 to 10-21 05:01 | the seeded crystallisation probe | D26, D27 (`probe-d26.sh`): 2 of 8 |
| D28 | 2026-10-20, about 19:00 | the crucible at the cap | `--warm-up-screen 1.01`: the warm-up runs its 8 steps |
| D29 | 2026-10-20, about 19:00 | the quasi-static cool | `--melt-gate 0.99`: the clock holds while the screen is below the hold's best |
| D30 | 2026-10-20, about 19:00 | the 3-octave window | the coupling's window at least 3 octaves |
| 4a'''' | 2026-10-20, 19:04-20:30 | the quasi-static crucible probe | D28-D30 on 4a''''s six failures (`probe-d29.sh`): 8 of 8 |
| D31 | 2026-10-21 | diagnose before a remedy | options 2-4 shelved; the panel's failures classified by kind first |
| D32 | 2026-10-21 | adaptive cooling | Huang et al.'s schedule beside the quasi-static crucible in the panel |
| D32's understanding step | 2026-10-21, 05:06-07:12 | the heat-capacity logging run | `thermals` on three certified runs (`probe-d32.sh`) |
| D33 | 2026-10-21 | the adaptive cool as built | `--melt-shape adaptive`: the equilibrium test, `T exp(-pace / sqrt(C))`, `pace` 0.03, `settle` 10 |
| D33's probe | 2026-10-21, from 07:24 | the adaptive cool's existence probe | S3 seeds 0-7, mtf-4 seed 3, a convex control (`probe-d33.sh`) |
| D34 | 2026-10-21, 08:30 | the bounded handover | `--cool-rise 0.02`: the adaptive step no colder than raises the coupling by 0.02 |
| D34's probe | 2026-10-21, from noon | the bounded handover's probe | S3 seeds 0, 1, 3-7 under D34 (`probe-d34.sh`) |
| the reasoning review | 2026-10-21 | gaps and inconsistencies | ten items (`reasoning-review-2026-10-21.md`); its re-reading of existing data in the findings |
| D35 | 2026-10-21 | the answers-only stop | `--state-bound N`: the snapped machine's answers all right within the bound; the certificate only judges |
| D36 | 2026-10-21 | crucible against whole | the panel's arms: the quasi-static crucible and the same cool with `--whole` (the owner; replaces D32's second arm) |
| D37 | 2026-10-21 | the panel's cells | D11's, fresh S3 seeds 8-15, random-q16-0 seeds 0 and 1 and a random-q12-0 control; the harm cells' expectation registered |
| D38 | 2026-10-21 | the coupling probe | `--coupling-cap 0`: the codes anneal on the liquid, never coupled; S3 seeds 0-7, move-to-front-4 seed 3 |
| D39 | 2026-10-21 | the design probes | after the panel: an approximation beside S3, a random machine's melting curve, runs past the first certificate |
| the consistency review | 2026-10-21, afternoon | rules against practice | ten items (`consistency-review-2026-10-21.md`), for the panel's registration |
| D40 | 2026-10-21, afternoon | the panel's verdict | D12 replaced for 4b: a harm loss goes to the diagnosis; rejected only if no failure-A run moves |
| D41 | 2026-10-21, afternoon | one success | certified at both verdicts on the stop's network; exact at 1.0 alone reported, not counted |
| D42 | 2026-10-21, afternoon | the bound as trained | each certified run's training length reported; the exclusion bound the config's; Chow's check for stage B |
| D43 | 2026-10-21, afternoon | the exact state bound | the panel at `N = q`, claimed so; slack read offline on its step networks |
| D44 | 2026-10-21, afternoon | the crucible's cap from the bound | after the panel (D39's (iv)): the cap from the state bound, not the target |
| D45 | 2026-10-21, 13:00 | D38's reading and D34's order | the gate's holds and the snapped answers read; D34's probe only if D38 is refuted |
| D46 | 2026-10-21, 16:30 | the core matrix | 4b-core: D36's arms on S3 and mtf-4, fresh seeds 8-23, 600 epochs on every machine; the large tier set aside |
| 4b-core | 2026-10-21, registered 16:30 | the core matrix | 48 runs (`probe-4b-core.sh`) on stone, jebel and pyramidon, from about 20:00 |
| D47 | 2026-10-21, evening | the matrix set aside | the owner: 4b-core not launched; the continuous method replaces it |
| D48 | 2026-10-21, evening | the continuous crystallisation | the owner: every switch of the profile made continuous, all at once (`--continuous`) |
| D49 | 2026-10-21, evening | the continuous rate | the clock advances by `g`, from the screen against the hold's level (0.95-0.99); one exponential cool |
| D50 | 2026-10-21, evening | the coupling from sharpness | the coupling read live from the codes' mean largest soft weight, from the cool's start |
| D51 | 2026-10-21, evening | no hard switch | soft to the end with a straight-through gradient (the crate's `straight`); `T` held where the codes are hard to 0.1% |
| D52 | 2026-10-21, evening | continuous growth | the longest length rising in `ln L` at `0.02 g c` from the crucible's to the cap |
| D53 | 2026-10-21, evening | the readings every epoch | the certificate's breakdown, snapped answers, sharpness, `g`, `L` each epoch; the melting curve every 25 |
| D54 | 2026-10-21, evening | the continuous probe | S3 seeds 0-7 and move-to-front-4 seeds 0-3 under D48 (`probe-d54.sh`) |
| D55 | 2026-10-21, 20:50 | the fresh-seed matrix | D54's cell on S3 seeds 24-39 and move-to-front-4 seeds 24-31 (`probe-d55.sh`) |
| D56 | 2026-10-22, 00:40 | the unstructured probe | D54's cell on random-q16-0 seeds 0-3 (`probe-d56.sh`) |
| D57 | 2026-10-22, 00:50 | the S4 probe | D54's cell on word-problem-s4 seeds 0-2, the harm cells without the commitment (`probe-d57.sh`) |
| D58 | 2026-10-22, 02:50 | the random control probe | D54's cell on random-q12-0 seeds 0-3 (`probe-d58.sh`) |
| D59 | 2026-10-22, 04:20 | the margin stop | D35's stop waits until no near tie splits the machine (`--margin-wait 200`); its probe on the 12 strict-only stops (`probe-d59.sh`) |
| D60 | 2026-10-22, 08:00 | the margin deadline | the margin stop's wait counts every step past the first pass, ending on the best pass's network |
| D61 | 2026-10-22, 08:00 | the paced coupling | the coupling paced by a trust region (`--coupling-pace`) or read from a Savitzky-Golay line (`--sharpness-smoothing`); its probe (`probe-d61.sh`) |
| D62 | 2026-10-22, 08:00 | the run's state object | `run`'s loop-carried state one dataclass, the loop's body split into functions of it (planned) |
| D63 | 2026-10-22, 08:15 | the collapse | reheat (the rate below the hold's level negative, the clock run back) or kill-and-resample, on cost per certified machine (planned, after D62) |
| D64 | 2026-10-22, 09:00 | the fresh-seed rate | the current method (D54's cell, the margin stop under D60) on 22 fresh seeds across five backdrops (`probe-d64.sh`) |
| D65 | 2026-10-22, 10:45 | the order-biased cool | the cool's halving time times `1 + w (s - 1)`: w the machine's order from answers (0 when ordered), s the slowdown from the melting curve and relaxation; set aside at 11:05 on its stage 0 (the failures freeze, the cool does not outrun them) |
| D66 | 2026-10-22, 11:20 | the two-phase cool | a second gate on the clock, quasi-static in the state's order (the snapped machine's long answers): slowed while the order rises, today's cool where it is settled (planned, after D63's reading) |
| 4b | registered later | the panel | D36's arms (the crucible against `--whole`, both quasi-static) on D37's cells, D35's stop |

From earlier plans:

| ID | Plan (when) | Short label | What it is |
|---|---|---|---|
| L3 | division ladder | the division rung | division `p` 13, factors 32 wide, window 16 |
| W11 | window branches (2026-10-19) | the base window | three blocks, `D` 16, 2 heads, FFN 32 |
| W1 | window branches | the 32-wide window | L3's cell at window width 32: 5 of 6 |
| W2 | window branches | the size control | L3's cell matched to B1's size: 3 of 6 |
| B1-B3 | window branches | the window branches | B1 on division, B2 on S5 and mtf-5, B3 no harm (written b3 in 4a'''s readings) |
| D2'' | commitment control (2026-10-19) | the thermostat's screen | restored once the step's screen reaches 0.95 |
| P1 | code consolidation (2026-10-18) | the guard | A with the certificate guard |
| P2 | code consolidation | restore the step's best | A with the step's best network restored: 0 of 6 |
| R1-R4 | one code, one state (2026-10-19) | the extra-codes remedies | R1 the answers' own branch, R2 pairs per code logged, R3 keep the best machine, R4 counterexample training |
| C0 | one code, one state | the base at width 32 | rounds 1 and 2's cells at width 32, seeds 0-5 (not the claims report's C0) |
| C2 | one code, one state | R3 and R4 on the base | S5 2 of 6, mtf-5 3 of 6 at width 32 |

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
  of C0's and C2's); **codes standing for several states** (extra codes) moved only with R3/R4.

## The options, in order (D1)

1. **The annealed initial snap** (the review's options 1 and 2 joined, at the owner's suggestion): the
   state continuous for a warm-up, then the codes initialised from the states seen and cooled from a
   `tau_0` set by the outputs' scale to hard, the commitment off until hard and ramped back after.
2. **Noise in the cell's pre-activation** (Oliva & Lago-Fernández 2020): saturated, attracting states
   and wider margins.
3. **The rotation trick** (Fifty et al. 2025): a gradient that can separate windows on one code
   (extra codes).
4. **FSQ-style fixed-grid factors** (Mentzer et al. 2024): no codebook to learn, no commitment.

Each option in turn: **understand** (a measurement on existing networks that tests the literature's
mechanism here, before any build), **build** (tests, parity, the golden run), **test** (the panel,
D11), **accept or reject** (D12), recorded in this plan's findings before the next option starts. An
accepted option becomes part of the base the next one is tested on.

## Decisions

- **D1. One option at a time, in the order above.**
  - (a) *Chosen.* Pros: the owner's direction; each result read against one change; the order is the
    review's, by fit to the failure kind seen most (formation) and by cost (harness, then a layer, then
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
  - (a) *Chosen.* Pros: answers-only (the fork's and the thermostat's screen's, the commitment-control plan's D2''); the diagnostic's states formed
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
  - formation or conversion (failure A): word-problem-s3 seeds 0, 1, 4 (0 certified with the commitment off; 1 and 4 never
    formed); move-to-front-4 seed 3; move-to-front-5 seeds 0, 1;
  - harm: word-problem-s4 seeds 0, 1 (C0 certified both), move-to-front-4 seed 0 (C0), division `p` 13
    seeds 0, 1 (W1, `--factor 32 38` twice);
  - extra codes: word-problem-s5 seed 1.
  Controls by the by-bits rule where they exist (C0, W1); S3 at 32 has none, so S3 seeds 0, 1, 4 run
  without the option beside the panel once (the base control). Feasibility: the cells' registered
  bounds (S3 q 6, 8 codes, m 64, cap 69; mtf-4 24, 15, 225, 248; S4 the same; S5 and mtf-5 120, 32,
  1,024, 1,143; division 169, 38, 1,444, 1,612), each width 32 certified before.
  - Pros: every failure kind and both kinds of harm in one batch, about 3 h on `stone` at night. Cons:
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

**Vocabulary (agreed with the owner, 2026-10-20).** Rose's terms where his definitions hold exactly
(deterministic annealing on fixed data): *temperature*, *critical temperature* (`2 lambda_max`),
*phase transition* (a split of the codes); the melting point and the last transition are the first and
last of the measured curve's transitions (D21). The rest is an explicit analogy with a crystallising
material, not literature's terms:

| Term | The profile | The codes and the state |
|---|---|---|
| crystallisation | the whole profile, the warm-up's end to hard codes | (the anneal) |
| heating | the temperature doubling to `T_hot`, the coupling 0 | the codes soften |
| melt | the hold above the melting point | the codes merge toward the mean (the lattice dissolves); the forward path carries the cells' outputs (coupling 0, D25) |
| nucleation | the cool passing the melting point | the first phase transition: the first codes part |
| growth | the cool down to the last transition | the later phase transitions; the coupling rises to 1, locking the state onto the codes |
| solid | hard codes, coupling 1 | the crystal; its certificate says whether it is the target machine |

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
- **D20. Each factor's profile logged each epoch** (its phase, temperature, hardness, commitment and its
  codes' groups closer than `T / 100` squared), and the warm-up's end row with each factor's distinct
  states, critical temperatures and `T_hot`. Pros: the melt, each split and the crystal are read from the
  log. Cons: a little more per epoch (the codes only).
- **D20 amended: the groups' distance is `T max(1/100, 10 kappa)` squared,** above the thermal noise
  (4a's `full` arm unreadable at `T / 100`).
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
### Seeded crystallisation (2026-10-20, after 4a''s readings)

The owner: "continuing with the crystallization analogy - how can we form seed nuclei and then use them
to seed the melt when we start cooling?"; on a seed freezing wrong (a polycrystal): "isn't this why we
need a statistical/parallel approach?"; and "I would hold off on the Hukushima variation until we have
done the initial baseline and interrogated the results".

**Understanding first (the readings, `option1/separation/separation.txt`):** on word-problem-s3 the
no-curriculum warm-up never formed the state (4a'', step 8: continuous centroid accuracy at chance from
length 16, nearest neighbour 0.51 at 64 against chance 0.17: a drifting liquid, nothing to crystallise).
The curriculum's warm-up did form it (4a: 1.00 to length 32, 0.94 at 64), but **snapped onto its own
k-means codes with no training it fell to chance** (joint 0.90, 0.41, 0.20 at lengths 4, 16, 64 against
0.26, 0.20, 0.17; 7 of 145 pairs right), where move-to-front-4's snap kept 1.00. A permutation machine has
no synchronising word: one wrong snap stays wrong to the word's end, where a reset machine forgets it
within a few tokens. So a group's seed can't be a snapshot of the continuous state: it must be
crystallised, with the handover, on the material where it formed, before the long words arrive.

- **D26. The seed grown in its own crucible, then grown on the whole distribution (`--crucible`).** The
  curriculum's warm-up (D14, without `--whole`), then the material held at the warm-up's last step's
  lengths through the whole profile (D15-D25: the full melt, the handover, convex, the commitment never
  back) until every factor is hard; then growth: every block the whole distribution's (D22), the codes
  hard, the forward path coupled to them (1).
  - (a) *Chosen.* Pros: each part worked before (the curriculum formed S3's state, 4a; the handover
    crystallised mtf-4's, 4a''); the material is fixed through the anneal (D22's reason holds: one
    material a profile); growth trains to the cap, so the run's training meets the exclusion bound.
    Cons: the crucible trains below the bound, so a seed may be a memorised machine (growth on the whole
    distribution is its correction; a certificate is a proof whatever the lengths); growth at a frozen
    temperature can't dissolve a wrong seed ((b) would).
  - (b) Seed the melt: after the crucible, reheat on the whole distribution to just below the seed's
    melting point and cool again (the owner's picture, and their earlier "heat to just below the melt").
    Pros: a wrong seed can dissolve. Cons: here a temperature in the forward path blurs the state, which
    costs a group's (the understanding above), and D25's coupling is 0 at the melting point, unplugging
    the seed while the long words arrive: it needs a coupling rule of its own. The next variant, chosen
    from the baseline's readings.
  - (c) The curriculum warm-up's state snapped, no crucible anneal. Cons: the snap leaves S3 at chance.
- **D27. A population of independent crucibles, the certificate the selector:** word-problem-s3 seeds
  0-7, one cell, each run its own warm-up, codes and noise; existence (a certificate in any run), the
  rate a screen (Wilson 95%).
  - (a) *Chosen.* Pros: the owner's statistical approach; independent by construction (no shared seed,
    so a polycrystal in one run says nothing of another); with the rate at 0.2, 8 runs give a
    certificate with probability 0.83; 8 fills `stone` with the handover probe's two runs (8-10 workers its
    best throughput). Cons: one task; 8 seeds a screen, not a rate.
  - (b) Population annealing (Hukushima and Iba 2003; Machta 2010: replicas resampled by fitness at each
    temperature). Deferred at the owner's word until this baseline is read; its resampling signal must
    be the answers' screen, never the certificate (discovery: the target unknown).
  - (c) Fewer seeds. Cons: the rate unmeasured.
- **The cell (feasibility, docs/information-flow.md):** two factors, `--codes-ratio 32/11` (8 codes a
  factor, `m` 64 against `q` 6, row 1), reading the output, width 32 (S3 certified at 32, row 3), cap 69
  (`m + q - 1`, row 2, reached by growth), 1,600 epochs or 690 minutes; `T_hot` 2 `T_c`, hold 20,
  `kappa` 1e-2, convex, tail 40, the handover, the commitment never back. Budget: the crucible's 666
  epochs at about 10 s (4a's S3 at lengths 1-9), growth's 930 at about 28 s (D24's 18.6 s, slower with 10
  runs on `stone`): about 10 h a run, in `stone`'s window (18:00-07:00).
- **Expectations (registered in `probe-d26.sh`'s header before its data):** seed formation: at the
  crucible's last block (before growth), the snapped joint nearest-neighbour separation at length 16
  >= 0.9 in at least half the runs; growth: at least one certifies. Refuted: no crucible crystal
  separates (seeds don't form: the crucible's design is wrong); seeds form but none certifies (growth
  loses them: (b) next, where the logs say).

### The quasi-static crucible (2026-10-20, evening, from the population of 4a''' in flight)

By the owner's framing (path dependence expected; a failure that correlates with something observable
early is a design flaw, to fix), the six failures of 4a''' (`population` on its logs at about 19:00, 55
minutes in; findings below once it ends) fall in three systematic kinds, each seen from the answers or
the run's own measurements, none from the target:

- **The crucible too short:** D14 ended three warm-ups at step 3 (screen and distinct states passed at
  once); in each the cool's best state share stayed 0.36-0.38. The five crucibles of length 5 and 8 all
  formed a state in the cool (0.75-1.00).
- **An open-loop cool:** the profile's clock is the epochs trained, so a step that dips trains more
  epochs and the cool runs on through the dip. Each loss began with a dip at an unchanged schedule
  (seed 2, step 85: screen 1.000 to 0.704; seed 6, step 69: 1.000 to 0.657; seed 1, step 101: 1.000 to
  0.923), the coupling rising through it (seed 1 0.84 to 1.0 in 20 epochs, seed 2 0.80 to 1.0 in 11),
  and no run recovered once at 1. The certified runs' screens never fell below 0.99 in the cool.
- **A degenerate coupling window:** seed 6's factor 1 melted in one transition (1 to 8 groups at 2.14),
  so D25's window fell back to an octave, and its coupling went 0 to 1 within 12 steps of the cool's
  start, D25's abrupt handover in small; its loss came there.

D28-D30 fix one each. Seeds 0 and 5 are untouched by all three (their warm-ups ended at the cap, their
windows span 3.25-4.25 octaves, no screen in their cools fell below 0.99), so under them they are the
same runs, and the follow-up reruns the six others.

- **D28. The crucible at the warm-up's cap** (`--warm-up-screen 1.01`: only D14's cap of 8 steps ends
  the warm-up).
  - (a) *Chosen.* Pros: the three length-8 crucibles each formed the state (two certified); a setting,
    no build; generic (the cap is D5's, every backdrop's). Cons: the cap a number, not read from the
    run; a longer continuous warm-up may drift (D4), and on a backdrop whose warm-up passes early it
    costs epochs.
  - (b) A stricter D14 (more distinct states a code). Cons: S3's length-3 crucibles had 8 of 8 codes in
    use; the count wasn't what was short.
  - (c) The crucible grown from length 3 by a second anneal. Cons: a new mechanism, untested.
- **D29. A quasi-static cool (`--melt-gate 0.99`): the profile's clock holds through an epoch whose
  screen before it is below 0.99 of the hold's best**; temperature and coupling wait while the network
  recovers at fixed values, the noise still on. As adaptive schedules in simulated annealing cool only
  near equilibrium (Huang, Romeo and Sangiovanni-Vincentelli 1986), here the answers' screen is the
  equilibrium's reading.
  - (a) *Chosen.* Pros: answers only; compatible with D18 (nothing undone: the crystal waits rather than
    being restored); time isn't the constraint (the owner); off while the screen holds, so a run that
    never dips is unchanged. Cons: a run that never recovers stalls at its temperature until the budget
    (logged: `held`); the threshold a number (the certified runs never fell below it; the losses fell
    to 0.66-0.92).
  - (b) Slow the whole cool. Cons: the dips came at every speed of the convex cool; the cost paid by
    every run, and an open loop still.
  - (c) The fork (restore and cool slower). Cons: D18, the owner's.
- **D30. Each factor's coupling window at least 3 octaves below its melting point
  (`--handover-octaves 3`).**
  - (a) *Chosen.* Pros: the certified factors' windows were 3.25-4.25 octaves (unchanged); a curve with
    one transition hands over gradually. Cons: below a single transition the window's end isn't the
    material's; 3 a number.
  - (b) Remeasure the curve until it shows a cascade. Cons: the material is what it is.
- **D31. After option 1, the panel diagnoses before any next option (the owner, 2026-10-21).** Once
  option 1 (the quasi-static crucible) is decided, options 2-4 go on the shelf, not in turn as D1 had
  them. The panel (4b) runs the method against the full backdrop to find the failing cases; those are
  analysed in detail for a common failure mechanism; only then is a remedy chosen, by its fit to that
  mechanism, from the shelf or elsewhere.
  - (a) *Chosen.* Pros: a remedy picked for a measured mechanism, not by the review's order (the lesson
    of formation against conversion: remedies judged on runs failing at the other problem); the
    instruments exist (`population`, `state-separation`, `approximations`, `breaks`); no option built
    for a wall not yet met. Cons: options 2-4 untested for now; a mechanism the diagnosis can't name
    leaves no ranking.
  - (b) Options 2-4 in turn (D1). Cons: each tested on whatever fails, fitting or not; three builds
    before the panel's failures are read.
  - Refinements: the diagnosis presumes no mechanism (extra codes likely, not given): each failing run
    is classified by formation (`state-separation`, continuous), conversion (snapped against
    continuous), and extra codes (`breaks`, codes standing for several states); the shelf records each
    option's wall (option 2, noise: thin margins and near ties; option 3, the rotation trick: windows of
    different states sharing a code, extra codes' own; option 4, FSQ factors: a codebook that
    collapses), so the diagnosis can call one down; the panel adds fresh word-problem-s3 seeds (8-15),
    since D28-D30 were designed from seeds 1-7's failures.
- **D32. Adaptive cooling, beside the quasi-static crucible in the panel (the owner, 2026-10-21).** In
  the adaptive schedule of Huang, Romeo and Sangiovanni-Vincentelli (developed in Romeo 1989, Ch. 7)
  the cooling curve isn't precomputed: the temperature steps only once an equilibrium test passes (the
  cost's moving average within 0.75 of its predicted standard deviation, then Kolmogorov-Smirnov), and
  each step is sized by the cost's measured spread, `T_{k+1} = T_k exp(-lambda T_k / sigma_T)`, so it
  slows by itself where the variance (the specific heat) peaks, at the phase transitions; a cap on moves
  a temperature, with the predictions re-estimated when it is hit. D29 is a one-sided form of the test
  on a precomputed curve. The panel (4b) runs two arms: the quasi-static crucible (D26-D30) and the
  crucible with adaptive cooling (the shape, octaves and cool length replaced by the rule and `lambda`).
  - (a) Adaptive cooling as option 1's form before the panel, one arm. Pros: the panel tests one
    finished form; one panel. Cons: the panel waits on the build and its existence probe.
  - (b) The panel with D29 alone, adaptive cooling shelved. Pros: fastest to the failures. Cons: a
    panel rerun if the failures are the cool's.
  - (c) *Chosen.* Both arms in the panel. Pros: the two schedules compared on every backdrop at once;
    a failure read against both (a cooling failure in one arm and not the other names itself); the
    shape question answered by the data. Cons: twice the panel's runs; the panel waits on the build.
  - The order: the understanding step first (the cost's spread per epoch logged: the batch loss and the
    soft assignment's distortion, Rose's energy, whose variance is the specific heat; read on a
    certified S3 seed and move-to-front-4 seed 3 for peaks at the codes' splits, which picks the cost);
    then the build (`--cool adaptive`, the equilibrium test, the step rule, the cap); then a short
    existence probe on S3 seeds 0-7 and move-to-front-4 seed 3; then the panel, both arms, registered
    before its data. Open in the build: the weights learn while the codes anneal, so the landscape moves
    and stationarity at fixed `T` may not hold; the cap and re-estimation are the guard.
- **D33. The adaptive cool as built (`--melt-shape adaptive`; D32's build, settled 2026-10-21 under the
  owner's delegation).** The cost is the soft assignment's distortion, Rose's energy (`thermals`), each
  factor its own; the understanding step's S3 reading (below) shows its capacity low in the melt, rising
  at nucleation and peaking within the cascade of splits. One measurement an epoch (the energy and
  capacity over the step's screen at its three lengths), so Huang et al.'s statistics over many moves
  at a temperature become statistics over epochs:
  - **The equilibrium test:** a temperature's first epoch passes if its energy is within `within` 0.75
    sigma (`sigma = T sqrt(C)`, the assignment's spread) of the energy predicted from the last
    temperature (`E - C dT`, Huang et al.'s prediction); a later epoch, if within 0.75 sigma of the
    epoch's before (stationary at `T`). (a) *Chosen.* Pros: their test's form and constant; the
    prediction lets a material that tracks equilibrium step every epoch; stationarity is what the
    weights' moving landscape breaks, so the test holds `T` exactly then. Cons: an epoch's energy moves
    with training as well as heat, so in the cold tail (sigma small) it holds often, until the cap. (b)
    A Kolmogorov-Smirnov test of the epochs' energies. Cons: needs many epochs a temperature; their
    second stage, for many moves. (c) No test, the step alone. Cons: open loop again (D29's lesson).
  - **The step:** `T exp(-pace / sqrt(C))`, `C` the capacity's mean over the epochs at `T` (their
    `T exp(-lambda T / sigma)` exactly), at most an octave. (a) *Chosen.* Pros: slow where the capacity
    peaks (the splits), fast through the melt's flat top and below the last split; the octave bounds a
    step where `C` is near 0. Cons: a factor 0.5 a choice; the mean smooths the epoch's noise (19-22 local
    maxima a factor in S3 seed 4's cool) only over the epochs at `T`.
  - **`pace` 0.03:** the convex cool's own speed through the cascades it certified with (implied
    `-d ln T sqrt(C)` an epoch, the splits' medians 0.027-0.034 on S3 seeds 4 and 7), so the adaptive arm
    is no faster than convex where the transitions are and faster elsewhere. Pros: one known-good
    anchor; mtf-4's cascades agree (0.029-0.037, findings below). Cons: one schedule's speed, not
    an optimum.
  - **`settle` 10 epochs:** the cap on epochs at a temperature, then a step regardless, sized by the
    epochs measured there (their re-estimation). Pros: bounds a stall (the cold tail) at about the
    exponential's 15 epochs a halving. Cons: a number.
  - **Kept:** D29's gate (it holds the adaptive cool too: no step on a held epoch), D25's coupling
    (log-linear in `T`, whatever schedule sets `T`), the hold and heat; hardness read from the cool's
    start (as the exponential: no curve to wait for).
  - **Resumption:** the cool's state (each factor's temperature, its epochs' measurements, its
    prediction) is in the checkpoint; building its resumption test found that the couplings, training
    state not saved with the network, weren't restored on resume, so a soft, coupled run resumed at a
    step's end or start screened at coupling 1 (logged screens only, and D29's first gate reading);
    fixed (the checkpoint keeps them). No earlier probe's certificate is touched (certificates read the
    snapped network).
- **D34. The adaptive cool's handover bounded (`--cool-rise 0.02`; from the D33 probe's in-flight
  reading below, settled 2026-10-21 under the owner's delegation): a step no colder than raises a
  factor's coupling by 0.02.** D33's step reads the codes' capacity, which falls to about 0.01 below the
  cascade while the coupling (D25, log-linear in `T`) is still about half way, so the steps there reach
  the octave cap and the coupling rises up to 0.65 in an epoch; each jump drops the screen, the gate
  (D29) holds, the network recovers, the next jump drops it again, until the state erodes. The
  quasi-static crucible's coupling never rose more than 0.02 an epoch (0.009-0.020 on seeds 0, 1, 3, 4,
  5).
  - (a) *Chosen.* The bound on the coupling's rise a step, 0.02. Pros: bounds the forward path's change,
    which is what broke, where it changes (inside the window only: above it and below it the step is
    D33's); 0.02 the most the 8-of-8 cools used; the cascade's pacing kept; `rise` 0 is D33 exactly
    (bits unchanged). Cons: slower than D33 below the cascade (about 50 epochs across a full window);
    a number, from one schedule.
  - (b) A smaller octave cap (a step at most e.g. 0.1 in ln `T`). Cons: blind to the window's width (a
    3-octave window and a 10-octave one need different caps); slows the melt's flat top too.
  - (c) The coupling on its own clock, a fixed rate from the cool's start. Cons: unlinks it from the
    temperature (the owner's D25: "would that be linked to temperature").
  - (d) The forward path's response in the capacity (the screen's variance with the energy's). Cons:
    a new measurement, untested; (a) bounds the same thing directly.
### The program restructured (2026-10-21, after the reasoning review)

The reasoning review ([reasoning-review-2026-10-21.md](reasoning-review-2026-10-21.md)) and its
re-reading of existing data (findings below) change what the panel (4b) tests and what comes before
it. D36 is the owner's; D35 and D37-D39 settled under the owner's delegation.

- **D35. The stop: answers-only under the state bound, the certificate only judging it**
  (`--state-bound N`; the review's item 1). A run stops at the first step whose
  snapped machine's answers are all right (the step's screen and lengths 8, 16, 24, 48, read on the
  snapped network, `mf.snapped_answers`) and whose machine, read off and minimised, has at most `N`
  states (the target's size where the target is known). The stopped network is certified at the
  default snap ratio and at 1.0, both recorded; a run that certifies before the rule passes trains
  on (recorded: `missed`).
  - (a) *Chosen.* Pros: the target leaves the training loop, as discovery needs (machine-discovery.md);
    the re-reading lost no certificate under it (121 kept, 8 earlier on networks exact at 1.0, no false
    discovery, none missed), so it costs the panel nothing; the panel tests it prospectively. Cons: a
    network exact only at 1.0 now ends a run (both verdicts reported); the bound must be tight (1.1
    times the size: 25 false stops), so a loose-bound study is its own question.
  - (b) Keep the stop at the first certificate, the rule logged beside it. Pros: no change. Cons: the
    target stays in the loop; every rate stays a known-target rate.
  - (c) Train a fixed number of steps past the first certificate. Pros: measures rules that wait for
    a stable machine (the re-reading's limit). Cons: training past a proof destroys it (P1, 3b); its
    own probe, after the panel (D39).
- **D36. The panel's arms: the crucible against the whole distribution, both quasi-static (the
  owner, 2026-10-21; replaces D32's second arm).** Arm 1 the quasi-static crucible (D26-D30); arm 2
  the same cool on the whole length distribution from the first epoch (`--whole`, no crucible). The
  review's items 4 and 5: the bounded handover (D34) ties the adaptive cool to the quasi-static
  cool's coupling rate where the failures came, so D32's arms barely differ there, while the crucible
  hasn't certified move-to-front-4, which certified only with `--whole` (4a''). A failure in the
  crucible arm alone names the crucible's harm on a backdrop. The adaptive cool waits as a generality
  refinement (no shape, octave or cool-length constants); D34's probe still runs as registered (it
  reads whether that refinement holds) but no longer gates the panel.
- **D37. The panel's cells: D11's, fresh S3 seeds 8-15 (D31), and random machines** (the review's
  items 7 and 10). Added: random-q16-0 seeds 0 and 1 (0 of 6 at round 1's cell, by extra codes and lost
  near solutions) and a random-q12-0 seed that certified at round 1's cell (the control). Every cell's
  bounds (codes, the exclusion bound, row 15's crucible check) written beside it at registration.
  Registered with it, before its data: **the harm cells' expectation** (S4 seeds 0 and 1,
  move-to-front-4 seed 0, division `p` 13 seeds 0 and 1, which certified only with the commitment):
  at least one loses its certificate in each arm, the commitment never back; each loss read for thin
  margins (near ties, option 2's wall) against extra codes (pairs per code, option 3's), which
  decides the follow-up.
  - (a) *Chosen.* Pros: discovery's likely targets (unstructured) in the panel; the harm question
    registered, not read after. Cons: more runs (sized at registration; about two nights on `stone`).
  - (b) D11's cells only. Cons: option 1 judged on structured machines alone.
- **D38. The coupling probe, before the panel (`--coupling-cap 0`; the review's item 3).** The
  re-reading: every certificate in the quasi-static crucible came at a coupling of 0.06-0.90, its codes
  at their final groups at 0.00-0.50. The probe holds each factor's coupling at 0 (the codes anneal on
  the liquid and never enter the forward path) and certifies the snapped network, everything else as
  the quasi-static crucible (D26-D30), on S3 seeds 0-7, with move-to-front-4 seed 3 under `--whole`.
  Expectations registered in its script before its data: at least 6 of 8 S3 certified; refuted at 4
  or fewer (the coupling matters, and where the snapped and coupled screens part says why).
  - (a) *Chosen.* Pros: if it holds, conversion needs no handover (one schedule fewer, a simpler
    method for every backdrop) and the panel tests that; short (S3's crucible runs are 4-28 minutes).
    Cons: a build (`--coupling-cap`, pure Python); the panel waits about a night.
  - (b) After the panel. Cons: the panel then tests a method that may carry a part it doesn't need.
  - (c) Not at all (D25 as built). Cons: the review's item 3 stays open.
- **D39. After the panel's diagnosis (D31): three short design probes, in order.** (i) An
  approximation's output beside S3's answers (the sign, an item's place) under `--whole`: does an
  approximation form the state without a seed (items 2 and 9; joins P2's finding to formation)?
  (ii) The melting curve (D21) on random-q16-0's warm-up: a crowded band of transitions against an
  algebra's cascade (the random-machine conjecture). (iii) Runs past their first certificate under
  D35 (D35's (c)): does a rule that waits for a stable machine stop on the target? Each registered in
  its own script before its data.

### The core matrix (2026-10-21, 16:30; before the coupling probe's data)

The panel's timing (2026-10-21, one cell at a time on `pyramidon`, scaled to `stone`): under
`--whole`, S5 and move-to-front-5 epochs take about 300-440 s and 2.6 GB, division 13's over 1,000 s
and over 3.1 GB, so a run of 1,600 epochs is 5.5-8 days (S5, mtf-5) and 18-27 days (division). The
owner: "we don't have the infrastructure for the large tier ... is there an affordable matrix we can
run to rather just provide a better statistical basis to what we already know?"; and "let's include
jebel at night, but running at 50% capacity".

- **D46. The core matrix (4b-core) before the panel:** D36's arms (the quasi-static crucible, and the
  same cool with `--whole`) on word-problem-s3 and move-to-front-4, fresh seeds (S3 crucible and mtf-4
  whole 8-23, S3 whole and mtf-4 crucible 8-15: 48 runs), 600 epochs on every machine, D35's stop at
  the exact bound, D41's success, D45's coupling rule; expectations, refutations and analysis in
  `~/code/probe-4b-core.sh`'s header (registered before its data). The large tier (S5, mtf-5, division)
  waits for a machine that can run it or a cheaper length distribution (a design question, not
  settled); the harm and random-machine cells (D37) are add-ons, not in this matrix.
  - (a) *Chosen* (the owner's question; settled under the delegation). Pros: every claim option 1
    rests on (the crucible's rate on S3, groups against resets, the crucible's harm on a reset
    machine, the stop's safety) gets a rate on fresh seeds and two Fisher contrasts; about 1.5 days on
    three machines, under 0.6 GB a run. Cons: two backdrops only; the harm cells and the random
    machines wait.
  - (b) The full panel (D37's cells). Cons: weeks for the large tier on this hardware.
  - (c) The small tier (S3, mtf-4, S4, random; D37's cells without the large ones). Pros: the harm
    question too. Cons: about 3 days, and fewer seeds a cell for the same cost.
  - **The budget, 600 epochs on every machine (minutes 2,400, a safety):** the machines run at
    different speeds (`pyramidon` about twice `stone`'s seconds an epoch), so a minutes budget would
    give a run's outcome its machine; every certified run of option 1 so far took at most 294 epochs.
    Pros: machine-neutral; about twice the slowest success. Cons: a run that would certify later than
    600 epochs is a failure here (logged: its course to the budget).

### The continuous crystallisation (2026-10-21, evening; the owner)

A detailed reading of the coupling probe's (D38) seed 3, in flight (not a verdict; the probe runs to
its registered end): at coupling 0 the continuous network separated S3's six states perfectly from
step 9 through the melt and the cool, while its codes, annealed on its outputs, re-formed to a
snapped separation of 1.0 only at step 108, where the certificate at ratio 1.0 found no wrong state
and 2 wrong outputs (the readout, trained on continuous states alone, misreads some codes), then
froze from step 120 (22 codes, 12 states minimised) for want of any signal; its hardness unread for
600 epochs by the convex shape (D24), growth waited on it. The readings that showed this were all
rebuilt offline. The owner: "stop the matrix. make everything we can, that still makes sense to do
so, continuous. all at once - the whole point of the crystallization is to be quasi-static so that
we can smoothly transition down the energy profile. then let's just test that." D47 and D48 are the
owner's; D49-D54 settled under the delegation.

- **D47. The core matrix (D46) set aside, unlaunched.** Its registration stands in
  `~/code/probe-4b-core.sh`; the continuous method replaces the method it would have measured. The
  coupling probe (D38) and the adaptive cool's probe (D33's) run to their registered ends.
- **D48. The continuous crystallisation (`--continuous`): every switch of option 1's profile that can
  be continuous made so, all at once, and tested as one method.** One event stays: the warm-up's end,
  where the codes are made (k-means, D6) and the profile starts, continuous in the forward path
  because the coupling is 0 there. Its parts are D49-D53; its test D54.
- **D49. The cool's rate, continuous (replaces D29's binary gate and D24's shapes).** The profile's
  clock advances each cool epoch by `g = clip((s / level - 0.95) / (0.99 - 0.95), 0, 1)`, `s` the
  screen before the epoch, `level` the hold's best (D29's): full speed at D29's 0.99, slowing linearly
  to a stop at 0.95. The cool is exponential in that clock, halved every 30 epochs (D8's halving), with
  no shape; heat and hold as D15.
  - (a) *Chosen.* Pros: the rate falls as the network leaves equilibrium, not on or off, so the path
    down the profile is quasi-static by construction; one shape, no shape constants (D24's octaves and
    cool length). Cons: two thresholds (0.95, 0.99) for D29's one.
  - (b) D29's binary gate. Cons: a switch; the clock runs at full speed the epoch after any recovery,
    which is where the D33 probe's losses came (a run of jumps).
  - (c) The adaptive cool's step (D33) scaled by `g`. Pros: the heat capacity's information too.
    Cons: the D33 probe lost 4 of 7 by its steps' size; two mechanisms at once.
- **D50. The coupling read live from the codes' sharpness (replaces D25's window, D21's once-measured
  melting curve and D30's 3 octaves).** Each epoch, each factor's sharpness `S` is the mean over the
  screen's windows (D8's sample) of the largest soft weight at its temperature; its coupling
  `l = clip((S - S_0) / (1 - S_0), 0, 1)`, `S_0` its sharpness at the cool's first epoch; 0 through the
  heat and the hold (D25's), so it starts at 0 at the cool's start.
  - (a) *Chosen.* Pros: the codes take the state as fast as their assignments become definite, read
    on the material as it is (the 15:15 readings: the melting curve moves by up to 5 octaves as
    training sharpens the clusters); no melting points, windows or octaves; continuous wherever the
    sharpness is. Cons: steep where a split happens (D49's rate slows it there); new, untested.
  - (b) D25 with the melting curve re-measured every k epochs. Pros: D25's tested form. Cons: a curve a
    measurement (2-8 s), stepwise between them, and D30's window constant.
  - (c) No coupling (D38 at cap 0). Cons: D38's seed 3: the readout and the cell never train on the
    codes, so the snapped machine never answers right.
- **D51. No hard switch; a straight-through gradient at every temperature (replaces D8's hardness
  test, D19's tail and the levels).** The snapped factors stay soft for the rest of the run, at their
  profile's temperature, until each factor's sharpness first reaches 0.999 (its codes hard to 0.1% of
  windows), where its temperature stays. The soft codes' gradient to the cell is straight through
  (the crate's `straight`: `dh = dq` plus the commitment's, the softmax's term dropped), so with D25's
  split the cell gets `dq` whole at every coupling, and as `T` falls the soft path becomes the hard
  path's exactly. The codes keep their soft gradients.
  - (a) *Chosen.* Pros: no switch anywhere from the melt to the crystal; the cell is never starved
    (with the soft gradient its share of `dq` falls to `1 - l` while the softmax's term vanishes away
    from boundaries, near 0 at full coupling, then jumps to the whole at the hard switch); the
    straight-through estimator is the literature's (van den Oord et al. 2017). Cons: a crate build;
    the softmax's term (where moving `h` changes its code) no longer reaches the cell.
  - (b) The soft gradient, `T` to no floor, no hard switch. Cons: the cell starved at full coupling;
    `2/T` gradients near boundaries, unclipped (Adam, no clipping).
  - (c) Keep the hardness switch, read every epoch. Cons: a switch, and the gradient's jump.
- **D52. Growth continuous (replaces D26's switch from the crucible to the whole distribution).** The
  longest length `L` (real) starts at the crucible's and rises each epoch by `0.02 g c` in `ln L`, `c`
  the smallest factor's coupling, to the cap; an epoch's examples are the mixture
  `(1 - f) U(1..n) + f U(1..n+1)`, `n = floor(L)`, `f = L - n`, and its screen is
  `whole_lengths(n)`. At full rate S3 grows from 8 to 69 in about 108 epochs, move-to-front-4 to 248
  in about 172.
  - (a) *Chosen.* Pros: growth only as the crystal holds the state (coupled) and the screen stays at
    equilibrium (`g`); the material changes as slowly as the temperature does. Cons: a rate constant
    (0.02); growth can stall where `c` stays low.
  - (b) Growth once every coupling reaches 1. Cons: a switch again.
  - (c) The whole distribution from the cool's start (D36's arm 2). Cons: not the crucible; tested
    separately by the panel's arms.
- **D53. The readings every epoch (the owner: "shouldn't these readings be continuous?").** After
  the warm-up a step is one epoch, so every epoch ends with the step's readings: the certificate at
  both verdicts (D41) with its wrong states, wrong outputs and near ties apart, the snapped answers
  (`snapped_answers`), and the machine read off; each epoch's row logs every factor's sharpness, its
  hardness (D8's share, ungated), the coupling, `g`, the clock and `L`. The melting curve (D21) is read
  on the current network every 25 epochs (2-8 s a curve); the network is saved every 10 epochs and at
  every certificate and stop (each 2 MB). D35's stop at the exact bound reads every epoch.
  - (a) *Chosen.* Pros: a failure's mechanism read from the log; the time series `population` reads.
    Cons: each epoch's certificate and answers cost seconds (measured at the build).
  - (b) Readings offline on saved networks, as now. Cons: what the owner named: most of the path unseen.
- **D54. The continuous crystallisation's existence probe (`~/code/probe-d54.sh`, registered before
  its data).** word-problem-s3 seeds 0-7 (the crucible's design seeds: 4a'''' 8 of 8, D38 in flight)
  and move-to-front-4 seeds 0-3 (the crucible's control on seed 3 never certified; its whole runs
  certified seed 3), each D48's method from the crucible's cell, the commitment never back, D35's stop
  at the exact bound, D41's success; 1,600 epochs or 690 minutes a run.
  - (a) *Chosen.* Pros: the method on the seeds where the quasi-static crucible's rate is known, plus
    the reset machine the crucible failed. Cons: design seeds; a rate on fresh seeds waits for a
    matrix of the method that survives.
  - (b) Fresh seeds at once. Cons: existence first (the memory's rule): one working cell before a rate.
- **D55. The continuous crystallisation's fresh-seed matrix (`~/code/probe-d55.sh`, registered before
  its data, 2026-10-21, 20:50).** D54's existence bar met on its design seeds (S3 6 of 8 by then,
  move-to-front-4 seed 0 certified; its last two S3 runs still running, its reading to come), the
  owner asking for the night's spare capacity: D54's cell unchanged on seeds no run has touched,
  word-problem-s3 24-39 and move-to-front-4 24-31 (D54 used 0-7 and 0-3; the stopped core matrix
  (D46) 8-23, under other arms). Expectations: S3 at least 12 of 16 succeed (D41), refuted at 8 or
  fewer; move-to-front-4 at least 2 of 8, its strict-only certifications counted apart, 0 read, not a
  refutation; any false discovery refutes; each rate with its Wilson 95% interval, the mechanics and
  failures read as D54's. On `stone` (8 at once, stopped at 06:55 and resumed the next evening),
  `pyramidon` (3) and `klip` (as its heat test allows).
  - (a) *Chosen.* Pros: the rate D54 can't give (its seeds were the design's); runs of minutes make
    it a night's work; the same cell, so nothing in it is tuned on what it reads. Cons: run before
    D54's last two S3 runs end (they can't move its bar: 6 of 8 already) and before the margin
    question (near ties) is settled, which it measures rather than fixes.
  - (b) Wait for D54's reading and the margin decision. Cons: a night's machines idle for a reading
    that can't change the cell.
- **D56. The continuous crystallisation on an unstructured machine (`~/code/probe-d56.sh`, registered
  before its data, 2026-10-22, 00:40).** `jebel`'s four night cores freed by D38's S3 arm: D54's cell
  unchanged on random-q16-0 seeds 0-3 (D37's random cells, seeds 0 and 1: 0 of 6 at round 1's cell
  by extra codes and lost near solutions; the consolidation plan's control, B and A each 0 of 6).
  Feasibility: q 16, 12 codes a factor, m 144 (9.0 q), cap 159, width 32; row 15: depth 4 + k 1.
  Expectations: at least 1 of 4 succeed (D41), random-q16-0's first certificate in any arm; 0 of 4
  read, not a refutation, each failure classified by kind (extra codes against near ties); any false
  discovery refutes D35's stop on this backdrop; the mechanics read as D54's.
  - (a) *Chosen.* Pros: the method on discovery's likely target (the memory's rule: a working cell is
    checked on every backdrop), on cores otherwise idle; existence first, four seeds. Cons: before
    D55's rate; a backdrop whose failures so far were extra codes, which a schedule may not reach.
  - (b) More fresh seeds for D55. Cons: grows a registered matrix (the memory's rule: no growing
    matrices); D55's 24 runs already size its rate.
- **D57. The continuous crystallisation on S4 (`~/code/probe-d57.sh`, registered before its data,
  2026-10-22, 00:50).** `pyramidon`'s three slots freed by D55's slice: D54's cell unchanged on
  word-problem-s4 seeds 0-2 (seeds 0 and 1 D37's harm cells, certified before only with the
  commitment, which D48 never turns on). Feasibility: q 24, 15 codes a factor, m 225 (9.4 q), cap 248,
  width 32; row 15: depth 4 + k 1. Expectations: at least 1 of 3 succeed (D41), the harm cells' first
  certificate without the commitment; 0 of 3 read, not a refutation, each failure classified by kind
  (near ties against extra codes, D37's harm question), strict-only certifications apart; any false
  discovery refutes D35's stop on this backdrop.
  - (a) *Chosen.* Pros: the harm question asked of the method that dropped the commitment; a second
    backdrop beside D56's, on cores otherwise idle. Cons: three seeds, existence only.
  - (b) Division `p` 13 (the other harm cell). Cons: its 169 states make a run memory-heavy, past
    `pyramidon`'s three slots.
- **D58. The continuous crystallisation on the random control (`~/code/probe-d58.sh`, registered
  before its data, 2026-10-22, 02:50).** `klip`'s slots freed by D55's slice: D54's cell unchanged on
  random-q12-0 seeds 0-3 (D37's control backdrop; the general algebras' consolidation certified 2 of 3
  at about 12 states). Feasibility: q 12, 11 codes a factor, m 121 (10.1 q), cap 132, width 32; row 15:
  depth 3 + k 1. Expectations: at least 2 of 4 succeed (D41); refuted at 0 of 4 (the method loses
  what the earlier arms had on an unstructured machine); strict-only certifications apart; any false
  discovery refutes D35's stop on this backdrop.
  - (a) *Chosen.* Pros: D56's reading needs it (a failure on random-q16-0 means little if the method
    fails where the earlier arms didn't); cores otherwise idle. Cons: four seeds, a coarse rate.
  - (b) move-to-front-5 or S5. Cons: 120 states, memory-heavy on `klip`; existence on the smaller
    unstructured machine first.
- **D59. The margin stop (`--margin-wait`; D35's (c) and D39's (iii); the owner, 2026-10-22: "yes,
  register the waiting stop probe").** By 04:00, 12 runs of D54 and D55 had stopped at D35's first pass
  on networks exact at the strict verdict (0 wrong states and outputs, the target's size minimised) but
  not the default: each a near tie's branch into another state (1-13 wrong states, 2-188 near ties),
  so D41 counted none. Near ties alone don't decide it: the 10 certified stops had 2-125. What does is
  whether a tie splits the machine: the certificate now counts `split_ties`, the (state, window) pairs
  whose near ties branch into another state of its own minimised machine than the snap's, read
  without the target; on D55's 19 stopped networks it is above 0 on each of the 9 early stops and 0 on
  each of the 10 certified. The stop: a step D35's stop passes ends the run only when no tie splits
  and no label is a near tie, at most 200 steps after the first such pass ("waited"), that pass's
  network kept. Its probe (`~/code/probe-d59.sh`, registered before its data): the 12 re-run from
  their seeds (word-problem-s3 27, 29, 33, 35, 36 and move-to-front-4 24, 26, 29, 30 from D55;
  move-to-front-4 1, 2, 3 from D54), each bit-identical to its first run up to its first pass.
  Expectations: at least 8 of 12 succeed (D41), refuted at 4 or fewer; a run exact at 1.0 at its first
  pass whose stop's network isn't is a lost machine, refuted if the lost outnumber the gained; any false
  discovery refutes; read: the steps waited, the ties over the wait, the trained length.
  - (a) *Chosen.* Pros: target-free, as D35 is (discovery); it reads, on this data, exactly the
    default verdict's failures; a wait that ends (200 steps) so a run that never clears still stops.
    Cons: training past an exact machine can lose it (P1, 3b), which the probe counts; the rule's
    separation shown on 19 networks of two machines only.
  - (b) Zero near ties. Cons: every certified stop had some (2-125): it would wait past certificates.
  - (c) A fixed number of steps past the first pass (D35's (c)). Cons: blind to the margin it waits for.
- **D60. The margin deadline (`--margin-wait`'s wait, revised; the owner, 2026-10-22: "make the wait
  also count steps outside the pass").** D59's wait counted only the steps D35's stop passed: S3 35 and
  D54's move-to-front-4 3 passed twice with split ties, then never again, so neither run's wait ended
  (each past step 250, toward the 1,600-epoch budget). Now every step past the first pass counts; a run
  no pass cleared stops 200 steps after it ("waited") on the pass with the fewest split and label ties
  (each such pass's network kept, `best-pass-step-S.network.json`), restored and judged as the stop's
  in the run's last row (`end` "waited", `restored` its step); the margin stop's state is now `Margin`.
  A run whose pass clears inside the wait is D59's, bit for bit. No probe of its own: on D59's data it
  changes only the two runs, read from their logs and kept networks, and D61's runs use it.
  - (a) *Chosen.* Pros: every run's stop bounded (200 steps past its first pass); the stop's network
    is one D35's stop passed, the fewest ties being the nearest to a clean pass. Cons: a run that
    would clear later is stopped (none of D59's ten cleared past step 72 of the wait).
  - (b) The wait ending on the network as it stands. Cons: it may pass nothing (S3 35's didn't).
  - (c) The first pass's network (D59's kept one). Cons: ignores a pass with fewer ties since.
- **D61. The paced coupling (the owner, 2026-10-22: "i would like to test both b and c", each
  "dynamically calculated", no fixed constant).** The coupling, read straight from the sharpness
  (D50), jumps when a split moves it: every run that spent its budget (D54's S3 1, 4; D55's S3 24, 25;
  D56's random-q16-0 0-3) moved it by 0.1 or more in an epoch 58-201 times (largest 0.34-0.77), the
  certified ones at most 11 (largest 0.20); S3 35 lost D35's pass two steps after a rise of 0.23. D49's
  rate slows the clock when the screen drops, but not the coupling. Two arms, each with no scale of
  its own, in `Continuous` (off by default, the golden run unchanged):
  - (b) *Tested,* `--coupling-pace`: each coupling moves toward its sharpness's at most its allowance
    an epoch, the allowance its first rise, then times `2 ** (2 rate - 1)` (doubled while the screen
    holds, halved when the clock stops). Pros: the network's response sets the pace (a trust region);
    every control on D49's one signal. Cons: a long-stopped clock freezes the coupling, its retreat too.
  - (c) *Tested* (the owner: "swap c to savitzky-golay"), `--sharpness-smoothing`: the coupling read
    from the sharpness smoothed by a one-sided Savitzky-Golay filter of degree 1: its least-squares
    line over the readings of the cool's last e-folding time of clock (`halving / ln 2`, about 43 at
    halving 30), read at the clock's end. Pros: no lag on a steady rise, a jump spread over the time
    the temperature takes to move; offline on four runs' logged sharpness it removed every move of
    0.1 or more (largest 0.02-0.04) with a mean lag of -0.02 to 0.05. Cons: the one-sided line is
    noisier than a centred one, and damps a jump less than an average would.
  - (c') The moving average first proposed (gain `rate ln 2 / halving`). Cons: offline, it held the
    coupling 0.16-0.53 below the sharpness's, and froze it with the clock (random-q16-0's).
  - (a) A fixed cap on the move (0.05, the pre-continuous bound). Cons: a constant (the owner: "we
    can't [use] a fixed constant").
  Its probe (`~/code/probe-d61.sh`, registered before its data): both arms, D54's cell with D59's
  margin stop under D60, on the failures word-problem-s3 24, 25, 35, random-q16-0 0, 1 and
  move-to-front-4 3 (lost its pass with only 6 such moves: the jumps' counter-case) and the clean
  word-problem-s3 26 and move-to-front-4 24, 16 runs on `jebel` (8 at once by day, a queued run
  launched after 18:00 only below 4), each control its seed's earlier run.
  Expectations: each arm's coupling moves below 0.1 an epoch in every cool, fewer such moves than its
  control in each failure; an arm's mechanism reads as promising if it rescues (D41) at least 2 of the
  6 failures and loses neither clean seed, refuted if it rescues none or loses both; any false
  discovery refutes the stop; D60: every run ends at a clean pass or 200 steps past its first; read:
  rows up to the cool's start against the control's (bit-identical but for the arm's fields).
  Prediction from the failures' order (written before the data: the first coupling move of 0.1 or
  more against the screen's first fall below 0.9): jump first in S3 35 (outside the probe, also in
  random-q16-0 2, 3 and D54's S3 1), the same epoch in S3 25 and random-q16-0 0, 1, drop first in S3 24 (the
  screen at 0.74 with the coupling 0, the cool's start) and D57's S4 0 and 2 (a collapse at a steady
  0.97, the trained length 16.6-18.8, after which the rate held the clock at 0 for 1,400 epochs). An
  arm should rescue jump-first and same-epoch failures, not S3 24 (the counter-case); every failure
  shares the frozen clock after its drop, which neither arm changes.
  Diagnosis, not adoption (the owner, 2026-10-22: "only a clear majority of them. if we tinker to
  every seed we will over-tune the method ... the final method [is] statistical"): this probe reads
  whether an arm removes the jumps and what that does on chosen failures. An arm is adopted only on a
  rate over fresh seeds against the method's current rate (D54's cell with the margin stop on D55's
  seeds: S3 11 of 16, move-to-front-4 7 of 8 at D59's reading), in a batch registered after this
  reading, if its mechanism reads as promising.
- **D62. The run's state, one object (the owner, 2026-10-22: "do we not need a restructure (maybe DTOs
  or request objects) to simplify typing ? this is now becoming a recurring problem").** `run` is one
  function of about 700 lines carrying some 40 locals from step to step; past pyright's code-flow
  limit its inferred types silently become partly Unknown (twice now: D59's first build, D60's, each
  fixed by moving logic out). `Place` already mirrors those locals for the checkpoint, by hand, so
  every new switch touches three places (the local, `Place`, the checkpoint's call).
  - (a) *Chosen,* after D61's launch, its own PR: the loop-carried state one dataclass (the run's
    state, `Place`'s fields as one object that is checkpointed as it stands), the loop's body split
    into functions of it (the epoch's setting and training, the step's end and stop, the warm-up's
    end), each a request in and a result out. Pros: types stay inferred as the method grows; a new
    switch is one field; the checkpoint can't drift from the state. Cons: a large mechanical change
    to the most-run code: checked as a pure refactor (the golden run and every resume test
    bit-identical, a D54 cell's run row for row).
  - (b) Raise pyright's limits or silence its reports there. Cons: hides the unknowns rather than
    removing them; the three-place mirror stays.
  - (c) Split `modular_family.py` into modules first. Cons: moves the problem; `run`'s flow is the cause.
- **D63. The collapse: reheat or resample (planned; the owner, 2026-10-22: "yes, plan d63").** Every failure of D54-D59 shares
  one end: once the screen falls, D49's rate holds the clock at 0, the temperature stays where it was
  and the coupling, read from a sharpness that stays high, holds a wrong, nearly hard machine for the
  rest of the budget (D57's S4 0 and 2: a collapse to 3-17 states at a steady coupling 0.97, then
  1,400 epochs frozen; S3 24: the screen at 0.74 at the cool's start). Nothing in the method can undo a
  collapse. The rate extended below the hold's level would run the clock back: the temperature rises,
  the codes soften, the sharpness and so the coupling fall, and the cool resumes once the screen
  recovers.
  - (a) *Planned.* The rate's line extended below `low` to -1 (at `2 low - high` times the level, 0.91
    at the defaults), the clock moving back by it, never past the cool's start; the held temperatures
    (D51) released where the clock passes back over them; growth only forward. Pros: continuous, on
    D49's one signal, no constant of its own; a collapse becomes a local re-melt rather than the run's
    end. Cons: a run can cycle (reheat, cool, collapse); the epochs it spends; S3 24's drop at the
    cool's start leaves no room to run back (the hold's).
  - (b) A re-melt to T_hot on collapse. Cons: a switch (D48 removed them); it discards what formed.
  - (e) *Planned beside (a)* (the owner, 2026-10-22: "add the kill-and-resample option"): end a run
    whose clock has held at 0 past a window (its collapse read from the rate, D49's signal, over the
    cool's e-folding time) and start a fresh seed in its slot. Pros: the method is a batch (path
    dependence, D41's rate): a doomed run's budget goes to a new sample (D57's S4 0 spent about 1,400
    epochs frozen); no change to training, so nothing tuned to a seed. Cons: discards a run that might
    have recovered; needs the batch runner, not the run; selects for paths that never collapse, so
    it may discard a category of solutions only a collapse and recovery reach (the owner, 2026-10-22:
    "the reheat may be worth it, otherwise we may be discarding a category of solutions that we may
    need"), likeliest on the larger machines, where collapses are commonest. (a) and (e) compared on
    cost per certified machine (epochs, or minutes, over certificates) on the same fresh seeds, and on
    which runs each certifies: a certificate reached only after a reheat (a collapse, then recovery)
    is the category (e) loses, read apart, by backdrop (random-q16-0 and S4 first).
  - (c) Restore the best step's network (`--restore-best`, the guard). Cons: a discontinuity, and the
    same training from there can collapse again.
  - (d) Lower only the coupling when the screen falls (D61 (b) with its rate signed). Cons: the codes
    stay hard at the frozen temperature; half the move.
  Build after D62 (on the run's state object), tested, the golden run unchanged without it; its probe
  registered after D61's reading, on the drop-first failures (S3 24, D54's S3 4, D57's S4 0, 2) and
  the jump-first ones, with D61's arm if one is adopted: a diagnosis; adopted only on a fresh-seed
  rate against the method's current one (a clear majority certifying is the aim, not every seed).
- **D64. The method's fresh-seed rate (`~/code/probe-d64.sh`, registered before its data, 2026-10-22,
  09:00; the owner: "you can run stone at a half-load today").** The method is statistical, so a
  remedy (D61's arms, D63) is adopted only on a fresh-seed rate against the current method's, which is
  known only from D59's re-runs of D55's seeds (S3 11 of 16, move-to-front-4 7 of 8); and "in general"
  covers every backdrop class, where the larger and unstructured machines are the gap (random-q16-0 0
  of 4, D56; random-q12-0 1 of 4, D58, read 08:58; S4 1 of 3, D57). The current method unchanged
  (D54's cell, `--margin-wait 200` under D60, no D61 arm) on untouched seeds: word-problem-s3 40-47,
  move-to-front-4 32-35, word-problem-s4 3-5, random-q16-0 4-6 on `stone` (6 at once by day, 10 at
  night; stopped at 06:55 Friday, resumed that evening), random-q12-0 4-7 on `klip` (3 at once); 22
  runs. Feasibility as D54's and D56-D58's. Expectations: each backdrop's rate (D41) with its Wilson
  interval; S3 and move-to-front-4 together at least 8 of 12 (the working rate), refuted as working
  in general on them at 6 or fewer; the larger and unstructured backdrops read, not barred (their
  rates against S3's: the scaling question); any false discovery refutes the stop; read: each
  failure's kind and order (jump or drop first, the frozen clock's epochs), the margin's waits, the
  deadline's stops, the seconds to each certificate. Its seeds pair any later arm's batch.
  - (a) *Chosen.* Pros: the comparison every adoption needs, on fresh seeds; weights the gap; the
    half-load `stone` and the free `klip` today. Cons: a coarse rate per backdrop (3-8 seeds).
  - (b) S3 and move-to-front-4 only, 24 seeds (a sharper working rate). Cons: silent on the gap.
  - (c) Wait for D61's reading, then one batch with its arm. Cons: no current rate to compare it to.
- **D65. The order-biased cool (planned 2026-10-22, 10:45, set aside at 11:05 on its stage 0, below; the owner: "plan this work in detail, and
  schedule it immediately after the completion of our current planned work").** The scaling
  conjecture's reading (below) puts the random machines' failures at forming the state, their melting
  curves the most crowded, while every ordered machine forms at today's pace (S3 19 of 24,
  move-to-front-4 11 of 12, S4 3 of 3; S4's losses are the collapse, D63's). The owner: keep the
  quasi-static curve and bias only its pace by the machine's order, read from answers alone; a
  correction that slows random machines and vanishes for ordered ones ("where we seem to be spot on
  already"); a constant only if derived from theory, ideally read dynamically. The mechanism: the
  cool's halving time (D49) becomes `halving x (1 + w (s - 1))`, its shape and gate unchanged:
  - **w, the order weight (once, at the run's start, answers only):** the answer's entropy given the
    last k tokens (k = 0-6) on the run's own uniform words and answers (`mf.answer_profile`), against
    an ensemble of random machines of the state bound's size and the machine's tokens and classes
    (`mf.order_against_random`). Read offline on the five targets (60 random machines each): rms z
    S3 3.34, move-to-front-4 6.69, S4 3.31, random-q12-0 0.31, random-q16-0 1.10; the ensembles'
    own medians 0.68-0.93 and maxima 2.46-4.39. A tail probability is no weight (uniform for random
    machines: random-q16-0 0.28); the candidate is a ramp read from the ensemble alone, 1 up to its
    median and 0 at its maximum (S3 0, move-to-front-4 0, random-q12-0 1, random-q16-0 0.85), but S4
    reads 0.29 (one 24-state random machine at 4.39 stretches the envelope): the statistic is stage
    0's to settle. w exactly 0 leaves a run bit-identical to today's.
  - **s, the slowdown's size (dynamic, from the run):** constant thermodynamic speed (Salamon and
    Berry; Nulton and Salamon): an interval of the cool gets time in proportion to what it must
    release. `s = max(1, splits x tau / halving)`, the splits those of the warm-up's last melting curve
    (D21, D53) in its densest octave (the factors' largest), tau the network's relaxation time in
    epochs (stage 0 picks its reading: the hold's epochs to its best screen, or live in the cool as the
    epochs from a clock move to the gate's return to 1, only ever raising s: the adaptive cool, D33,
    failed by speeding up on a live reading).
  - **Stage 0, offline (allowed now: existing data; the owner's rule):** (0a) the order statistic on
    the catalogue's machines to 169 states (S3-S5, move-to-front-4 and 5, division p 5-13,
    random-q12 and random-q16 seeds 0-9, random-4) with ensembles of 60, at the bound and 1.1 and 1.5
    times it; criteria registered now: every algebra w = 0, at least 90% of the random machines (not
    the ensemble's) w >= 0.8; the candidates rms z, |drop z| and the whitened (Mahalanobis) distance,
    the first to meet both chosen; none meeting them goes to the owner. (0b) s and tau on the 47
    completed runs' logs: if s alone is 1 on every ordered run and above 1 on the random ones, w is
    dropped (option (c)). (0c) the cool's length under s on random-q12-0 and random-q16-0 against
    the budget (1,600 epochs, 690 minutes), the budget derived from it if short. (0d) feasibility of
    the new random machines (docs/information-flow.md's register: codes, exclusion bound, width).
  - **Stage 1, the build (code only, while the current work runs):** `--order-bias` (with
    `--continuous`): an `order` row at the run's start (profile, statistic, w), s fixed at the cool's
    start, each anneal row's `halving`, `w`, `s`, `tau`; `Place` fields, an identity key, a resumable
    kind; a test that w = 0 leaves a run row-identical to the control; strict pyright, the full suite,
    the golden run; its PR, CI, merge.
  - **Stage 2, the probe (registered before its data; launched when the current planned work has
    ended: D61 read, D63's probe and D64 ended, D59's last run):** random machines only (an ordered
    machine's w = 0 is checked in stage 1, no run): random-q12-0 seeds 4-7 and random-q16-0 seeds 4-6
    under the bias, paired with D64's controls on the same seeds; random-q16-1 and random-q16-2 (not
    yet run: not one machine's tuning) seeds 0-2, arm and control; 19 runs, S4 seeds 3-5 added (paired
    with D64's) only if stage 0 leaves S4 a w above 0, as a harm check. Expectations (to fix at
    registration): formation and certificate rates, arm against control; promising at the arm forming
    on at least half its runs with the controls at a quarter or fewer, refuted at an arm rate no
    better than its controls'; any false discovery refutes the stop; read: w, s, tau, the cool's
    length, cost per certified machine (with D63 (e)'s kill offline). A diagnosis; adoption only on a
    fresh-seed rate. Machines: `jebel` (at most 4 at night), `klip`, `stone` on weekday nights,
    `pyramidon`. Expected start Saturday 2026-10-24 (after D64's Friday night on `stone` and D63's
    probe), read Saturday evening to Sunday.
  - (a) *Chosen:* w and s together. Pros: what the owner asked: one consistent mechanism for both
    limits, answers-only, the curve kept; ordered runs unchanged exactly; s derived and dynamic, w
    read from an ensemble, no fitted constant. Cons: two readings to validate; w needs the state bound
    (a loose bound shifts the ensemble: stage 0a reads it); an ensemble of 60 costs seconds a run.
  - (b) w with a slowdown fixed from theory. Cons: a constant, and nothing in hand derives its size.
  - (c) s alone. Pros: one reading, fully dynamic. Cons: S3's cascade is as crowded as random-q16-0's,
    so it may slow ordered machines that need no slowing; chosen instead of (a) only if stage 0b
    shows s collapses on ordered runs by itself.
  - (d) More seeds instead (the owner's conjecture). Cons: random-q16-0's four plateaus point at the
    pace, not the sample; the owner: the sampling unchanged for now.
  - **Stage 0 read (11:05): the slowdown set aside** (the owner: "write it up that way"); the freeze
    (findings, below) is the common failure, D63's target. (0b) s is about 1 on every run, random or
    ordered (at most 1.6, an S3 run; random-q16-0 1.0-1.17): before the freeze the gate's holds last
    1-5 epochs against a halving of 30, so the cool does not outrun the network; and the frozen runs
    sat at one temperature for about 1,400 epochs, an infinitely slow cool there, without forming.
    (0a) No statistic meets the criteria: |drop z| puts every algebra (S3-S5, move-to-front-3 to 5) at
    w = 0 at every bound, but 16 of 22 random machines at most reach w >= 0.8 (73%, at 1.1 times the
    bound; 15 of 22 at the bound), and at 1.5 times the bound 5 of 22; rms z and the whitened distance
    do worse; random-perm-4-0 (every token a permutation, a group's action) reads ordered under all
    three, rightly. The order reading separates the algebras reliably; a weight that is 1 on random
    machines does not follow from it, and a loose bound blurs it. Kept as tools (`order`,
    `formation`); nothing built on them. Revisit only if the reheat (D63 (a)) fails on the random
    machines.
- **D66. The two-phase cool: quasi-static in the order too (planned 2026-10-22, 11:20; the owner:
  "we need to split the cool into two portions - an initial slow cool ... and a second quasi-static
  cool like we already have ... the ordered machines only need the second phase"; "yes" to its
  plan).** Replaces D65's mechanism. The onset reading (findings, the freeze, below): the state begins
  to form (state share first >= 0.6) at 2.2-3.1 times the factors' melting point on move-to-front-4
  and mostly 1.7-2.3 on S3, but only at 1.0-1.4 on the random machines, which then freeze at about 1
  before forming: they start ordering late and run out of room. Today's gate (D49) is quasi-static in
  the screen, which a network answers right on short words while its state is still disordered; the
  cool is not quasi-static in the state's order. The owner's aside (raise the melt for random
  machines) answered: the melting point is measured, not set, the start is already 2-5 times it, and
  in an exponential cool a hotter start adds epochs only at the top, not around the melt.
  - **The mechanism:** a second gate on the clock, `rate = min(g_screen, g_order)`, the order read
    from answers alone: `A`, the snapped machine's accuracy at the longest answers length (48; already
    read at every step's end for the answers-only stop, D35, on the same fixed words each step, so a
    change is the network's, not the sample's). `g_order = clip(1 - rise / se)`, `rise` the epoch's
    increase of `A` (falls ignored: a fall is the screen gate's), `se = sqrt(A (1 - A) / n)` the
    standard error of `A` over its `n` answers: the clock runs at full speed while the order is
    settled (flat: not yet begun, or formed) and slows while it is still rising faster than it can be
    told from its own sampling. No new constant: the scale is the reading's own error. Phase 1 is
    then not a schedule but whatever stretch of the cool the order rises in: an ordered machine that
    has formed above the zone meets it only briefly (its rise passes early, at 2-3 times the melt), a
    random machine's begins at the melt, where it is slowed.
  - **Stage 0, offline (now: existing data):** (0a) `g_order` replayed on the 47 completed runs'
    logged `A` (step ends): where it would slow each (temperature over the melting point), for how
    many epochs, how much of an ordered run's cool it would stretch; (0b) `n` (the answers behind
    `A`) and whether the step-end `A` is per epoch in the cool (the cool's steps are one epoch); the
    cost of reading `A` every epoch (seconds); (0c) the budget under the replayed slowing; (0d)
    feasibility of random-q16-1 and -2 (docs/information-flow.md's register). Criteria registered now:
    the replay slows every random run between its onset and its freeze; on ordered runs it adds at
    most a quarter to the cool's epochs before formation; else the owner sees it before stage 1.
  - **Stage 1, the build (code only, while the current work runs):** `--order-gate` (with
    `--continuous`): `A` each cool epoch, `g_order` and `rate` in the anneal row; `Place` fields, an
    identity key, a resumable kind; tests: the gate's formula, and off leaves a run bit-identical;
    strict pyright, the full suite, the golden run; PR, CI, merge.
  - **Stage 2, the probe (registered before its data, after D63's reading; expected Saturday
    2026-10-24 once D64 and D63's probe end):** D63's reading sets its arms: if the reheat re-freezes
    the random runs at the same point on its re-cool, the pace through the zone is implicated (for
    D66); if the reheat certifies them, D66's probe is a diagnosis of a second route. Cells: the gate
    on random-q12-0 seeds 4-7 and random-q16-0 seeds 4-6 (paired with D64's controls and D63's
    reheat), random-q16-1 and -2 seeds 0-2 with and without (12 runs: not one machine's tuning), and a
    harm check on S3 seeds 40-43 and move-to-front-4 32-33 (paired with D64's): 25 runs. Expectations
    (fixed at registration): promising at the gate forming on at least half its random runs with
    their controls at a quarter or fewer; refuted at a random rate no better than the controls';
    harm: the ordered arm losing more than one of six against its pairs; any false discovery refutes
    the stop; read: where `g_order` slowed each run (temperature over the melting point), the epochs
    added, the onsets and freezes, cost per certified machine. A diagnosis; adoption only on a
    fresh-seed rate, the gate with the reheat as the candidate method.
  - (a) *Chosen:* the order gate. Pros: answers-only; one reading already taken at every step; no
    order weight (D65's stage 0 found none reliable); slows only while the order rises, so random
    machines are slowed where they start; derived scale. Cons: slows ordered machines during their own
    formation too (harmless in principle; the harm check reads it), not exactly zero for them; reads
    `A` every epoch (cost: stage 0b).
  - (b) A fixed slow band around the melting point (from the warm-up's melting curve). Cons: a band's
    width is a constant; slows ordered machines that have long formed.
  - (c) The order weight (D65) on the pace. Cons: no weight met its criteria.
  - (d) A hotter start (the owner's aside). Cons: adds time only at the top of an exponential cool.

### The consistency review settled (2026-10-21, afternoon)

The consistency review ([consistency-review-2026-10-21.md](consistency-review-2026-10-21.md)), its
items 1-8, settled under the owner's delegation ("go ahead and settle them under your delegation"),
before the panel's (4b) registration and before the coupling probe's (D38) data.

- **D40. The panel's verdict (item 1): D12 replaced for 4b.** Each arm is accepted as the base
  candidate when it certifies more failure-A runs (D11's S3 seeds 0, 1, 4, move-to-front-4 seed 3,
  move-to-front-5 seeds 0, 1) than their controls; a lost harm certificate doesn't reject it, but
  sends that cell to D31's diagnosis (near ties against pairs per code, D37's expectation), whose
  remedy is tested before adoption; rejected when it moves no failure-A run. Fresh S3 seeds 8-15
  measure `p` against a bar (at least 4 of 8, as D33's), not a control. The random cells get a
  control: round 1's cell at width 32, the option off (`--fork`, the commitment on, as C0), once
  each, beside the panel (3 short runs). Adoption as the general method stays the owner's (6 seeds a
  backdrop).
  - (a) *Chosen.* Pros: consistent with D31 (the panel diagnoses) and with D37's registered harm
    expectation, which under D12 would reject option 1 on a loss the programme already predicts;
    a failure-A gain is still required, so an option that moves nothing is still rejected. Cons:
    an option can be accepted while it harms, pending the remedy (a base candidate, not adopted).
  - (b) D12 as written. Pros: the rule set before option 1's first panel. Cons: rejects on the
    expected loss, before the diagnosis D31 asks for.
  - (c) The commitment back on the harm cells only. Cons: a method chosen per backdrop, against one
    general architecture.
- **D41. One success definition (item 5): a run succeeds when the network it stops on certifies at
  both verdicts** (the default snap ratio, near ties branched, and 1.0), as D12 and `population`
  (`VERDICTS`) count; a stop exact only at 1.0 is reported ("exact as it computes"), not counted.
  - (a) *Chosen.* Pros: every earlier rate's definition, so the panel compares with them; the
    default ratio's branching is the robust proof. Cons: under D35 a run stopped on a network exact
    at 1.0 counts as a failure (4 of the re-reading's 8 earlier stops were of that kind; reported).
  - (b) 1.0 counts. Pros: the machine is right as the network computes. Cons: breaks comparability;
    near ties are the margin option 2 is shelved for, so they should show.
  - (c) The stop's later certificate counts. Cons: under D35 the run stops; nothing later exists.
- **D42. The exclusion bound as trained (item 2).** The register's row 2 is a property of the cell's
  config (growth trains to `m + q - 1`), not of a run that certifies before growth; each certified
  run's longest trained length is reported beside it (from its log), and a claim says "certified,
  trained to length L". Under D35, a stop below the bound rests on the state bound and sampled
  answers (a screen), with the certificate the judge. The exact answers-only form is Chow's test set
  from the read machine under the bound (machine-discovery.md's check 4: with `N = n` it is the
  transition cover times a characterization set, small): built as stage B's first tool, run offline
  on the panel's stops when it is read.
  - (a) *Chosen.* Pros: no claim says more than the run did; the panel isn't delayed; the exact
    check arrives as a durable tool with the panel's reading. Cons: the panel's stop stays a screen
    until then.
  - (b) The crucible's lengths to the bound (69 on S3). Cons: undoes the crucible (D26, D28).
  - (c) Chow's test set as the stop, built before the panel. Pros: an exact answers-only stop at
    once. Cons: a new tool, untested, delaying the panel; D35's re-reading found no false discovery
    at the exact bound, so it can follow.
- **D43. The exact state bound (item 4): the panel stops at `N = q`, the target's size, and its
  claims say "under an exact state bound".** The slack's cost is read offline on the panel's saved
  step networks (`answers-stop` at 1.1 and 1.5 times the size, as the re-reading did); the
  population's selector under a slack bound (runs stopping on different machines) goes to stage C.
  - (a) *Chosen.* Pros: the bound the re-reading validated; slack read for free from the same runs.
    Cons: the best case for discovery; a slack bound's false stops would have ended runs the panel
    reads past.
  - (b) The panel at a slack bound. Cons: its false stops end runs early, confounding option 1's
    failures with the stop's.
- **D44. The crucible's cap (item 3): after the panel, as D39's (iv).** The cap set from the state
  bound (row 15's depth and `k`, each at most `N - 1`) or the dependence stated as a known-structure
  assumption, decided by the panel's crucible arm. For the panel, row 15's check stands as a
  feasibility check (the target known).
  - (a) *Chosen.* Pros: the panel's crucible-against-whole arms (D36) say first whether the crucible
    is kept at all. Cons: the panel's crucible is the S3-tuned 8.
  - (b) The cap from the bound now. Cons: at division's 169 the worst case is the curriculum's whole
    range, not a crucible.
- **D45. D38's reading and D34's order (items 6-8; registered here before D38's data, 13:00).**
  D38's reading adds, per run: whether D29's gate held and how often (at cap 0 it reads the
  continuous network), and the snapped answers on each step's network (`answers-stop --snapped`)
  against the logged screen; whichever way it comes out, both accounts (D25's, the re-reading's)
  written up. The bounded handover's probe (D34's probe) runs only if D38 is refuted on S3 (read by
  about 20:00); if D38 holds, it is set aside unrun, recorded.
  - (a) *Chosen.* Pros: a failure can be told apart from the blind gate; no `pyramidon` night on a
    refinement of a part D38 may shelve. Cons: D34's probe waits an evening.
  - (b) D34's probe now, as the handover had it. Cons: may answer a question the shelf has closed.

- **4a''''. The quasi-static crucible probe:** word-problem-s3 seeds 1, 2, 3, 4, 6, 7, the cell and
  profile of 4a''' with D28-D30, on `stone` from about 19:30 (13 workers on its 12 cores until 4a''' ends);
  registered in `probe-d29.sh`'s header before its data. Seeds 0 and 5 carried over from 4a''' (the
  same runs, above), so the population is 0-7.

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

## Findings: stage 4a', the shapes probe's warm-ups (2026-10-20)

`probe-4a2.sh` at 45f7c51 (D22-D24, the abrupt snap still at the warm-up's end): started 12:15, stopped
at 12:53 at the owner's direction once its warm-ups were read; D25 superseded its anneal. Data in
data/window-recurrent/state-formation/option1/probe-4a2/ (`s3/`, `mtf4/`).

- **The no-curriculum warm-up on move-to-front-4 seed 3:** screen 1.0 from its first block (a block an
  epoch from step 2), its continuous states 50-55 a factor through step 5 and 75 at step 6, so D14 ended
  the warm-up at step 6 (60 needed); 15 of 15 codes a factor in use, `d_eff` 8. Its states carry the
  target's at every length (`--live`, joint nearest neighbour 1.00 at lengths 4-128, chance 0.05-0.19;
  each factor alone 0.91-0.99): the long words the curriculum held back cost the warm-up nothing here.
- **Its material:** whole critical temperatures 1.48 and 1.76 (`T_hot` 3.0, 3.5), melting points 1.10
  and 1.30, last transitions (14 to 15 groups) 0.029 and 0.020: about 5.5 octaves, against S4's 11 (4a).
- **Word-problem-s3 seed 0 was still in its warm-up at the stop:** step 1 ended on its plateau (20
  epochs, screen about 0.7, 28 distinct states a factor against 32 needed).
- The three shapes are one run until their cool starts (the shape only sets the cool), so a probe's
  three arms read alike through the heat and the hold.

## A flaw in 4a'', fixed by resumption (2026-10-20, 14:05)

D22's blocks still ran from step 1 to the cap (the curriculum's last length): with no curriculum a block
isn't a length, and a run whose screen reached 1.0 passed a block an epoch, so move-to-front-4 seed 3
(screen 1.0 from its warm-up's block 5) would have ended at block 248, about 240 epochs after the
warm-up, before its cool (626 epochs) and, for the s and concave shapes, before the coupling reached 1;
S3 at block 69. Fixed (3ac2ada: blocks go on until the budget or a certificate). The six runs were
stopped at 14:05-14:08 (mtf-4 at blocks 31-36, at the cool's start; S3 in its warm-up, block 4) and
resumed from their checkpoints at 14:09-14:10: a resumed run is the uninterrupted one by bits up to the
stop (the consolidation workplan's stage 3), and the fix changes nothing before the cap, so no setting
changed after the data; the registered expectations stand.

## Readings: stage 4a'', the handover probe, in flight (2026-10-20, 15:00-17:00; not a verdict)

`probe-4a3.sh` (resumed as `probe-4a3-resume.sh`, 3ac2ada; expectations in its header): S3 seed 0 on
`pyramidon`, move-to-front-4 seed 3 on `stone`, each shape. The commitment never returns
(`--anneal-never-commit`, D24); the coupling runs log-linear in `T` from each factor's melting point to
its last transition (D25). Readings from the logs only; the separation through the cool is read when the
runs end (the header's rule).

- **Move-to-front-4's warm-up as 4a''s** (step 6, the same material), the phases by block: heat steps
  7-11 (`T` 0.05 to 3.0), hold 12-31 (`T_hot`), cool from 32.
- **The k-means snap at the warm-up's end nearly certified:** the snapped network (the codes read, the
  forward path still continuous at coupling 0) at step 7 share 0.94, 39 of 67 pairs right; at step 8
  (`T` 0.09-0.11, early in the heat, codes 15 and 15) share 1.00, 43 of 43 pairs right, uncertified
  (minimised 32 states against the target's 24). Unlike 4a's S4 (snap 0.82 and 0.25 joint separation),
  here the no-curriculum warm-up's states snap almost into the target machine.
- **The melt took it away, on the codes alone:** the screen held at 1.0 through the heat and the hold
  (the header's first expectation met: the forward path didn't melt), while the codes merged (groups 15
  to 12, 9 at the hold's start, 4-8 through it) and the snapped share fell to 0.64-0.87, pairs right 0.
- **The cool, so far (steps 32-52, coupling still 0):** the codes split back above the melting points
  measured at the warm-up's end (convex at `T` 1.43 and 1.70: 14 and 11 groups; the curve said 1.10 and
  1.30), so the material has moved since. Convex, the fastest early fall, leads: share 0.94, codes 15 and
  15, 4-6 of 77-80 pairs right (steps 44-48); s 0.92 from step 49; concave about 0.79, still near the melt.
- **Move-to-front-4 convex certified at step 104** (16:00; 168 min): minimised 24 states, the target's,
  51 of 51 pairs right, `T` 0.11 and 0.13, coupling 0.63 and 0.55, still soft, the commitment never on;
  the run stopped on it (D19). The first certificate of option 1. The cool re-formed the structure the
  melt erased and bettered it (step 8's snap minimised to 32).
- **Separation through it** (`state-separation`, each length alone; `option1/separation/separation.txt`):
  the continuous state joint 1.00 at lengths 4, 16 and 64 at every step read (6-104), through the heat,
  the hold and the coupling's rise; snapped, joint at length 16 1.00 at step 8, 0.74 at the hold's end
  (31), 0.87 at 48, 0.93 at 63 (coupling 0.11), 0.98 at 80 (0.30), 1.00 at 95 (0.49) and 104. The
  header's expectations: the screen held (met), joint >= 0.9 at length 16 as `l` rose (met), a state
  share >= 0.5 and a certificate (met); each factor's alone toward 1.00 **not met** (snapped centroid
  0.74-0.86 throughout): the certified machine holds the state jointly.
- **What each factor holds** (`approximations`, built for this reading; exact over the pairs the start
  reaches; `option1/approximations/`): neither factor's partition is a congruence, but the finest
  approximation each determines has 10 and 7 blocks, and refines named congruences of move-to-front:
  factor 0 the relative order of items {0, 1}, {0, 3} and {2, 3}, factor 1 of {0, 2} and {1, 2}. Each
  relative order of two items is a flip-flop (access one, it leads; access the other, it trails; else
  hold): an LRU's age matrix, five of its six bits held exactly, the sixth ({1, 3}) only jointly (the
  approximations' meet 21 of 24 states, so not a cover). Each factor's code nearly a machine of its own
  (autonomous 0.96, 0.88).
- **The same reading on 13 earlier certified mtf-4 networks** (window branches b3, the never probe,
  one-code-one-state C0, consolidation P1, general algebras round 1; the curriculum and, but for the never
  probe, the commitment): every factor that determines a named congruence determines a relative order,
  of two items or of three (an LRU of three items, 6 blocks, in 6 of the 13). Meets 2-24: one cover
  (general algebras round 1 seed 1: an LRU of {0, 1, 3} in factor 0 and an 8-block congruence in factor 1,
  both exact congruences, meet 24: the target a subdirect product of the two); then 4a'' (21), the never
  probe (19), C0 seed 5 (18); the rest 2-12. So relative-order approximations are what this
  architecture forms on move-to-front, whatever the method; the anneal's certificate is among the most
  decomposed, not unique in kind.
- **S3** (all three shapes one run until the cool): the warm-up reached its cap (step 8) at best share
  0.22; the hold at `T` about 5.6, codes melted to 2 groups, screen 0.53-0.55 (as the warm-up's); the cool
  begun at 16:30. Its epochs take about 52 s on `pyramidon` (three at once), so the budget runs to about
  noon on 2026-10-21.

## Readings: the coupling probe's preview and the moving material (2026-10-21, 15:15; before D38's data)

Two readings of existing data, the quasi-static crucible's saved step networks (4a'''', seeds 1, 3,
4, 6, 7; seed 2 certified in the heat), while the panel's timing ran; written before the coupling
probe's (D38) data (its runs launch at 19:00). Data and script in
data/window-recurrent/state-formation/option1/readings-2026-10-21/.

- **The coupling probe's preview: existing data can't separate the coupling from the cooling.** At
  each run's last step at coupling 0 its codes hadn't reached their final groups (seed 4 [2, 7]
  against [8, 8], seed 1 [7, 2], seed 3 [2, 3] against [4, 6], seed 6 [5, 6]); D25 starts the
  coupling at the melting point, where the splits begin, so the two move together by design. Through
  the hold and the cool at coupling 0 the snapped machine is at chance (snapped answers 0.43-0.49 at
  length 8, joint separation at 16 0.23-0.31 against 0.20: the codes melted); it improves only as
  coupling and splits advance together, and certifies at couplings 0.12-0.84. **One weak clue:** seed 7
  reached its final groups ([3, 4]) at step 59, coupling 0.02, and its snapped machine stayed near
  chance through step 64 (coupling 0.08, answers 0.54 at length 8), exact by 67 (0.12): the right
  groups alone weren't the machine; the coupling's shaping, or more training, completed it. The
  completion is abrupt in seeds 3 and 7 (about 0.6 to exact in 3 steps), gradual in 1, 4, 6 (20-50
  steps). So D38 is the only test; this reading leans, weakly, against its registered expectation
  (at least 6 of 8 at cap 0). `coupling_preview.py` (snapped answers read uncoupled, `mf.snapped_answers`,
  and `state-separation` at length 16, at about a dozen steps a run).
- **The splits come early because the material moves (D32's open item, answered for S3).** The
  melting curve (D21) re-measured with `melting-curve --step 8` on later step networks tracks
  training's splits; the warm-up's curve doesn't. Seed 4: the warm-up's curve puts factor 0's full
  split (8 groups) at T 0.015 and factor 1's never; on step 50's network 0.79 and 0.77, on step 69's
  0.50 and 1.74, where training reached 8 groups (step 69) at 1.05 and 0.63. Seed 7: training's final
  3 groups in factor 0 at T 3.58 (step 56), the curve on step 59's network 3.58 (the warm-up's 1.22);
  factor 1's 4 groups at 1.72 in training, 1.1 on step 59's network. So the curve on the current
  network is within 0-1 octave of training (about 5 of the 6 octaves removed); the remaining octave
  or less is the thermal noise or the curve's own lag (critical slowing, D21). As training sharpens
  the states into clusters, their splits move hotter. **Bearing:** the coupling window (D25, D30) and
  the adaptive cool's step are tied to a curve measured once, on a material that moves; if the
  coupling survives D38, its window would be read from a curve re-measured as the cool runs (seconds
  a curve: 2-8 s here). Not a remedy chosen: a design note for after D38 and the panel.

## Findings: stage 4a'''s S3 arm, the handover with no curriculum (2026-10-21, read 13:40)

`probe-4a3.sh` (resumed at 3ac2ada) on `pyramidon`, word-problem-s3 seed 0 under each shape, 2026-10-20
13:04 to 2026-10-21 13:26; each run to its 1,440-minute budget (steps 88-97). Read by its header's rule
(`population` on each log, `state-separation` snapped on convex's step networks around its best).
Data in data/window-recurrent/state-formation/option1/probe-4a3/s3/; separation in
`option1/separation/separation.txt`.

- **No certificate in any shape.** The warm-up reached its cap (step 8) at screen 0.75, 8 of 8 codes
  a factor in use, `d_eff` 6; trained to the cap (69) throughout (`--whole`), the gate absent (D29
  came later; `held` 0); the coupling's largest rise an epoch 0.006-0.010.
- **States: met only formally.** Best state shares concave 0.356 (step 17), convex **0.566** (step 12,
  coupling 0.35), s 0.251 (step 13), each lost within 1-5 steps. Convex's share passes the header's
  0.5, but its snapped joint nearest-neighbour separation at length 16 is 0.35 at step 12 (chance 0.20;
  0.23-0.30 at steps 8, 13, 20 and 91) and at chance at 64 throughout (0.17-0.21); the continuous
  network itself (`--live`, pooled over lengths 4, 16, 64) 0.48-0.52 at steps 8-12, 0.35 at 13. The
  share comes from short words; no state formed.
- **So formation, as registered and expected** (the reasoning review: "no curriculum on a group"):
  the handover never had a state to convert. By the header's rule, convex is the only shape to go
  forward, and it is already the panel's: D36's `--whole` arm is this cell with D29's gate. Concave
  and s don't go forward. The arm's reading is the panel's expectation for S3 under `--whole`: the
  formation failure, unless the gate changes it.

## Findings: stage 4a''', seeded crystallisation (2026-10-21, read 05:20)

`probe-d26.sh` at f0064cb on `stone`, word-problem-s3 seeds 0-7, 18:05-05:01. Read by its header's rule
(`state-separation` snapped at length 16 on the crucible's last block, the step before the first
"grown" block; a run certified in its crucible read on its certified network). Data in
data/window-recurrent/state-formation/option1/probe-d26/s3/.

- **Seed formation 2 of 8 (4 needed): not met.** Seeds 0 and 5 joint nearest-neighbour 1.00 (centroid
  0.99-1.00); the other six 0.24-0.30 against chance 0.20 on their crucible's last block (steps 101-191).
- **Growth: met only formally.** Seeds 0 and 5 certified in the crucible (steps 85, 117); growth (from
  steps 102-192) recovered none of the six, each to the epoch budget (577-656 minutes).
- **The six failures as read in flight** (`population`): crucibles of length 3 never formed the state
  (seeds 3, 4, 7, best share 0.36-0.38); length 5 and 8 formed it and lost it as the open-loop cool ran
  to coupling 1 (seeds 1, 2, 6: best 0.75-0.93, lost at steps 81-106, `T` 0.002-0.02); seed 6's factor 1
  a single transition. Neither refutation applies (two crystals separated, both certified). These are
  the three kinds D28-D30 fixed, and 4a'''' (below) reran the six under them: 8 of 8.

## Findings: stage 4a'''', the quasi-static crucible (2026-10-20, read 20:30)

`probe-d29.sh` at 3fb967a on `stone` (expectations in its header): word-problem-s3 seeds 1, 2, 3, 4, 6, 7,
19:04-19:33; seeds 0 and 5 carried over from 4a''' (the same runs under D28-D30). Read by the header's
rule: `population` on each log; `state-separation` snapped on the crucible's last block's network (every
run certified in its crucible, none reached growth, so the certified network); `approximations` on each
certified network. Data in data/window-recurrent/state-formation/option1/probe-d29/s3/; separation in
`option1/separation/separation.txt`, approximations in `option1/approximations/`.

- **Every run certified: 6 of 6, so `p` 8 of 8 (Wilson 95% 0.68-1.00), against 2 of 8 (0.07-0.59) in
  4a'''.** Steps 13-123, 4.1-28.4 min. The six seeds were 4a''''s three systematic failures.
- **D28 met:** every crucible length 8 (screen 1.0, 8 of 8 codes a factor in use, `d_eff` 3-9); each
  run's best state share 1.000; snapped joint nearest-neighbour separation 1.00 at lengths 16 and 64 in
  all six (chance 0.20, 0.17; centroid 0.84-1.00), each factor alone 1.00 by nearest neighbour. Seed
  formation in 6 of 6 (4 needed).
- **D29 met:** two runs dipped in the cool, and the clock held through each: seed 1 at steps 49 and 87
  (screen 0.94, 0.99; `held` 2 by its end), seed 4 at 48, 53 and 83 (0.63, 0.94, 0.80; `held` 6), each
  recovering to the gate in 1-3 epochs at fixed `T` and coupling. No run lost a formed state (`lost -`
  in all six), none stalled at a held clock (refutations not met). The hold's own dips (seed 1 step 17,
  0.88) are outside the gate by design and recovered within the step.
- **D30:** seed 6's factor 1 again melted in one transition (at 2.14); its window held at 3 octaves, and
  it certified at step 92 (coupling 0.67, `T` 0.42).
- **Seed 2 certified in the heat** (step 13, coupling 0, `T` 0.33): the length-8 crucible alone formed
  its machine; 4a''''s length-3 crucible had not (best share 0.36-0.38).
- **What the factors hold** (`approximations`; S3 on three points): seed 7 is a cover (meet 6): factor
  0 is exactly the congruence `places {1}`, factor 1 `places {2}`, the places of two items, which
  together fix the permutation; the target a subdirect product of the two factors. Seeds 2 and 6:
  factor 0 `places {1}` (meet 3); 4a''''s seed 0: factor 1 `places {2}`. Seeds 1, 3, 4 and 4a''''s seed
  5: no named congruence (meet 1), the state held jointly.
- **Cascades** (`approximations`' cascade reading, added 2026-10-21: a factor's approximation
  nontrivial while the joint codes determine the state): seeds 0, 2 and 6 are cascades, the place of
  an item on top and the other factor below reading it (two states a block: the other two items'
  order), S3's chain through a point's stabiliser, not Krohn and Rhodes's through the sign (no network
  holds `parity`); seed 7 parallel (either on top); seeds 1, 3, 4, 5 none. Move-to-front-4's three
  certificates: either factor on top. Whether the parts are irreducible isn't read yet (next-steps.md,
  "the components' kind").
- **Verdict:** the header's claims met, no refutation; `p` lifted above the bar (4 of 8). By the
  handover's rule, the panel (4b) next, with the quasi-static crucible (4a''' and D28-D30) as option 1's
  candidate, once 4a''' and 4a'' end and are read.

## Readings: the D33 probe, in flight (2026-10-21, 08:15; not a verdict)

`probe-d33.sh` at 9acc6be on `stone` from 07:24 (`population` on the logs, fixed at 17acb63 for a loss
after every factor is hard). Its launcher was stopped at 08:18 with seeds 6 and 7 unstarted (5 at once,
the running five at their 690-minute budgets till about 19:00); `probe-d33-rest.sh` (the same flags)
started seed 6 then and seed 7 at 18:00, a schedule change only.

- **Certified: S3 seeds 0, 2, 4** (steps 71, 13, 104; 11.5, 2.4, 21.5 min; quasi-static 4a'''/4a''''
  85, 13, 123). Seed 2 in the heat (as 4a''''), so it reads neither cool.
- **Lost in the cool: S3 seeds 1, 3, 5,** each after forming the state (best share 0.73, 0.90, 0.99 at
  steps 69, 60, 80; codes in use halved at 73, 65, 89), then grown; the same seeds certified
  quasi-static. **mtf-4 seed 3:** adaptive best 0.925 at 90, grown at 99; convex control best 0.929
  at 114, still cooling.
- **The mechanism (seed 3, epoch by epoch):** below the cascade (groups 8, 8 by step 61) the capacity
  fell to 0.01-0.02, the step `pace / sqrt(C)` rose to 0.14-0.27 in ln `T` (0.04 through the cascade)
  and the coupling, at 0.5-0.6, rose in jumps (0.59, 0.64, 0.68, 0.75); after each the screen fell to
  0.54-0.85, the gate held 1-18 epochs, recovered to 0.99, and the next step dropped it again; share
  0.90 to 0.30 in five steps.
- **Across seeds:** the largest coupling rise in an epoch, quasi-static against adaptive: seed 1 0.013
  and 0.28, seed 3 0.014 and 0.25, seed 5 0.015 and 0.65; certified seed 0 0.020 and 0.055 (it
  certified as its steps grew), seed 4 0.009 and 0.56 (it survived them). Rises of 0.03 or more: none
  quasi-static, 7-15 adaptive. An early observable before every loss: D34.
- **Seed 6 (read 10:00) lost the same way, more gently:** best share 0.78 at step 68, the coupling's
  rises growing steadily from 0.030 to 0.059 an epoch over steps 71-77 (quasi-static: at most 0.021),
  the screen falling to 0.87 at 77, the codes in use halved there. So 3 certified and 4 lost of 7;
  the registered bar (4 of 8) now needs seed 7 (from 18:00).

## Findings: D32's understanding step (2026-10-21, read 07:30)

`probe-d32.sh` at 364a09e on `stone`, 05:06-07:12: S3 seeds 4 and 7 (as 4a'''') and mtf-4 seed 3 convex
(as 4a''), the `thermals` logging only. Read by its header's rule with `heat-capacity` (built for it) on
each log. Data in data/window-recurrent/state-formation/option1/probe-d32/.

- **Passive: met.** Each run certified at its original step: S3 seed 4 at 123, seed 7 at 67, mtf-4 at
  104 (126 min); the golden run bit-identical at 364a09e.
- **The capacity peaks at the splits: met by runs (2 of 3), narrowly; by factors 2 of 6.** The largest
  capacity within 2 epochs of the largest split: S3 seed 4 factor 0 (1 epoch after), mtf-4 factor 1 (at
  it); not S3 seed 4 factor 1 (3), seed 7 (9 and 12: it certified mid-cascade, at 3 and 4 groups), mtf-4
  factor 0 (its peak at the cool's first epoch: its hold left 7 groups, not a full melt).
- **The shape the step needs holds where the cool passes a cascade:** the capacity sits at 0.01-0.06
  through the melt (S3), jumps 1.2-2.6 times at nucleation, stays high across the cascade (peaks 0.6-1.0
  at T 1.4-2.7) and falls to about 0.001 below the last split. Not a spike at one split: a hump over the
  cascade, so `pace / sqrt(C)` slows the whole cascade. The refutation (flat or monotone) not met.
- **Noisy epoch to epoch** (11-22 local maxima a factor), hence D33's mean over a temperature's epochs.
- **`pace` (D33):** the convex cool's implied `-d ln T sqrt(C)` through each cascade, S3 0.027-0.034, mtf-4
  0.029-0.037: 0.03 stands.
- **The splits come early:** training reaches its fine group counts 2-6 octaves hotter than the
  melting curve measured at the warm-up's end predicts (mtf-4 factor 0: 15 groups at `T` 1.19,
  predicted 0.029). Not the grouping threshold (it biases the other way); 4a's `quiet` arm (`kappa`
  1e-4) split within a fifth of an octave of the curve, so the material moving after the warm-up or
  the thermal noise (`kappa` 1e-2) parting codes; unread, a look in the panel.
- **Tools:** `heat-capacity` (each factor's splits and capacity peak); `scripts/profile_page.py`, a
  run's course as one page (stacked time series of T, capacity, energy, groups, coupling and screen;
  their overlay rescaled; and each factor's phase diagram, its codes on their principal axis against T
  from the saved step networks, beside the melting curve).

## Findings: the reasoning review's re-reading of existing data (2026-10-21)

Re-read offline while the adaptive cool's existence probe (D33's probe) and the handover probe's S3
arm (4a'') ran, for the review's items 1, 3 and 8 ([reasoning-review-2026-10-21.md](reasoning-review-2026-10-21.md)).
The tool: `answers-stop` (the probe script; `mf.answers_stop`), a line per run.

- **An answers-only stop finds the target's machine when the state bound is tight (item 1).** The
  rule: the first step whose answers are all right (the step's own screen and the long lengths 24 and
  48) and whose machine, read off and minimised, has at most the bound's states; the certificate read
  only to judge it. Over every run log with step ends (the studies under `data/window-recurrent/`,
  option 1's apart), with the bound at the target's size: 121 runs stop on their certified network,
  8 stop earlier on an uncertified one, none stops in a run that never certified (no false
  discovery), none trains past its certificate. Each of the 8 has exactly the target's states; the
  four whose stop networks were saved (move-to-front-4 three times, word-problem-s4) **certify at
  snap ratio 1.0** (the network as it computes) and fail at 1.05 only on near ties, so the rule found
  the target's machine there too; the other four (division `p` 5) saved only their certified network.
  - **The bound must be tight:** at 1.1 times the target's size, 25 stop on an uncertified network
    and 1 is a false discovery; at 1.5, 44 and 4; with no bound, 55 and 10. Those networks answer every
    sampled word to length 48 and are larger machines (e.g. 26-165 states for division's 25, 533-613
    for S5's 120): the samples don't tell them from the target, their size does. Discovery's state
    bound ([machine-discovery.md](machine-discovery.md)) is therefore the stop's, not only the
    exclusion bound's, and its slack costs.
  - **Option 1's logged answers are the coupled network's**, not its snapped machine's (a run at
    coupling 0.06 logs right answers while its snapped machine has 2-3 states), so the rule re-reads
    them on each step's saved network (`--snapped`; a loaded network carries no coupling). On option
    1's S3 runs (4a''' and 4a''''), 7 of 8 certified runs stop on their certified network and one
    (4a''' seed 5) at step 108 against 117, whose network certifies at 1.0 (logit lead 13.1); no false
    discovery in the 21 runs (S3 and move-to-front-4) that never certified.
  - **Limits:** every run stopped at its first certificate, so a rule that waits (e.g. for a machine
    stable over steps) can't be read: the certified step's machine usually differs from the step
    before's, and nothing after it was trained. The near ties are the certificate's margin, not the
    machine.
- **The certificates come long before the coupling is complete (item 3).** In the quasi-static
  crucible's S3 runs (4a'''' and 4a'''s seeds 0 and 5), every factor's codes reach their final number
  of groups at a coupling of 0.00-0.50, and the run certifies at couplings of 0.06-0.90 (seed 7:
  0.06 and 0.17; seed 3: 0.27 and 0.23). The certificate reads the snapped network; the network
  trained is the coupled one. So the codes come to quantise the continuous dynamics exactly while
  the forward path is still mostly continuous: the handover's completion isn't what conversion needs.
- **The crucible's lengths reach and tell apart every state on every machine backdrop (item 8).**
  The deepest state from the start and the length that tells every pair apart (Chow's `k`), against
  the crucible's 8: word-problem-s3 2 and 1; S4 4 and 1; S5 5 and 1; move-to-front-4 3 and 2;
  move-to-front-5 4 and 3; random-q8-1 2 and 1; random-q12-0 3 and 1; random-q16-0 4 and 1. Every
  sum is at most 7. Division (arithmetic, no machine table) isn't read. The register's new row 15.

## Findings: the probes D33-D59, final (read 2026-10-21 evening to 2026-10-22, 09:45)

Each probe read by its registered header; a success is a certificate at both verdicts (D41). Counts
from each run's last line (`seed N: step S, END, M min`); Wilson 95% intervals.

- **The adaptive cool's probe (D33's probe): S3 4 of 8, its bar met exactly** (seeds 0, 2, 4, 7 at
  steps 71, 13, 104, 60; 2-22 minutes). Seeds 1, 3, 5, 6 lost in the cool as read in flight (the
  coupling's rises of 0.25-0.65 an epoch below the cascade) and ran to the 690-minute deadline;
  move-to-front-4 seed 3 likewise (deadline at step 105), its convex control spent at step 173.
- **The coupling probe (D38): refuted.** With the coupling held at 0, S3 4 of 8 (seeds 0, 2, 5, 7;
  0.22-0.78); seeds 1, 3, 4, 6 spent at steps 698-709, never once answering right on their snapped
  machine; move-to-front-4 seed 3 to its deadline. The handover is needed; the panel's handover
  question was then superseded by the coupling read from sharpness (D50).
- **The continuous probe (D54):** S3 6 of 8 (seeds 0, 2, 3, 5, 6, 7; 0.41-0.93; 11-57 minutes),
  seeds 1 and 4 spent; move-to-front-4 1 of 4 (seed 0, the crucible's first move-to-front
  certificate), seeds 1-3 stopped by the answers-only stop (D35) on machines exact only at the strict
  verdict (seeds 1, 2 certified under the margin stop, D59).
- **The fresh-seed matrix (D55): S3 7 of 16 (0.44, 0.23-0.67), refuted as registered (8 or fewer);
  move-to-front-4 3 of 8 (0.38, 0.14-0.69), its bar (2) met.** Most losses were strict-only stops,
  not failures to form the state: with the margin stop the same seeds read S3 11 of 16 (0.69,
  0.44-0.86) and move-to-front-4 7 of 8 (0.88, 0.53-0.98).
- **The unstructured probe (D56): random-q16-0 0 of 4,** every seed spent (steps 1523-1561).
  Conversion failures: the state formed in the warm-up and was lost to coupling jumps of 0.20-0.48,
  then the clock held at 0 to the budget.
- **The S4 probe (D57): 1 of 3, its bar (1) met** (seed 1 at step 106, now without the commitment
  term). Seeds 0 and 2 formed the state (seed 0 strict-exact with 24 states at steps 130-134), D35
  never passed (the default verdict's minimised machine about 116 states: 100-130 near ties), then
  **collapsed at a steady coupling of 0.97 and length 16.6-18.8** (screen 0.99 to 0.42 in one or two
  epochs, to 3-17 states), and the clock held at 0 for about 1,400 epochs.
- **The random control probe (D58): random-q12-0 1 of 4** (seed 0 at step 127), below its bar of 2,
  not refuted (0); seeds 1-3 spent.
- **The margin stop (D59): success by its rules, 10 of the 11 read** (S3 27, 29, 33, 36; mtf-4 24,
  26, 29, 30; D54's mtf-4 1, 2), every replay identical to its first run up to the first pass; S3 35
  spent (its first pass, exact at the strict verdict, lost); no false discovery. The split ties
  (`Certificate.split_ties`, target-free) separate D55's 19 stopped networks exactly; waits 1-72
  steps. The twelfth run (D54's mtf-4 seed 3) passed at steps 143-144 with split ties and runs to
  its deadline; under D60 it would end 200 steps past its first pass on the best pass.

### The coupling's jumps and the order of failure (offline, the spent runs of D54-D59)

A jump is a coupling move of 0.1 or more in an epoch. Every spent run had 58-201 of them (largest
0.34-0.77); certified runs at most 11 (largest 0.20). Against the screen's first fall below 0.9:

| order | runs |
|---|---|
| jump first | S3 35 (D59's replay), S3 1 (D54), random-q16-0 2, 3 (D56) |
| the same epoch | S3 25 (D55), random-q16-0 0, 1 (D56) |
| drop first | S3 24 (D55), S3 4 (D54; coupling about 0), S4 0, 2 (D57; the late collapse) |

Shared by all: after the drop the clock's rate (D49) holds at 0 and the run sits frozen to its
budget, the motive of the collapse (D63). The jump-first and same-epoch failures are the paced
coupling's (D61) target; the drop-first ones are not.

### Savitzky-Golay against a moving average (offline, for D61 (c))

On four runs' logged sharpness, a one-sided degree-1 Savitzky-Golay line over the cool's e-folding
time removed every jump with a mean lag of -0.02 to 0.05; a moving average over the same window
lagged 0.16-0.53 and, its window counted in clock, froze with the clock.

## Findings: the scaling conjecture (offline, 2026-10-22, 10:30)

The owner's conjecture: larger machines, and more random ones, need the phase space sampled more (a
larger batch, more seeds). Read offline on every completed run of the current method (D54-D58, with
D59's margin stop), 47 runs; the tool: `formation` (`mf.formation_course`), a line per run. The
owner: the sampling is not changed yet.

- **A run's success is two probabilities: forming the state** (a step's end with state share >= 0.95)
  **and keeping it** to a certificate:

  | backdrop | states | formed | kept, of those formed | certified |
  |---|---|---|---|---|
  | S3 | 6 | 19 of 24 | 17 of 18 (S3 35 open) | 17 of 24 |
  | move-to-front-4 | 24 | 11 of 12 | 10 of 10 (D54's seed 3 running) | 10 of 12 |
  | S4 | 24 | 3 of 3 | 1 of 3 | 1 of 3 |
  | random-q12-0 | 12 | 1 of 4 | 1 of 1 | 1 of 4 |
  | random-q16-0 | 16 | 0 of 4 | - | 0 of 4 |

- **Size is not the variable:** move-to-front-4 (24 states) is the best backdrop, S4 (24) and
  random-q12-0 (12) poor. **The larger machine fails at keeping:** S4 formed 3 of 3 and lost 2 to the
  late collapse (D57), one mechanism, a design signal (D63); more seeds would also do at p about 1/3
  (12 runs for 99%). **The random machines fail at forming:** their failures plateaued at a state
  share of 0.82-0.86 (random-q12-0) and 0.68-0.76 (random-q16-0) over the whole budget (about 1,550
  steps), further from forming than any S3 failure (0.80-0.95); not a time limit.
- **The first melting curve (D21, at step 25) reads the backdrops' cascades:** the octaves below the
  melting point in which a factor makes three quarters of its clusters, mean (range): move-to-front-4
  4.3 (1.3-9.3), random-q12-0 3.3 (0.8-11.3), S4 1.3 (0-2.0), S3 1.0 (0-4.3), random-q16-0 0.8
  (0-1.8); the share made at the first transition 0.31, 0.40, 0.51, 0.53, 0.54. The most crowded
  (random-q16-0) is the worst backdrop and the most spread (move-to-front-4) the best, as the note on
  random machines (below) conjectured; within each backdrop the certified and failed runs' cascades
  are alike, so it is the machine's, not the seed's: more seeds do not change it, a slower cool might.
  Against it: S3 is as crowded and certifies (few states to sort, untested), and random-q12-0 is
  spread yet poor.
- **The cost:** the 12 runs that never formed spent about 65 hours, 86% of it after step 200; of the
  runs not formed by step 200 one formed later (S3 seed 30, at step 853, certified). A kill at step
  200 with a fresh seed in its slot (D63 (e)) would lose about 1 in 13 of them and most of their cost:
  a larger batch is affordable with it.
- **So:** for S4 and random-q12-0 a larger batch suffices at today's rates (with the kill, and D63 for
  S4's collapse); for random-q16-0 the evidence (four plateaus far from forming, the most crowded
  cascade) points at the cool's pace, not the sample, though 0 of 4 (Wilson 0-0.49) cannot exclude a
  usable p. Small samples throughout (3-4 runs on the new backdrops); D64 adds 3-4 a backdrop.

## Findings: the freeze, every failure's end (offline, 2026-10-22, 11:00)

Read for D65's stage 0 on the same 47 runs' per-epoch rows (`anneal`: `rate`, `clock`,
`temperature`, `coupling`, `sharpness`, `length`) and step ends.

- **Every spent run ends frozen:** all 16 (S3 1, 4, 24, 25, 31, 39; move-to-front-4 25; S4 0, 2;
  random-q12-0 1-3; random-q16-0 0-3) hold the rate at 0 for 738-1,472 epochs. The random machines
  freeze early, before the state forms (clock 71-91, temperature 1.4-1.9, at 0.54-1.16 of the factor's
  melting point, mostly 0.7-0.95); S4's freeze deep after its collapse (0.33-0.42 of the melting
  point). S3 seed 30 froze for 445 epochs, thawed and certified (formed at step 853).
- **Not a cool outrunning the network:** before the freeze the gate holds 1-5 epochs at a time
  (halving 30); the frozen runs then spent about 1,400 epochs at one temperature without forming.
- **Frozen is not still:** the clock stops but the coupling, read from the sharpness, mostly rises to
  1 while frozen (random-q16-0 seeds 0, 3, random-q12-0 seed 2: 0.12-0.81 to 0.99-1.00): a quench by
  the coupling at fixed temperature, a glass. Two runs stayed soft: random-q16-0 seeds 1 and 2 sat at
  0.87-1.16 of the melting point with coupling 0.30-0.68 unchanged for 1,405-1,472 epochs, a hold in
  the window between the melt and the freeze, in effect, and did not form either.
- **Not the curriculum:** the never-formed runs froze at lengths 8.0-9.4 and ended at 9.2-11.6; the
  formed runs formed at 8.2-15.6 (S3 8.2-11.2), so the frozen runs sat at lengths where others form.
- **Onset (the owner's two-phase question):** the state begins to form (share first >= 0.6) at
  2.2-3.1 times the factors' melting point on move-to-front-4, 0.8-2.5 (mostly 1.7-2.3) on S3,
  1.2-1.4 on S4, and 1.0-1.4 on random-q12-0 and random-q16-0, whose failures then freeze at about 1
  before forming (random-q12-0 seed 0 formed at 0.59). Ordered machines order above the zone; random
  ones only begin there. The frozen holds above were of a glass already formed: they say nothing
  about a slower approach before it (D66).
- **The snapped machine's long answers read formation from answers alone:** its accuracy at length 48
  tracks the state share (random failures flat at 0.51-0.60 for 1,400 steps; S3 seed 34 0.35 to 1.00
  as it formed), an answers-only observable for a freeze.
- **So:** the remedy is escaping the freeze, not slowing the cool: a reheat (D63 (a): the clock run
  back, the held temperatures released, a partial re-melt, a fresh attempt at nucleation) or a fresh
  seed (D63 (e)). A plain hold in the window (the owner's question, from glass-ceramics, where
  crystallisation is fastest between the glass transition and the melt) has in effect been tried by
  two runs and did not form; glass-ceramics also need nucleating agents, which points at nucleation,
  not time, as the barrier: a reheat or a new seed is a new nucleation attempt.

## A note on random machines (2026-10-20; a note, not a plan)

The owner: "this approach, if it works, is likely to also work for random machines"; and a conjecture,
which this note shares: **the more structure a machine has, the quicker it can be cooled (crystallised);
a random machine would just need a very slow cool.**

- **Why the method should carry over:** nothing in it is the task's. It assumes only what any cell must
  hold (codes enough for the states, training lengths to the exclusion bound); every temperature it
  uses (the melting point, the transitions, the coupling's window) is measured from the run's own states;
  Rose's theory is about clustering, not algebra. The handover (D25) addresses the joint encoding found
  in the warm-up's states, which should be at its worst on a machine without product structure for the
  factors to align with.
- **Why random machines may be harder:** our random backdrops (random-4-0, random-q16-0) failed by
  extra codes (codes whose outputs go wrong within `d` tokens; [failure-kinds.md](failure-kinds.md)),
  not by formation or conversion, so annealing helps there only if the splits stop where the target's states stop
  (plausible: codes part only as far as the states' clusters do; untested, and the thermal noise could
  split codes within a cluster).
- **The conjecture's reasoning:** an algebra's structure (subgroups, quotients) gives a cascade of
  transitions well apart in temperature (S4's from 2.5 to 0.0015, 4a), each passed on its own; a
  random machine's clusters likely part closer together, in one crowded or sharp transition, where a
  fast cool freezes a disordered assignment. The physical counterpart: strongly ordered materials
  crystallise even when cooled fast, glass formers need a very slow cool or they freeze into a glass
  (an amorphous solid, no long-range order); a frozen, wrongly split code assignment would be the
  glass here.
- **What would test it, with the measurements in hand:** the melting curve (D21) on a random machine's
  warm-up: transitions crowded into a narrow band of temperature (against an algebra's cascade) would
  support the reasoning; then the cooling rate needed to certify against the band's width.
- **The owner's bias (2026-10-22; a note, not a plan, after the planned work):** keep the
  quasi-static curve (D49's clock, its gate, its shape) and bias only its pace (the halving time) by
  the target's order, read from answers alone, so one mechanism covers both limits: a strongly ordered
  machine cooled faster, a glass former slower. Two answers-only readings, measured offline on the
  five targets and runs:
  - *From the data, before training:* the answer's conditional entropy given the last k tokens,
    k = 0-6, on uniform words (bits): S3 1.58 flat, S4 2.0 to 1.95 (a group: no finite memory),
    move-to-front-4 2.0 to 0.25 (a reset machine: its last few distinct tokens fix the state),
    random-q12-0 1.77 to 1.08 and random-q16-0 1.61 to 1.02 (a slow, steady decay). Not a difficulty
    (the owner: order is what is measured, not our experience of difficulty): against random machines
    of the same size (the state bound) it reads order itself, a relative entropy against randomness:
    the random targets sit inside their ensembles (rms z 0.31, 1.10), the algebras outside (S3 3.3,
    S4 3.3, move-to-front-4 6.7), in two directions, a group conserving its past (the profile flat)
    and a reset machine forgetting it (the profile falling fast); `order` (the probe's subcommand). Moore's refinement (states told apart by suffixes of length
    k, which needs chosen words) is no use here: every target but move-to-front-4 separates at k = 1.
  - *From the network, in the warm-up:* the melting curve (D21) is answers-only too (the network saw
    only answers), and reads the property the cool meets: `log2 groups(T)` is the code assignment's
    configurational entropy, its rise per octave the entropy the cool must release there (a heat
    capacity, C = dS/d ln T). Crowded (random-q16-0) releases it in under an octave, a cascade
    (move-to-front-4) over four.
  - *A derived bias, no new constant:* constant thermodynamic speed (Salamon and Berry; Nulton and
    Salamon) gives each interval of the cool time in proportion to what it must release; with the
    network's relaxation time tau (the epochs its screen takes to recover after the clock moves, read
    live as the gate already does) the halving time becomes `max(halving, splits per octave x tau)`, the
    splits per octave from the warm-up's melting curve at its densest octave; the curve's shape is
    unchanged, its pace slowed where the machine is crowded. Cautions: the adaptive cool (D33) failed by
    reading the capacity live (its steps outran the handover below the cascade): a bias fixed from the
    warm-up's curve avoids that; a network that has not separated the states may show a crowded curve
    as a symptom, not a cause (random-q16-0's never formed); and the speed-up side (shortening for
    ordered machines) changes what works today, so it would be tested apart.

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
   - **4a'''. The seeded crystallisation probe** (D26, D27): word-problem-s3 seeds 0-7 on `stone`,
     `--crucible`; existence; registered before its data; read 2026-10-21: 2 of 8 (findings above).
   - **4a''''. The quasi-static crucible probe** (D28-D30): 4a''''s six failed seeds on `stone`;
     registered in `probe-d29.sh`; read 2026-10-20: 8 of 8 certified (findings above).
   - **D32's understanding step:** `thermals` logged on S3 seeds 4, 7 and mtf-4 seed 3 convex
     (`probe-d32.sh`), read with `heat-capacity`; then D33's build.
   - **D33's existence probe:** the adaptive crucible on S3 seeds 0-7 and mtf-4 seed 3 (with a convex
     crucible control on mtf-4), registered in `probe-d33.sh`.
   - **D34's probe:** the bounded handover on S3 seeds 0, 1, 3-7 (seed 2 carried over), registered in
     `probe-d34.sh`, on `pyramidon` once 4a'''s S3 arm ends; it no longer gates the panel (D36).
   - **The restructured order (D35-D39):** the builds (`--state-bound`, the snapped
     answers at each step's end, `--coupling-cap`; pure Python, tested; built 2026-10-21); D38's coupling
     probe (`~/code/probe-d38.sh`, registered before its data; on `stone` from 18:00); then:
  - **The continuous crystallisation (D47-D54):** the core matrix set aside; the crate's `straight`
    (`indrajala-math-rust` first, then a "Bump rust/" PR), then `--continuous` (D49-D53, pure Python,
    tested, the golden run unchanged without it), then D54's probe, registered in its script.
    Then D55's fresh-seed matrix, the same cell, registered in its script.
    Then D56's random-q16-0 probe, the same cell, registered in its script.
    Then D57's S4 probe, the same cell, registered in its script.
    Then D58's random-q12-0 probe, the same cell, registered in its script.
    Then the margin stop (D59, `--margin-wait`, tested, the golden run unchanged) and its probe.
    Then the margin deadline (D60) and the paced coupling (D61, both arms; tested, the golden run
    unchanged), D61's probe registered in its script; then the run's state object (D62).
    Then D64's fresh-seed rate, registered in its script; then the collapse (D63).
  - **4b. The panel** (D11) at 4a's `T_hot`, its arms D36's, its cells D37's, its stop D35's, registered
    before its data; D12; then D31's diagnosis and D39's probes.
5-7. **Shelved by D31** (diagnose before a remedy): options 2-4 are called down only by the panel's
   diagnosis, by fit to the mechanism found; their stages as first planned:
5. **Option 2 (noise):** understanding (the certified networks' snap margins against the failures'),
   build (`--cell-noise`, numpy and crate), the panel, decision.
6. **Option 3 (the rotation trick):** understanding (on extra-codes failures, whether windows of
   different target states share codes, from `breaks`), build (the crate's straight-through gradient),
   the panel, decision.
7. **Option 4 (FSQ factors):** understanding, design (a fixed grid a factor, its levels from the
   codes ratio), build, the panel, decision.
8. **Findings** in [failure-kinds.md](failure-kinds.md) and the probe script; this plan retired, leftovers
   to [next-steps.md](next-steps.md).

## Machines

`stone`: about half its 12 cores by day, all at night (the panel's 12 runs at night). `pyramidon`:
development, up to about 4 small workers. `jebel`: up, 06:00-18:00 only (its power load overnight), timing.
