# Workplan: snapshot and fork on collapse (the division runs' abrupt collapse)

**Status: written 2026-10-15 at the owner's request ("a snapshot and backtrack to fork on collapse
... a very good general mechanism"), after the collapse was traced as abrupt
(`scripts/modular_family_probe.py`'s findings, "the collapse, traced"). Decisions D1-D6 settled
(the owner delegates a workplan's decisions; each with its options' pros and cons). Probe-driven
([experiment-design.md](experiment-design.md), section 2). It comes before division at larger `p`
(next-steps.md, "From the division work": the owner, the fix first, to maximise the chance of `p`
11 on `jebel`).**

## Why

Every division run that failed collapsed, and every run that solved and trained on collapsed too.
The trace (15 events in 11 runs) shows each collapse **abrupt and large**: within 5 epochs the codes
in use halve or worse and every answer falls to about chance, with no slow fall before it; 11 of
15 mid-step. Whatever its cause (an optimiser instability is the likeliest: no gradient clipping, a
constant Adam rate, sequences lengthening each step), a mechanism that notices the fall and goes
back to before it doesn't need the cause: it needs the fall to be fast and large, which it is.

Because runs are deterministic, going back alone would replay the same fall. So the mechanism
**forks**: it restores the snapshot and continues on a different path (a new data stream, a cooler
rate).

## The mechanism

Within each curriculum step, with `fork` on:

- **the reference:** the screen score (the step's training lengths, the same per-epoch accuracy the
  plateau rule reads) at the step's start and after each epoch; the step's best so far is the peak;
- **the snapshot:** the network (weights, optimiser state) at the peak, saved to
  `out/snapshot.network.json` whenever the peak rises (the step's start included);
- **a collapse:** an epoch's score below `(1 - drop)` of the peak;
- **a fork:** restore the snapshot, multiply the step's rate by `cool`, and take the epoch's data
  from a new stream (the fork's index in the generator's seed); a line `{"fork": f, ...}` in the
  log; at most `limit` forks a step, then training goes on unforked.

With `fork` off the run is exactly as before (the same seeds, the same rate).

## Decisions

- **D1. What detects a collapse: the screen score falling below `(1 - 0.4)` of the step's peak,
  after every epoch.**
  - (a) *Chosen.* Pros: every epoch (the diagnosis is every 5); the training answers only (the
    owner's "learn the method": no target state read); relative, so it scales with the accuracy
    reached; the step's start counts, so a fall at a step's first epoch (4 of 15) is caught. Cons:
    a threshold: the screens' noise between epochs is a few points, a collapse 0.2-0.7 (e.g. 0.55 to
    0.20), so 0.4 sits between.
  - (b) The codes in use halving. Pros: the trace's marker. Cons: known only at a diagnosis (every
    5 epochs, a full forward over the diagnosis cases); a code-count change can be benign.
  - (c) Both. Cons: the codes add little to (a) and cost a diagnosis an epoch.
- **D2. The snapshot: the step's peak network, on disk** (`net.save`, format 2 with the optimiser's
  state, about 0.3 MB at these sizes).
  - (a) *Chosen.* Pros: the best point the step reached, so a fork never restores something worse;
    format 2 round-trips weights and optimiser state (the harness's resume relies on it). Cons: a
    save each time the peak rises (milliseconds).
  - (b) A ring of the last few epochs. Pros: always recent. Cons: may hold the fall's own start.
- **D3. The fork: restore, a new data stream, and the rate halved for the rest of the step.**
  - (a) *Chosen.* Pros: the new stream changes the path (without it the fall replays exactly); the
    cooler rate is the standard remedy for an instability and costs nothing if the cause is
    elsewhere; per step, so the next step starts at the configured rate. Cons: halving twice or more
    slows learning within the step (`limit` bounds it).
  - (b) A new stream only. Pros: the purest fork. Cons: if the instability is the rate's, it likely
    recurs.
  - (c) Also reset the optimiser's moments. Cons: Adam's first steps after a reset are large: the
    shock the fork means to avoid.
- **D4. At most 4 forks a step, then unforked.** Pros: bounded cost (each fork costs the epochs since
  the peak); recorded, so a run that exhausts them is visible. Cons: a step that falls more than 4
  times trains on as now.
- **D5. Built in the probe tool** (`modular_family.run`'s `fork`, the script's `--fork`), the
  runner of every division stage (division workplan, D10); the harness gains it only when a harness
  study needs it. Pros: no second path; division is where the collapse is. Cons: the harness's
  studies don't have it yet.
- **D6. The probe: `p` 7, two factors of 32 codes (`s` 16), the cover (`--arm family`), reading the
  output, seeds 0-3, 400 epochs each (`--budget-epochs`, so the load doesn't matter), stop at
  certified; two arms on the same seeds: with `--fork` and without.** On `jebel` after E2b's
  continuous arm (8 runs at once, about 1-1.5 h).
  - (a) *Chosen.* Pros: the cell the trace says needs both (saturated codebooks, collapses); the
    without arm isolates the fork on the same seeds; 32 codes a factor is division's D7 (b), wanted
    anyway; epochs, not minutes, so the arms are matched. Cons: 4 seeds an arm is a screen.
  - (b) Replays of the old `p` 7 cell (16 a factor) with the fork. Pros: the exact runs that
    collapsed. Cons: the saturated codebook confounds it; we'd run 32 next anyway.
  - (c) `p` 5 first. Cons: `p` 5 already succeeds often; the test is where it fails.

## Feasibility

| cell | target states | the model's states | per factor | exclusion bound `m + q - 1` | cap |
| --- | --- | --- | --- | --- | --- |
| the cover, `p` 7, two factors of 32 | 49 | 1,024 joint | 32 (a counter 7) | 1,072 | 1,072 |

No run approaches the cap within 400 epochs; every success is certified, which is exact at every
length.

## The probe (stage 3), registered before its data

**Works (the fork helps):** the fork arm certifies more networks than the arm without, or as many
with fewer collapses; and in the fork arm no run ends collapsed (its last step's state minimised to
fewer than half the target's 49 states). **Fails:** the fork arm certifies none and its runs end
collapsed despite forks (then the fork's restore isn't enough: the measurement probe of next-steps.md,
the gradient's norm per batch, before anything else). Reports: each run's certified step, epochs,
forks (count, steps, scores before and after), its trace.

**Noted before the probe's data was read** (2026-10-15, from division stage 5's runs, the probe
script's findings): stage 5's family failures lost near solutions over several steps, the codes
collapsing only later, and the fork's snapshot is the current step's. So in reading the forks, note
each one's step against the run's best step (`readout`'s peak): a fork after the best step restores
a network already past it. This changes nothing registered.

**Next, by outcome:** works: the `p` 11 ladder (two factors of 32, then 64; 4-6 seeds, the fork on);
fails: the measurement probe, then clipping (next-steps.md).

## Stages

1. **This plan.**
2. **The build:** `modular_family.Fork` and `run(..., fork=...)`, the script's `--fork`, tests (a
   fork restores the snapshot and cools the rate; the limit; without `fork` the run is unchanged).
3. **The probe** (D6), its findings in the probe script's docstring.
4. **Retire this plan;** what stays open to next-steps.md (the harness's fork; the cause).

Tier 0 throughout: the study module and its script, pure Python; no timed path.
