# Workplan: a population of runs, selected by their machines (level 2: tempering across runs)

**Status: drafted 2026-10-19 at the owner's request ("if we ran a large number of these in
parallel - could we use an analogy like crystallization - i.e temperature profile that slowly
cools"; then: "sounds good to me" to two levels, the temperature inside a run (level 1: the
commitment plan's thermostat and its quantiser, the annealing) and a population across runs, this
plan). D1-D8 settled under the owner's delegation, each with its options' pros and cons. P1
registered before its data. Starts after the current batch (C0, R1's probes, the thermostat's
second probe).**

## Why

Today's successes on the hard backdrops were lucky escapes: B2 certified S5 and move-to-front-5 in
one seed of six each, and near solutions were found and lost (B2's S5 seed 2 at 1.65 pairs per
code, then gone; [one-code-one-state-workplan.md](one-code-one-state-workplan.md), "Why"). Six
independent seeds give six chances and keep none of what the others found. A population that keeps
and copies its best machines turns one member's escape into every member's starting point, and
explores around it.

Two established methods do this:
- **population-based training** (Jaderberg et al. 2017): at intervals, the worst members copy the
  best one's network and perturb its settings;
- **parallel tempering** (replica exchange): members at different temperatures swap networks, so
  what hot members find flows to cold ones.

Both need an **energy**, to rank members, and a **temperature**, a setting to vary. We have both:
- **The energy: pairs per code**, how many target states a code stands for. It is 1.0 in every
  certified network, exact, about a second to read, and logged at every step's end since R2.
- **The temperature: the commitment β**, which pulls a code's outputs to it (failure A's cause, the
  commitment-control workplan), and the learning rate. With the quantiser (level 1's annealing) τ
  becomes the natural one.

No GPU is needed. A member is one CPU process of about 100-150 MB at these sizes, and the cores are
the limit, not memory: 8 on `jebel`, about 8 on `stone`.

## Decisions

- **D1. The method: (a) population-based training, exploit and explore.**
  - (a) *Chosen.* At each interval, the bottom quarter copies a top-quarter member (its whole
    checkpoint: network, optimizer, place) and perturbs β and the rate. Pros: one rule; robust with
    8 members; the energy decides directly what survives. Cons: copies can crowd out diversity
    (perturbation is the counter).
  - (b) Parallel tempering: a fixed ladder of β, swaps between neighbours by a Metropolis rule on
    the energy. Pros: the crystallisation analogy exactly. Cons: the swap rule needs an energy
    scale; with 8 members the ladder is coarse; natural with τ, which isn't built.
  - (c) Independent seeds with the best kept (R3 across runs). Cons: no member ever profits from
    another's discovery.
- **D2. The energy: pairs per code at the member's last diagnosis; ties to more right pairs; a
  certified member wins outright and the population stops at the first certificate.** Pros: the
  measure every certified network satisfies (one code, one state). Cons: the target's machine is
  needed to read it, as the certificate already is (a training control, as stop at certified).
- **D3. The temperature: β (a multiple of each factor's specs' commitment, 0 to 1) and the rate (a
  multiple, 0.5 to 2), each perturbed by a factor of 0.8 or 1.25.** Pros: both exist; β is failure
  A's lever. Cons: τ waits for level 1's quantiser.
- **D4. The population: 8 members on one machine (`jebel`), all 8 at once, a process a member.**
  Pros: no cross-machine sync; 8 cores. Cons: one machine's study at a time.
- **D5. The interval: every 20 epochs.** Pros: about a step's length at the backdrops' plateaus
  (patience 10, steps 10-50 epochs). Cons: guessed; the probe shows whether members move between
  intervals.
- **D6. The mechanism: (b) a coordinator over checkpoints.**
  - (a) One process holding every member's network. Cons: a rewrite of `run`'s loop.
  - (b) *Chosen.* The coordinator runs each member as today's `run --resume` for 20 more epochs
    (`--budget-epochs` raised each round), reads each one's energy from its log, then replaces a
    losing member's checkpoint with a copy of a winner's, its β and rate perturbed. `run` gains
    `--adopt` (a copied checkpoint's identity taken as the member's own) and `--commitment-scale`
    and `--rate-scale`, read from a member's settings file each round. Pros: reuses `run`,
    checkpoints and resume (all tested by bits); a stopped population resumes. Cons: process start
    per round (seconds against 20 epochs' minutes).
- **D7. The base window: the one the one-code-one-state plan adopts (its D4': W11 at width 32
  unless C0 shows harm).** Pros: a population on a window with capacity; the general architecture.
  Cons: waits for C0.
- **D8. The comparison: the same compute, independent.** A population of 8 against 8 independent
  seeds with the same total epochs (8 x 600), several times each, so "the chance a population of 8
  yields a certificate" is compared with "the chance any of 8 independent seeds does".

## P1 (registered before its data)

- **Cells:** word-problem-s5 and move-to-front-5 (round 2's cell: two factors, `--codes-ratio
  32/11`, reading the output, the fork, the base window of D7), the backdrops for bottlenecks 2
  and 1.
- **Arms:** 3 populations of 8 per cell (members' seeds 0-23), each to 8 x 600 epochs in all or its
  first certificate; and 24 independent seeds per cell (the same seeds, no exchange, the same
  budget each), so the independent arm is 3 groups of 8.
- **Works:** a population certifies (any member, both verdicts, recertified).
- **Helps:** the populations certify in more of their 3 than the independent groups of 8 do (each
  group "certifies" if any of its 8 does); reported with the independent arm's per-seed rate and
  the probability it implies for a group of 8 (`1 - (1 - r)^8`). Three populations a cell is an
  existence-and-direction test, not a significance test: stated so in the findings.
- **Read every failure** by its members' pairs per code over the rounds: did copies spread a near
  solution, and was it lost after spreading?

**Feasibility (experiment-design.md, section 1; [information-flow.md](information-flow.md)):** the
cells' codes and caps are round 2's (`q` 120, 32 codes a factor, `m` 1,024, cap 1,143); the window
the base's (D7); a population member is an ordinary run, so every row of the register holds as it
does for one.

**Budget:** C0 gives the base window's epoch time on these cells; 24 runs a cell an arm, 96 in all,
is about two nights on `jebel` and `stone`. The probe (stage 3) gives the population's own overhead.

## Stages

1. **This plan.**
2. **The build (tier 0):** `run --adopt`, `--commitment-scale`, `--rate-scale` (tested: a member
   adopting a checkpoint repeats the source's next epochs by bits at scale 1); the coordinator
   (`population` in the probe script), its rounds, the energy read, copy and perturb, and its
   resume (tested on a small cell).
3. **The probe:** one population of 8 on S5, 600 epochs a member, its expectation written first;
   the overhead measured.
4. **P1.**
5. **Findings;** this plan retired, its leftovers (parallel tempering with τ once the quantiser
   exists; more members across machines) to next-steps.md.

Related: [commitment-control-workplan.md](commitment-control-workplan.md) (level 1: the thermostat,
and stage 3's quantiser, the annealing), [one-code-one-state-workplan.md](one-code-one-state-workplan.md)
(the energy, R2; R3 keeps the best machine within a run, which this generalises across runs),
[machine-discovery.md](machine-discovery.md) (reliability, stage A).
