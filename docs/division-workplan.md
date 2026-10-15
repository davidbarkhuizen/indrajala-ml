# Workplan: division, from a certified network to a claim

**Status: written 2026-10-15 at the owner's request, after the modular-family probe certified
division at `p` 5 ([modular-family-workplan.md](modular-family-workplan.md), stage 2). Decisions
D1-D10 settled (the owner delegates a workplan's decisions; each with its options' pros and cons).
Probe-driven ([experiment-design.md](experiment-design.md), section 2): each stage is short
probes, its next step chosen from its result; one longer run, stage 5's, with its reason stated.
Stages 1-3 done (2026-10-15); stage 4's probe certified 1 of 4 at `p` 7 (its findings to
follow); stage 5 registered (D11-D14), its build done.**

## Where this starts

- **Division at `p` 5 certified, 2 of 2** (`scripts/modular_family_probe.py`'s findings): a
  window-recurrent network of two factors of 16 codes, its poolings reading the window's output,
  trained on the modular family (each window answering the running quotient, sum and difference
  of two numbers mod 5) by the curriculum, certified against division's 25-state machine in both
  verdicts, solved in 14-17 minutes of a 60-minute probe (4 runs sharing `jebel`).
- **What it took, as far as 2 seeds an arm can say:** the poolings reading the output (D10's
  reading before the state blocks even one counter: the D10 check, 0 of 2 against 2 of 2 on E8c's
  modular sum) and the family (the quotient alone, reading the output: 0 of 2).
- **What it isn't:** factored by residue. Each certified state is spread over both codebooks
  (33-53 states reached, minimised to 25; no factor a function of a residue).
- **What breaks it:** training on past the solution; every run that got there collapsed within 3-5
  steps. The probe now stops at its first certified step.

Open: the rate (2 of 2 is a screen); the long tests; whether the family is needed (the quotient
alone had 2 seeds and an hour); whether two codebooks are (the certified states don't use them as
factors); whether it climbs to `p` 7 (49 states; E7 and E7', the quotient alone with one codebook,
0 of 80).

## The order (the owner's, 2026-10-15)

1. **Consolidate:** more seeds of the certified configuration; the long tests.
2. **Isolate:** the family with one codebook; the quotient alone, more seeds.
3. **Step up:** `p` 7.
4. **A registered claim:** a rate, under the configuration stages 2-4 single out.
5. **Write it up.**

## Decisions

- **D1. Every run stops at its first certified step, its network saved** (the probe's
  `certified-step-<j>.network.json`).
  - (a) *Chosen.* Pros: the solution kept (every run trained on lost it); a run's cost ends when
    it has answered. Cons: a run's later steps unseen (whether a solution survives is a separate
    question, Option 1's thermostat if ever needed).
  - (b) Train on, save every step. Cons: the collapse; the disk.
- **D2. Probes are screens; only stage 5 states a rate.** Each probe: 2-6 seeds, a 60-minute
  budget a run, its success and failure written in this plan before its data (below).
  - (a) *Chosen.* Pros: the owner's probe-driven mode; minutes to an answer. Cons: counts at 2-6
    seeds don't separate rates (experiment-design.md, section 3: 20 a cell tells 0.8 from 0.4).
  - (b) Rates from the start. Cons: a long run before the configuration is settled.
- **D3. Stage 2's seeds: 2-7, all six at once on `jebel`** (6 processes; a probe run takes one core and
  90-160 MB, measured at 43 minutes: the family's 156 MB, the quotient's 91-97 MB).
  - (a) *Chosen.* Pros: 8 family seeds in all with the first two; one hour. Cons: 6 of 8 cores, so
    each run is a little slower than the first probe's.
  - (b) 4 seeds. Pros: lighter. Cons: 6 in all says less.
- **D4. The long tests: the claims workplan's U1 and S1 at 128, 256 and 1,024 columns** on every
  certified network, the quotient slot read, one process on `pyramidon`.
  - (a) *Chosen.* Pros: the project's independent check (E8c's long tests: every case passed);
    comparable with every earlier long test. Cons: a long-test path for the query encoding to build
    (division's catalogue cases through `QueryMachine`'s window), and `pyramidon`'s memory (one
    memory-heavy job at a time: working-notes).
  - (b) 1,024 only. Cons: not the earlier tests' shape.
- **D5. The single-codebook cell: one factor, `s` 32, `m` 128** (the two factors' width; E7''s
  128 codes at `p` 5, about 5 times the 25 states).
  - (a) *Chosen.* Pros: E7''s codebook, so "the family on E7''s state" is the one change from E7'
    besides the output reading; the same width as the two-factor cell. Cons: one codebook of 128 is
    not the two factors' 256 joint states; a failure could be the count, so (b) as its variation.
  - (b) `s` 32, `m` 256. Pros: the joint count. Cons: E7' also had 256 at `p` 5 and failed (the
    quotient alone).
  - (c) `s` 16, `m` 64. Cons: below the owner's floor of 3-5 times the states by the joint count.
- **D6. The quotient-alone control: 4 more seeds (2-5), 60 minutes, stop at certified.**
  - (a) *Chosen.* Pros: 6 seeds in all to the family's 8; seeds, not hours (both of its first
    runs had plateaued by about step 5 and collapsed later, so a longer run adds little). Cons: a
    late success could still be missed.
  - (b) 2 seeds, 3 hours. Cons: a long run without a stated reason; the collapse.
- **D7. The `p` 7 cell: the certified configuration, two factors of 16** (256 joint states for 49,
  5.2 times), 4 seeds, 60 minutes; the cap at the exclusion bound `256 + 49 - 1 = 304` columns.
  - (a) *Chosen.* Pros: one change (the modulus); the owner's floor met by the joint count. Cons:
    a step costs more (7 + 1 classes, more states to learn), so 60 minutes may be short: the
    probe's budget may be extended by the conditional rule below.
  - (b) Two factors of 32. Pros: a surplus like `p` 5's. Cons: two changes at once.
- **D8. Where the work runs: probes on `jebel`** (from a worktree there: working-notes), **the
  certificates and long tests on `pyramidon`**, results copied to both with manifests.
- **D9. The factored state isn't a goal here.** The certified networks aren't factored by residue.
  - (a) *Chosen.* Keep the two-factor cell as the working configuration; ask whether it is needed
    (stage 3's single codebook) rather than force it to factor. Pros: existence first; a simpler
    architecture if one codebook works. Cons: factored readouts (each factor a residue) wait; to
    get them back, D10 of the factored-state workplan reopens ((a) or (c) there).
  - (b) Make the factors split by residue now. Cons: no evidence yet that it helps learning; D10's
    (b) was the way to force it and blocked learning.
- **D10. The runner stays the probe tool**, `scripts/modular_family_probe.py`, for every stage, the
  registered claim included; the harness gains the family only if stage 5 needs what the probe
  tool lacks.
  - (a) *Chosen.* Pros: built, tested, certifies and stops; no second path. Cons: not the
    harness's `status`, `report`, resume or machine lock (a probe run is short: its log is enough;
    the lock: nothing else runs on `jebel` during a probe, checked by hand).
  - (b) Move the family into the harness first. Cons: a package change before it is needed.
- **D11. The claim's comparison: the family against the quotient alone, `p` 5, two factors of 16,
  reading the output** (the screens' cells: 5 of 8 against 0 of 6).
  - (a) *Chosen.* Pros: the comparison the screens point to, with the most data; the working
    configuration (D9 (a)); the quotient alone is the project's earlier division (C9-E7'), so the
    claim says what the family adds. Cons: every failure is a collapse, so the rate is this
    recipe's, not the architecture's.
  - (b) At `p` 7 (1 of 4). Pros: the harder modulus. Cons: a rate near 0.25 needs more seeds to
    separate from 0, and a `p` 7 run costs more.
  - (c) One codebook of 128 (4 of 4). Pros: the highest screen. Cons: not the working configuration
    (D9 (a), factored); the quotient alone wasn't screened on it.
- **D12. Seeds: 20 a cell, fresh, 100-119 in both arms.**
  - (a) *Chosen.* Pros: section 3's 20 a cell (tells 0.8 from 0.4); fresh seeds, so the claim isn't
    measured on the seeds that chose its configuration. Cons: 40 runs.
  - (b) 20 including the screens' 14. Pros: 14 fewer runs. Cons: the screens selected the
    configuration on those seeds.
- **D13. The budget: 400 epochs trained in all (`--budget-epochs 400`), 180 minutes of wall clock
  only as a guard; stop at certified.**
  - (a) *Chosen.* Pros: the same training in every run whatever the machine or its load, so seeds
    can run on both machines; every success seen needed 73-215 epochs, and an hour's failures
    reached 424-432 (family) and 417-599 (quotient alone), so 400 is about the screens' hour.
    Cons: a build (`--budget-epochs`, with `--workers` to queue seeds); the arms are matched on
    examples, not minutes (the quotient's windows are a third of the family's tokens).
  - (b) 60 minutes, as the screens. Pros: no build. Cons: the result depends on the load and the
    machine, so seeds couldn't be spread fairly.
- **D14. Both machines, by seeds, each seed's two arms on the same machine:** `jebel` seeds 100-113
  (each arm `--workers 3`, 6 processes), `pyramidon` 114-119 (each arm `--workers 2`, 4 processes;
  about 160 MB a run). About 3-3.5 hours wall clock (a family run about 15 minutes when it
  succeeds, 55 when it fails; a quotient run about 45, on `jebel`).
  - (a) *Chosen.* Pros: about half `jebel`'s time alone; D13 makes the machines equivalent.
    Cons: `pyramidon` busy for its share (development there slows).
  - (b) `jebel` alone. Pros: one machine. Cons: about 5 hours.

## Feasibility (each cell; experiment-design.md, section 1)

| cell | target states | the model's states | per factor | exclusion bound `m + q - 1` | cap |
| --- | --- | --- | --- | --- | --- |
| family, `p` 5, two factors of 16 | 25 | 256 joint | 16 (a counter 5) | 280 | 280 |
| family, `p` 5, one factor `m` 128 | 25 | 128 | - | 152 | 152 |
| quotient and sum, `p` 5, one factor `m` 128 | 25 | 128 | - | 152 | 152 |
| quotient alone, `p` 5, two of 16 | 25 | 256 joint | 16 | 280 | 280 |
| family, `p` 7, two factors of 16 | 49 | 256 joint | 16 (a counter 7) | 304 | 304 |
| stage 5: family and quotient alone, `p` 5, two of 16 | 25 | 256 joint | 16 | 280 | 280 |

No cap is reached in an hour (the certified runs stopped at steps 11-12); a probe claims nothing
about memorising, and every success is certified, which is exact at every length. Through stage 3
`run` capped every cell at 280 (the default, so 152 wasn't enforced; no run passed step 34); from
stage 4 the cap is the cell's exclusion bound (`modular_family.exclusion_bound`), so `p` 7's 304
is reachable.

## Stages

1. **This plan.** With it, the modular-family workplan's stage 3 (the certificate's adapter) is
   recorded done, its leftovers here.
2. **Consolidate.**
   - *Build:* the probe tool's `long-test` (division's U1 and S1 cases at a length, the quotient
     slot read, chunked as the harness's `long_test`; JSON as `long-test`'s), with a test; the
     probe's `--first-seed`. *Done:* the tool's `long-test` (about 19 µs a case-column on `pyramidon`,
     so about 55 minutes a network at 128, 256 and 1,024) and `run --first-seed F --seeds N` (the
     seeds at once, a process each).
   - *Probe 2a, registered:* family, reading the output, `p` 5, two factors of 16, seeds 2-7, 60
     minutes each, stop at certified, `jebel`. **Reports** the count certified (both verdicts) and
     each run's certified step and minutes. No success threshold: a screen of the rate (D2).
     *Result:* 3 of 6 (seeds 5, 7, 4 at steps 7, 8, 10, within 25 minutes); with the first
     probe's two, 5 of 8. Every failure collapsed (the state minimised to 6-11); seed 3 was right
     on every sampled answer for four steps without certifying, then collapsed.
   - *Probe 2b, registered:* the long tests (D4) on every certified network (the two saved and
     2a's). **Works:** every case passes. **Fails:** any case fails on a certified network: a bug
     in the certificate's adapter or the long test, to be found before anything else.
     *Result: works.* All five networks pass every one of their 375,000 cases.
3. **Isolate.**
   - *Probe 3a, registered:* family, reading the output, one factor `s` 32 `m` 128 (D5), seeds 0-3,
     60 minutes. **Works:** at least one certifies: the family is enough, the second codebook
     isn't needed. **Fails:** none does; then its variation, `m` 256 (D5 (b)), 2 seeds.
     *Result: works,* 4 of 4 (steps 5-8, within 18 minutes), both verdicts; every long-test case
     passes. The working configuration stays two factors (D9 (a)), so stage 4's cell (D7) is
     unchanged.
   - *Probe 3b, registered:* the quotient alone, reading the output, two factors of 16, seeds 2-5
     (D6). **Reports** the count; with the first two, 6 seeds against the family's 8, a screen.
     **If any certifies,** the family's part is a rate question for stage 5, not existence.
     *Result:* 0 of 4; with the first two, 0 of 6 against the family's 5 of 8 (Fisher's exact p
     0.031, a screen).
   - (Optional, if 3a works) the family without the difference (the sum and the quotient), 2
     seeds: is one dense counter enough.
   - *Probe 3c, registered (3a worked: 4 of 4):* the quotient and the sum (`--arm sum`), reading
     the output, 3a's factor (`s` 32, `m` 128; codes 5.1 times the 25 states; the exclusion
     bound 128 + 25 - 1 = 152 columns, the feasibility table), seeds 0-1, 60 minutes, stop at
     certified (the certificate against division's machine, both verdicts). **Works:** at least
     one certifies: one dense counter is enough. **Fails:** none does: the difference's counter is
     part of what helps (a screen, 2 seeds).
     *Result: works,* 2 of 2 (steps 9-10, within 13 minutes), both verdicts; every long-test
     case passes.
4. **Step up: `p` 7.**
   - *Probe 4, registered:* family, reading the output, `p` 7, the cell of D7, seeds 0-3, 60
     minutes. **Works:** at least one certifies against division's 49-state machine. **Fails:**
     none. **Conditional rule (registered now):** if a run's screen is still rising when its hour
     ends, rerun that seed with 120 minutes; otherwise the next probe is D7 (b).
5. **The registered claim** (the one longer run, its reason: a rate is the claim the project
   reports; a probe can't give one).
   - *Build:* `run --budget-epochs N` (the run stops when N epochs are trained in all, end
     `spent`) and `--workers W` (W runs at once, the rest queued). *Done.*
   - *Claim, registered:* the family against the quotient alone (D11), `p` 5, two factors of 16,
     reading the output, seeds 100-119 in each arm (D12), 400 epochs (D13), stop at certified,
     the certificate against division's 25-state machine in both verdicts (each certified network
     recertified on `pyramidon`), the machines of D14. **Reports** each arm's count certified with
     its Wilson interval (95%), and each success's step, epochs and minutes. **Works (the family
     helps):** the family's count exceeds the quotient alone's, Fisher's exact test one-sided p <
     0.05. **Fails:** otherwise; the screens' difference doesn't hold at 20 seeds. Then the long
     tests (D4) on every certified network: any failing case is a bug, found before the claim is
     written up.
6. **Write it up.** The findings in the probe script's docstring at each stage; the D10 finding,
   division's certified networks and the collapse past a solution as notes for the paper
   (`~/code/indrajala-papers`); next-steps.md for what stays open (factored readouts, D10 (a)/(c),
   Option 1's thermostat).
7. **Retire this plan and the modular-family plan.**

Tier 0 throughout (the study module and its script, pure Python; no timed path), unless stage 5
moves the family into the harness (then the golden run, as for any package change that could reach
training results).
