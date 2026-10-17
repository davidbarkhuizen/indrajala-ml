# The window-recurrent claims: the report

The report of the claims workplan (D11), written 2026-10-17 when its last claim (C6) came in and
the plan was retired: `git show fea9de1:docs/window-recurrent-claims-workplan.md` has each
claim's registration, the experiments E1-E8 with their designs, the decisions D1-D39 with their
options' pros and cons, and every stage's record. Each experiment's findings, with its data,
manifests and commits, are in [scripts/window_recurrent_study.py](../scripts/window_recurrent_study.py)'s
docstring (E1's certificate also in
[window_recurrent_certificate.py](../indrajala_ml/studies/window_recurrent_certificate.py)). The
paper is P1, `snapped-window-recurrence.tex` in indrajala-papers; the literature is in
[window-recurrent-literature.md](window-recurrent-literature.md).

The question: the window-recurrent adder ([window-recurrent-adder-workplan.md](window-recurrent-adder-workplan.md))
added two long base-3 numbers to 64 columns in about half its runs. The claims say what that
result is and why, each registered with its refutation before its data (D4: 20 seeds a cell;
Wilson intervals; Fisher's exact test; Cochran-Armitage for a trend; Holm over C2-C4 and C6).

## The claims and their verdicts

| id | claim | experiment | verdict | evidence |
| --- | --- | --- | --- | --- |
| C0 | The runs reproduce | E3's reruns of the matrix | **holds** | 15 of 15 reruns byte-identical: results, and every step's weights, optimizer state and generator |
| C1 | A working network is a correct adder at every length | E1, the certificate | **refuted for the sampled "works"; restated** (D13): a certified network is correct at every length, and the sampled tests overstate working | 52 of 57 working networks certified (48 robustly); 2 passed every test to 64 columns and are wrong (at 8 and 12); no failing network certifies; every certified machine minimises to the 3-state carry automaton |
| C2 | The snap is what makes the recurrence extrapolate | E2, the ablation | **holds** | continuous 0 of 20 at 1,024 columns against snapped 8 of 20, p 0.0033 (Holm 0.013); continuous fits its training lengths about as often, then breaks at the longest length trained |
| C2b | The same on the plain recurrence (no transformer in the window) | E2b' (D38, D39) | **holds** | snapped 20 of 20 at 1,024 against continuous 0 of 20, one-sided p 7.3e-12; every snapped network certified, all long-tested |
| C3 | More codes than the automaton needs make training find it more often | E3, the code count | **holds for working, not for certified** | working rate rises with `m` (Cochran-Armitage p 0.014, Holm 0.042); certified doesn't (p 0.43): the extra codes buy networks that pass the tests without being the automaton |
| C4 | The curriculum isn't needed: all at once works as often | E4, the arms | **holds** (not refuted after Holm) | all at once 19 of 20 against the curriculum's 13 of 20, p 0.044 raw, 0.088 adjusted; on about 20 times fewer examples (Mann-Whitney p 2e-6); certified 12 against 8 (p 0.34). Stage 7 read it raw as refuted; Holm, registered, corrects that |
| C5 | The seed decides: a cell's working rate is a stable fraction | E3 and E4 | **a measurement** | rates with Wilson intervals, about 0.4 wide at 20 seeds |
| C6 | Failing runs reach states training never produced (Buitrago Ruiz & Gu) | E5, saved networks | **not supported** | 2 of 21 failing runs reach a new code past training against 3 of 100 working, p 0.21 (Holm 0.21): failures are wrong transitions among visited states |
| C7 | The design learns modular division (`p` 5, 7; `p`² states) | E7 | **refuted** | 0 of 80 working or certified (and `p` 7 at `m` 32 had fewer codes than states: D23's rule followed) |
| C7' | With enough codes it learns modular division | E7' (D23-D26) | **refuted** | 0 of 80; division was learned later with a factored state and a cover of the target machine (the division workplan, P2) |
| C8 | It learns a machine of a few states beyond 3: the modular sum at `q` 4 and 5 | E8, E8b, E8c (D27-D35) | **holds** | E8c: at least 3 of 6 certified at `q` 4 and 6 of 6 at `q` 5, each minimised to `q` states, every one passing every long-test case; E8 and E8b recorded memorising below the exclusion bound `m + q - 1` (D23) |

E6, an existence test (D37): the window without its transformer (the linear window) certifies
the carry automaton at `k` 1 (at least 2 of 6 seeds), so the FFN window wasn't needed.

## Holm's correction (D4)

| claim | raw p | Holm-adjusted | at 0.05 |
| --- | --- | --- | --- |
| C2 | 0.0033 | 0.013 | holds |
| C3 | 0.014 | 0.042 | holds |
| C4 (its refutation) | 0.044 | 0.088 | not refuted |
| C6 | 0.21 | 0.21 | not supported |

## Against the literature

- **Proof, not sampling.** Extraction work recovers small automata from recurrent networks and
  checks them by sampling or by equivalence queries; the certificate here enumerates every code a
  snapped network reaches against every window, so a certified network is correct at every length
  (C1), and sampled tests to 64 columns passed two networks that are wrong.
- **Spare codes.** Wang et al. (state-regularized RNNs) gave their networks spare centroids and
  reported how many were used; C3 measures success against the codebook's size: more codes, more
  networks that pass, not more that are correct.
- **Unexplored states.** Buitrago Ruiz & Gu attribute recurrent models' failures past training to
  states training never visits; here that isn't the failure (C6).

The comparisons in full, with citations, are P1's related work and the literature review.

## Where the evidence is

Every experiment's data under `data/window-recurrent/` with a sha256 manifest beside it, named in
its findings; E1's certified networks and its two counterexamples archived in indrajala-models
(`window-recurrent/2026-10-13-e1`, with their certificates and long tests, D12).

## Left over (in next-steps.md)

- E3's dropped cells (D34): `m` 32 and every `k` 1 cell; the curriculum against all at once at
  more cells, if the curriculum's role is asked again (C4 holds, with all at once leaning higher).
- The transformer against the linear window at `k` 1 `s` 16 `m` 16, trained to the bound with every
  step certified (E2b snapped's cell with only the window changed): not registered.
