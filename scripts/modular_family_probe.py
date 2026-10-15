"""
The modular family's probe (docs/modular-family-workplan.md, stage 2): one network answering the
running quotient, sum and difference of two numbers mod p at every window (query tokens), against
the quotient alone; a factored state of C9''s cell, trained by the curriculum within a wall-clock
budget, diagnosed as it trains (indrajala_ml/studies/modular_family.py).

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/modular_family_probe.py run \\
        --arm family --seed 0 --budget-minutes 60 --out data/window-recurrent/modular-family-probe/family-seed0

Each run writes `log.jsonl` (a line per diagnosis: each operation's accuracy, the quotient by
class, each factor's codes used and its share of A, B, the sum S, the difference D and the zero
tests; at each step's end, the quotient's accuracy at 24 and 48 columns) and `network.json` (the
network at its last step's end).

Findings (stage 2 of docs/modular-family-workplan.md), on jebel (i7-9700K), one process a run,
OPENBLAS_NUM_THREADS=1, 2026-10-15; p 5, two factors of 16 codes, s 16 each, k 1 column (D3), the
probe's settings (D4), a 60-minute budget a run. Data in data/window-recurrent/modular-family-probe/
(a directory a run: log.jsonl, network.json; manifest `modular-family-probe.sha256`,
4d465007...a9847c25, 41 files, run from data/window-recurrent), the D10 check's in
data/window-recurrent/probe-d10/ (`probe-d10.sha256`, 4236a059...557ae9c3, 81 files), the certified networks and their certificates in
modular-family-probe/certify/ (in the first manifest); both machines.

- **Registered arms (the poolings before the state, D10, as C9's cell), indrajala-ml 9b4be51,
  06:08-07:09: none works.** family seeds 0 and 1 and quotient seeds 0 and 1 all end their hour
  with the nonzero quotients at 0.33-0.37 (a guess 0.25), the state's share of the pair 0.22-0.27,
  24-30% right at 24 columns; every factor's share of A, B, S or D at most 0.18, the shares frozen
  to three decimals from step to step. The family made no difference.
- **The cause is D10 (the D10 check: the harness, E8c's own cell and task, `modular-sum-5`, seeds 0
  and 1, 30 minutes each, the same commit):** one factor whose pooling reads before the state passes
  steps 1-2, then plateaus at every step from 3 (31-91 epochs each, break 0): 0 of 2 certified; the
  same factor reading the window's output certifies as E8c did, step for step (seed 0 at step 4,
  seed 1 at step 5): 2 of 2. Reading before the state, a factor's cell gets only the embedded column
  and must compute the whole transition itself; reading the output, the window's three blocks do
  most of it (a reading, not tested apart). Every factored division run (C9, C9', C9'') read before
  the state.
- **Reading the output (indrajala-ml 600b19b, 06:53-07:54), the family certified 2 of 2; the
  quotient alone 0 of 2.** family seed 1 was right on every answer at 3-12 columns and at 24 and 48
  from step 7, seed 0 from step 6. Their networks saved at steps 11 (seed 1, sha256 718b382d...4bd595be)
  and 12 (seed 0, 77b558a0...838bc3ef) **certify against division's machine, both verdicts** (certify,
  2da6d0b, pyramidon): seed 1 branched 33 states reached, minimised to 25, 0 wrong, 2 near ties,
  logit lead >= 2.03; strict 32 reached, minimised to 25, snap ratio >= 1.018, lead >= 3.81; seed 0
  branched 53 reached, minimised to 25, 6 near ties, lead >= 2.21; strict 52, minimised to 25, snap
  ratio >= 1.025, lead >= 2.21. Their sums certify against the modular sum too (minimised to 5). The
  first certified networks for division. seed 1's step-6 network, 98.7% right at 48 columns on
  samples, fails the certificate (1,025 wrong outputs): sampled tests overstate "works". The
  quotient alone reached nonzero quotients of at most 0.64 and at most 36% at 24 columns; seed 1
  collapsed at step 9 (codes 2 and 5).
- **The state isn't factored by residue.** At seed 1's step 6 each factor uses 14 codes and holds
  0.37-0.71 of every direction xA + yB (no factor a function of one), the pair of codes 63 states
  for the 25; the certificate's readout agrees (neither factor's code a function of the target's
  state). So the registered success's factor clause (a factor at 0.9 of A, B, S or D) isn't met;
  its quotient clause is: an outcome the plan didn't name, division solved with its state spread
  over both codebooks.
- **Training past a solution destroyed it, in every run that got far:** family seed 1 collapsed at
  step 12 (codes 4 and 4), seed 0 at step 14, both quotient runs by the hour's end (all four ended
  at nonzero quotients 0.13-0.31, 20-30% at 24 columns). The certified networks had been saved first.
  The probe now certifies each step's quotient and stops at the first certified (38f787c), as the
  harness's `stop_at_certified` does (E8b).
- Two counts at 2 seeds an arm are a screen: the family's part (2 of 2 against 0 of 2) is
  suggestive, not shown. Before launch the probe's examples lacked the harness's final zero window
  and the state never split in either arm (the step at one column passed on the column alone); found
  in local checks, fixed in 9b4be51 before any registered run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from indrajala_ml.studies import modular_family as mf
from indrajala_ml.studies import window_recurrent_certificate as wc

ARMS = {"family": mf.FAMILY, "quotient": mf.QUOTIENT}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="one probe run")
    run.add_argument("--arm", choices=sorted(ARMS), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--p", type=int, default=5)
    run.add_argument("--factors", type=int, default=2, help="factors of 16 codes, s 16 each")
    run.add_argument(
        "--reads", choices=["before_state", "output"], default="before_state", help="the poolings' tokens (D10)"
    )
    run.add_argument("--budget-minutes", type=float, default=60.0)
    run.add_argument("--out", type=Path, required=True)
    certify = commands.add_parser("certify", help="a saved family network's certificate, one slot")
    certify.add_argument("network", type=Path)
    certify.add_argument("--task", required=True, help="modular-division-p (the quotient) or modular-sum-p (the sum)")
    certify.add_argument("--snap-ratio", type=float, default=wc.SNAP_RATIO)
    certify.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "certify":
        p = int(args.task.rsplit("-", 1)[1])
        result = mf.certify_file(args.network, mf.Encoding(p, mf.FAMILY), args.task, args.snap_ratio)
        args.out.write_text(json.dumps(result, indent=1))
        print(wc.summary(result))
        return 0
    encoding = mf.Encoding(args.p, ARMS[args.arm])
    mf.run(encoding, [(16, 16)] * args.factors, args.seed, args.budget_minutes * 60, args.out, reads=args.reads)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
