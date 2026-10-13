"""
The window-recurrent adder (docs/window-recurrent-adder-workplan.md): whether a network that reads a
window of k columns and hands a snapped state from window to window learns, through a curriculum of
window counts, to add base-3 numbers of any length, under indrajala_ml/studies/window_recurrent_study.py's
harness and indrajala_ml/data/addition_data.py's catalogue.

    python scripts/window_recurrent_study.py run CONFIG.json
    python scripts/window_recurrent_study.py status CONFIG.json
    python scripts/window_recurrent_study.py report CONFIG.json [--md FILE]
    python scripts/window_recurrent_study.py check NETWORK.json --n N [N ...] [--properties ID ...]
        [--sampled S] [--per-stratum P] [--doubling] [--json FILE]
    python scripts/window_recurrent_study.py certify NETWORK.json [...] [--out DIR] [--workers W] [--snap-ratio R]

A config names the study's directory (results/, one JSON file per run, and models/, each run's steps
and their networks in format 2), its workers, its settings and its matrix (load_config's
docstring). `run` resumes from what is on disk, a run from its last saved step, and prints a line
per finished run, then the report; it holds the machine lock, as every study. `check` loads a saved
step's network and runs the catalogue (without B3) at each length: its properties, B1-B3's,
A3's, M1's and M5's samples, M3's and M4's cases per stratum and M3 at 1, 2, 4, ..., n only (the
long tests of docs/window-recurrent-claims-workplan.md, D9) as given. `certify` writes each saved
network's certificate (indrajala_ml/studies/window_recurrent_certificate.py) as
DIR/<run>-<step>.json and prints a line for each, over W processes; --snap-ratio 1 certifies the
network's own snaps only, branching no near tie (strict, D3 as first written).

OPENBLAS_NUM_THREADS=1, as every study.

Findings (stage 9), on jebel (i7-9700K), indrajala-ml 6b94933, crate 8c20e510c9a9,
`scripts/window_recurrent_matrix.json`: the curriculum arm, `k` 1, 2, 4 x `s` 4, 16 x `m` 4, 8,
16, 5 seeds, 3 steps (1 window, then 1-2, then 1-3), at most 200 epochs a step, patience 30, the
catalogue's default sizes, each step's test pass up to its first failing length (W12), 6 workers.
90 runs, none raised. Wall time 2026-10-12 21:43 to 2026-10-13 05:14 (7.5 h; 43.4 h of runs, 24.6
of them training and 18.8 testing); a run took 2-67 minutes (median 27). The results are on jebel
and pyramidon in `data/window-recurrent/matrix/results/` (90 files, listed with their sha256 in
`data/window-recurrent/matrix/results.sha256`, itself sha256 407d6ec3...7941cab9), every step's
network beside them in `models/` (156 MB), and the log in `data/window-recurrent-matrix.log`
(sha256 b47626f0...68a7dd59).

- Half the runs work (W10: a break of 16 columns or more): 45 of 90. 43 of them pass every
  property at every test length up to 64 columns, after training on at most 3 windows: 3, 6 and 12
  columns at `k` 1, 2 and 4, window ratios of 21.3, 10.7 and 5.3. The other two break at 24 and 32
  (`k` 4, one B1 case in 100,000 wrong at the next length). In every working run the carry reaches the
  whole length (`L*` = n at every test length).
- It takes the recurrence: no run works after step 1, which trains one window and so never hands
  a state on; 9 work from step 2 (2 windows; at `k` 1, two columns of training add 64), 36 more
  from step 3, and none that worked at step 2 stopped working at step 3. A working run's step 3
  passed its check in a median of 2 epochs.
- Failure is mostly all or nothing: 43 of the 45 failing runs fail at 8 columns already, most
  across B1, A2, M2-M5 (their step 3 best screens: median 0.924). A few are near misses: two `k` 4
  runs break at 12 (their trained length) with 1-2 cases in 100,000 wrong at 16, and one `k` 2 run
  misses 4 of B1's 100,000 at 8 (and a few of A1's, M3's and M5's).
- Passing the check isn't working at `k` 1: 5 runs passed their step 3 check (every triple at 1-3
  columns) and still got 0.2-8.3% of B1 wrong at 8.
- By axis: `k` 1, 2, 4 work in 16, 11 and 18 of 30 runs; `s` 4 and 16 in 24 and 21 of 45 (the
  state's width barely matters); `m` 4, 8 and 16 in 10, 16 and 19 of 30 (more codes, more working
  runs). No cell works on every seed: the best, 4 of 5, are `k` 1 `s` 16 `m` 8, `k` 4 `s` 4 `m` 16
  and `k` 4 `s` 16 `m` 16; the worst, `k` 2 `s` 16 `m` 4, on none. Within a cell the seed decides,
  so what fails is the optimisation, not the capacity: every cell has a state that can hold a
  carry (3 values), and 6 working runs use exactly 3 codes.
- The codes used by working runs: 3-11 (median 5) over the lengths tested, so a working network
  uses more codes than the carry's 3 values; the probes (stage 10) say what the others hold.
- Cost: a step 3 epoch trains in a median 6.8 s; a working run's step 3 test pass (every length)
  took a median 17.6 minutes, a failing one's (length 8 alone) about a minute.

Per cell, its seeds working and each seed's break (in columns, seeds 0-4) after steps 2 and 3, and
the codes the working seeds used:

| k | s | m | parameters | seeds working | breaks after step 2 | breaks after step 3 | codes used (working) |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 4 | 7,683 | 2/5 | 0/64/64/0/0 | 0/64/64/0/0 | 4,4 |
| 1 | 4 | 8 | 7,699 | 3/5 | 0/0/0/0/0 | 0/64/64/0/64 | 8,6,7 |
| 1 | 4 | 16 | 7,731 | 2/5 | 0/0/0/0/0 | 0/64/64/0/0 | 11,11 |
| 1 | 16 | 4 | 9,651 | 2/5 | 64/0/0/64/0 | 64/0/0/64/0 | 4,4 |
| 1 | 16 | 8 | 9,715 | 4/5 | 0/0/64/0/0 | 64/64/64/0/64 | 6,4,4,7 |
| 1 | 16 | 16 | 9,843 | 3/5 | 0/0/0/0/0 | 64/64/64/0/0 | 7,7,6 |
| 2 | 4 | 4 | 7,699 | 1/5 | 0/0/0/0/0 | 0/0/64/0/0 | 3 |
| 2 | 4 | 8 | 7,715 | 3/5 | 0/64/0/0/0 | 0/64/64/0/64 | 4,6,4 |
| 2 | 4 | 16 | 7,747 | 3/5 | 0/0/0/0/0 | 0/64/0/64/64 | 8,7,5 |
| 2 | 16 | 4 | 9,667 | 0/5 | 0/0/0/0/0 | 0/0/0/0/0 | - |
| 2 | 16 | 8 | 9,731 | 1/5 | 0/0/0/0/0 | 64/0/0/0/0 | 6 |
| 2 | 16 | 16 | 9,859 | 3/5 | 0/0/0/0/0 | 0/0/64/64/64 | 9,4,8 |
| 4 | 4 | 4 | 7,731 | 3/5 | 8/0/64/0/64 | 64/0/64/0/64 | 4,3,3 |
| 4 | 4 | 8 | 7,747 | 3/5 | 0/0/12/0/0 | 0/64/64/12/64 | 4,5,5 |
| 4 | 4 | 16 | 7,779 | 4/5 | 0/0/0/0/8 | 64/64/12/32/64 | 7,5,7,7 |
| 4 | 16 | 4 | 9,699 | 2/5 | 0/0/0/0/8 | 0/64/0/0/64 | 3,3 |
| 4 | 16 | 8 | 9,763 | 2/5 | 64/0/0/0/0 | 64/0/0/64/0 | 4,3 |
| 4 | 16 | 16 | 9,891 | 4/5 | 8/0/0/0/0 | 64/24/64/64/0 | 5,6,4,6 |

Findings (the control, W16 and W30), on jebel, indrajala-ml 1dd22f9, crate 8c20e510c9a9,
`scripts/window_recurrent_control.json`: the all-at-once arm (one step on window counts 1-3
together, at most 200 epochs) at the matrix's three best cells, seeds 0-4, the matrix's settings
otherwise. 15 runs, none raised. Wall time 2026-10-13 05:28 to about 06:29 (5.3 h of runs; a run
took 1-33 minutes, median 23). The results are on jebel and pyramidon in
`data/window-recurrent/control/results/` (15 files, listed with their sha256 in
`data/window-recurrent/control/results.sha256`, itself sha256 6eed795f...a6facf9b), the networks
beside them in `models/` (9.2 MB), and the log in `data/window-recurrent-control.log` (sha256
913e996f...c52662dd).

- The curriculum isn't needed here: the control works as often, 12 of 15 runs against the
  curriculum's 12 of 15 at the same cells and seeds. At `k` 4 it works on all 10 (curriculum 8 of
  10; Fisher's exact test p 0.47); at `k` 1 `s` 16 `m` 8 on 2 of 5 (curriculum 4 of 5; p 0.52).
  At 5 seeds neither difference is decided; the claims workplan's E4 measures it at 20.
- It needs far fewer examples where it works: at `k` 4 it passes its check within 4-16 epochs,
  0.26-1.05 M examples (median 0.39 M), against the curriculum's 4.8-12.3 M over its three steps
  for the runs that work (median 6.4 M), about 16 times fewer. The curriculum spends 78-99% of
  its examples on its first two steps (one window, then one or two; 61 of the matrix's 90 runs
  stop step 1 on the plateau), and once it trains on 1-3 windows it passes in a few epochs, as the
  control does from the start.
- Which seeds work isn't shared between the arms: at `k` 4 the curriculum's two failing seeds
  (seed 2 of `s` 4, seed 4 of `s` 16) work all at once, and at `k` 1 the arms fail on different
  seeds (all at once: 0, 2, 4; curriculum: 3).
- Exact on short sums still isn't generalising: `k` 1's seed 2 passed its check (every sum of 1-3
  columns) in 3 epochs and failed at 8, as five of the matrix's `k` 1 runs did.

| cell | all at once: working, breaks (seeds 0-4) | epochs | curriculum: working, breaks after step 3 |
|---|---|---|---|
| `k` 1 `s` 16 `m` 8 | 2/5, 0/64/0/64/0 | 42, 1, 3, 3, 46 | 4/5, 64/64/64/0/64 |
| `k` 4 `s` 4 `m` 16 | 5/5, 64/64/64/48/64 | 13, 6, 7, 5, 6 | 4/5, 64/64/12/32/64 |
| `k` 4 `s` 16 `m` 16 | 5/5, 64/64/64/64/64 | 5, 4, 16, 6, 9 | 4/5, 64/24/64/64/0 |
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import multiprocessing
import sys
import time
from pathlib import Path

from indrajala_ml.data import addition_data as ad
from indrajala_ml.measurement import machine_lock
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_recurrent_study as wr


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "report"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "report":
            command.add_argument("--md", type=Path, help="write the report to this file too")
    check = commands.add_parser("check")
    check.add_argument("network", type=Path)
    check.add_argument("--n", type=int, nargs="+", required=True, help="the lengths to check")
    check.add_argument("--properties", nargs="+", help="these properties only (default: all but B3)")
    check.add_argument("--sampled", type=int, default=ad.SAMPLED, help="B1-B3, A3, M1 and M5's sample")
    check.add_argument("--per-stratum", type=int, default=ad.PER_STRATUM, help="M3's per distance, M4's per carry")
    check.add_argument("--doubling", action="store_true", help="M3 at distances 1, 2, 4, ..., n only")
    check.add_argument("--json", type=Path, help="write each length's results and seconds to this file")
    certify = commands.add_parser("certify")
    certify.add_argument("networks", type=Path, nargs="+")
    certify.add_argument("--out", type=Path, help="write each certificate here, as <run>-<step>.json")
    certify.add_argument("--workers", type=int, default=1)
    certify.add_argument(
        "--snap-ratio", type=float, default=wc.SNAP_RATIO, help="near ties branch under this ratio (1: strict)"
    )
    args = parser.parse_args(argv)

    if args.command == "check":
        catalogue = ad.catalogue(args.sampled, ad.RELATION_SAMPLED, args.per_stratum, doubling=args.doubling)
        properties = [p for p in wr.test_properties(catalogue) if args.properties is None or p.id in args.properties]
        network = load_network(str(args.network))
        model = wr.WindowAdder(network, network.columns)
        results: dict[str, object] = {}
        for n in args.n:
            start = time.perf_counter()
            checked = ad.evaluate(model.adder(n), n, properties, shrink=False)
            for result in checked:
                print(
                    f"n {n}, {result.id}: {result.passed}/{result.cases}"
                    + ("" if result.holds else f", e.g. {result.failures[0]}"),
                    flush=True,
                )
            results[str(n)] = {
                "properties": {r.id: wr.result_json(r) for r in checked},
                "seconds": time.perf_counter() - start,
            }
        if args.json is not None:
            args.json.write_text(json.dumps({"network": str(args.network), "lengths": results}, indent=1))
        return 0
    if args.command == "certify":
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("forkserver")
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            for path, result in zip(
                args.networks,
                pool.map(functools.partial(wc.certify_file, snap_ratio=args.snap_ratio), args.networks),
                strict=True,
            ):
                name = f"{path.parent.name}-{path.name.removesuffix('.network.json')}"
                print(f"{name}: {wc.summary(result)}", flush=True)
                if args.out is not None:
                    (args.out / f"{name}.json").write_text(json.dumps(result, indent=1))
        return 0
    study = wr.load_config(args.config)
    if args.command == "run":
        with machine_lock.study():
            wr.run(study, log=lambda line: print(line, flush=True))
        print(wr.report(study))
    elif args.command == "status":
        print(wr.status(study))
    else:
        text = wr.report(study)
        print(text)
        if args.md is not None:
            args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
