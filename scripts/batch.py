"""
The batch runner (docs/batch-runner-workplan.md): a controller on one machine and a worker daemon on
each, over newline-delimited JSON on a TCP socket on 127.0.0.1 (indrajala_ml.batch).

    python scripts/batch.py register DIR --name NAME --mode measure|find --cell CELL --attempts SPEC
                                     [--cutoff BACKDROP=STEPS ...] [--k K] [--commit COMMIT]
    python scripts/batch.py prepare --hosts pyramidon,klip,stone --commit COMMIT
    python scripts/batch.py controller DIR [--port PORT] [--hosts pyramidon,klip,stone]
    python scripts/batch.py fleet --hosts pyramidon,klip,stone [--port PORT]
    python scripts/batch.py worker ROOT --slots N --code WORKTREE [--python PYTHON] [--host NAME] [--port PORT]
                                   [--pythonpath PATH] [--wrapper ARG ...]
    python scripts/batch.py status DIR
    python scripts/batch.py report DIR

SPEC is `s3:40-47,mtf4:32-35` (TASKS' short names); a cutoff is a backdrop's Luby t* in steps (the
run's --cutoff-steps). The controller serves until the batch is done; a worker runs its attempts in
ROOT/<attempt>/ from WORKTREE's scripts and reconnects after a dropped connection. Across machines
(stage 4): `prepare` puts each host's worktree at the batch's commit, `fleet` keeps a reverse ssh tunnel
to each remote host and starts its worker (HOSTS: slots, worktree, python, hours), and the controller
with `--hosts` gives work only inside each host's hours and pulls each remote attempt back, checked.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from indrajala_ml import batch as b

PORT = 47111


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="a new batch's directory and ledger")
    register.add_argument("dir", type=Path)
    register.add_argument("--name", required=True)
    register.add_argument("--mode", choices=("measure", "find"), required=True)
    register.add_argument("--cell", choices=sorted(b.CELLS), required=True)
    register.add_argument("--attempts", required=True, help="s3:40-47,mtf4:32-35")
    register.add_argument("--cutoff", action="append", default=[], help="BACKDROP=STEPS, each backdrop's t*")
    register.add_argument("--k", type=int, help="find mode: certificates a backdrop needs")
    register.add_argument("--commit", help="the commit every worker must run (D6)")
    prepare = commands.add_parser("prepare", help="each host's worktree at a commit")
    prepare.add_argument("--hosts", required=True)
    prepare.add_argument("--commit", required=True)
    fleet = commands.add_parser("fleet", help="each host's tunnel and worker")
    fleet.add_argument("--hosts", required=True)
    fleet.add_argument("--port", type=int, default=PORT)
    controller = commands.add_parser("controller", help="serve a batch until it is done")
    controller.add_argument("dir", type=Path)
    controller.add_argument("--port", type=int, default=PORT)
    controller.add_argument("--hosts", help="the fleet's hosts: their hours and pulls (D7)")
    worker = commands.add_parser("worker", help="run attempts on this machine")
    worker.add_argument("root", type=Path)
    worker.add_argument("--slots", type=int, required=True)
    worker.add_argument("--code", type=Path, required=True, help="the worktree whose scripts run")
    worker.add_argument("--python", default=sys.executable)
    worker.add_argument("--host")
    worker.add_argument("--port", type=int, default=PORT)
    worker.add_argument("--pythonpath", default="", help="before the worktree on PYTHONPATH (stone's crate)")
    worker.add_argument("--wrapper", action="append", default=[], help="an argument before each attempt's command")
    for name in ("status", "report"):
        commands.add_parser(name).add_argument("dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "register":
        cutoffs = {k: int(v) for k, v in (c.split("=") for c in args.cutoff)}
        made = b.attempts(args.attempts, cutoffs)
        b.register(args.dir, b.Batch(args.name, args.mode, args.cell, made, args.k, args.commit))
        print(f"registered {args.dir}: {len(b.State.read(args.dir).batch.attempts)} attempts")
    elif args.command == "prepare":
        repo = Path(__file__).resolve().parent.parent
        for name in args.hosts.split(","):
            for command in b.prepare(b.HOSTS[name], args.commit, repo, Path("/tmp")):
                subprocess.run(command, check=True)
            print(f"{name}: at {args.commit}")
    elif args.command == "fleet":
        asyncio.run(b.fleet([b.HOSTS[n] for n in args.hosts.split(",")], args.port))
    elif args.command == "controller":
        hosts = {n: b.HOSTS[n] for n in args.hosts.split(",")} if args.hosts else None
        asyncio.run(b.Controller(args.dir, hosts).serve(args.port))
        print("\n".join(b.report(args.dir)))
    elif args.command == "worker":
        worker = b.Worker(
            args.root,
            args.slots,
            args.python,
            args.code,
            host=args.host,
            pythonpath=args.pythonpath,
            wrapper=args.wrapper,
        )
        asyncio.run(worker.serve(args.port))
    elif args.command == "status":
        state = b.State.read(args.dir)
        print(f"{len(state.ended)} ended, {len(state.assigned)} running ({dict(state.assigned)}), done {state.done}")
    else:
        from indrajala_ml.studies import modular_family as mf  # the reading's cost, not the daemons'

        state = b.State.read(args.dir)
        print("\n".join(b.report(args.dir)))
        for backdrop in sorted({a.backdrop for a in state.batch.attempts}):
            ended = [state.ended[a.id] for a in state.batch.attempts if a.backdrop == backdrop and a.id in state.ended]
            if ended:
                print(f"  {backdrop}: {mf.restart_summary(mf.restart_cost(b.runs_of(ended)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
