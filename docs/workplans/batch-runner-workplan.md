# Workplan: the batch runner (restarts at a cutoff, across machines)

**Status: written 2026-10-22, at the owner's request ("plan the missing batch-running infrastructure:
should it be able to split across multiple machines ?"). D1-D3 and D5-D8 settled under the owner's
delegation, each with its options' pros and cons; D4, the architecture, is the owner's: a controller on
`pyramidon` and a socket-connected worker daemon on each machine.**

## Why

The method is statistical (the owner, 2026-10-22: a parallel batch, a clear majority certifying), and
the literature review of 2026-10-22 showed that its runs are Las Vegas runs: a certified run is right
whenever it stops. So the cheapest way to a certificate is fresh-seed restarts at a fixed cutoff
`t*` (Luby, Sinclair and Zuckerman 1993), read per backdrop from the runs themselves
(`modular_family_probe.py restart-cost`): on the completed runs, 3.5 times cheaper per certificate
than running each seed out on S3, 2.2 times on move-to-front-4, about 9 times on S4 (in epochs, the
time axis, re-read 2026-10-23; first read in steps as 4.4, 1.9 and 10). The collapse's
probe (D63 in [state-formation-workplan.md](state-formation-workplan.md)) decides whether the reheat
joins the restarts; either way the method then runs as a batch of attempts cut at a cutoff, which no
current tool does: the probe scripts run a fixed list of seeds each to its budget, on one machine.

What is missing: a run that stops itself at a cutoff, and a runner that launches attempts, starts a
fresh seed in a slot whose attempt was cut, stops by the batch's rule, records every attempt, and uses
the machines we have, each in its own hours.

## Decisions

- **D1. Where the cutoff lives.**
  - (a) *Chosen:* in the run: `--cutoff-epochs T` (first `--cutoff-steps`; epochs since stage 5),
    the run ending "cut" at the first step's end with `T` epochs trained if not certified (D41) by then; off by default, so the golden run is unchanged. Pros: the stop is
    deterministic (a step's end, the same on every machine, so an attempt's outcome doesn't depend on
    its host or the runner's timing); the end label tells a cut attempt from a spent one; a single run
    can be cut without the runner. Cons: a package change to the most-run code (the run's state
    object, D62, makes it one field and one check).
  - (b) The runner kills the process at `T` steps read from the log. Cons: the stop lands wherever the
    poll does; the last step's network and certificate may be half-written.
- **D2. The cutoff's value.**
  - (a) *Chosen:* per backdrop and arm, `t*` from `restart-cost` on that arm's completed runs, read
    before a batch and fixed through it (registered with the batch); a backdrop or arm with no runs
    yet gets an uncut first batch (as every probe so far), whose runs then give its `t*`. Pros: no
    constant (the owner's rule); the cutoff is the one the theory says is optimal for that arm's own
    run times. Cons: `t*` from few runs is rough (S4's from 3); a first uncut batch costs what the
    probes cost today.
  - (b) Luby's universal sequence (cutoffs 1, 1, 2, 1, 1, 2, 4, ... times a unit). Pros: needs no
    runs, within a log factor of optimal. Cons: its unit is a constant; it wastes most attempts on
    cutoffs far below any success (none of ours before step 13).
  - (c) A fixed cutoff (the first sizing's 200 steps). Cons: a constant.
- **D3. The batch's modes and stopping rules.**
  - (a) *Chosen:* two modes. **Measure:** a fixed number of attempts per backdrop, registered before
    the data, each cut at `t*`: the rate per attempt (Wilson), `l(t*)` per certificate, and the paired
    verdicts as now; no optional stopping, so the rates are unbiased. **Find:** attempts until `k`
    certificates (machine discovery's use: one proved machine, stages C-F of
    [machine-discovery.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/machine-discovery.md)), with a cap on attempts. Pros: each question gets
    the rule that answers it. Cons: two code paths.
  - (b) Measure only. Cons: discovery wants the first certificate, not a rate.
  - (c) Until `k` certificates for both. Cons: the rate from a stop-at-`k` batch is biased unless read
    as negative binomial; the registered claims would change form.
- **D4. Multiple machines: a controller and workers over sockets (the owner, 2026-10-22: "ideally
  i would like to have a low level socket-based server-client architecture - running at low resources
  on 1 core on each machine"; "so there is a controller (pyramidon), and worker machine (with multiple
  cores each)").** A batch runs across every available core of every machine.
  - (a) *Chosen, the owner's:* **a controller on `pyramidon`** (the server: the batch's queue, its
    ledger, D5, and the hosts' hours, D7) and **a worker daemon on each machine** (`stone`, `jebel`,
    `klip`, and `pyramidon` itself), each a single low-resource process (Python's standard library,
    asyncio, niced, idle in `select` between messages: one core's worth at most, the rest of the
    machine's cores its run slots). A worker connects to the controller, says what it is (host, commit,
    crate hash, free slots, memory), asks for an attempt whenever a slot frees (pull: self-balancing,
    and the controller assigns each attempt once, so no claim race), launches it as its own detached
    process group, reports its start, its steps and its end, and re-adopts its running attempts from
    their pid files after its own restart. The runs survive a dropped connection; the worker reconnects
    and reports. **Transport:** newline-delimited JSON over TCP, one versioned message per line
    (worker to controller: `hello`, `ready`, `started`, `progress`, `ended`, `heartbeat`; controller to
    worker: `attempt` (batch, backdrop, a registered cell by name, seed, cutoff), `stop` (a window's
    end), `drain`); an attempt names a registered cell, never a command line, so a worker runs nothing
    else. **Security and `stone`'s sandbox:** the controller listens on `pyramidon`'s 127.0.0.1 only and
    no worker opens a port; each worker reaches it through an ssh reverse tunnel the controller opens
    (`ssh -N -R PORT:127.0.0.1:PORT host`), re-opened when it drops (`stone`'s wifi). Authentication is
    ssh's, already in use; nothing new is installed or authenticated from `stone` (no token, no
    outbound login). **Bulk data** (logs, step networks: about 2 MB a network) by `rsync` from the
    controller, with checksums, after each attempt ends; only the summary travels on the socket.
    Pros: every core used, each machine in its own hours; a cut attempt is short (S3's `t*` about 137
    steps, under an hour), so a closing window loses at most one; cross-machine bit parity means an
    attempt's outcome doesn't depend on its host, and a lost attempt re-run elsewhere with its seed
    gives the same result; the daemons cost almost nothing at rest. Cons: the most code of the
    options (a protocol, two daemons, tunnels); a socket protocol to test (stubs for the run and the
    network).
  - (b) Push over ssh from `pyramidon` (the first draft): the controller launches and polls each
    attempt by ssh. Cons: an ssh session per launch and poll; no worker to re-adopt runs; balancing
    by the controller's guess of each host's load.
  - (c) One machine per batch, split by hand (today's practice). Cons: a machine's queue can't drain
    into another's; every split a manual plan.
  - (d) Workers claiming from a shared queue on disk (by `mkdir` over ssh). Cons: the claim race seen
    2026-10-16; `stone` may not authenticate outward.
- **D5. The ledger.** (a) *Chosen:* an append-only `ledger.jsonl` per batch on `pyramidon`: the batch's
  registration (backdrops, arms, seed ranges, cutoffs, mode, commit, crate hash), then one line per
  event (an attempt queued, launched on a host, ended with its outcome: certified, cut, spent, stopped,
  lost). The controller rebuilds its state from it on start, so it resumes after its own restart or a
  reboot; the session's waiters read it. Pros: durable, inspectable, the batch's record for its
  reading. Cons: none found against a database for this size. (b) State in memory. Cons: lost with the
  process.
- **D6. Code and machines pinned.** (a) *Chosen:* every host runs the batch's commit, from a worktree
  (`stone` from a bundle, per its sandbox), with the shared `.venv`'s crate; the controller checks each
  host's commit and the crate extension's hash before each launch and refuses a mismatch. Pros:
  pairing and parity hold by construction. Cons: a host updated mid-batch stalls until fixed.
- **D7. Hosts' hours and resources.** (a) *Chosen:* a host table (slots, hours by day of week, memory
  per run by backdrop, a wrapper: `klip`'s thermal guard; `jebel`'s PL1 checked at its first launch of
  the day); an attempt is launched only if its expected time (its `t*` in steps times the host's
  measured seconds per step) ends inside the host's window; an attempt still running at a window's end
  is stopped and re-queued as lost (its seed redone elsewhere, identically). The owner's rules as
  data: `jebel` off overnight (solar), the laptops at full load overnight, `stone` weekdays 18:00-07:00.
  Pros: the rules live in one place, read by the controller. Cons: seconds per step vary by backdrop
  and length; the estimate is refined from the ledger.
- **D8. Seeds.** (a) *Chosen:* each batch draws from registered, never-used seed ranges per backdrop,
  in order; arms compared on the same seeds (pairing). A lost attempt keeps its seed. Pros: no seed
  chosen after the data (over-tuning); paired verdicts as McNemar needs. Cons: a range's seeds are
  spent even when an attempt is cut early.

## Stages

1. **This plan.**
2. **The cutoff in the run** (D1): `--cutoff-steps` (in epochs since stage 5), its end label "cut", a field of the run's state;
   tests (a cut run's rows identical to the uncut run's up to the cutoff); the golden run unchanged;
   `restart-cost` reading "cut" as a finished failure. Tier 0 by its timing (no timed path), the
   golden run because it touches the run's loop.
3. **The controller and the worker, on one machine** (D3, D4, D5, D8): `scripts/batch.py` with
   `controller` (the queue, the ledger, the protocol's server on 127.0.0.1), `worker` (the daemon:
   slots, launch, report, re-adopt), `register`, `status`, `report` (the measure mode's rates and `l`,
   the find mode's certificates); the protocol versioned and tested with a stub run (fast,
   deterministic) and a stub worker: the ledger's resume, both modes and their stopping rules, a
   dropped connection, a worker's restart re-adopting its runs, a controller's restart rebuilding from
   the ledger; then one real S3 attempt end to end on `pyramidon`.
4. **Every machine** (D4, D6, D7): the host table (slots, memory per run by backdrop, hours, `klip`'s
   thermal guard, `jebel`'s PL1), the reverse tunnels and their re-opening, workers on `klip`, `jebel`
   and `stone` (by bundle) in pinned worktrees, the `rsync` pulls with checksums, the windows' stops and
   the lost attempts' re-queue; checked end to end with short S3 attempts (cut at a few steps) on all
   four, each attempt's log identical to a local run of its seed, and the daemon's own CPU and memory
   measured at rest and under load.
5. **The cutoff in epochs, then first use.** Epochs trained are the time axis (the owner, 2026-10-23;
   [experiment-design.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/protocols/experiment-design.md), section 1): the run's cutoff becomes `--cutoff-epochs`
   (epochs trained in all at a step's end), the attempts' costs and the report's `l` in epochs. Steps
   understate epochs early in a run (the warm-up and hold train several a step; D64's S3 43 certified at
   step 11, epoch 23), exactly where cutoffs sit. **Built** (2026-10-23): `--cutoff-epochs`; the
   attempts' summaries carry `epochs` (success and last step's end in epochs trained), `runs_of` and the
   report in epochs, a host's time estimate from its seconds an epoch. Then the first use: the method's next fresh-seed batch in measure mode, with D63's verdict deciding the
   arm (the plain method with restarts, or the reheat with restarts), registered as its own decision
   in the state-formation workplan.
6. **Docs and retire:** the runner in [working-notes.md](../working-notes.md) (its hosts, its ledger, its
   gotchas); this plan retired, its leftovers to [next-steps.md](../next-steps.md).

## Known limits

- Cutoffs and costs were in steps in stages 2-4; stage 5 moved them to epochs, the time axis. A ledger
  from before it (the stage-4 checks) holds summaries without `epochs`: `runs_of` leaves them out.

- Restarts assume attempts independent and identically distributed: true of fresh seeds by
  construction, not of attempts on different backdrops (each backdrop has its own `t*`).
- A cut attempt's networks are kept (its last step's), but a certificate that would have come after
  the cutoff is lost by design: the reheat's late recoveries are the case D63 reads apart.
- `stone`'s flaky wifi makes its pulls the most likely to fail; a pull is retried, never assumed.
