import time

from indrajala_ml.benchmark_sweep import estimate_sweep_wallclock, run_parameter_sweep, summarize_sweep_results


def _toy_worker(shared_context: object, config: int, seed: int) -> float:
    # deterministic in (config, seed): tests the runner's dispatch, seeding and aggregation
    return config * 10.0 + seed


def _sleepy_worker(shared_context: object, config: int, seed: int) -> float:
    time.sleep(0.05)
    return float(config + seed)


def _shared_context_worker(shared_context: dict[str, float], config: int, seed: int) -> float:
    # shared_context must reach every worker as passed; a module-level global would work only
    # under fork, not spawn
    return shared_context["offset"] + config + seed


def test_run_parameter_sweep_dispatches_every_config_seed_pair_exactly_once():

    results = run_parameter_sweep(configs=[1, 2, 3], seeds=[0, 1], worker_fn=_toy_worker, report_progress=False)

    assert set(results.keys()) == {1, 2, 3}
    for config in (1, 2, 3):
        assert sorted(results[config]) == sorted(config * 10.0 + seed for seed in (0, 1))


def test_run_parameter_sweep_result_count_matches_configs_times_seeds():

    results = run_parameter_sweep(configs=[5, 9], seeds=[0, 1, 2], worker_fn=_toy_worker, report_progress=False)

    for config in (5, 9):
        assert len(results[config]) == 3


def test_run_parameter_sweep_delivers_shared_context_to_every_worker():

    results = run_parameter_sweep(
        configs=[1, 2],
        seeds=[0, 1],
        worker_fn=_shared_context_worker,
        shared_context={"offset": 100},
        report_progress=False,
    )

    assert sorted(results[1]) == [101.0, 102.0]
    assert sorted(results[2]) == [102.0, 103.0]


def test_estimate_sweep_wallclock_projects_from_a_single_real_run():

    projected = estimate_sweep_wallclock(
        _sleepy_worker, sample_config=0, sample_seed=0, planned_run_count=20, worker_count=4
    )

    # 20 jobs / 4 workers = 5 serial batches, each taking roughly _sleepy_worker's own ~0.05s
    assert 0.15 <= projected <= 0.5


def test_estimate_sweep_wallclock_passes_shared_context_through_to_the_worker():

    # returns a timing, not the worker's value, so the check is that it doesn't raise: a
    # missing shared_context makes the worker's lookup raise TypeError
    projected = estimate_sweep_wallclock(
        _shared_context_worker,
        sample_config=0,
        sample_seed=0,
        planned_run_count=1,
        shared_context={"offset": 100},
        worker_count=1,
    )

    assert projected >= 0.0


def test_summarize_sweep_results_renders_a_markdown_table():

    table = summarize_sweep_results({"lr=0.1": [0.9, 0.92, 0.94], "lr=0.5": [0.5, 0.5, 0.5]})

    lines = table.splitlines()
    assert lines[0] == "| config | mean | stdev |"
    assert lines[1] == "|---|---|---|"
    assert any(line.startswith("| lr=0.1 |") for line in lines)
    assert any(line.startswith("| lr=0.5 | 50.00% | 0.00% |") for line in lines)
