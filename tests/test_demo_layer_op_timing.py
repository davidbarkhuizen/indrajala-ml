from indrajala_ml.demos import demo_layer_op_timing as demo
from indrajala_ml.demos.registry import DEMOS


def test_is_registered():

    assert demo.__name__ in {info.module for info in DEMOS}


def test_every_case_runs_on_both_backends():

    # timings aren't asserted - only that every op runs on both backends and yields a row
    for case in demo.all_cases(batch_sizes=(1, 2)):
        row = demo.time_case(case, calls=1, loops=1)
        assert row.numpy_us > 0.0 and row.rust_us > 0.0, (case.shape, case.op, case.batch)
        assert demo.format_row(row).startswith(case.shape)


def test_covers_every_shape_and_single_and_batch_ops():

    cases = demo.all_cases(batch_sizes=(1, 32))
    assert {case.shape for case in cases} == {label for label, *_ in demo.DENSE_SHAPES} | {
        label for label, _side in demo.CONV_SHAPES
    }
    dense_ops = {case.op for case in cases if case.shape == demo.DENSE_SHAPES[0][0]}
    assert {"downstream", "accumulate_gradient", "sgd step", "accumulate_gradient_batch", "forward_batch"} <= dense_ops
    assert {case.batch for case in cases} == {None, 1, 32}


def test_batch_ops_get_proportionally_fewer_calls():

    assert demo.calls_per_loop(None, 300) == 300
    assert demo.calls_per_loop(1, 300) == 300
    assert demo.calls_per_loop(32, 300) == 10
    assert demo.calls_per_loop(512, 300) == 10
    assert demo.calls_per_loop(4, 300) == 75
