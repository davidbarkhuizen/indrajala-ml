from indrajala_ml.studies.common import mean_sd, table


def test_mean_sd_formats_the_mean_and_sample_sd():
    assert mean_sd([0.9, 0.95, 1.0]) == "95.00% ± 5.00%"
    assert mean_sd([1.0, 2.0, 3.0], ".1f") == "2.0 ± 1.0"


def test_mean_sd_of_one_value_has_no_spread():
    assert mean_sd([0.5]) == "50.00% ± 0.00%"


def test_table_is_markdown_with_a_separator_row():
    assert table(["rate", "accuracy"], [["0.1", "90%"], ["0.2", "91%"]]) == (
        "| rate | accuracy |\n|---|---|\n| 0.1 | 90% |\n| 0.2 | 91% |"
    )
