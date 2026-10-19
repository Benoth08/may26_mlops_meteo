from datetime import datetime, timedelta, timezone

import pytest

from models.training_condition import evaluate_training_condition

NOW = datetime(2026, 10, 18, 12, tzinfo=timezone.utc)


def decide(**overrides):
    args = dict(
        current_row_count=5000,
        last_training_row_count=4500,
        last_training_date=NOW - timedelta(hours=1),
        min_new_rows=1000,
        max_interval_hours=24,
        now=NOW,
    )
    args.update(overrides)
    return evaluate_training_condition(**args)


def test_no_data_never_trains():
    d = decide(current_row_count=0, last_training_date=None)
    assert not d.should_train and d.reason == "no_data"


def test_first_training():
    assert decide(last_training_date=None).reason == "first_training"


def test_new_rows_threshold():
    d = decide(last_training_row_count=3000)
    assert d.should_train and d.reason == "new_rows_threshold" and d.new_rows == 2000


def test_max_interval():
    d = decide(last_training_date=NOW - timedelta(hours=25))
    assert d.should_train and d.reason == "max_interval"


def test_threshold_not_reached():
    assert not decide().should_train


def test_naive_datetime_is_treated_as_utc():
    d = decide(last_training_date=(NOW - timedelta(hours=2)).replace(tzinfo=None))
    assert d.elapsed_hours == pytest.approx(2)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        decide(min_new_rows=0)
