"""Tests for windmojo.training (time-tolerant objective, trainer)."""

import numpy as np
import pandas as pd
import pytest

from windmojo.training import UnifiedTrainer
from windmojo.training.objectives import time_tolerant_objective

HOURS = pd.date_range("2026-07-01 10:00", periods=6, freq="h")
PEAK_AT_13 = np.array([1.0, 1.0, 1.0, 7.0, 1.0, 1.0])


def gradient(target, prediction, times=HOURS, **tolerance):
    objective = time_tolerant_objective(times, target, **tolerance)
    return objective(target, np.asarray(prediction, dtype=float))


class TestTimeTolerantObjective:
    def test_zero_tolerance_is_squared_error(self):
        prediction = np.array([2.0, 0.5, 3.0, 4.0, 1.0, 0.0])
        grad, hess = gradient(PEAK_AT_13, prediction, hours=0)
        np.testing.assert_allclose(grad, 2 * (prediction - PEAK_AT_13))
        np.testing.assert_allclose(hess, 2.0)

    def test_peak_one_hour_early_is_not_punished(self):
        early = np.array([1.0, 1.0, 7.0, 1.0, 1.0, 1.0])
        grad, _ = gradient(PEAK_AT_13, early, hours=1, exact_weight=0.0)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_missed_peak_stays_expensive(self):
        flat = np.ones(6)
        grad, _ = gradient(PEAK_AT_13, flat, hours=1, exact_weight=0.0)
        # The measured peak pulls a forecast within +-1 h up by the full error
        assert grad.min() == pytest.approx(-6.0)
        assert np.argmin(grad) in (2, 3, 4)

    def test_shift_beyond_tolerance_is_punished(self):
        late = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 7.0])  # two hours late
        grad, _ = gradient(PEAK_AT_13, late, hours=1, exact_weight=0.0)
        assert grad[5] == pytest.approx(6.0)  # false alarm
        assert grad.min() == pytest.approx(-6.0)  # missed peak

    def test_gaps_are_no_neighbours(self):
        times = pd.DatetimeIndex(["2026-07-01 23:00", "2026-07-02 06:00"])
        target = np.array([1.0, 7.0])
        grad, _ = gradient(target, np.array([7.0, 1.0]), times=times, hours=1)
        np.testing.assert_allclose(grad, 2 * (np.array([7.0, 1.0]) - target))


class TestTrainerTimeTolerance:
    def make_data(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        data = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=n, freq="h"),
                "a": rng.normal(size=n),
                "b": rng.normal(size=n),
            }
        )
        data["target"] = 3 + data["a"] + rng.normal(scale=0.3, size=n)
        return data

    def test_trains_saves_and_loads(self, tmp_path):
        data = self.make_data()
        config = {
            "xgboost": {"n_estimators": 50, "early_stopping_rounds": 5},
            "time_tolerance": {"hours": 2, "shift_penalty": 0.25, "exact_weight": 0.2},
        }
        trainer = UnifiedTrainer(config)
        trainer.train(data[:300], data[300:350], data[350:], ["a", "b"])
        trainer.save(tmp_path / "model.json", tmp_path / "metadata.json")

        loaded = UnifiedTrainer({})
        loaded.load(tmp_path / "model.json", tmp_path / "metadata.json")
        assert loaded.time_tolerance == config["time_tolerance"]
        np.testing.assert_allclose(
            loaded.predict(data[350:]), trainer.predict(data[350:])
        )
        assert (
            np.corrcoef(trainer.predict(data[350:]), data["target"][350:])[0, 1] > 0.8
        )

    def test_needs_time_column(self):
        data = self.make_data().drop(columns="time")
        trainer = UnifiedTrainer(
            {"xgboost": {"n_estimators": 5}, "time_tolerance": {"hours": 2}}
        )
        with pytest.raises(ValueError, match="time"):
            trainer.train(data[:300], data[300:350], data[350:], ["a", "b"])
