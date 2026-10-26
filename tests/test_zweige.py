"""Zweig-Tests (Nullspalten-Regel, DEMO-PLAYBOOK §0): Jede Variante weicht auf mindestens einer Instanz von jeder anderen ab; die Klippen liegen, wo die Demo sie nennt (als Bänder).

Eine Messreihe mit lauter gleichen Spalten hieße: eine Variante läuft nie."""

import itertools

import numpy as np
import pytest

import thu_curves as K
import thu_robust as R
import thu_scenario as S

SPECS = ("mean", "median", "trim:0.1", "winsor:0.1", "huber:1.345", "hl", "catoni:1")


def _batch(n=100, runs=200, m=10, d=5.0):
    return S.Contamination("Normal", runs, n, S.Stream(77)).at(m, d)


def test_every_pair_of_estimators_differs_on_contaminated_data():
    x = _batch()
    est = {s: R.estimate(s, x) for s in SPECS}
    for a, b in itertools.combinations(SPECS, 2):
        assert np.abs(est[a] - est[b]).max() > 1e-6, (a, b)


def test_trimming_five_percent_equals_the_mean_below_n_20_and_differs_at_100():
    for n in (10, 19):
        x = S.Contamination("Normal", 50, n, S.Stream(5)).at(1, 5.0)
        assert R.trimmed_mean(x, 0.05) == pytest.approx(R.mean(x), abs=1e-14)
    x = S.Contamination("Normal", 50, 100, S.Stream(5)).at(5, 5.0)
    assert np.abs(R.trimmed_mean(x, 0.05) - R.mean(x)).max() > 1e-3


def test_winsorizing_differs_from_trimming_when_something_is_cut():
    x = _batch()
    assert np.abs(R.winsorized_mean(x, 0.1) - R.trimmed_mean(x, 0.1)).max() > 1e-3


def test_zero_scale_branch_returns_the_median_and_differs_from_the_normal_branch():
    zero = np.array([[0.0, 0.0, 0.0, 1.0, 7.0]])
    assert R.huber_location(zero) == pytest.approx(R.median(zero)) and R.mean(zero) != R.median(zero)
    x = S.zero_inflated_batch(100, 50, 0.6, 5.0, S.Stream(2))
    zero_rows = R.mad_scale(x) <= 0
    assert zero_rows.any()
    assert np.array_equal(R.huber_location(x)[zero_rows], R.median(x)[zero_rows])


def test_known_scale_differs_from_mad_scale_on_contaminated_data():
    x = _batch(m=30, d=5.0)
    assert np.abs(R.huber_location(x, 1.345, None) - R.huber_location(x, 1.345, 1.0)).max() > 1e-3


def test_catoni_multiplier_and_delta_change_the_estimate():
    x = _batch()
    assert np.abs(R.catoni_location(x, 0.05, 1.0) - R.catoni_location(x, 0.05, 8.0)).max() > 1e-3
    assert np.abs(R.catoni_location(x, 0.05, 1.0) - R.catoni_location(x, 0.2, 1.0)).max() > 1e-6


def test_the_cliff_of_trimming_sits_at_epsilon_greater_than_alpha_as_a_band():
    """Trimmen 10 % bei n = 100, d = 1000: bei ε = 10 % (m = 10, k = 10) hält es, bei 11 % bricht es (Bänder)."""
    cell = K.shift_cell(("trim:0.1", "median"), 100, (0.10, 0.11), (1000.0,), 100, (1, 2))
    row = {r["spec"]: r for r in cell["rows"]}
    assert row["trim:0.1"]["shift"][0][0][0] < 1.0 and row["trim:0.1"]["shift"][0][1][0] > 5.0
    assert row["median"]["shift"][0][1][0] < 1.0


def test_hodges_lehmann_breaks_between_25_and_35_percent_and_median_does_not():
    cell = K.shift_cell(("hl", "median"), 100, (0.20, 0.25, 0.35), (1000.0,), 100, (1, 2))
    hl = {r["spec"]: r for r in cell["rows"]}["hl"]["shift"][0]
    assert hl[0][0] < 1.5 and hl[2][0] > 100.0
    assert {r["spec"]: r for r in cell["rows"]}["median"]["shift"][0][2][0] < 1.5


def test_every_distribution_changes_the_sample_and_the_mean():
    seen = set()
    for dist in S.DISTRIBUTIONS:
        x = S.base_batch(dist, 20, 50, S.Stream(3))
        seen.add(round(float(x.mean()), 6))
    assert len(seen) == len(S.DISTRIBUTIONS)


def test_two_sided_contamination_changes_the_rmse_of_the_mean():
    one = K.shift_cell(("mean",), 50, (0.2,), (10.0,), 200, (1, 2), two_sided=False)["rows"][0]["rmse"][0][0][0]
    two = K.shift_cell(("mean",), 50, (0.2,), (10.0,), 200, (1, 2), two_sided=True)["rows"][0]["rmse"][0][0][0]
    assert one > 1.5 and two < 1.5
