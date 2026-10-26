"""Analyse einer Stichprobe: die sieben Schätzer mit den Reglerwerten, Zählungen, Zielgrößen."""

import math

import numpy as np
import pytest

import thu_constants as C
import thu_evaluation as ev
import thu_robust as R


def test_specs_follow_the_regler_values():
    s = ev.Settings(alpha=25, c=0.5, mult=8.0)
    assert ev.specs_of(s) == ("mean", "median", "trim:0.25", "winsor:0.25", "huber:0.5", "hl", "catoni:8")
    assert ev.specs_of(ev.Settings(alpha=5))[2] == "trim:0.05"


def test_analyse_counts_outliers_and_matches_the_estimators_applied_directly():
    s = ev.Settings(n=50, eps=20, d=100.0, seed=3)
    a = ev.analyse(s)
    assert a.x.shape == (50,) and int(a.is_outlier.sum()) == 10 and (a.x[a.is_outlier] > 90).all()
    for spec, value in zip(a.specs, a.values):
        assert value == pytest.approx(R.scalar(R.estimate(spec, a.x)), abs=1e-12)
    assert a.cut == 5 and a.labels[0] == "Mittel" and a.target_median == 0.0 and a.target_mean == 0.0
    assert a.mad == pytest.approx(R.scalar(R.mad_scale(a.x)))


def test_known_scale_changes_huber_only():
    base = ev.analyse(ev.Settings(eps=30, d=1000.0))
    known = ev.analyse(ev.Settings(eps=30, d=1000.0, scale=C.SCALES[1]))
    assert base.values[4] != known.values[4]
    assert [v for i, v in enumerate(base.values) if i != 4] == [v for i, v in enumerate(known.values) if i != 4]
    assert ev.known_scale(ev.Settings()) is None and ev.known_scale(ev.Settings(scale=C.SCALES[1])) == 1.0


def test_two_sided_outliers_have_both_signs_and_one_sided_only_one():
    one = ev.analyse(ev.Settings(eps=20, d=100.0))
    two = ev.analyse(ev.Settings(eps=20, d=100.0, side=C.SIDES[1]))
    assert (one.x[one.is_outlier] > 0).all()
    assert (two.x[two.is_outlier] > 50).any() and (two.x[two.is_outlier] < -50).any()


def test_skewed_distributions_have_different_targets():
    a = ev.analyse(ev.Settings(dist="Lognormal (σ = 1)", eps=0))
    assert a.target_median == 1.0 and a.target_mean == pytest.approx(math.exp(0.5))
    e = ev.analyse(ev.Settings(dist="Exponential", eps=0))
    assert e.target_median == pytest.approx(math.log(2)) and e.target_mean == 1.0


def test_catoni_alpha_matches_the_definition_with_the_mad_variance_and_the_multiplier():
    s = ev.Settings(mult=4.0, delta=0.1)
    a = ev.analyse(s)
    expected = 4.0 * float(R.catoni_alpha(s.n, a.mad ** 2, 0.1))
    assert a.catoni_alpha == pytest.approx(expected, rel=1e-12)


def test_analysis_is_deterministic_and_the_seed_matters():
    assert ev.analyse(ev.Settings(seed=5)).values == ev.analyse(ev.Settings(seed=5)).values
    assert ev.analyse(ev.Settings(seed=5)).values != ev.analyse(ev.Settings(seed=6)).values


@pytest.mark.parametrize("field,value", [("dist", "Laplace"), ("n", 50), ("eps", 20), ("d", 100.0), ("alpha", 25), ("c", 0.5), ("mult", 8.0), ("delta", 0.1), ("seed", 8)])
def test_every_regler_changes_at_least_one_estimate(field, value):
    """Nullspalten-Regel: ein Regler ohne Wirkung wäre ein toter Zweig."""
    base = ev.analyse(ev.Settings(eps=15, d=30.0))
    changed = ev.analyse(ev.Settings(**{**dict(eps=15, d=30.0), field: value}))
    assert base.values != changed.values


def test_side_and_scale_change_something_where_they_apply():
    base = ev.analyse(ev.Settings(eps=30, d=30.0))
    assert base.values != ev.analyse(ev.Settings(eps=30, d=30.0, side=C.SIDES[1])).values
    assert base.values != ev.analyse(ev.Settings(eps=30, d=30.0, scale=C.SCALES[1])).values


def test_n_boundaries_and_eps_boundaries_run_through():
    for s in (ev.Settings(n=C.N_MIN, eps=0), ev.Settings(n=C.N_MIN, eps=C.EPS_MAX, delta=0.01, alpha=C.ALPHA_MAX), ev.Settings(n=500, eps=C.EPS_MAX, d=1e6)):
        a = ev.analyse(s)
        assert all(np.isfinite(a.values))
