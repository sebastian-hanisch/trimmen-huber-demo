"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): genau zwei Werte, ein Wert, Gleichstände, inklusive Grenzen, Anteil 0 und Anteil 1."""

import numpy as np
import pytest

import thu_neighbors as N
import thu_private as P
import thu_robust as R
import thu_scenario as S


def test_exactly_two_values():
    """n = 2: der kleinste Fall, in dem es etwas zu vergleichen gibt (der Zweig „n < 2“ darf nicht greifen)."""
    x = [1.0, 5.0]
    assert R.scalar(R.mean(x)) == 3.0 and R.scalar(R.median(x)) == 3.0
    assert R.scalar(R.trimmed_mean(x, 0.45)) == 3.0 and R.scalar(R.winsorized_mean(x, 0.45)) == 3.0       # n_cut(2, 0,45) = 0: es bleiben beide Werte
    assert R.scalar(R.hodges_lehmann(x)) == 3.0                                                         # Walsh-Mittel 1, 3, 5
    assert R.scalar(R.huber_location(x, 1.345, 1.0)) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        R.catoni_location(x)                                                                           # n = 2 ≤ 2·log(1/δ)


def test_a_single_value_and_the_empty_cut():
    assert R.scalar(R.trimmed_mean([4.0], 0.45)) == 4.0 and R.scalar(R.winsorized_mean([4.0], 0.45)) == 4.0
    assert R.n_cut(1, 0.45) == 0 and R.n_cut(3, 0.45) == 1 and R.n_cut(2, 0.45) == 0
    assert R.scalar(R.hodges_lehmann([4.0])) == 4.0 and R.scalar(R.huber_location([4.0])) == 4.0


def test_ties_everywhere():
    x = np.full(9, 2.5)
    for spec in ("mean", "median", "trim:0.1", "winsor:0.1", "huber:1.345", "hl", "catoni:1"):
        assert R.scalar(R.estimate(spec, x)) == pytest.approx(2.5), spec
    y = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
    assert R.scalar(R.median(y)) == 1.0 and R.scalar(R.hodges_lehmann(y)) == 1.0 and R.scalar(R.huber_location(y)) == 1.0         # MAD = 0: Huber = Median


def test_bounds_of_alpha_are_inclusive_on_the_zero_side_and_cut_the_most_on_the_top_side():
    rng = np.random.default_rng(1)
    x = rng.normal(size=21)
    assert R.scalar(R.trimmed_mean(x, 0.0)) == pytest.approx(R.scalar(R.mean(x)))
    assert R.scalar(R.trimmed_mean(x, 0.5)) == pytest.approx(R.scalar(R.median(x)))                     # größtes α: nur der mittlere Wert bleibt


def test_outlier_share_zero_and_one():
    assert S.n_outliers(10, 0.0) == 0 and S.n_outliers(10, 1.0) == 10
    src = S.Contamination("Normal", 4, 10, S.Stream(3))
    assert np.array_equal(src.at(0, 5.0), src.base)
    assert (src.at(10, 1000.0) > 990).all()                                                           # alle Werte sind Ausfälle


def test_the_clip_boundary_is_inclusive():
    assert P.clip_mean([4.0, 4.0], 0.0, 4.0) == 4.0 and P.clip_mean([0.0, 0.0], 0.0, 4.0) == 0.0     # Werte genau auf der Grenze bleiben
    assert P.sensitivity(2.0, 2.0, 5) == 0.0                                                          # lo = hi: kein Spielraum
    with pytest.raises(ValueError):
        P.laplace_scale(0.0, 1.0, 5, 0.0)


def test_enumeration_with_one_value_and_a_domain_of_two():
    assert list(N.neighbors((1,), (1, 2))) == [(2,)]
    worst, pairs = N.max_log_ratio_laplace(1, range(0, 3), 0.0, 2.0, 1.0)
    assert pairs == 3 * 2 and worst == pytest.approx(1.0)
