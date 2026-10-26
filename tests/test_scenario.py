"""Szenario: der blockweise berechnete Zufallsstrom stimmt mit der Python-Klasse überein, Verteilungen haben die dokumentierten Lagen, Ausfälle sind genau m an der Zahl."""

import math

import numpy as np
import pytest

import thu_scenario as S
from thu_rng import SplitMix64


def test_block_stream_equals_the_python_class_value_by_value():
    for seed in (0, 7, 123456789, 2 ** 63 + 5):
        rng = SplitMix64(seed)
        expected = [rng.next() for _ in range(50)]
        assert [int(v) for v in S.mix_words(seed, 0, 50)] == expected
        assert [int(v) for v in S.mix_words(seed, 20, 10)] == expected[20:30]


def test_stream_continues_where_it_stopped():
    a, b = S.Stream(9), S.Stream(9)
    first = np.concatenate([a.uniform(7), a.uniform(5)])
    assert first == pytest.approx(b.uniform(12), abs=0)
    assert a.pos == 12


def test_uniform_ranges_and_normal_moments():
    st = S.Stream(3)
    u = st.uniform_open(100000)
    assert 0 < u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.01
    z = S.Stream(4).normal(200000)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01 and len(S.Stream(4).normal(7)) == 7


def test_cell_seed_is_stable_and_distinguishes_tags():
    assert S.cell_seed(5, "a", 1.5) == S.cell_seed(5, "a", 1.5)
    assert len({S.cell_seed(5, "a", 1), S.cell_seed(5, "b", 1), S.cell_seed(5, "a", 2), S.cell_seed(6, "a", 1)}) == 4


@pytest.mark.parametrize("dist", S.DISTRIBUTIONS)
def test_every_distribution_has_the_documented_median_and_mean(dist):
    mean_, median_ = S.population(dist)
    x = S.base_batch(dist, 1, 400000, S.Stream(S.cell_seed(1, dist)))[0]
    assert np.median(x) == pytest.approx(median_, abs=0.02)
    if "σ = 2" not in dist and dist != "t(3)":                       # Lognormal(2) und t(3): Stichprobenmittel streut zu stark
        assert x.mean() == pytest.approx(mean_, abs=0.05 * max(1.0, abs(mean_)))


def test_poisson_values_are_non_negative_integers_with_documented_share_of_zero():
    x = S.draw("poisson", 0.5, S.Stream(2), 100000)
    assert (x >= 0).all() and (x == np.floor(x)).all() and np.mean(x == 0) == pytest.approx(math.exp(-0.5), abs=0.01)


def test_pareto_has_minimum_one_and_the_tail_exponent():
    x = S.draw("pareto", 3.0, S.Stream(8), 200000)
    assert x.min() >= 1.0 and np.mean(x > 2.0) == pytest.approx(2.0 ** -3, abs=0.005)


def test_unknown_distribution_is_rejected():
    with pytest.raises(ValueError):
        S.draw("cauchy", 1.0, S.Stream(1), 3)


def test_number_of_outliers_is_round_half_up_and_bounded():
    assert [S.n_outliers(100, e) for e in (0.0, 0.1, 0.15, 0.45)] == [0, 10, 15, 45]
    assert S.n_outliers(10, 0.05) == 1 and S.n_outliers(10, 0.04) == 0 and S.n_outliers(10, 2.0) == 10 and S.n_outliers(10, -0.1) == 0


def test_contamination_replaces_exactly_m_values_in_every_row():
    src = S.Contamination("Normal", 50, 40, S.Stream(5))
    clean, dirty = src.at(0, 0.0), src.at(6, 1000.0)
    assert np.array_equal(clean, src.base)
    assert (dirty[:, :6] > 900).all() and np.array_equal(dirty[:, 6:], clean[:, 6:])
    assert np.mean(dirty[:, :6]) == pytest.approx(1000.0, abs=1.0)


def test_two_sided_contamination_has_both_signs_and_shares_the_noise():
    src = S.Contamination("Normal", 200, 20, S.Stream(6))
    one, two = src.at(10, 100.0), src.at(10, 100.0, True)
    assert (one[:, :10] > 50).all()
    assert (two[:, :10] > 50).any() and (two[:, :10] < -50).any()
    assert np.abs(np.abs(two[:, :10]) - 100.0 - 0).max() < 8         # derselbe Normalwert z, nur das Vorzeichen ändert sich
    assert abs(np.sign(two[:, :10]).mean()) < 0.1


def test_one_sample_flags_the_outliers_and_is_reproducible():
    x, flag = S.one_sample("Normal", 100, 0.1, 50.0, False, 7)
    assert flag.sum() == 10 and (x[flag] > 40).all() and (x[~flag] < 10).all()
    x2, flag2 = S.one_sample("Normal", 100, 0.1, 50.0, False, 7)
    assert np.array_equal(x, x2) and np.array_equal(flag, flag2)
    assert not np.array_equal(S.one_sample("Normal", 100, 0.1, 50.0, False, 8)[0], x)
    assert not flag[:10].all()                                         # gemischt, nicht vorn gestapelt


def test_zero_inflated_batch_has_the_requested_share_of_zeros():
    x = S.zero_inflated_batch(200, 100, 0.6, 5.0, S.Stream(3))
    assert np.mean(x == 0) == pytest.approx(0.6 + 0.4 * math.exp(-5.0), abs=0.01)
