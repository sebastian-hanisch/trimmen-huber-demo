"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für jeden Lageschätzer gelten müssen, ohne dass man das richtige Ergebnis kennt.

Je Eigenschaft 30 Zufallsinstanzen mit festem Seed. Gelten sie nur unter Voraussetzungen, stehen diese im Test:
* Verschiebungs- und Skalenäquivarianz, Spiegelung und Permutationsinvarianz gelten für alle sieben Schätzer, auch für Huber und Catoni mit der MAD als Skala bzw. Varianz
  (MAD und Varianz verschieben und skalieren mit);
* Monotonie in einem Wert gilt für Mittel, Median, Trimmen, Winsorisieren und Hodges–Lehmann sowie für Huber mit FESTER Skala und Catoni mit festem α, nicht für Huber mit
  geschätzter Skala (die Skala selbst hängt vom Wert ab)."""

import numpy as np
import pytest

import thu_robust as R

SPECS = ("mean", "median", "trim:0.1", "winsor:0.1", "huber:1.345", "hl", "catoni:1")
MONOTONE = ("mean", "median", "trim:0.1", "winsor:0.1", "hl")


def _instances(seed, count=30, n=25):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        x = rng.standard_t(3, size=n) * 2 + rng.normal() * 5
        if i % 3 == 0:
            x[: n // 5] += 30
        out.append(x)
    return out


@pytest.mark.parametrize("spec", SPECS)
def test_shift_and_scale_equivariance_and_reflection(spec):
    for x in _instances(1):
        t = R.scalar(R.estimate(spec, x))
        assert R.scalar(R.estimate(spec, x + 3.5)) == pytest.approx(t + 3.5, abs=1e-8)
        assert R.scalar(R.estimate(spec, x * 2.5)) == pytest.approx(t * 2.5, rel=1e-8, abs=1e-8)
        assert R.scalar(R.estimate(spec, -x)) == pytest.approx(-t, abs=1e-8)


@pytest.mark.parametrize("spec", SPECS)
def test_permutation_invariance(spec):
    rng = np.random.default_rng(2)
    for x in _instances(2):
        t = R.scalar(R.estimate(spec, x))
        assert R.scalar(R.estimate(spec, rng.permutation(x))) == pytest.approx(t, abs=1e-8)


@pytest.mark.parametrize("spec", SPECS)
def test_constant_sample_gives_the_constant_and_every_estimate_lies_between_min_and_max(spec):
    assert R.scalar(R.estimate(spec, np.full(15, 4.25))) == pytest.approx(4.25)
    for x in _instances(3):
        t = R.scalar(R.estimate(spec, x))
        assert x.min() - 1e-9 <= t <= x.max() + 1e-9


@pytest.mark.parametrize("spec", MONOTONE)
def test_raising_one_value_never_lowers_the_estimate(spec):
    rng = np.random.default_rng(4)
    for x in _instances(4):
        i = int(rng.integers(len(x)))
        y = x.copy()
        y[i] += float(rng.uniform(0.1, 50))
        assert R.scalar(R.estimate(spec, y)) >= R.scalar(R.estimate(spec, x)) - 1e-12


def test_huber_with_fixed_scale_and_catoni_with_fixed_alpha_are_monotone():
    rng = np.random.default_rng(5)
    for x in _instances(5):
        i = int(rng.integers(len(x)))
        y = x.copy()
        y[i] += float(rng.uniform(0.1, 50))
        assert R.scalar(R.huber_location(y, 1.345, 2.0)) >= R.scalar(R.huber_location(x, 1.345, 2.0)) - 1e-9
        assert R.scalar(R.catoni_root(y, 0.3)) >= R.scalar(R.catoni_root(x, 0.3)) - 1e-12


def test_trimming_all_but_the_middle_value_is_the_median_for_odd_n():
    for x in _instances(6, n=25):
        assert R.scalar(R.trimmed_mean(x, 0.499)) == pytest.approx(R.scalar(R.median(x)))
        assert R.scalar(R.winsorized_mean(x, 0.499)) == pytest.approx(R.scalar(R.median(x)))


def test_one_value_is_its_own_estimate_for_every_estimator():
    for spec in ("mean", "median", "trim:0.1", "winsor:0.1", "huber:1.345", "hl"):
        assert R.scalar(R.estimate(spec, [3.75])) == 3.75


def test_the_hodges_lehmann_estimate_is_the_median_of_the_walsh_averages_so_it_lies_between_median_and_extremes():
    for x in _instances(7):
        assert np.min(R.walsh_averages(x)) == x.min() and np.max(R.walsh_averages(x)) == x.max()


def test_same_seed_same_sample_and_estimates_do_not_depend_on_other_rows():
    from thu_scenario import Stream, base_batch
    a = base_batch("Normal", 20, 30, Stream(9))
    b = base_batch("Normal", 20, 30, Stream(9))
    assert np.array_equal(a, b)
    for spec in SPECS:
        full = R.estimate(spec, a)
        part = R.estimate(spec, a[:5])
        assert part == pytest.approx(full[:5], rel=1e-12, abs=5 * R.HUBER_TOL)
