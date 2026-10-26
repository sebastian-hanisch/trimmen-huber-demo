"""Orakel der Schätzer: ein anderer Rechenweg als der Kern (ORACLE-PLAYBOOK §1).

* getrimmtes und winsorisiertes Mittel gegen scipy (trim_mean, mstats.winsorize); scipy zählt ⌊α·n⌋ mit Gleitkomma, deshalb wird α als (k + 0,5)/n übergeben (genau k Werte je Seite).
  Die 13 Kanten-Paare, an denen int(α·n) und die Toleranz auseinanderlaufen, werden einzeln geprüft;
* Hodges–Lehmann gegen die Aufzählung aller Paarmittel mit itertools und statistics.median (auch mit Gleichständen);
* Huber gegen Bisektion auf Σψ und gegen scipy.optimize.minimize_scalar (Zielfunktion Σρ);
* Catoni gegen scipy.optimize.brentq und gegen eine Fixpunktiteration.
Zufallsinstanzen stammen aus numpy (hier ohne Plattformbezug: es zählt nur der Vergleich)."""

import itertools
import statistics

import numpy as np
import pytest

import thu_robust as R

scipy_stats = pytest.importorskip("scipy.stats")
scipy_opt = pytest.importorskip("scipy.optimize")


def _limits(n, k):
    return (k + 0.5) / n


@pytest.mark.parametrize("n", [5, 8, 20, 21, 50, 101])
def test_trimmed_and_winsorized_means_against_scipy(n):
    rng = np.random.default_rng(100 + n)
    for k in range(0, (n - 1) // 2 + 1):
        for _ in range(8):
            x = np.round(rng.normal(size=n) * 3, 1) if _ % 2 else rng.standard_t(3, size=n)        # mit und ohne Gleichstände
            alpha = k / n + 1e-12
            assert R.n_cut(n, alpha) == k
            assert R.scalar(R.trimmed_mean(x, alpha)) == pytest.approx(scipy_stats.trim_mean(x, _limits(n, k)), abs=1e-12, rel=1e-12)
            if k * 2 < n:
                w = scipy_stats.mstats.winsorize(x, limits=(_limits(n, k), _limits(n, k)))
                assert R.scalar(R.winsorized_mean(x, alpha)) == pytest.approx(float(np.mean(w)), abs=1e-12, rel=1e-12)


def test_float_edge_pairs_are_checked_one_by_one():
    """α = 0,29 und n = 100: 0,29 · 100 = 28,999999999999996, int() gibt 28. Der Kern rechnet mit Toleranz (29); scipy mit 29 Werten je Seite stimmt damit überein."""
    rng = np.random.default_rng(5)
    for alpha, n, k in ((0.29, 100, 29), (0.35, 180, 63), (0.57 / 2, 100, 28)):
        assert R.n_cut(n, alpha) == k, (alpha, n)
        x = rng.normal(size=n)
        assert R.scalar(R.trimmed_mean(x, alpha)) == pytest.approx(scipy_stats.trim_mean(x, _limits(n, k)), abs=1e-12)


def _hl_by_enumeration(values):
    pairs = [(a + b) / 2 for a, b in itertools.combinations_with_replacement(values, 2)]
    return statistics.median(pairs)


def test_hodges_lehmann_against_the_enumeration_of_all_pair_means():
    rng = np.random.default_rng(7)
    for n in range(1, 10):
        for trial in range(60):
            x = rng.integers(-4, 5, size=n).astype(float) if trial % 2 else rng.normal(size=n)       # ganze Zahlen: viele Gleichstände
            assert R.scalar(R.hodges_lehmann(x)) == pytest.approx(_hl_by_enumeration(list(x)), abs=1e-12), (n, trial)


def _huber_bisect(x, c, s, iters=200):
    lo, hi = float(x.min()), float(x.max())
    for _ in range(iters):
        mid = (lo + hi) / 2
        if np.clip((x - mid) / s, -c, c).sum() > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _rho(r, c):
    a = np.abs(r)
    return np.where(a <= c, 0.5 * r * r, c * a - 0.5 * c * c)


def test_huber_against_bisection_and_against_minimize_scalar():
    rng = np.random.default_rng(11)
    for trial in range(120):
        n = int(rng.integers(5, 60))
        x = rng.standard_t(3, size=n) * 2 + 1
        x[: int(rng.integers(0, n // 3 + 1))] += 20                                       # Ausreißer
        c = float(rng.choice([0.5, 1.0, 1.345, 3.0]))
        s = float(R.scalar(R.mad_scale(x)))
        if s <= 0:
            continue
        mine = R.scalar(R.huber_location(x, c))
        assert mine == pytest.approx(_huber_bisect(x, c, s), abs=1e-9 * s)
        res = scipy_opt.minimize_scalar(lambda m: float(_rho((x - m) / s, c).sum()), bounds=(x.min(), x.max()), method="bounded", options={"xatol": 1e-12})
        assert mine == pytest.approx(res.x, abs=1e-5 * s)


def _catoni_brentq(x, alpha):
    f = lambda t: float(R.catoni_psi(alpha * (x - t)).sum())            # noqa: E731
    return scipy_opt.brentq(f, float(x.min()), float(x.max()), xtol=1e-14)


def _catoni_fixed_point(x, alpha, iters=5000):
    th = float(np.median(x))
    for _ in range(iters):
        step = float(R.catoni_psi(alpha * (x - th)).sum()) / (alpha * len(x))
        th += step
        if abs(step) < 1e-14:
            break
    return th


def test_catoni_against_brentq_and_the_fixed_point_iteration():
    rng = np.random.default_rng(13)
    for trial in range(100):
        n = int(rng.integers(8, 80))
        x = rng.standard_t(3, size=n) + 0.5
        v = float(np.var(x, ddof=1))
        alpha = float(R.catoni_alpha(n, v, 0.05)) * float(rng.choice([1.0, 4.0, 8.0]))
        mine = R.scalar(R.catoni_root(x, alpha))
        assert mine == pytest.approx(_catoni_brentq(x, alpha), abs=1e-10)
        assert mine == pytest.approx(_catoni_fixed_point(x, alpha), abs=1e-8)


def test_hodges_lehmann_equals_the_median_of_the_walsh_averages_for_larger_samples():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(5, 40))
    assert R.hodges_lehmann(x) == pytest.approx([np.median(R.walsh_averages(row)) for row in x], abs=1e-12)
