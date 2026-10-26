"""Orakel für den Abschnitt „Kappen und privates Mittel“ (anderer Rechenweg als der Kern):

* gekappte Momente gegen numerische Integration (scipy.integrate.quad);
* die Zerlegung MSE = Verzerrung² + Varianz/n + 2b² gegen eine Simulation mit dem Zufallsstrom (Verhältnis um 1);
* die Empfindlichkeit (hi − lo)/n gegen die Aufzählung aller benachbarten Datensätze, einschließlich der absichtlichen Fehler;
* das Gegenbeispiel „C aus den Daten“: das größte log-Verhältnis wächst linear in y (keine endliche Schranke);
* der Laplace-Sampler gegen die Verteilungsfunktion; Kappen auf Daten-Schwellen gegen scipy.stats.mstats.winsorize."""

import math

import numpy as np
import pytest

import thu_neighbors as N
import thu_private as P
import thu_robust as R
from thu_scenario import Stream, draw

quad = pytest.importorskip("scipy.integrate").quad
winsorize = pytest.importorskip("scipy.stats").mstats.winsorize


def _lognormal_pdf(sigma):
    return lambda x: math.exp(-0.5 * (math.log(x) / sigma) ** 2) / (x * sigma * math.sqrt(2 * math.pi))


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
def test_lognormal_clipped_moments_against_numerical_integration(sigma):
    pdf = _lognormal_pdf(sigma)
    for c in (0.7, 1.0, 3.0, 25.0):
        m1 = quad(lambda x: x * pdf(x), 0, c, limit=200)[0] + c * quad(pdf, c, math.inf, limit=200)[0]
        m2 = quad(lambda x: x * x * pdf(x), 0, c, limit=200)[0] + c * c * quad(pdf, c, math.inf, limit=200)[0]
        g1, g2 = P.clip_moments(("lognormal", sigma), [c])
        assert g1[0] == pytest.approx(m1, rel=1e-7) and g2[0] == pytest.approx(m2, rel=1e-7), (sigma, c)


@pytest.mark.parametrize("alpha", [1.5, 3.0])
def test_pareto_clipped_moments_against_numerical_integration(alpha):
    def pdf(x):
        return alpha * x ** (-alpha - 1)

    for c in (1.0, 2.0, 40.0):
        m1 = quad(lambda x: x * pdf(x), 1, c, limit=200)[0] + c * quad(pdf, c, math.inf, limit=200)[0]
        m2 = quad(lambda x: x * x * pdf(x), 1, c, limit=200)[0] + c * c * quad(pdf, c, math.inf, limit=200)[0]
        g1, g2 = P.clip_moments(("pareto", alpha), [c])
        assert g1[0] == pytest.approx(m1, rel=1e-7) and g2[0] == pytest.approx(m2, rel=1e-7), (alpha, c)
    g1, g2 = P.clip_moments(("pareto", alpha), [0.8])                    # unter dem kleinsten Wert 1 ist alles gekappt
    assert (g1[0], g2[0]) == (0.8, pytest.approx(0.64))


@pytest.mark.parametrize("spec,n,eps,c", [(("lognormal", 1.0), 200, 1.0, 8.0), (("pareto", 3.0), 200, 0.5, 4.0), (("lognormal", 0.5), 100, 0.2, 2.0)])
def test_mse_decomposition_against_simulation(spec, n, eps, c):
    runs = 20000
    x = draw(spec[0], spec[1], Stream(41), runs * n).reshape(runs, n)
    noise = draw("laplace", 0.0, Stream(42), runs)
    err = P.release(np.minimum(x, c), 0.0, c, eps, noise) - P.pop_mean(spec)
    exact = float(P.exact_mse(spec, [c], n, eps)[0][0])
    assert float(np.mean(err ** 2)) / exact == pytest.approx(1.0, abs=0.06)


def test_sensitivity_is_sharp_by_enumeration_and_the_deliberate_errors_are_caught():
    for n, eps in ((3, 0.5), (3, 1.0)):
        worst, pairs = N.max_log_ratio_laplace(n, range(0, 7), 0.0, 3.0, eps)
        assert worst == pytest.approx(eps) and pairs == 7 ** n * n * 6            # eingehalten und scharf
        assert N.max_log_ratio_laplace(n, range(0, 7), 0.0, 3.0, eps, "empfindlichkeit_n_plus_1")[0] == pytest.approx(eps * (n + 1) / n)
        assert N.max_log_ratio_laplace(n, range(0, 7), 0.0, 3.0, eps, "ohne_kappung")[0] == pytest.approx(2 * eps)     # 0 ↔ 6 bewegt das Mittel um 6/n, die Skala erlaubt 3/n


def test_two_sided_clipping_by_enumeration_has_sensitivity_2c_over_n():
    worst, _ = N.max_log_ratio_laplace(3, range(-3, 4), -2.0, 2.0, 1.0)
    assert worst == pytest.approx(1.0)
    wrong, _ = N.max_log_ratio_laplace(3, range(-3, 4), -2.0, 2.0, 1.0, "empfindlichkeit_n_plus_1")
    assert wrong > 1.0


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        N.max_log_ratio_laplace(3, range(0, 3), 0.0, 1.0, 1.0, "falsch")


def test_neighbors_differ_in_exactly_one_position():
    d = (1, 2, 3)
    nb = list(N.neighbors(d, range(0, 4)))
    assert len(nb) == 3 * 3 and len(set(nb)) == 9
    assert all(sum(a != b for a, b in zip(d, e)) == 1 for e in nb)


def test_data_dependent_c_has_no_finite_bound():
    """C = dritt-kleinster von vier Werten (Rang 2): das größte log-Verhältnis wächst linear in y — die Steigung 10/3 ist die Differenz der Kehrwerte der Skalen im Paar
    D = (1, 1, 6, 6), D′ = (1, 1, 1, 6): 1/b′ − 1/b mit b = 6/4 und b′ = 1/4, also 4 − 2/3."""
    vals = {y: N.max_log_ratio_data_dependent_c(4, range(1, 7), 1.0, y)[0] for y in (10.0, 100.0, 1000.0)}
    assert vals[10.0] > 4 and vals[1000.0] > 3000
    assert (vals[100.0] - vals[10.0]) / 90 == pytest.approx(10 / 3, abs=1e-9)
    assert (vals[1000.0] - vals[100.0]) / 900 == pytest.approx(10 / 3, abs=1e-9)


def test_laplace_sampler_against_the_distribution_function():
    x = np.sort(draw("laplace", 0.0, Stream(5), 200000))
    for q in (-3.0, -1.0, -0.2, 0.0, 0.5, 2.0):
        cdf = 0.5 * math.exp(q) if q < 0 else 1 - 0.5 * math.exp(-q)
        assert np.searchsorted(x, q) / len(x) == pytest.approx(cdf, abs=0.004)


def test_clipping_at_the_data_thresholds_equals_scipy_winsorize():
    rng = np.random.default_rng(23)
    for n in (7, 20, 51):
        for k in range(0, (n - 1) // 2 + 1):
            x = rng.lognormal(0, 1, n)
            w = winsorize(x, limits=((k + 0.5) / n, (k + 0.5) / n))
            assert P.clip_at_data_bounds_mean(x, k) == pytest.approx(float(np.mean(w)), abs=1e-12)
            assert P.clip_at_data_bounds_mean(x, k) == pytest.approx(R.scalar(R.winsorized_mean(x, k / n + 1e-12)), abs=1e-12)


def _brute_force_data_c(n, domain, eps, y, rank):
    """Unabhängige Fassung der Aufzählung mit reinem Python (keine Bibliothek, keine Hilfsfunktion der Demo)."""
    import itertools
    import math as m
    best = -m.inf
    for data in itertools.product(domain, repeat=n):
        c = sorted(data)[rank]
        mean_ = sum(min(v, c) for v in data) / n
        b = c / (n * eps)
        for i in range(n):
            for v in domain:
                if v == data[i]:
                    continue
                other = data[:i] + (v,) + data[i + 1:]
                c2 = sorted(other)[rank]
                mean2 = sum(min(w, c2) for w in other) / n
                b2 = c2 / (n * eps)
                best = max(best, m.log(b2 / b) + abs(y - mean2) / b2 - abs(y - mean_) / b)
    return best


@pytest.mark.parametrize("n,domain,eps,y,rank", [(4, range(1, 7), 1.0, 10.0, 2), (3, range(1, 5), 0.5, 5.0, 1), (3, range(1, 5), 2.0, 0.5, 0), (4, range(1, 5), 1.0, 3.0, 3)])
def test_data_dependent_c_against_an_independent_enumeration(n, domain, eps, y, rank):
    assert N.max_log_ratio_data_dependent_c(n, domain, eps, y, rank)[0] == pytest.approx(_brute_force_data_c(n, domain, eps, y, rank), abs=1e-9)


def test_data_dependent_c_by_hand_for_the_worst_pair():
    """D = (1, 1, 6, 6), D′ = (1, 1, 1, 6), y = 10: C = 6, Mittel 3,5, b = 1,5; C′ = 1, Mittel 1, b′ = 0,25. log(0,25/1,5) + 9/0,25 − 6,5/1,5 = −1,7918 + 36 − 4,3333 = 29,8749."""
    import math
    worst, d, d2 = N.max_log_ratio_data_dependent_c(4, range(1, 7), 1.0, 10.0)
    assert worst == pytest.approx(math.log(0.25 / 1.5) + 9 / 0.25 - 6.5 / 1.5, abs=1e-9) and d == (1, 1, 6, 6) and d2 == (1, 1, 1, 6)
