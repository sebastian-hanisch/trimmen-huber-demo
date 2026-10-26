"""Kappen und Rauschen an der Handinstanz (conftest.py): Mittel gekappter Werte, Empfindlichkeit, Skala, Zerlegung, Winsorisieren."""

import math

import numpy as np
import pytest

import thu_private as P


def test_hand_instance_clipping_sensitivity_and_scale():
    x = (0.0, 2.0, 10.0)
    assert P.clip_mean(x, 0.0, 4.0) == pytest.approx(2.0)                      # (0 + 2 + 4)/3
    assert P.sensitivity(0.0, 4.0, 3) == pytest.approx(4 / 3)
    assert P.laplace_scale(0.0, 4.0, 3, 1.0) == pytest.approx(4 / 3)
    assert P.laplace_scale(0.0, 4.0, 3, 0.5) == pytest.approx(8 / 3)
    neighbor = (0.0, 2.0, 0.0)
    assert P.clip_mean(neighbor, 0.0, 4.0) == pytest.approx(2 / 3)
    assert abs(P.clip_mean(x, 0.0, 4.0) - P.clip_mean(neighbor, 0.0, 4.0)) == pytest.approx(P.laplace_scale(0.0, 4.0, 3, 1.0) * 1.0)   # scharf: b · ε_DP


def test_two_sided_clipping_has_twice_the_sensitivity():
    assert P.sensitivity(-4.0, 4.0, 10) == pytest.approx(0.8) == pytest.approx(2 * 4.0 / 10)


def test_release_adds_scaled_noise():
    x = np.array([0.0, 2.0, 10.0])
    assert P.release(x, 0.0, 4.0, 1.0, 0.0) == pytest.approx(2.0)
    assert P.release(x, 0.0, 4.0, 1.0, 1.5) == pytest.approx(2.0 + 1.5 * 4 / 3)
    batch = P.release(np.array([x, x]), 0.0, 4.0, 2.0, np.array([1.0, -1.0]))
    assert batch == pytest.approx([2.0 + 2 / 3, 2.0 - 2 / 3])


def test_clip_mean_works_row_wise():
    assert P.clip_mean(np.array([[0.0, 2.0, 10.0], [5.0, 5.0, 5.0]]), 0.0, 4.0) == pytest.approx([2.0, 4.0])


def test_winsorizing_is_clipping_at_the_data_thresholds():
    x = (1.0, 2.0, 3.0, 4.0, 100.0)
    assert P.winsor_bounds(x, 1) == (2.0, 4.0)
    assert P.clip_at_data_bounds_mean(x, 1) == pytest.approx(3.0)             # (2 + 2 + 3 + 4 + 4)/5
    assert P.clip_at_data_bounds_mean(x, 0) == pytest.approx(P.clip_mean(x, 1.0, 100.0))


def test_replacement_sensitivity_above_one_with_thresholds_from_the_data():
    """Ersetzt man einen der Randwerte, wandern die Schwellen mit: die Änderung kann größer sein als (hi − lo)/n."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    ratio = P.replacement_sensitivity_ratio(x, 1, (0.0, 3.0, 50.0))
    assert ratio > 1.0
    flat = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    assert P.replacement_sensitivity_ratio(flat, 1, (5.0,)) == 0.0


def test_pop_mean_and_domain_errors():
    assert P.pop_mean(("lognormal", 1.0)) == pytest.approx(math.exp(0.5)) and P.pop_mean(("pareto", 3.0)) == pytest.approx(1.5)
    for bad in (("pareto", 1.0), ("cauchy", 1.0)):
        with pytest.raises(ValueError):
            P.pop_mean(bad)
    with pytest.raises(ValueError):
        P.clip_moments(("pareto", 2.0), [3.0])


def test_exact_mse_pieces_for_pareto_by_hand():
    """Pareto(1; 3), C = 2: E[min(X, 2)] = 1,5 − 2⁻²/2 = 1,375; E[min(X,2)²] = 3·(2⁻¹ − 1)/(−1) + 2⁻¹ = 1,5 + 0,5 = 2."""
    m1, m2 = P.clip_moments(("pareto", 3.0), [2.0])
    assert m1[0] == pytest.approx(1.375) and m2[0] == pytest.approx(2.0)
    mse, bias2, samp, noise = P.exact_mse(("pareto", 3.0), [2.0], 100, 1.0)
    assert bias2[0] == pytest.approx((1.375 - 1.5) ** 2) and samp[0] == pytest.approx((2.0 - 1.375 ** 2) / 100) and noise[0] == pytest.approx(2 * (2 / 100) ** 2)
    assert mse[0] == pytest.approx(bias2[0] + samp[0] + noise[0])


def test_below_the_smallest_pareto_value_everything_is_clipped():
    m1, m2 = P.clip_moments(("pareto", 3.0), [0.5])
    assert m1[0] == 0.5 and m2[0] == 0.25


def test_c_star_is_the_minimum_of_the_mse_and_grows_with_n():
    spec = ("lognormal", 1.0)
    c = P.c_star(spec, 1000, 1.0)

    def mse(v):
        return float(P.exact_mse(spec, [v], 1000, 1.0)[0][0])

    assert mse(c) <= mse(c * 0.9) and mse(c) <= mse(c * 1.1)
    assert P.c_star(spec, 10000, 1.0) > c > P.c_star(spec, 100, 1.0)


def test_replacement_sensitivity_by_hand_with_a_single_candidate():
    """x = (1, 2, 3, 4, 100), k = 1: Schwellen 2 und 4, gekapptes Mittel 3. Ersetzt man einen Wert durch 50: den 1 → (50, 2, 3, 4, 100), Schwellen 3 und 50, Mittel (50 + 3 + 3 + 4 + 50)/5 = 22, Änderung 19;
    die 2 → ebenfalls 22 (19); die 3 → (1, 2, 50, 4, 100), Schwellen 2 und 50, Mittel 21,6 (18,6); die 4 → 21,4 (18,4); die 100 → Mittel 3 (0). Größte Änderung 19, geteilt durch (4 − 2)/5 = 0,4: 47,5."""
    assert P.replacement_sensitivity_ratio([1.0, 2.0, 3.0, 4.0, 100.0], 1, (50.0,)) == pytest.approx(47.5)
    # Kandidat 3: die 3 selbst wird übersprungen; ersetzt man 1, 2, 4 oder 100 durch 3, wird das Mittel 3,4 bzw. 2,6, Änderung stets 0,4, geteilt durch 0,4: 1
    assert P.replacement_sensitivity_ratio([1.0, 2.0, 3.0, 4.0, 100.0], 1, (3.0,)) == pytest.approx(1.0)


def test_c_star_at_the_edge_of_the_grid_returns_the_edge_value():
    """Liegt das Minimum am Rand des Gitters (hier: n so groß, dass C* über 10⁷ liegt), gibt es keine Verfeinerung; der Randwert wird zurückgegeben."""
    assert P.c_star(("pareto", 1.5), 10 ** 13, 1.0) == pytest.approx(1e7, rel=1e-9)


def test_c_star_refinement_beats_the_grid_spacing():
    """Die parabolische Verfeinerung trifft das Minimum der MSE genauer als das Gitter (Abstand 2,3 % in C)."""
    scipy_opt = pytest.importorskip("scipy.optimize")
    worst_grid_error = 0.0
    for spec, n, eps in ((("lognormal", 1.0), 1000, 1.0), (("pareto", 1.5), 300, 0.5), (("lognormal", 0.5), 5000, 2.0), (("pareto", 3.0), 777, 0.3)):
        res = scipy_opt.minimize_scalar(lambda t: float(P.exact_mse(spec, [np.exp(t)], n, eps)[0][0]), bounds=(np.log(0.5), np.log(1e5)), method="bounded", options={"xatol": 1e-10})
        assert P.c_star(spec, n, eps) == pytest.approx(float(np.exp(res.x)), rel=2e-3)
        grid = np.exp(np.linspace(np.log(P.C_GRID_LOG[0]), np.log(P.C_GRID_LOG[1]), P.C_GRID_LOG[2]))
        worst_grid_error = max(worst_grid_error, abs(float(grid[int(np.argmin(P.exact_mse(spec, grid, n, eps)[0]))]) / float(np.exp(res.x)) - 1))
    assert worst_grid_error > 2e-3            # das Gitter allein wäre ungenauer: die Verfeinerung zählt


def test_c_star_at_the_lower_edge_of_the_grid_returns_the_edge_value():
    """Mit sehr wenig Daten und winzigem Budget ist das Rauschen so groß, dass die kleinste Grenze (0,3) die beste ist: Randwert ohne Verfeinerung."""
    assert P.c_star(("lognormal", 1.0), 1, 1e-6) == pytest.approx(P.C_GRID_LOG[0], rel=1e-9)


def test_replacement_sensitivity_does_not_depend_on_the_position_of_the_values():
    """Jede Rotation derselben Daten hat dasselbe Ergebnis (alle Positionen werden ersetzt, auch die letzte)."""
    rng = np.random.default_rng(31)
    x = rng.lognormal(0, 1, 12)
    values = [P.replacement_sensitivity_ratio(np.roll(x, r), 1, (0.0, 10.0, 100.0)) for r in range(12)]
    assert values == pytest.approx([values[0]] * 12, rel=1e-12)
    assert values[0] > 1.0


def test_replacement_sensitivity_checks_the_last_position_too():
    """k = 0 (Schwellen = kleinster und größter Wert), x = (10, 1, 2, 3, 0), Kandidat 100, Mittel 3,2. Ersetzt man die 10: (100, 1, 2, 3, 0) → 21,2, Änderung 18; die 1: (10, 100, 2, 3, 0) → 23, Änderung 19,8;
    die 2 und die 3: 19,6 bzw. 19,4; die 0 (letzte Position): (10, 1, 2, 3, 100) → 23,2, Änderung 20. Größte Änderung 20, geteilt durch (10 − 0)/5 = 2: 10."""
    assert P.replacement_sensitivity_ratio([10.0, 1.0, 2.0, 3.0, 0.0], 0, (100.0,)) == pytest.approx(10.0)
