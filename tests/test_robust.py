"""Einheiten der Schätzer an den Handinstanzen (conftest.py): jede Regel für sich."""

import math

import numpy as np
import pytest
from conftest import LOG_2_5

import thu_robust as R


def test_hand_instance_all_estimators(hand_x):
    assert R.scalar(R.mean(hand_x)) == 20 / 5
    assert R.scalar(R.median(hand_x)) == 2.0
    assert R.scalar(R.trimmed_mean(hand_x, 0.2)) == pytest.approx(8 / 3)
    assert R.scalar(R.winsorized_mean(hand_x, 0.2)) == pytest.approx(14 / 5)
    assert R.scalar(R.hodges_lehmann(hand_x)) == 3.0


def test_walsh_averages_of_the_hand_instance(hand_x):
    w = np.sort(R.walsh_averages(hand_x))
    assert list(w) == [0, 0.5, 1, 1, 1.5, 2, 2.5, 3, 3.5, 5, 6, 6.5, 7, 8.5, 12] and len(w) == 5 * 6 // 2


def test_hodges_lehmann_of_two_values_is_their_midpoint_group():
    # (1, 2): Walsh-Mittel 1; 1,5; 2 → Median 1,5
    assert R.scalar(R.hodges_lehmann([1.0, 2.0])) == 1.5


def test_huber_hand_instance():
    assert R.scalar(R.huber_location([0.0, 1.0, 10.0], 1.0, 1.0)) == pytest.approx(1.0, abs=1e-9)


def test_huber_with_huge_c_is_the_mean_and_with_tiny_c_the_median(hand_x):
    assert R.scalar(R.huber_location(hand_x, 1e9, 1.0)) == pytest.approx(4.0)
    assert R.scalar(R.huber_location(hand_x, 1e-6, 1.0)) == pytest.approx(2.0, abs=1e-3)


def test_zero_scale_falls_back_to_the_median(zero_scale_x):
    assert R.scalar(R.mad_scale(zero_scale_x)) == 0.0
    assert R.scalar(R.huber_location(zero_scale_x)) == 0.0 == R.scalar(R.median(zero_scale_x))
    assert R.scalar(R.mean(zero_scale_x)) == pytest.approx(8 / 5)


def test_catoni_psi_values():
    assert R.catoni_psi(1.0) == pytest.approx(LOG_2_5)
    assert R.catoni_psi(-1.0) == pytest.approx(-LOG_2_5) and R.catoni_psi(0.0) == 0.0
    assert R.catoni_psi(2.0) == pytest.approx(math.log(5.0))          # 1 + 2 + 2


def test_catoni_hand_instance_by_antisymmetry():
    assert R.scalar(R.catoni_root([0.0, 1.0], 1.0)) == pytest.approx(0.5, abs=1e-12)


def test_catoni_alpha_formula_and_domain():
    big_l = math.log(1 / 0.05)
    n, v = 100, 2.0
    expected = math.sqrt(2 * big_l / (n * v * (1 + 2 * big_l / (n - 2 * big_l))))
    assert float(R.catoni_alpha(n, v, 0.05)) == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError):
        R.catoni_alpha(5, 1.0, 0.05)                                 # n <= 2 log(20) = 5,99
    R.catoni_alpha(6, 1.0, 0.05)


def test_n_cut_is_a_staircase_with_tolerance():
    assert [R.n_cut(10, a) for a in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)] == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert R.n_cut(10, 0.05) == 0 and R.n_cut(20, 0.05) == 1
    assert R.n_cut(100, 0.29) == 29 and int(0.29 * 100) == 28         # Gleitkomma-Kante: 28,999999999999996
    assert R.n_cut(5, 0.45) == 2 and R.n_cut(3, 0.45) == 1 and R.n_cut(1, 0.45) == 0
    assert R.n_cut(4, 0.6) == 1                                       # höchstens so viele, dass ein Wert bleibt


def test_trimming_five_percent_is_the_mean_below_n_20():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 19))
    assert R.trimmed_mean(x, 0.05) == pytest.approx(R.mean(x), rel=1e-14, abs=1e-15)
    y = rng.normal(size=(50, 20))
    assert not np.allclose(R.trimmed_mean(y, 0.05), R.mean(y))


def test_batch_rows_are_independent():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(6, 25))
    for spec in ("mean", "median", "trim:0.1", "winsor:0.1", "huber:1.345", "hl", "catoni:1"):
        batch = R.estimate(spec, x)
        single = np.array([R.scalar(R.estimate(spec, x[i])) for i in range(6)])
        # Huber bricht erst ab, wenn die ganze Zeilengruppe unter HUBER_TOL liegt: im Stapel laufen manche Zeilen länger als allein (Unterschied < Toleranz)
        assert batch == pytest.approx(single, rel=1e-12, abs=5 * R.HUBER_TOL), spec


def test_parse_spec_and_labels():
    assert R.parse_spec("trim:0.1") == ("trim", 0.1) and R.parse_spec("hl") == ("hl", None)
    for bad in ("foo", "foo:1", "trim", "mean:1"):
        with pytest.raises(ValueError):
            R.parse_spec(bad)
    assert R.label("mean") == "Mittel" and R.label("trim:0.1") == "Trimmen 10 %" and R.label("winsor:0.25") == "Winsorisieren 25 %"
    assert R.label("huber:1.345") == "Huber (c = 1,345)" and R.label("catoni:8") == "Catoni (λ × 8)"
    assert R.label("hl") == "Hodges–Lehmann" and R.label("median") == "Median"


def test_rows_rejects_empty_and_three_dimensional_input():
    with pytest.raises(ValueError):
        R.rows(np.zeros((2, 0)))
    with pytest.raises(ValueError):
        R.rows(np.zeros((2, 2, 2)))


def test_catoni_alpha_boundary_n_equal_to_two_log_is_rejected():
    """δ = e⁻³: 2·log(1/δ) = 6 genau; n = 6 verletzt „n größer als 2L“, n = 7 nicht."""
    delta = math.exp(-3)
    with pytest.raises(ValueError):
        R.catoni_alpha(6, 1.0, delta)
    assert float(R.catoni_alpha(7, 1.0, delta)) > 0


def test_catoni_location_with_the_sample_variance():
    rng = np.random.default_rng(8)
    x = rng.standard_t(3, size=(4, 40))
    for mult in (1.0, 5.0):
        expected = R.catoni_root(x, mult * R.catoni_alpha(40, x.var(axis=1, ddof=1), 0.05))
        assert R.catoni_location(x, 0.05, mult, "stichprobe") == pytest.approx(expected, rel=1e-12)
    assert np.abs(R.catoni_location(x, 0.05, 1.0, "stichprobe") - R.catoni_location(x, 0.05, 1.0, "mad")).max() > 1e-4
    with pytest.raises(ValueError):
        R.catoni_location(x, 0.05, 1.0, "unbekannt")


def test_huber_iteration_count_starts_at_one():
    """Symmetrische Stichprobe, riesiges c: schon die erste Iteration ändert nichts (Median = Mittel), also genau eine Iteration."""
    mu, iterations = R.huber_location([-1.0, 0.0, 1.0], 1e9, 1.0, return_iterations=True)
    assert iterations == 1 and R.scalar(mu) == 0.0
    _, more = R.huber_location([0.0, 1.0, 10.0, 11.0, 50.0], 0.5, None, return_iterations=True)          # Skala aus dem MAD: die Wurzel liegt nicht im Median, mehrere Schritte
    assert 2 < more < 100                                                                          # konvergiert lange vor der Grenze von 500 Schritten
    assert R.scalar(R.huber_location([0.0, 1.0, 10.0, 11.0, 50.0], 0.5, 1.0)) == 10.0                  # sehr kleines c gegen die Skala: schon der Median ist die Wurzel (ein Schritt)


def test_catoni_with_a_single_value_and_with_two_values_at_a_high_delta():
    """δ = 0,9: L ≈ 0,105, also genügt schon n = 1 (n > 2L). Bei n = 1 ist die Stichprobenvarianz nicht definiert und wird durch 1 ersetzt (ohne Warnung), bei n = 2 gilt die echte."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert R.scalar(R.catoni_location([[5.0]], 0.9, 1.0, "stichprobe")) == pytest.approx(5.0)
    x = np.array([[0.0, 1.0]])
    expected = R.catoni_root(x, R.catoni_alpha(2, x.var(axis=1, ddof=1), 0.9))
    assert R.catoni_location(x, 0.9, 1.0, "stichprobe") == pytest.approx(expected, rel=1e-12)
    assert R.scalar(R.catoni_location(x, 0.9, 1.0, "stichprobe")) == pytest.approx(0.5, abs=1e-9)


def test_catoni_with_a_small_sample_variance_uses_it_and_not_the_fallback_one():
    """Varianz unter 1: nur Varianz 0 wird durch 1 ersetzt, nicht jede kleine."""
    rng = np.random.default_rng(9)
    x = 0.1 * rng.standard_t(3, size=(3, 30)) + np.array([[1.0], [2.0], [3.0]])
    variance = x.var(axis=1, ddof=1)
    assert (variance < 1.0).all() and (variance > 0.0).all()
    expected = R.catoni_root(x, 4.0 * R.catoni_alpha(30, variance, 0.05))
    assert R.catoni_location(x, 0.05, 4.0, "stichprobe") == pytest.approx(expected, rel=1e-12)
