"""Die vorgerechneten Messreihen (data/thu_results.json): vollständig, aktuell gegenüber den Quelldateien und in sich stimmig; Zellen werden live nachgerechnet."""

import json

import numpy as np
import pytest

import thu_constants as C
import thu_curves as K
import thu_experiments as X
import thu_kappen as KP
import thu_results as R


def test_results_file_is_current_with_the_generator_sources():
    """Schlägt fehl, sobald sich eine der Quelldateien ändert (thu_rng, thu_scenario, thu_robust, thu_private, thu_neighbors, thu_curves, thu_kappen, thu_experiments, tools/make_results.py):
    dann `sh tools/run_parts.sh` (alle Teile und Zusammenfügen) neu laufen lassen."""
    assert R.meta()["erzeuger_hash"] == X.source_hash()
    assert R.meta()["quellen"] == list(X.SOURCES) and R.meta()["seeds"] == list(C.SEEDS)


def test_results_file_is_valid_utf8_json_with_all_parts():
    text = R.PATH.read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {"meta", *X.PARTS}
    assert "Infinity" not in text and "NaN" not in text and "Lognormal (σ = 1)" in text           # Umlaute und Sonderzeichen im Klartext, gültiges JSON


def test_grids_in_the_file_equal_the_constants():
    for n in C.N_RASTER:
        cell = R.efficiency_cell(n)
        assert [r["spec"] for r in cell["rows"]] == list(C.FAMILY_SPECS) and cell["runs_per_seed"] == X.M1_RUNS[n]
    for n in X.M2_N:
        cell = R.shift_cell(n)
        assert cell["eps"] == list(C.EPS_GRID) and cell["d"] == list(C.D_GRID) and [r["spec"] for r in cell["rows"]] == list(C.CURVE_SPECS)
    assert R.fine_cell(100)["eps"] == list(C.EPS_FINE) and R.distance_cell()["d"] == list(C.D_CURVE) and R.distance_cell()["eps"] == [0.1, 0.2, 0.3]
    assert [c["dist"] for c in R.tail_cells()] == [d for d in C.TAIL_DISTS for _ in C.N_RASTER]
    assert [R.skew_part(i)["dist"] for i in range(4)] == list(C.SKEW_DISTS)
    assert [tuple(c["spec"]) for c in R.scaling_rows()][:1] == [C.KAPPEN_SCALING_SPECS[0]] and len(R.scaling_rows()) == 5 * 7 * 4
    for i in range(3):
        assert [c["n"] for c in R.ucurve_cells(i)] == list(C.KAPPEN_N) and R.ucurve_cell(i, 100)["cgrid"] == pytest.approx(KP.c_grid())


def test_cells_have_four_seeds_worth_of_spread_and_values_are_finite():
    for n in C.N_RASTER:
        for r in R.efficiency_cell(n)["rows"]:
            assert 0.0 < r["eff"][0] <= 1.01 and r["eff"][1] >= 0.0 and all(np.isfinite(s[0]) for s in r["shifts"])
    for n in X.M2_N:
        cell = R.shift_cell(n)
        assert all(np.isfinite(v[0]) for r in cell["rows"] for row in r["shift"] for v in row)
        assert R.shift_series(cell, "mean", 5.0)[0] == pytest.approx(0.0, abs=0.05)           # ohne Ausfälle keine Verschiebung


def test_mean_shift_is_epsilon_times_d_up_to_the_sampling_error():
    cell = R.shift_cell(100)
    series = R.shift_series(cell, "mean", 1000.0)
    for e, v in zip(cell["eps"], series):
        assert v == pytest.approx(e * 1000.0, abs=0.5)                                       # E[x̄] = m/n · d, Std der Mitte höchstens 0,1


def test_stored_cells_are_reproduced_by_the_live_functions():
    """Das Nachmessen in der App liefert dieselben Zahlen wie die vorgerechnete Zelle (gleiche Seeds)."""
    cell = R.efficiency_cell(10)
    again = K.efficiency_bias_cell(("mean", "median", "huber:1.345"), 10, cell["runs_per_seed"], C.SEEDS)
    for r in again["rows"]:
        stored = next(s for s in cell["rows"] if s["spec"] == r["spec"])
        assert r["eff"] == pytest.approx(stored["eff"]) and np.asarray(r["shifts"]) == pytest.approx(np.asarray(stored["shifts"]))
    shift = R.shift_cell(20)
    live = K.shift_cell(("median", "hl"), 20, C.EPS_GRID, C.D_GRID, shift["runs_per_seed"], C.SEEDS)
    for r in live["rows"]:
        stored = next(s for s in shift["rows"] if s["spec"] == r["spec"])
        assert np.asarray(r["shift"]) == pytest.approx(np.asarray(stored["shift"]))
    uc = R.ucurve_cell(1, 100)
    again_u = KP.ucurve_cell(("pareto", 1.5), 100, uc["runs_per_seed"], C.SEEDS, (1.0,), KP.c_grid())
    assert np.asarray(again_u["eps"][0]["rmse"]) == pytest.approx(np.asarray(R.ucurve_eps(1, 100, 1.0)["rmse"]))
    assert np.asarray(again_u["clip_abs"]) == pytest.approx(np.asarray(uc["clip_abs"]))


def test_exact_mse_stored_in_the_file_matches_the_closed_form():
    import thu_private as P
    row = R.ucurve_eps(0, 1000, 1.0)
    assert row["exact_mse"] == pytest.approx(list(P.exact_mse(("lognormal", 1.0), R.ucurve_cell(0, 1000)["cgrid"], 1000, 1.0)[0]))


def test_scaling_and_wrong_c_rows_match_the_closed_form():
    import thu_private as P
    row = R.scaling_row(("pareto", 1.5), 1000, 1.0)
    assert row["c_star"] == pytest.approx(P.c_star(("pareto", 1.5), 1000, 1.0)) and 0.0 < row["noise_share"] < 1.0
    wc = R.wrong_c(("lognormal", 1.0), 1.0)
    assert [r["factor"] for r in wc["rows"]] == [0.25, 1.0, 4.0, 16.0] and wc["rows"][1]["rmse_ratio"] == pytest.approx(1.0)
    assert all(r["rmse_ratio"] >= 1.0 for r in wc["rows"])


def test_enumeration_part_has_the_documented_shape():
    e = R.enumeration()
    assert [(r["n"], r["eps"]) for r in e["laplace"]] == [(3, 0.5), (3, 1.0), (4, 0.5), (4, 1.0)]
    assert [r["y"] for r in e["data_c"]] == [3.0, 10.0, 100.0, 1000.0]
    ws = R.winsor_sensitivity()
    assert ws["n"] == 100 and ws["k"] == 1 and ws["winsor_equals_clip_gap"] < 1e-12 and ws["min_ratio"] <= ws["mean_ratio"][0] <= ws["largest_ratio"]


def test_pitfalls_part_is_complete():
    p = R.pitfalls()
    assert len(p["stairs"]) == 5 * 9 and R.cut(10, 0.05) == 0 and R.cut(20, 0.05) == 1 and R.cut(10, 0.45) == 4
    assert [c["kind"] for c in p["zero_scale"]][::2] == list(X.ZERO_KINDS) and R.iterations(0.5) > R.iterations(1.345) > R.iterations(3.0)
