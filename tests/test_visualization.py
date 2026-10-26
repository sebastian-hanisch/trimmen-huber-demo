"""Figuren: jede wird für typische und Randwerte gebaut, alle Achsen sind fest (fixedrange), Beschriftungen stehen in den Daten."""

import numpy as np
import pytest

import thu_constants as C
import thu_evaluation as V
import thu_private as P
import thu_results as R
import thu_visualization as viz


def _fixed(fig):
    return fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


def test_sample_chart_for_presets_and_with_far_outliers():
    for preset in C.PRESETS.values():
        a = V.analyse(V.Settings(**preset))
        fig = viz.build_sample_chart(a.x, a.is_outlier, list(a.labels), list(a.specs), list(a.values), a.target_mean, a.target_median)
        assert _fixed(fig) and len(fig.data) >= 8
    far = V.analyse(V.Settings(eps=20, d=1e6))
    fig = viz.build_sample_chart(far.x, far.is_outlier, list(far.labels), list(far.specs), list(far.values), far.target_mean, far.target_median)
    lo, hi = fig.layout.xaxis.range
    assert hi < 100 and any("außerhalb" in (t.name or "") for t in fig.data)                       # Ausfälle bei 10⁶ liegen außerhalb des Bildes und stehen als Pfeil am Rand


def test_view_range_includes_near_outliers_but_not_far_ones():
    x = np.array([-1.0, 0.0, 1.0, 5.0, 1e6])
    flag = np.array([False, False, False, True, True])
    lo, hi = viz._view_range(x, flag, [0.0])
    assert hi > 5.0 and hi < 100.0 and lo < -1.0


def test_experiment_charts_build_from_the_stored_results():
    fig = viz.build_pareto_chart(R.efficiency_cell(100), 0, {"trim": "trim:0.1", "huber": "huber:1.345"})
    assert _fixed(fig) and {t.name for t in fig.data} >= {"Trimmen", "Winsorisieren", "Huber", "Catoni", "Mittel", "Median", "Hodges–Lehmann"}
    cell = R.shift_cell(100)
    shift = viz.build_shift_chart(cell, 1000.0)
    assert _fixed(shift) and shift.layout.yaxis.type == "log" and len(shift.data) == len(C.CURVE_SPECS) and 0.0 not in shift.data[0].x
    assert viz.build_shift_chart(cell, 5.0, key="rmse") is not None
    assert _fixed(viz.build_distance_chart(R.distance_cell(), 0.1)) and _fixed(viz.build_fine_chart(R.fine_cell(100), ["hl", "median"]))
    part = R.skew_part(1)
    assert _fixed(viz.build_target_chart(R.target_cell(C.SKEW_DISTS[1], 100), part["population"]))
    series = {s: [R.target_value(C.SKEW_DISTS[1], k, s, "rmse_mean") for k in C.N_RASTER] for s in ("mean", "median")}
    assert _fixed(viz.build_rmse_by_n_chart(C.SKEW_DISTS[1], list(C.N_RASTER), series))
    assert _fixed(viz.build_tail_chart([c for c in R.tail_cells() if c["n"] == 100], 100, ["median", "hl"]))
    assert _fixed(viz.build_stairs_chart([10, 20], [0.05, 0.1, 0.25])) and _fixed(viz.build_catoni_distance_chart(R.catoni_part()["distance"]))


def test_kappen_charts_build_and_do_not_put_zero_on_the_log_axis():
    cell = R.ucurve_cell(0, 1000)
    parts = P.exact_mse(C.KAPPEN_SPECS[0], cell["cgrid"], 1000, 1.0)
    fig = viz.build_ucurve_chart(R.ucurve_eps(0, 1000, 1.0), cell["cgrid"], parts, R.ucurve_eps(0, 1000, 1.0)["rmse"], cell["cgrid"][12])
    assert _fixed(fig) and fig.layout.yaxis.type == "log" and all(v is None or v > 0 for t in fig.data for v in t.y)
    assert _fixed(viz.build_scaling_chart(R.scaling_rows(), list(C.KAPPEN_SCALING_SPECS), 1.0))
    assert _fixed(viz.build_wrong_c_chart(R.wrong_c_rows())) and _fixed(viz.build_enumeration_chart(R.enumeration()["data_c"]))


@pytest.mark.parametrize("spec,expected", [("trim:0.1", "solid"), ("trim:0.25", "dash"), ("huber:3", "dot"), ("catoni:8", "dash"), ("hl", "solid")])
def test_dash_styles_separate_family_members(spec, expected):
    assert viz._dash(spec) == expected
