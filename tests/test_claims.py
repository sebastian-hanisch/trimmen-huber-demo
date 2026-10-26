"""Jede Zahl der README wird hier aus der vorgerechneten Messreihe (data/thu_results.json) gebildet und in der README gesucht (DEMO-PLAYBOOK §4).

Die Datei selbst ist durch tests/test_results.py gegen die Quelldateien abgesichert; dort werden Zellen auch live nachgerechnet."""

from pathlib import Path

import numpy as np
import pytest

import thu_constants as C
import thu_evaluation as V
import thu_private as P
import thu_results as R

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def de(x, nd=3):
    return V.de(x, nd)


def has(text):
    assert text in README, text


def eff(s, n=100):
    return R.efficiency(n, s)


def sh(s, j=0, n=100):
    return R.efficiency_shift(n, s, j)


def test_h1_the_three_families_are_on_one_curve():
    has(f"Huber {de(eff('huber:1.345'))} / {de(sh('huber:1.345'))}, Hodges–Lehmann {de(eff('hl'))} / {de(sh('hl'))}, Trimmen 10 % {de(eff('trim:0.1'))} / {de(sh('trim:0.1'))}; "
        f"Mittel 1 / {de(sh('mean'))}, Median {de(eff('median'))} / {de(sh('median'))}")
    shifts = [sh(s) for s in ("huber:1.345", "hl", "trim:0.1")]
    effs = [eff(s) for s in ("huber:1.345", "hl", "trim:0.1")]
    assert max(shifts) - min(shifts) <= 0.02 and min(effs) >= 0.945 and max(effs) <= 0.96
    has(f"{de(min(shifts))} (Huber) bis {de(max(shifts))} (Hodges–Lehmann und Trimmen 10 %), Unterschied {de(max(shifts) - min(shifts))}")
    has(f"Hodges–Lehmann {' / '.join(de(eff('hl', k)) for k in C.N_RASTER)} und Huber (c = 1,345) {' / '.join(de(eff('huber:1.345', k)) for k in C.N_RASTER)} bei n = 10 / 30 / 100 / 300; "
        f"der Median fällt von {de(eff('median', 10))} auf {de(eff('median', 300))}")
    assert R.efficiency(100, "median") < 0.67 and R.efficiency(100, "median") > 0.64


def test_h2_trimming_below_the_outlier_share_and_winsor_dominance():
    has(f"Trimmen 5 % bei 10 % Ausfall: Verschiebung {de(sh('trim:0.05'))} (Mittel {de(sh('mean'))})")
    has(f"Winsorisieren 10 % ({de(eff('winsor:0.1'))} / {de(sh('winsor:0.1'))}) wird von Huber c = 2 ({de(eff('huber:2'))} / {de(sh('huber:2'))}) dominiert")
    assert eff("huber:2") > eff("winsor:0.1") and sh("huber:2") < sh("winsor:0.1")
    dominated = []
    for a in C.TRIM_ALPHAS:
        w = f"winsor:{a:g}"
        if any(eff(f"huber:{c:g}") >= eff(w) and sh(f"huber:{c:g}") <= sh(w) for c in C.HUBER_CS):
            dominated.append(a)
    assert dominated == [0.05, 0.1, 0.15, 0.25, 0.3]
    has("nur 5 von 8 Winsor-Anteilen haben einen dominierenden Huber-Wert im Raster der Messung (nicht bei 20, 40 und 45 %)")
    has("gilt nur für 5 von 8 Winsor-Anteilen (siehe H2)")


def test_h3_h4_cliffs_and_hl_beyond_the_cliff():
    cell = R.shift_cell(100)
    v = lambda s, e: R.shift_value(100, s, e, 1000.0)          # noqa: E731
    has(f"d = 1000: Trimmen 10 % {de(v('trim:0.1', 0.1), 2)} bei 10 % und {de(v('trim:0.1', 0.15), 2)} bei 15 %; Trimmen 25 % {de(R.fine_value(100, 'trim:0.25', 0.25), 2)} bei 25 % und {de(R.fine_value(100, 'trim:0.25', 0.26), 2)} bei 26 %; "
        f"Hodges–Lehmann {de(R.fine_value(100, 'hl', 0.29), 2)} bei 29 % und {de(R.fine_value(100, 'hl', 0.3), 2)} bei 30 %; Huber (c = 1,345) {de(v('huber:1.345', 0.45), 2)} und Median {de(v('median', 0.45), 2)} bei 45 %")
    series = {s: R.shift_series(cell, s, 1000.0) for s in C.CURVE_SPECS}
    assert [s for s, ser in series.items() if V.cliff_eps(cell["eps"], ser) is not None] == ["trim:0.1", "trim:0.25", "trim:0.4", "winsor:0.1", "winsor:0.25", "hl"]
    has(f"ε = 35 %, d = 1000: {de(v('hl', 0.35), 2)} gegen {de(v('mean', 0.35), 2)} beim Mittel")
    assert v("hl", 0.35) > v("mean", 0.35)


def test_h5_bounded_estimators_do_not_depend_on_d():
    ds = list(R.distance_cell()["d"])
    huber, mean_ = R.distance_series("huber:1.345", 0.1), R.distance_series("mean", 0.1)
    cat = R.distance_series("catoni:1", 0.1)
    has(f"ε = 10 %: Huber {de(min(huber))} bis {de(max(huber))} von d = {V.fmt_g(ds[0])} bis 10⁶, Median {de(R.distance_series('median', 0.1)[0])}; Catoni (λ × 1) {de(cat[0])} bei d = 3, "
        f"{de(cat[5])} bei d = 1000, {de(cat[6])} bei d = 10⁶; Mittel {de(mean_[0])} bis {de(mean_[-1])}")
    assert ds[5] == 1000.0 and ds[6] == 1e6 and max(R.distance_series("median", 0.1)) - min(R.distance_series("median", 0.1)) < 0.01


def test_h6_tails():
    t = R.tail_ratio
    has(f"Hodges–Lehmann {de(t('Normal', 10, 'hl'))} / {de(t('Normal', 100, 'hl'))}, Huber {de(t('Normal', 10, 'huber:1.345'))} / {de(t('Normal', 100, 'huber:1.345'))}, "
        f"Median {de(t('Normal', 10, 'median'))} / {de(t('Normal', 100, 'median'))}; Laplace: Median {de(t('Laplace', 100, 'median'))}, Hodges–Lehmann {de(t('Laplace', 100, 'hl'))}; t(5): Median {de(t('t(5)', 100, 'median'))} (schlechter als das Mittel)")
    assert t("Normal", 10, "hl") < 1.04 and t("Normal", 10, "huber:1.345") < 1.041 and t("t(5)", 100, "median") > 1.0
    has(f"t(3): RMSE-Verhältnis zum Mittel bei n = 100: Hodges–Lehmann {de(t('t(3)', 100, 'hl'))}, Huber {de(t('t(3)', 100, 'huber:1.345'))}, Trimmen 25 % {de(t('t(3)', 100, 'trim:0.25'))}, Median {de(t('t(3)', 100, 'median'))}; "
        f"Laplace bei n = 300: Median {de(t('Laplace', 300, 'median'))}")


def test_h7_skew_target():
    d1, d2, dx = C.SKEW_DISTS[1], C.SKEW_DISTS[2], C.SKEW_DISTS[3]
    bias = [R.target_value(d1, k, "trim:0.1", "bias") for k in C.N_RASTER]
    has("Bias von Trimmen 10 % gegen den Mittelwert " + " / ".join(de(b).replace("-", "−") for b in bias) + " bei n = 10 / 30 / 100 / 300")
    rm = {s: [R.target_value(d1, k, s, "rmse_mean") for k in C.N_RASTER] for s in C.TARGET_SPECS}
    cross = {s: V.crossing_n(list(C.N_RASTER), rm["mean"], rm[s]) for s in rm if s not in ("mean", "catoni:1")}
    assert cross == {"median": 10, "trim:0.1": 30, "trim:0.25": 30, "winsor:0.1": 100, "huber:0.5": 10, "huber:1.345": 30, "hl": 30}
    has("das Mittel hat ab n = 30 den kleineren RMSE als Median, Trimmen 10 und 25 %, Huber (c = 0,5 und 1,345) und Hodges–Lehmann, gegen Winsorisieren 10 % erst ab n = 100")
    has(f"Catoni (λ × 1) liegt bei n = 10 / 30 / 100 unter dem Mittel ({' / '.join(de(rm['catoni:1'][i]) for i in range(3))} gegen {' / '.join(de(rm['mean'][i]) for i in range(3))})")
    assert all(rm["catoni:1"][i] < rm["mean"][i] for i in range(3))
    rm2 = {s: [R.target_value(d2, k, s, "rmse_mean") for k in C.N_RASTER] for s in C.TARGET_SPECS}
    cross2 = {s: V.crossing_n(list(C.N_RASTER), rm2["mean"], rm2[s]) for s in rm2 if s not in ("mean", "catoni:1")}
    assert cross2 == {"median": 100, "trim:0.1": 100, "trim:0.25": 100, "winsor:0.1": 300, "huber:0.5": 100, "huber:1.345": 100, "hl": 100}
    has(f"das Mittel gewinnt gegen Median, Trimmen 10 %, Hodges–Lehmann und Huber (c = 1,345) ab n = 100, gegen Winsorisieren 10 % ab n = 300; seine eigene Streuung ist dort groß "
        f"(SD {de(R.target_value(d2, 100, 'mean', 'sd'))} bei n = 100, Median {de(R.target_value(d2, 100, 'median', 'sd'))})")
    has(f"bei lognormal (σ = 2) ist sein RMSE gegen den Mittelwert bei n = 100 kleiner als der des Mittels ({de(rm2['catoni:1'][2])} gegen {de(rm2['mean'][2])}), bei n = 300 nicht ({de(rm2['catoni:1'][3])} gegen {de(rm2['mean'][3])})")
    assert rm2["catoni:1"][2] < rm2["mean"][2] and rm2["catoni:1"][3] > rm2["mean"][3]
    pop = lambda dist, s: de(R.population_location(dist, s))      # noqa: E731
    has(f"Mittelwert 1,649, Median 1,000; an der Population Trimmen 10 % {pop(d1, 'trim:0.1')}, Hodges–Lehmann {pop(d1, 'hl')}, Huber (c = 1,345) {pop(d1, 'huber:1.345')}, Catoni (λ × 1) {pop(d1, 'catoni:1')}")
    has(f"Mittelwert 7,389; Trimmen 10 % {pop(d2, 'trim:0.1')}, Hodges–Lehmann {pop(d2, 'hl')}, Huber (c = 1,345) {pop(d2, 'huber:1.345')}")
    rme = {s: [R.target_value(dx, k, s, "rmse_mean") for k in C.N_RASTER] for s in C.TARGET_SPECS}
    assert all(V.crossing_n(list(C.N_RASTER), rme["mean"], rme[s]) == 10 for s in rme if s not in ("mean", "catoni:1"))
    has(f"bei n = 10: {de(rme['mean'][0])} gegen {de(rme['median'][0])} beim Median und {de(rme['winsor:0.1'][0])} bei Winsorisieren 10 %")


def test_h8_catoni():
    has(f"Verschiebung {de(R.catoni_shift('catoni:1', 0))} gegen {de(R.catoni_shift('mean', 0))} bei ε = 10 %, d = 5), bei d = 1000 nicht: {de(R.catoni_shift('catoni:1', 1))} gegen {de(R.catoni_shift('mean', 1))} beim Mittel und {de(R.catoni_shift('huber:1.345', 1))} bei Huber")
    has(f"Mit λ × 32: Effizienz {de(R.catoni_efficiency('catoni:32'))}, Verschiebung {de(R.catoni_shift('catoni:32', 0))} / {de(R.catoni_shift('catoni:32', 1))}; "
        f"Huber ({de(R.catoni_efficiency('huber:1.345'))} / {de(R.catoni_shift('huber:1.345', 0))} / {de(R.catoni_shift('huber:1.345', 1))}) ist besser")
    assert R.catoni_efficiency("huber:1.345") >= R.catoni_efficiency("catoni:32") and R.catoni_shift("huber:1.345", 1) < R.catoni_shift("catoni:32", 1)
    has(f"(6,013 bei d = 1000 statt {de(R.catoni_shift('mean', 1))})")
    c = R.catoni_part()["distance"]
    series = {n: R.catoni_distance_series(n, "catoni:1") for n in (20, 100, 300)}
    has(f"wächst mit n: bei d = 1000 {de(series[20][5], 2)} / {de(series[100][5], 2)} / {de(series[300][5], 2)} für n = 20 / 100 / 300")
    assert [x["n"] for x in c] == [20, 100, 300]
    has(f"Effizienz {de(R.catoni_efficiency('catoni:1'))} / {de(R.catoni_efficiency('catoni:8'))} / {de(R.catoni_efficiency('catoni:32'))}")
    has(f"Verschiebung (ε = 10 %, d = 5) {de(R.catoni_shift('catoni:1', 0))} / {de(R.catoni_shift('catoni:8', 0))} / {de(R.catoni_shift('catoni:32', 0))} und (ε = 10 %, d = 1000) "
        f"{de(R.catoni_shift('catoni:1', 1))} / {de(R.catoni_shift('catoni:8', 1))} / {de(R.catoni_shift('catoni:32', 1))}")
    has(f"t(3): RMSE-Verhältnis {de(R.catoni_t3_ratio('catoni:1'))} (λ × 1), {de(R.catoni_t3_ratio('catoni:8'))} (λ × 8), Huber {de(R.catoni_t3_ratio('huber:1.345'))}")


def test_h9_pitfalls():
    p = R.pitfalls()
    has("n = 10: α = 5 / 10 / 15 / 20 % schneiden " + " / ".join(str(R.cut(10, a)) for a in (0.05, 0.1, 0.15, 0.2)) + " Werte je Seite ab")
    fe = p["float_edge"]
    has(f"`int(α·n)` weicht in {fe['differ']} von {fe['total']} Paaren ab (α = 0,29, n = 100: {fe['examples'][0][2]} statt {fe['examples'][0][3]})")
    has(f"Poisson(0,5): MAD = 0 in {de(100 * R.zero_scale_cell('poisson:0.5', 30)['zero_scale_share'][0], 1)} % (n = 30) und {de(100 * R.zero_scale_cell('poisson:0.5', 100)['zero_scale_share'][0], 1)} % (n = 100) der Stichproben")
    z = R.zero_scale_cell("poisson:0.5", 100)
    has(f"Poisson(0,5), n = 100: MAD = 0 in {de(100 * z['zero_scale_share'][0], 1)} % der Stichproben; Lage Median {de(z['location']['median'][0])}, Huber {de(z['location']['huber:1.345'][0])}, "
        f"Hodges–Lehmann {de(z['location']['hl'][0])}, Mittel {de(z['location']['mean'][0])}")
    zz30, zz100 = R.zero_scale_cell("zip:0.6:5", 30), R.zero_scale_cell("zip:0.6:5", 100)
    has(f"{de(100 * zz30['zero_scale_share'][0], 1)} % (n = 30) und {de(100 * zz100['zero_scale_share'][0], 1)} % (n = 100)")
    mad, known = R.scale_known_row("mad"), R.scale_known_row("known")
    has(f"Huber (c = 1,345) mit Skala 1: Effizienz {de(known['eff'][0])} gegen {de(mad['eff'][0])}; Verschiebung bei ε = 20 %, d = 5: {de(known['shifts'][0][0])} gegen {de(mad['shifts'][0][0])}; "
        f"bei ε = 30 %, d = 1000: {de(known['shifts'][1][0])} gegen {de(mad['shifts'][1][0])}")
    has(f"{de(R.iterations(0.5), 1)} bei c = 0,5, {de(R.iterations(1.345), 1)} bei c = 1,345, {de(R.iterations(3.0), 1)} bei c = 3")


def test_h10_hl_cliff_location_and_alpha_equals_epsilon():
    has(f"Sprung zwischen 29 % ({de(R.fine_value(100, 'hl', 0.29), 2)}) und 30 % ({de(R.fine_value(100, 'hl', 0.3), 2)}) bei n = 100 und 300")
    has(f"bei n = 300 ebenso ({de(R.fine_value(300, 'hl', 0.29), 2)} und {de(R.fine_value(300, 'hl', 0.3), 2)})")
    has(f"Trimmen 10 % hält bei ε = 10 % ({de(R.fine_value(100, 'trim:0.1', 0.1), 2)}) und bricht bei 11 % ({de(R.fine_value(100, 'trim:0.1', 0.11), 2)})")
    has(f"Klippe bei Trimmen 10 % zwischen ε = 10 % ({de(R.fine_value(100, 'trim:0.1', 0.1), 2)}) und 11 % ({de(R.fine_value(100, 'trim:0.1', 0.11), 2)}), bei Trimmen 25 % zwischen 25 % ({de(R.fine_value(100, 'trim:0.25', 0.25), 2)}) "
        f"und 26 % ({de(R.fine_value(100, 'trim:0.25', 0.26), 2)}), bei Winsorisieren 25 % zwischen 25 % ({de(R.fine_value(100, 'winsor:0.25', 0.25), 2)}) und 26 % ({de(R.fine_value(100, 'winsor:0.25', 0.26), 2)}), "
        f"bei Hodges–Lehmann zwischen 29 % ({de(R.fine_value(100, 'hl', 0.29), 2)}) und 30 % ({de(R.fine_value(100, 'hl', 0.3), 2)})")
    for spec, lo, hi in (("trim:0.1", 0.1, 0.11), ("trim:0.25", 0.25, 0.26), ("winsor:0.25", 0.25, 0.26), ("hl", 0.29, 0.3)):
        assert R.fine_value(100, spec, lo) < 2.0 < 10.0 < R.fine_value(100, spec, hi)


def test_befunde_table_distance_and_two_sided():
    e = 0.2
    huber, median_, hl_, mean_, trim = (R.distance_series(s, e) for s in ("huber:1.345", "median", "hl", "mean", "trim:0.1"))
    has(f"Bei ε = 20 %: Huber {de(huber[0])} bei d = 3 und {de(huber[-1])} ab d = 5, Median {de(min(median_))} bis {de(max(median_))}, Hodges–Lehmann {de(min(hl_))} bis {de(max(hl_))}; "
        f"Mittel {de(mean_[0])} bis {de(mean_[-1])}; Trimmen 10 % wächst bis {de(trim[-1])} (Anteil über α)")
    assert huber[1] == pytest.approx(huber[-1], abs=2e-3)
    m10, m20 = R.two_sided_value("mean", 0.1), R.two_sided_value("mean", 0.2)
    has(f"d = 10, ε = 10 %: RMSE um 0 beim Mittel {de(m10)} (nach der Formel √((1 + ε·d²)/n) {de(np.sqrt((1 + 0.1 * 100) / 100))}), Median {de(R.two_sided_value('median', 0.1))}, "
        f"Huber {de(R.two_sided_value('huber:1.345', 0.1))}, Trimmen 10 % {de(R.two_sided_value('trim:0.1', 0.1))}; bei ε = 20 % Winsorisieren 10 % {de(R.two_sided_value('winsor:0.1', 0.2))} gegen {de(m20)} beim Mittel")
    has(f"mit RMSE {de(R.two_sided_value('winsor:0.1', 0.2))} schlechter als das Mittel mit {de(m20)}")
    assert abs(m10 - float(np.sqrt((1 + 0.1 * 100) / 100))) < 0.01
    assert R.two_sided_value("winsor:0.1", 0.2) > m20


def test_kappen_u_curve_and_decomposition():
    cell = R.ucurve_cell(0, 1000)
    row = R.ucurve_eps(0, 1000, 1.0)
    cstar = R.scaling_row(("lognormal", 1.0), 1000, 1.0)
    k = int(np.argmin(row["exact_mse"]))
    has(f"Minimum bei C* = {de(cstar['c_star'], 2)} (Formel; Raster der Simulation {de(cell['cgrid'][k], 2)}), Anteil des Rauschens an der MSE dort {de(100 * cstar['noise_share'], 1)} %")
    has(f"RMSE der Formel dort {de(cstar['rmse'], 4)}, RMSE des nicht privaten Stichprobenmittels (Simulation) {de(cell['sample_rmse'][0], 4)}")
    has(f"RMSE²/MSE-Formel über das Raster: Median {de(row['ratio_median'])} (kleinster {de(row['ratio_min'])}, größter {de(row['ratio_max'])})")


def test_kappen_scaling_and_wrong_c():
    slopes = {}
    for spec in C.KAPPEN_SCALING_SPECS:
        per = [V.local_slopes(C.KAPPEN_NS_SCALING, [R.scaling_row(spec, k, e)["c_star"] for k in C.KAPPEN_NS_SCALING]) for e in C.KAPPEN_EPS]
        slopes[spec] = [float(np.mean(col)) for col in zip(*per)]
    p15, p3, ln1 = slopes[("pareto", 1.5)], slopes[("pareto", 3.0)], slopes[("lognormal", 1.0)]
    has(f"Pareto (α = 1,5) {de(min(p15), 2)} bis {de(max(p15), 2)}, Pareto (α = 3) {de(min(p3), 2)} bis {de(max(p3), 2)} (Herleitung 0,667 und 0,333); Lognormal (σ = 1) fällt von {de(ln1[0], 2)} auf {de(ln1[-1], 2)}")
    assert ln1[0] > ln1[-1] and abs(np.mean(p15) - 2 / 3) < 0.02 and abs(np.mean(p3) - 1 / 3) < 0.02
    c = [R.scaling_row(s, 1000, 1.0) for s in (("lognormal", 1.0), ("pareto", 1.5), ("pareto", 3.0))]
    has(f"Lognormal (σ = 1) {de(c[0]['c_star'], 2)}, Pareto (α = 1,5) {de(c[1]['c_star'], 2)}, Pareto (α = 3) {de(c[2]['c_star'], 2)}; Anteil des Rauschens an der MSE im Minimum "
        f"{de(100 * c[0]['noise_share'], 1)} % / {de(100 * c[1]['noise_share'], 1)} % / {de(100 * c[2]['noise_share'], 1)} %")
    w = {(tuple(r["spec"]), r["eps"]): {x["factor"]: x["rmse_ratio"] for x in r["rows"]} for r in R.wrong_c_rows()}
    ln = w[(("lognormal", 1.0), 1.0)]
    has(f"RMSE-Verhältnis {de(ln[0.25], 2)} (C*/4), {de(ln[4.0], 2)} (C* · 4), {de(ln[16.0], 2)} (C* · 16); bei Pareto (α = 1,5) ist C* · 4 teurer als C*/4 ({de(w[(('pareto', 1.5), 1.0)][4.0], 2)} gegen {de(w[(('pareto', 1.5), 1.0)][0.25], 2)})")
    worse = [k for k, v in w.items() if v[0.25] > v[4.0]]
    assert len(worse) == 4 and (("pareto", 1.5), 1.0) not in worse and (("pareto", 1.5), 0.1) not in worse
    has("**Ja (in 4 von 6 Zellen), nicht bei Pareto (α = 1,5)**")


def test_kappen_enumeration_and_winsor():
    e = R.enumeration()
    y = {r["y"]: r["worst"] for r in e["data_c"]}
    has("an der Stelle y = 3 / 10 / 100 / 1000: " + " / ".join(de(y[k], 2) for k in (3.0, 10.0, 100.0, 1000.0)) + " (wächst linear, keine Schranke)")
    lap = {(r["n"], r["eps"]): r for r in e["laplace"]}
    has(f"größtes log-Verhältnis = ε_DP ({de(lap[(4, 0.5)]['korrekt'], 4)} bei 0,5; {de(lap[(4, 1.0)]['korrekt'], 4)} bei 1); Fehler mit n + 1: {de(lap[(3, 0.5)]['empfindlichkeit_n_plus_1'], 4)} / {de(lap[(3, 1.0)]['empfindlichkeit_n_plus_1'], 4)} (n = 3)")
    has(f"{lap[(4, 0.5)]['pairs']} Paare bei n = 4): größtes log-Verhältnis genau ε_DP; Fehler mit n + 1: {de(lap[(4, 0.5)]['empfindlichkeit_n_plus_1'], 4)} / {de(lap[(4, 1.0)]['empfindlichkeit_n_plus_1'], 4)} (n = 4, ε_DP = 0,5 / 1); ohne Kappung {de(lap[(4, 0.5)]['ohne_kappung'], 4)} / {de(lap[(4, 1.0)]['ohne_kappung'], 4)}")
    ws = R.winsor_sensitivity()
    has(f"im Mittel das {de(ws['mean_ratio'][0], 1)}-fache von (hi − lo)/n, größter Datensatz {de(ws['largest_ratio'], 1)}-fach")


def test_sampling_spread_statement_is_consistent():
    assert V.de(R.shift_std_max(R.shift_cell(100)), 3) is not None and len(R.meta()["seeds"]) == 4


def test_the_preset_and_scale_statements():
    has("n = 3000" if C.N_MAX == 3000 else "n = ?")
    assert C.N_MAX == 3000 and len(C.SEEDS) == 4 and P.sensitivity(0.0, 1.0, 4) == 0.25


def test_mutation_state_matches_the_readme():
    """Der Mutationsstand der README („140 gefangen, 16 als äquivalent festgehalten, 0 offen“) wird mit tests/mutation_state.json abgeglichen."""
    import json
    state = json.loads((Path(__file__).resolve().parent / "mutation_state.json").read_text(encoding="utf-8"))
    killed = sum(1 for v in state.values() if v == "killed")
    equivalent = sum(1 for v in state.values() if v == "equivalent")
    assert set(state.values()) == {"killed", "equivalent"}
    has(f"gezogen wurden alle {killed + equivalent} Stellen")
    has(f"**{killed} gefangen, {equivalent} als äquivalent festgehalten, 0 offen**")
    assert {k.split("|")[0] for k in state} == {"thu_robust.py", "thu_private.py", "thu_neighbors.py"}
