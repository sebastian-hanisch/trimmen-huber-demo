"""Verdikte und Zusammenfassungen an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, nicht Seeds. Ohne diese Datei überleben fast alle Mutanten der Verdikt-Arithmetik."""

import numpy as np
import pytest

import thu_evaluation as ev


def _analysis(values, target_median=0.0, target_mean=0.0):
    specs = ("mean", "median", "trim:0.1", "winsor:0.1", "huber:1.345", "hl", "catoni:1")
    return ev.Analysis(ev.Settings(), np.zeros(10), np.zeros(10, dtype=bool), specs, tuple(f"S{i}" for i in range(7)), tuple(values), target_mean, target_median, 1.0, 1, 0.2)


def test_formatting_helpers():
    assert ev.de(1.2345, 2) == "1,23" and ev.de(0.5, 1) == "0,5" and ev.pct(0.1234, 1) == "12,3" and ev.pct(0.5, 0) == "50"
    assert ev.fmt_g(1.345) == "1,345" and ev.fmt_g(1000.0) == "1000" and ev.fmt_g(0.5) == "0,5"
    assert ev.fmt_g(1e6) == "10⁶" and ev.fmt_g(1e9) == "10⁹" and ev.fmt_g(1234567.0) == "1234567" and ev.fmt_g(100000.0) == "100000" and ev.fmt_g(10000.0) == "10000"


def test_deviations_are_measured_against_the_median_of_the_core():
    a = _analysis([1.0, 0.5, 0.25, -0.75, 0.0, 2.0, 3.0], target_median=0.5)
    assert ev.deviations(a) == {"mean": 0.5, "median": 0.0, "trim:0.1": 0.25, "winsor:0.1": 1.25, "huber:1.345": 0.5, "hl": 1.5, "catoni:1": 2.5}


def test_verdict_main_success_when_the_best_robust_estimator_is_at_least_twice_as_close():
    kind, text = ev.verdict_main(_analysis([1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))      # Mittel 1,0 gegen Median 0,5: genau die Hälfte
    assert kind == "success" and "**Mittel 1,00 Stunden**" in text and "**S1 nur 0,50**" in text


def test_verdict_main_info_when_the_advantage_is_smaller_than_a_factor_two():
    kind, text = ev.verdict_main(_analysis([1.0, 0.51, 0.6, 0.7, 0.8, 0.9, 1.0]))
    assert kind == "info" and "kein klarer Vorteil" in text and "S1 mit 0,51" in text and "Mittel 1,00" in text


def test_verdict_main_all_exact():
    kind, text = ev.verdict_main(_analysis([0.0] * 7))
    assert kind == "info" and "auf die Stelle genau" in text


def test_verdict_e1_threshold_is_inclusive_at_0_05_spread():
    rows = {"mean": (1.0, 0.5), "median": (0.66, 0.14), "huber:1.345": (0.95, 0.20), "hl": (0.96, 0.25), "trim:0.1": (0.94, 0.225)}
    kind, text = ev.verdict_e1(rows, 100)
    assert kind == "success" and "Bei n = 100" in text and "Effizienz 0,940 bis 0,960" in text and "Verschiebung 0,200 bis 0,250" in text and "(Unterschied 0,050)" in text
    assert "Effizienz 0,660 und Verschiebung 0,140" in text and "das Mittel bei 1 und 0,500" in text
    rows["hl"] = (0.96, 0.2501)
    assert ev.verdict_e1(rows, 100)[0] == "info"


def test_cliff_eps_needs_a_tenfold_jump_of_at_least_one_hour_and_ignores_zero():
    eps = [0.0, 0.05, 0.10, 0.15]
    assert ev.cliff_eps(eps, [0.0, 0.1, 0.2, 62.6]) == 0.15
    assert ev.cliff_eps(eps, [0.0, 50.0, 100.0, 150.0]) is None                 # das Mittel: kein Sprung, nur der Punkt ε = 0 fällt weg
    assert ev.cliff_eps(eps, [0.0, 0.01, 0.1, 0.9]) is None                     # zehnfach, aber unter einer Stunde
    assert ev.cliff_eps(eps, [0.0, 1.0, 2.0, 11.0]) is None                     # über eine Stunde, aber nicht das Zehnfache (genau 5,5)
    assert ev.cliff_eps(eps, [0.0, 1.0, 2.0, 20.0]) == 0.15                     # genau das Zehnfache und mehr als eine Stunde
    assert ev.cliff_eps(eps, [0.0, 0.1, 1.0, 11.0]) == 0.15                     # 0,1 → 1,0: zehnfach, aber nur 0,9 Stunden mehr; 1,0 → 11,0: zehnfach und 10 Stunden


def test_last_eps_below():
    eps = [0.0, 0.05, 0.1, 0.15]
    assert ev.last_eps_below(eps, [0.0, 0.5, 0.99, 62.0]) == 0.1
    assert ev.last_eps_below(eps, [0.0, 0.5, 1.0, 62.0]) == 0.05                 # genau 1 zählt nicht mehr als „unter 1“
    assert ev.last_eps_below(eps, [2.0, 3.0, 4.0, 5.0]) is None


def test_verdict_e2_names_cliffs_and_smooth_curves():
    eps = [0.0, 0.05, 0.10, 0.15]
    kind, text = ev.verdict_e2({"trim:0.1": [0.0, 0.1, 0.2, 62.0], "median": [0.0, 0.1, 0.2, 0.3]}, eps, lambda s: s.upper())
    assert kind == "info" and "TRIM:0.1 bei ε = 15 %" in text and "**Stetiger Anstieg** ohne Sprung: MEDIAN." in text and text.endswith("MEDIAN.")


def test_verdict_hl_cliff_texts():
    eps = [0.28, 0.29, 0.30]
    kind, text = ev.verdict_hl_cliff({"hl": [0.5, 1.6, 498.0]}, eps, str.upper)
    assert kind == "info" and "HL: letzter Anteil unter 1 Stunde Verschiebung 28 %, Sprung bei 30 %" in text
    _, none = ev.verdict_hl_cliff({"median": [0.5, 0.6, 0.7]}, eps, str.upper)
    assert "MEDIAN: kein Sprung bis 30 %, letzter Anteil unter 1 Stunde 30 %" in none


def test_verdict_e3_splits_bounded_from_growing():
    ds = [3.0, 30.0, 1000.0]
    kind, text = ev.verdict_e3({"mean": [0.3, 3.0, 100.0], "median": [0.14, 0.14, 0.14], "catoni:1": [0.3, 1.0, 6.0]}, ds)
    assert kind == "info" and "**Median**" in text and "**Catoni (λ × 1)**" in text and "3,0 bei d = 30, 100 bei d = 1000" in text
    assert "innerhalb des 1,5-Fachen (beschränkt)" in text and "wächst bei" in text
    _, edge = ev.verdict_e3({"mean": [1.0, 1.0, 1.0], "median": [1.0, 2.0, 3.0]}, ds)               # genau das 1,5-Fache zählt noch als beschränkt
    assert "**Median**" in edge.split("wächst bei")[0]


def test_crossing_and_slopes():
    assert ev.crossing_n([10, 30, 100], [1.0, 0.5, 0.2], [0.9, 0.6, 0.3]) == 30
    assert ev.crossing_n([10, 30, 100], [1.0, 0.5, 0.2], [0.9, 0.4, 0.3]) == 100         # gewinnt bei 30 nicht, also nicht „ab 10“
    assert ev.crossing_n([10, 30], [1.0, 1.0], [0.5, 0.5]) is None
    assert ev.local_slopes([10, 100, 1000], [1.0, 10.0, 1000.0]) == pytest.approx([1.0, 2.0])
    assert ev.ucurve_minimum([1.0, 2.0, 4.0], [3.0, 1.0, 2.0]) == (2.0, 1.0)
    assert ev.ucurve_plateau([1.0, 2.0, 4.0, 8.0], [3.0, 1.0, 1.05, 1.06]) == (2.0, 4.0)


def test_verdict_e4_reports_population_bias_and_crossing():
    ns = [10, 30, 100]
    rmse = {"mean": [1.0, 0.5, 0.2], "median": [0.9, 0.6, 0.3], "trim:0.1": [0.8, 0.6, 0.3], "hl": [0.8, 0.7, 0.4], "huber:1.345": [0.8, 0.7, 0.4], "catoni:1": [0.9, 0.4, 0.15]}
    pop = {"trim:0.1": 1.2, "hl": 1.25, "huber:1.345": 1.3}
    kind, text = ev.verdict_e4("Lognormal (σ = 1)", ns, rmse, pop, 1.65, 1.0, [-0.4, -0.41, -0.42])
    assert kind == "warning" and "Mittelwert 1,650, Median 1,000" in text and "Trimmen 10 % bei 1,200" in text and "Hodges–Lehmann bei 1,250" in text and "Huber (c = 1,345) bei 1,300" in text
    assert "-0,400 bei n = 10 und -0,420 bei n = 100" in text and "ab n = 30 einen kleineren RMSE" in text and "Catoni (λ × 1) hat bei n = 100 einen RMSE von 0,150 gegen 0,200" in text
    rmse["median"] = [0.1, 0.1, 0.1]
    _, never = ev.verdict_e4("X", ns, rmse, pop, 1.0, 1.0, [0.0, 0.0, 0.0])
    assert "nie gegen Median" in never
    for k in rmse:
        rmse[k] = [0.1, 0.1, 0.1]
    rmse["mean"] = [1.0, 1.0, 1.0]
    assert "nie einen kleineren RMSE" in ev.verdict_e4("X", ns, rmse, pop, 1.0, 1.0, [0.0, 0.0, 0.0])[1]


def test_verdict_e5_names_best_estimator_and_median_worse_than_mean():
    kind, text = ev.verdict_e5({"Normal": {"median": 1.25, "hl": 1.025}, "t(5)": {"median": 1.02, "hl": 0.90}})
    assert kind == "info" and "Normal: Hodges–Lehmann (1,025)" in text and "t(5): Hodges–Lehmann (0,900)" in text
    assert "Median auf Normal (1,250)" in text and "Median auf t(5) (1,020)" in text
    assert "Schlechter als das Mittel" not in ev.verdict_e5({"Laplace": {"median": 0.75, "hl": 0.82}})[1]


def test_verdict_stairs_branches():
    assert ev.verdict_stairs(10, 5)[0] == "warning" and "0 Werte" in ev.verdict_stairs(10, 5)[1]
    assert ev.verdict_stairs(20, 5) == ("info", "Bei n = 20 und α = 5 % werden **1 Wert** je Seite abgeschnitten, es bleiben 18.")
    assert "**3 Werte**" in ev.verdict_stairs(10, 30)[1]
    assert ev.verdict_stairs(10, 0)[0] == "info"                                                     # α = 0: nichts wird erwartet, also keine Warnung


def test_verdict_catoni_dominated_or_not():
    rows = {"mean": (1.0, 0.5, 100.0), "huber:1.345": (0.95, 0.2, 0.2), "catoni:1": (1.0, 0.47, 6.0), "catoni:32": (0.94, 0.25, 0.7)}
    kind, text = ev.verdict_catoni(rows)
    assert kind == "info" and "Effizienz 1,000, Verschiebung 0,470 bei d = 5 und 6,0 bei d = 1000 (Mittel: 0,500 und 100,0)" in text and "Mit λ × 32: 0,940, 0,250, 0,700" in text
    assert text.endswith("Huber ist in allen drei Kennzahlen besser als Catoni mit dem größten Faktor.")
    rows["catoni:32"] = (0.96, 0.25, 0.7)                   # Catoni hat die höhere Effizienz: nicht mehr dominiert
    assert not ev.verdict_catoni(rows)[1].endswith("Faktor.")


def test_verdict_ucurve_threshold_and_numbers():
    kind, text = ev.verdict_ucurve(14.5, 0.0682, 0.0680, 0.0999)
    assert kind == "success" and "C ≈ 14,5" in text and "RMSE von 0,0682" in text and "0,0680" in text and "0 % mehr" in text and "10 % der MSE" in text
    assert ev.verdict_ucurve(14.5, 0.0682, 0.0680, 0.25)[0] == "warning"                       # genau 25 %: nicht mehr „unter 25 %“
    assert "50 % mehr" in ev.verdict_ucurve(1.0, 0.15, 0.10, 0.5)[1]


def test_verdict_scaling_names_slopes_and_exponents():
    kind, text = ev.verdict_scaling({"Pareto (α = 1,5)": [0.66, 0.67], "Lognormal (σ = 1)": [0.38, 0.22]}, {"Pareto (α = 1,5)": 1 / 1.5, "Lognormal (σ = 1)": None})
    assert kind == "info" and "Steigung 0,66 bis 0,67 (Herleitung 1/α = 0,667)" in text and "Steigung 0,22 bis 0,38 (fällt mit n, kein Potenzgesetz)" in text


def test_verdict_wrong_c_orders_the_costs():
    rows = [{"factor": 0.25, "rmse_ratio": 3.99}, {"factor": 1.0, "rmse_ratio": 1.0}, {"factor": 4.0, "rmse_ratio": 1.59}, {"factor": 16.0, "rmse_ratio": 5.04}]
    kind, text = ev.verdict_wrong_c(rows)
    assert kind == "warning" and "das 4,0-fache" in text and "das 1,6-fache" in text and "das 5,0-fache" in text and "Zu klein ist teurer als zu groß" in text
    rows[0]["rmse_ratio"] = 1.2
    assert "Zu groß ist hier teurer als zu klein" in ev.verdict_wrong_c(rows)[1]


def test_verdict_data_c_lists_all_points_and_denies_a_bound():
    kind, text = ev.verdict_data_c({3.0: 6.39, 1000.0: 3329.87}, 1.0)
    assert kind == "warning" and "6,4 bei y = 3, 3329,9 bei y = 1000" in text and "ε_DP = 1" in text and "eine Schranke gibt es nicht" in text


def test_verdict_winsor_sensitivity_numbers():
    kind, text = ev.verdict_winsor_sensitivity(2.28, 1.07, 5.69)
    assert kind == "warning" and "**2,3-fachen**" in text and "1,1-fach" in text and "5,7-fach" in text


def test_spec_name():
    assert ev.spec_name(("lognormal", 1.5)) == "Lognormal (σ = 1,5)" and ev.spec_name(("pareto", 3.0)) == "Pareto (α = 3)"
