"""Auswertung: Einstellungen, die Analyse einer Stichprobe, Zusammenfassungen der Messreihen (Klippe, Kreuzungs-n, Minimum der U-Kurve, Steigungen) und die Verdikte.

Jedes Verdikt ist eine reine Funktion `verdict_*(…) -> (Art, Text)` (Art: success, info oder warning) und bekommt Zahlen oder Zeilen der Messreihen, nie Seeds:
tests/test_verdicts.py prüft Schwellen und jede Zahl im Text an handgesetzten Werten. Zahlen im Text kommen aus der Rechnung (f-String), nicht aus dem Gedächtnis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import thu_constants as C
import thu_robust as R
import thu_scenario as S


# --- Formatierung -------------------------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 1) -> str:
    """Deutsche Zahl: Komma, keine Tausendertrennung (Punkt wäre ein Dezimalpunkt)."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    """Anteil 0 … 1 als Prozentzahl mit Komma."""
    return de(100 * x, nd)


_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_g(x: float) -> str:
    """Zahl in kürzester Form mit Komma (0,5; 1,345; 1000); Zehnerpotenzen ab 10⁶ als 10⁶ (statt 1e+06)."""
    if x >= 1e6 and math.log10(x) == round(math.log10(x)):
        return "10" + str(round(math.log10(x))).translate(_SUPERSCRIPT)
    return f"{x:g}".replace(".", ",") if x < 1e6 else de(x, 0)


# --- Einstellungen und Analyse einer Stichprobe -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    dist: str = C._BASE["dist"]
    n: int = C.DEFAULT_N
    eps: int = C.DEFAULT_EPS                 # Prozent
    d: float = C.DEFAULT_D
    side: str = C._BASE["side"]
    alpha: int = C.DEFAULT_ALPHA             # Prozent
    c: float = C.DEFAULT_C
    scale: str = C._BASE["scale"]
    mult: float = C.DEFAULT_MULT
    delta: float = C.DEFAULT_DELTA
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    x: np.ndarray
    is_outlier: np.ndarray
    specs: tuple
    labels: tuple
    values: tuple
    target_mean: float
    target_median: float
    mad: float
    cut: int
    catoni_alpha: float


def specs_of(s: Settings) -> tuple:
    """Die sieben Schätzer der Hauptansicht mit den Reglerwerten: Mittel, Median, Trimmen α, Winsorisieren α, Huber c, Hodges–Lehmann, Catoni λ."""
    return ("mean", "median", f"trim:{s.alpha / 100:g}", f"winsor:{s.alpha / 100:g}", f"huber:{s.c:g}", "hl", f"catoni:{s.mult:g}")


def known_scale(s: Settings):
    return None if s.scale == C.SCALES[0] else 1.0


def analyse(s: Settings) -> Analysis:
    x, flag = S.one_sample(s.dist, s.n, s.eps / 100.0, s.d, s.side == C.SIDES[1], s.seed)
    specs = specs_of(s)
    values = tuple(R.scalar(R.estimate(spec, x, known_scale(s), s.delta)) for spec in specs)
    mad = R.scalar(R.mad_scale(x))
    v = mad * mad if mad > 0 else 1.0
    alpha0 = float(R.catoni_alpha(s.n, v, s.delta)) * s.mult
    mean_, median_ = S.population(s.dist)
    return Analysis(s, x, flag, specs, tuple(R.label(sp) for sp in specs), values, mean_, median_, mad, R.n_cut(s.n, s.alpha / 100.0), alpha0)


def deviations(a: Analysis) -> dict[str, float]:
    """Abstand jedes Schätzers zum Median des Kerns (dem Ziel der Hauptansicht), im Betrag."""
    return {sp: abs(v - a.target_median) for sp, v in zip(a.specs, a.values)}


# --- Zusammenfassungen der Messreihen ---------------------------------------------------------------------------------------------------

def cliff_eps(eps_list, series, factor: float = 10.0, floor: float = 1.0):
    """Kleinster Anteil ε > 0, bei dem die Verschiebung gegenüber dem vorherigen Punkt mit ε > 0 um mindestens `floor` Stunden UND um das `factor`-fache springt (Klippe);
    None bei stetigem Verlauf. Der Punkt ε = 0 zählt nicht (dort ist die Verschiebung 0, jeder Anstieg wäre ein Sprung)."""
    points = [(e, v) for e, v in zip(eps_list, series) if e > 0]
    for (_, v0), (e1, v1) in zip(points, points[1:]):
        if v1 - v0 >= floor and v1 >= factor * max(v0, 1e-3):
            return e1
    return None


def last_eps_below(eps_list, series, limit: float = 1.0):
    """Größter Anteil ε, bei dem die Verschiebung noch unter `limit` liegt (None, wenn schon der kleinste Anteil darüber liegt)."""
    good = [e for e, v in zip(eps_list, series) if v < limit]
    return max(good) if good else None


def crossing_n(ns, rmse_mean, rmse_other):
    """Kleinstes n des Rasters, ab dem das Mittel einen kleineren RMSE hat als der andere Schätzer und es auch für alle größeren n des Rasters hat; None, wenn nie."""
    wins = [m < o for m, o in zip(rmse_mean, rmse_other)]
    for i in range(len(ns)):
        if all(wins[i:]):
            return ns[i]
    return None


def local_slopes(ns, values) -> list[float]:
    """Steigung von ln(values) über ln(n) zwischen benachbarten n des Rasters."""
    ln, lv = np.log(np.asarray(ns, dtype=float)), np.log(np.asarray(values, dtype=float))
    return [float((lv[i + 1] - lv[i]) / (ln[i + 1] - ln[i])) for i in range(len(ns) - 1)]


def ucurve_minimum(cgrid, curve) -> tuple[float, float]:
    """(C, Wert) am Minimum der Kurve auf dem Raster."""
    k = int(np.argmin(curve))
    return float(cgrid[k]), float(curve[k])


def ucurve_plateau(cgrid, curve, tol: float = 0.05) -> tuple[float, float]:
    """Kleinstes und größtes C, deren Wert höchstens (1 + tol) mal das Minimum ist."""
    m = float(np.min(curve))
    idx = [i for i, v in enumerate(curve) if v <= m * (1 + tol)]
    return float(cgrid[min(idx)]), float(cgrid[max(idx)])


def spec_name(spec: tuple) -> str:
    kind, p = spec
    return f"Lognormal (σ = {fmt_g(p)})" if kind == "lognormal" else f"Pareto (α = {fmt_g(p)})"


# --- Verdikte -----------------------------------------------------------------------------------------------------------------------------

def verdict_main(a: Analysis) -> tuple[str, str]:
    """Hauptansicht: Abstand der Schätzer zum Median des Kerns für diese eine Stichprobe."""
    dev = deviations(a)
    mean_dev, best_spec = dev["mean"], min((sp for sp in a.specs if sp != "mean"), key=dev.get)
    best_dev = dev[best_spec]
    best = a.labels[a.specs.index(best_spec)]
    if mean_dev < 1e-9 and best_dev < 1e-9:
        return "info", "Alle Schätzer treffen das Ziel (Median des Kerns) auf die Stelle genau."
    if best_dev * 2 <= mean_dev:
        return "success", (f"In dieser Stichprobe liegt das **Mittel {de(mean_dev, 2)} Stunden** neben dem Ziel (Median des Kerns), **{best} nur {de(best_dev, 2)}**. "
                           "Das ist eine einzelne Stichprobe: Die Messreihen unten zeigen, wie sich das im Mittel über viele verhält.")
    return "info", (f"In dieser Stichprobe liegt das **Mittel {de(mean_dev, 2)} Stunden** neben dem Ziel (Median des Kerns), der nächste Schätzer ist **{best} mit {de(best_dev, 2)}**: "
                    "kein klarer Vorteil eines robusten Schätzers. Das ist eine einzelne Stichprobe.")


def verdict_e1(rows: dict, n: int) -> tuple[str, str]:
    """E1: Huber (c = 1,345), Hodges–Lehmann und Trimmen 10 % liegen bei fast gleicher Effizienz auf fast gleicher Verschiebung; rows: spec → (Effizienz, Verschiebung)."""
    names = ("huber:1.345", "hl", "trim:0.1")
    effs = [rows[s][0] for s in names]
    shifts = [rows[s][1] for s in names]
    spread = max(shifts) - min(shifts)
    text = (f"Bei n = {n} liegen **Huber (c = 1,345), Hodges–Lehmann und Trimmen 10 %** bei Effizienz {de(min(effs), 3)} bis {de(max(effs), 3)} und Verschiebung {de(min(shifts), 3)} bis {de(max(shifts), 3)} Stunden "
            f"(Unterschied {de(spread, 3)}), der Median bei Effizienz {de(rows['median'][0], 3)} und Verschiebung {de(rows['median'][1], 3)}, das Mittel bei 1 und {de(rows['mean'][1], 3)}.")
    return ("success", text) if spread <= 0.05 else ("info", text)


def verdict_e2(series: dict[str, list[float]], eps_list, label_of) -> tuple[str, str]:
    """E2: je Schätzer Klippe oder stetiger Anstieg; series: spec → Verschiebungen über eps_list."""
    cliffs = {sp: cliff_eps(eps_list, v) for sp, v in series.items()}
    with_cliff = [f"{label_of(sp)} bei ε = {pct(e, 0)} %" for sp, e in cliffs.items() if e is not None]
    smooth = [label_of(sp) for sp, e in cliffs.items() if e is None]
    return "info", ("**Klippe** (Sprung um mindestens das Zehnfache und eine Stunde): " + ("; ".join(with_cliff) if with_cliff else "keiner") + ". "
                    "**Stetiger Anstieg** ohne Sprung: " + (", ".join(smooth) if smooth else "keiner") + ".")


def verdict_hl_cliff(fine_series: dict[str, list[float]], eps_list, label_of) -> tuple[str, str]:
    """E2 Feinraster: genaue Lage der Klippen; Angabe: größter Anteil mit Verschiebung unter 1 und erster Anteil mit Sprung."""
    parts = []
    for sp, series in fine_series.items():
        last_ok = last_eps_below(eps_list, series)
        cliff = cliff_eps(eps_list, series)
        parts.append(f"{label_of(sp)}: " + (f"letzter Anteil unter 1 Stunde Verschiebung {pct(last_ok, 0)} %, Sprung bei {pct(cliff, 0)} %" if cliff is not None else
                                           f"kein Sprung bis {pct(max(eps_list), 0)} %, letzter Anteil unter 1 Stunde {pct(last_ok, 0) if last_ok is not None else '-'} %"))
    return "info", "Feines Raster (1-Prozent-Schritte, d = 1000): " + "; ".join(parts) + "."


def verdict_e3(shift_by_d: dict[str, list[float]], ds) -> tuple[str, str]:
    """E3: beschränkt (Verschiebung bei größtem d höchstens 1,5-mal die bei d = 30) oder wachsend."""
    i30 = ds.index(30.0)
    bounded = [sp for sp, v in shift_by_d.items() if sp != "mean" and v[-1] <= 1.5 * v[i30]]
    growing = [sp for sp, v in shift_by_d.items() if sp != "mean" and v[-1] > 1.5 * v[i30]]
    text = (f"Von d = 30 bis d = {fmt_g(ds[-1])} bleibt die Verschiebung bei **{', '.join(R.label(s) for s in bounded) or 'keinem'}** innerhalb des 1,5-Fachen (beschränkt); "
            f"sie wächst bei **{', '.join(R.label(s) for s in growing) or 'keinem'}**. Das Mittel wächst proportional zu d ({de(shift_by_d['mean'][i30], 1)} bei d = 30, {de(shift_by_d['mean'][-1], 0)} bei d = {fmt_g(ds[-1])}).")
    return "info", text


def verdict_e4(dist: str, ns, rmse: dict[str, list[float]], population: dict[str, float], mean_: float, median_: float, bias_trim: list[float]) -> tuple[str, str]:
    """E4: Zielgröße bei Schiefe. rmse: spec → RMSE gegen den Mittelwert über ns; bias_trim: Bias von Trimmen 10 % gegen den Mittelwert über ns."""
    others = [sp for sp in rmse if sp not in ("mean", "catoni:1")]
    wins = {sp: crossing_n(ns, rmse["mean"], rmse[sp]) for sp in others}
    never = [R.label(sp) for sp, n in wins.items() if n is None]
    reached = [n for n in wins.values() if n is not None]
    text = (f"{dist}: Mittelwert {de(mean_, 3)}, Median {de(median_, 3)}. An der Population liegen Trimmen 10 % bei {de(population['trim:0.1'], 3)}, Hodges–Lehmann bei {de(population['hl'], 3)} und "
            f"Huber (c = 1,345) bei {de(population['huber:1.345'], 3)}: dazwischen, aber **nicht beim Mittelwert**. Der Bias von Trimmen 10 % gegen den Mittelwert ist {de(bias_trim[0], 3)} bei n = {ns[0]} und "
            f"{de(bias_trim[-1], 3)} bei n = {ns[-1]}: er schrumpft nicht. ")
    if reached:
        text += f"Das Mittel hat ab n = {max(reached)} einen kleineren RMSE gegen den Mittelwert als {', '.join(R.label(sp) for sp in others if wins[sp] is not None)}"
        text += (f"; nie gegen {', '.join(never)}." if never else ".")
    else:
        text += "Im Raster hat das Mittel nie einen kleineren RMSE gegen den Mittelwert als die Robusten."
    if "catoni:1" in rmse:
        text += f" Catoni (λ × 1) hat bei n = {ns[-1]} einen RMSE von {de(rmse['catoni:1'][-1], 3)} gegen {de(rmse['mean'][-1], 3)} beim Mittel."
    return "warning", text


def verdict_e5(ratios: dict[str, dict[str, float]]) -> tuple[str, str]:
    """E5: kein Schätzer gewinnt überall. ratios: Verteilung → spec → RMSE-Verhältnis zum Mittel."""
    best = {dist: min(r, key=r.get) for dist, r in ratios.items()}
    names = [f"{dist}: {R.label(sp)} ({de(ratios[dist][sp], 3)})" for dist, sp in best.items()]
    worse = [f"{R.label(sp)} auf {dist} ({de(v, 3)})" for dist, r in ratios.items() for sp, v in r.items() if sp == "median" and v > 1.0]
    text = "Bester Schätzer je Rand (RMSE geteilt durch RMSE des Mittels): " + "; ".join(names) + ". " + ("Schlechter als das Mittel: " + "; ".join(worse) + "." if worse else "")
    return "info", text.strip()


def verdict_stairs(n: int, alpha_percent: int) -> tuple[str, str]:
    """E6 live: Abschneidezahl zu n und α."""
    k = R.n_cut(n, alpha_percent / 100.0)
    if k == 0 and alpha_percent > 0:
        return "warning", f"Bei n = {n} und α = {alpha_percent} % werden **0 Werte** je Seite abgeschnitten: das getrimmte Mittel ist das Mittel."
    return "info", f"Bei n = {n} und α = {alpha_percent} % werden **{k} {'Wert' if k == 1 else 'Werte'}** je Seite abgeschnitten, es bleiben {n - 2 * k}."


def verdict_catoni(rows: dict) -> tuple[str, str]:
    """E7: rows: spec → (Effizienz, Verschiebung bei ε = 10 %, d = 5, Verschiebung bei ε = 10 %, d = 1000), für mean, huber:1.345, catoni:1 und catoni:32 (n = 100)."""
    m, h, cat, big = rows["mean"], rows["huber:1.345"], rows["catoni:1"], rows["catoni:32"]
    dominated = h[0] >= big[0] and h[1] <= big[1] and h[2] <= big[2]
    text = (f"Catoni mit dem Wert λ × 1 aus der Definition: Effizienz {de(cat[0], 3)}, Verschiebung {de(cat[1], 3)} bei d = 5 und {de(cat[2], 1)} bei d = 1000 (Mittel: {de(m[1], 3)} und {de(m[2], 1)}). "
            f"Mit λ × 32: {de(big[0], 3)}, {de(big[1], 3)}, {de(big[2], 3)}. Huber (c = 1,345): {de(h[0], 3)}, {de(h[1], 3)}, {de(h[2], 3)}"
            + (": Huber ist in allen drei Kennzahlen besser als Catoni mit dem größten Faktor." if dominated else "."))
    return "info", text


def verdict_ucurve(c_min: float, rmse_min: float, rmse_sample: float, noise_share: float) -> tuple[str, str]:
    """E8: U-Kurve. Minimum bei C, Fehler dort gegen den nicht privaten Fehler, Anteil des Rauschens an der MSE."""
    extra = rmse_min / rmse_sample - 1 if rmse_sample > 0 else math.inf
    text = (f"Das Minimum liegt bei C ≈ {de(c_min, 1)} mit einem RMSE von {de(rmse_min, 4)} (nicht privates Stichprobenmittel: {de(rmse_sample, 4)}, also {de(100 * extra, 0)} % mehr). "
            f"Das Rauschen macht dort {pct(noise_share, 0)} % der MSE aus.")
    return ("success" if noise_share < 0.25 else "warning"), text


def verdict_scaling(slopes: dict[str, list[float]], exponents: dict[str, float | None]) -> tuple[str, str]:
    """E8: Steigungen d ln C*/d ln n; bei Pareto gegen 1/α, bei Lognormal fallend."""
    parts = []
    for name, sl in slopes.items():
        ex = exponents.get(name)
        parts.append(f"{name}: Steigung {de(min(sl), 2)} bis {de(max(sl), 2)}" + (f" (Herleitung 1/α = {de(ex, 3)})" if ex else " (fällt mit n, kein Potenzgesetz)"))
    return "info", "; ".join(parts) + "."


def verdict_wrong_c(rows: list[dict]) -> tuple[str, str]:
    """E8: Kosten einer falschen Grenze; rows: Einträge mit factor und rmse_ratio."""
    by = {r["factor"]: r["rmse_ratio"] for r in rows}
    return "warning", (f"Ein Faktor 4 zu klein kostet das {de(by[0.25], 1)}-fache des RMSE, ein Faktor 4 zu groß das {de(by[4.0], 1)}-fache, ein Faktor 16 zu groß das {de(by[16.0], 1)}-fache. "
                       f"{'Zu klein ist teurer als zu groß' if by[0.25] > by[4.0] else 'Zu groß ist hier teurer als zu klein'} (Faktor 4 in beide Richtungen).")


def verdict_data_c(worst_by_y: dict[float, float], eps: float) -> tuple[str, str]:
    """E8: C aus den Daten gewählt: das größte log-Verhältnis wächst mit y; es gibt keine endliche Schranke."""
    ys = sorted(worst_by_y)
    text = (f"Wird C aus den Daten gewählt und nur das Rauschen darauf skaliert, ist das größte log-Verhältnis {', '.join(de(worst_by_y[y], 1) + ' bei y = ' + fmt_g(y) for y in ys)} "
            f"(zugesagt wäre ε_DP = {fmt_g(eps)}): es wächst mit y, **eine Schranke gibt es nicht**. Dieser Mechanismus ist für kein ε_DP differentiell privat.")
    return "warning", text


def verdict_winsor_sensitivity(mean_ratio: float, min_ratio: float, largest_ratio: float) -> tuple[str, str]:
    return "warning", (f"Kappt man auf Schwellen aus den Daten (Winsorisieren), ändert sich das Mittel beim Ersetzen eines Werts bis zum **{de(mean_ratio, 1)}-fachen** (Mittel über die Datensätze; kleinster Datensatz "
                       f"{de(min_ratio, 1)}-fach, größter {de(largest_ratio, 1)}-fach) von (hi − lo)/n: Rauschen auf Basis von (hi − lo)/n wäre zu klein.")
