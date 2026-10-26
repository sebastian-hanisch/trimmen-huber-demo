"""Messreihen des Abschnitts „Kappen und privates Mittel“: U-Kurve (Simulation gegen die geschlossene Zerlegung), Skalierung der besten Kappungsgrenze, Kosten einer falschen
Grenze, Definition durch Aufzählung benachbarter Datensätze, Empfindlichkeit beim Winsorisieren. Reine Funktionen über feste Zufalls-Seeds."""

from __future__ import annotations

import math

import numpy as np

import thu_neighbors as N
import thu_private as P
import thu_robust as R
from thu_curves import mean_std
from thu_scenario import Stream, cell_seed, draw


def draw_spec(spec: tuple[str, float], stream: Stream, count: int) -> np.ndarray:
    """Werte der Verteilung aus P-Spezifikation: ("lognormal", σ) oder ("pareto", α)."""
    return draw(spec[0], spec[1], stream, count)


def c_grid(k_count: int = 25) -> list[float]:
    """Raster der Kappungsgrenzen 0,5 · 1,4^k, k = 0 … k_count − 1."""
    return [0.5 * 1.4 ** k for k in range(k_count)]


# ---------------------------------------------------------------- U-Kurve

def ucurve_cell(spec: tuple[str, float], n: int, runs: int, seeds, eps_list, cgrid) -> dict:
    """Simulation des privaten Mittels (Kappen auf [0, C], Laplace-Rauschen mit Skala C/(n·ε)) über das Raster der Kappungsgrenzen.

    Je ε und C: mittlerer Betragsfehler, RMSE und 95-%-Quantil des Betragsfehlers gegen das Populationsmittel; dazu der Fehler des ungekappten Stichprobenmittels und des gekappten
    Mittels ohne Rauschen. exact_mse ist die geschlossene Zerlegung (thu_private.exact_mse), ratio_* das Verhältnis RMSE²/MSE-Formel über das Raster."""
    cg = np.asarray(cgrid, dtype=np.float64)
    mu = P.pop_mean(spec)
    sample_abs, sample_rmse = [], []
    clip_abs = []
    acc = {e: {"abs": [], "rmse": [], "q95": []} for e in eps_list}
    rows_per_chunk = max(1, 1_000_000 // n)
    for seed in seeds:
        stream = Stream(cell_seed(seed, "m7", spec[0], spec[1], n))
        m_clip = np.empty((runs, len(cg)))
        xbar = np.empty(runs)
        for a in range(0, runs, rows_per_chunk):
            rows = min(rows_per_chunk, runs - a)
            x = draw_spec(spec, stream, rows * n).reshape(rows, n)
            xbar[a:a + rows] = x.mean(axis=1)
            for g, c in enumerate(cg):
                m_clip[a:a + rows, g] = np.minimum(x, c).mean(axis=1)
        noise = draw("laplace", 0.0, Stream(cell_seed(seed, "m7noise", spec[0], spec[1], n)), runs * len(cg)).reshape(runs, len(cg))
        sample_abs.append(float(np.abs(xbar - mu).mean()))
        sample_rmse.append(float(math.sqrt(np.mean((xbar - mu) ** 2))))
        clip_abs.append(np.abs(m_clip - mu).mean(axis=0))
        for e in eps_list:
            err = np.abs(m_clip + (cg / (n * e))[None, :] * noise - mu)
            acc[e]["abs"].append(err.mean(axis=0))
            acc[e]["rmse"].append(np.sqrt((err ** 2).mean(axis=0)))
            acc[e]["q95"].append(np.quantile(err, 0.95, axis=0))

    def curve(per_seed):
        arr = np.asarray(per_seed)
        return [[float(m), float(s)] for m, s in zip(arr.mean(axis=0), arr.std(axis=0))]

    out_eps = []
    for e in eps_list:
        exact = P.exact_mse(spec, cg, n, e)[0]
        rmse_mean = np.mean(acc[e]["rmse"], axis=0)
        ratio = rmse_mean ** 2 / exact
        out_eps.append({"eps": e, "abs": curve(acc[e]["abs"]), "rmse": curve(acc[e]["rmse"]), "q95": curve(acc[e]["q95"]), "exact_mse": [float(v) for v in exact],
                        "ratio_median": float(np.median(ratio)), "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max())})
    return {"spec": list(spec), "n": n, "runs_per_seed": runs, "cgrid": [float(c) for c in cg], "mu": mu, "sample_abs": mean_std(sample_abs), "sample_rmse": mean_std(sample_rmse),
            "clip_abs": curve(clip_abs), "eps": out_eps}


# ---------------------------------------------------------------- Skalierung von C*

def scaling_table(specs, ns, epss) -> list[dict]:
    """Beste Kappungsgrenze C* der geschlossenen MSE-Zerlegung und der Anteil des Rauschens an der MSE im Minimum, je Verteilung, n und ε."""
    out = []
    for spec in specs:
        for n in ns:
            for e in epss:
                c = P.c_star(spec, n, e)
                mse, _, _, noise = P.exact_mse(spec, [c], n, e)
                out.append({"spec": list(spec), "n": n, "eps": e, "c_star": c, "noise_share": float(noise[0] / mse[0]), "rmse": float(math.sqrt(mse[0]))})
    return out


def wrong_c_costs(spec: tuple[str, float], n: int, eps: float, factors=(0.25, 1.0, 4.0, 16.0)) -> dict:
    """Wie viel schlechter (RMSE-Verhältnis) ist das Mittel mit Kappungsgrenze f · C* als mit C*? Nach der geschlossenen Zerlegung."""
    c = P.c_star(spec, n, eps)
    best = float(P.exact_mse(spec, [c], n, eps)[0][0])
    rows = []
    for f in factors:
        mse = float(P.exact_mse(spec, [f * c], n, eps)[0][0])
        rows.append({"factor": f, "c": f * c, "rmse_ratio": math.sqrt(mse / best)})
    return {"spec": list(spec), "n": n, "eps": eps, "c_star": c, "rmse_best": math.sqrt(best), "rows": rows}


# ---------------------------------------------------------------- Definition durch Aufzählung

def enumeration_summary() -> dict:
    """Aufzählung aller benachbarten Datensätze auf kleinen Gittern.

    laplace: Werte 0 … 6, Kappung [0, 3], n = 3 und 4, ε = 0,5 und 1 — korrekte Skala und zwei absichtliche Fehler.
    symmetric: Werte −3 … 3, Kappung [−2, 2] (Empfindlichkeit 2C/n).
    data_c: C aus den Daten (Wert mit Rang 2 unter vier Werten aus 1 … 6), ε = 1, an den Stellen y = 3, 10, 100, 1000."""
    laplace = []
    for n in (3, 4):
        for eps in (0.5, 1.0):
            row = {"n": n, "eps": eps}
            for variant in ("korrekt", "empfindlichkeit_n_plus_1", "ohne_kappung"):
                worst, pairs = N.max_log_ratio_laplace(n, range(0, 7), 0.0, 3.0, eps, variant)
                row[variant] = worst
                row["pairs"] = pairs
            laplace.append(row)
    sym = {}
    worst, pairs = N.max_log_ratio_laplace(3, range(-3, 4), -2.0, 2.0, 1.0, "korrekt")
    sym = {"n": 3, "eps": 1.0, "worst": worst, "pairs": pairs, "sensitivity": P.sensitivity(-2.0, 2.0, 3)}
    data_c = []
    for y in (3.0, 10.0, 100.0, 1000.0):
        worst, d, d2 = N.max_log_ratio_data_dependent_c(4, range(1, 7), 1.0, y)
        data_c.append({"y": y, "worst": worst, "data": list(d), "neighbor": list(d2)})
    return {"laplace": laplace, "symmetric": sym, "data_c": data_c}


# ---------------------------------------------------------------- Kappen ist Winsorisieren

WINSOR_CANDIDATES = (0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0)


def winsor_sensitivity_cell(n: int, k: int, datasets: int, seeds) -> dict:
    """Empfindlichkeit beim Kappen auf Schwellen aus den Daten (Winsorisieren mit k Werten je Seite), lognormale Daten: das Verhältnis der größten Änderung des Mittels durch
    Ersetzen eines Werts (13 Kandidaten, alle Positionen) zu (hi − lo)/n. Über 1 heißt: das Rauschen auf Basis von (hi − lo)/n wäre zu klein."""
    ratios, max_ratios, all_ratios, equal_gap = [], [], [], 0.0
    for seed in seeds:
        stream = Stream(cell_seed(seed, "m9", n, k))
        vals = []
        for _ in range(datasets):
            x = draw("lognormal", 1.0, stream, n)
            vals.append(P.replacement_sensitivity_ratio(x, k, WINSOR_CANDIDATES))
            equal_gap = max(equal_gap, abs(P.clip_at_data_bounds_mean(x, k) - R.scalar(R.winsorized_mean(x, k / n))))
        all_ratios.extend(vals)
        ratios.append(float(np.mean(vals)))
        max_ratios.append(float(np.max(vals)))
    return {"n": n, "k": k, "datasets_per_seed": datasets, "mean_ratio": mean_std(ratios), "max_ratio": mean_std(max_ratios), "min_ratio": float(min(all_ratios)), "largest_ratio": float(max(all_ratios)),
            "winsor_equals_clip_gap": equal_gap}
