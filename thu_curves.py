"""Messreihen der Schätzer: Effizienz gegen Verzerrung, Verschiebung über dem Ausfallanteil und dem Abstand, Zielgröße bei Schiefe, Fallen. Reine Funktionen über feste
Zufalls-Seeds; sie erzeugen sowohl die vorgerechneten Tabellen (tools/make_results.py) als auch die Live-Knöpfe der App.

Jede Zelle läuft über mehrere Seeds (thu_constants.SEEDS); angegeben wird [Mittelwert, Standardabweichung über die Seeds]. Alle Schätzer einer Zelle sehen dieselben Stichproben
(gepaart), und alle Anteile und Abstände einer Zelle teilen sich dieselben Zufallszahlen (`Contamination`), die Kurven über ε und d sind also glatt.
Verschiebung = Mittelwert des Schätzers T über die Läufe (Ziel 0), Verzerrung ist hier dasselbe Wort."""

from __future__ import annotations

import math

import numpy as np

import thu_robust as R
from thu_scenario import DIST_INFO, Contamination, Stream, base_batch, cell_seed, draw, n_outliers, zero_inflated_batch


def mean_std(values) -> list[float]:
    v = np.asarray(values, dtype=np.float64)
    return [float(v.mean()), float(v.std())]


def _estimates(specs, x, scale, delta) -> dict[str, np.ndarray]:
    return {s: R.estimate(s, x, scale, delta) for s in specs}


# ---------------------------------------------------------------- Effizienz gegen Verzerrung

def efficiency_bias_cell(specs, n: int, runs: int, seeds, contamination=((0.10, 5.0), (0.20, 5.0)), scale=None, delta: float = 0.05) -> dict:
    """Je Schätzer: Effizienz auf sauberen Normaldaten (MSE des Mittels derselben Stichproben geteilt durch MSE des Schätzers) und die Verschiebung bei den
    angegebenen (ε, d)-Kontaminationen (einseitig)."""
    eff = {s: [] for s in specs}
    shifts = {s: [[] for _ in contamination] for s in specs}
    for seed in seeds:
        src = Contamination("Normal", runs, n, Stream(cell_seed(seed, "m1", n)))
        clean = src.at(0, 0.0)
        mse_mean = float(np.mean(clean.mean(axis=1) ** 2))
        for s, t in _estimates(specs, clean, scale, delta).items():
            eff[s].append(mse_mean / float(np.mean(t ** 2)))
        for j, (e, d) in enumerate(contamination):
            for s, t in _estimates(specs, src.at(n_outliers(n, e), d), scale, delta).items():
                shifts[s][j].append(float(t.mean()))
    return {"n": n, "runs_per_seed": runs, "contamination": [list(c) for c in contamination],
            "rows": [{"spec": s, "eff": mean_std(eff[s]), "shifts": [mean_std(v) for v in shifts[s]]} for s in specs]}


# ---------------------------------------------------------------- Verschiebung über Anteil und Abstand

def shift_cell(specs, n: int, eps_list, d_list, runs: int, seeds, two_sided: bool = False, dist: str = "Normal", scale=None, delta: float = 0.05) -> dict:
    """Verschiebung und RMSE um 0 für jeden Schätzer, jeden Abstand d und jeden Anteil ε (verschachtelt [d][ε] → [Mittel, Std über Seeds])."""
    shift = {s: [[[] for _ in eps_list] for _ in d_list] for s in specs}
    rmse = {s: [[[] for _ in eps_list] for _ in d_list] for s in specs}
    for seed in seeds:
        src = Contamination(dist, runs, n, Stream(cell_seed(seed, "m2", n, dist, int(two_sided))))
        for a, d in enumerate(d_list):
            for b, e in enumerate(eps_list):
                for s, t in _estimates(specs, src.at(n_outliers(n, e), d, two_sided), scale, delta).items():
                    shift[s][a][b].append(float(t.mean()))
                    rmse[s][a][b].append(float(math.sqrt(np.mean(t ** 2))))
    return {"n": n, "runs_per_seed": runs, "eps": list(eps_list), "d": list(d_list), "two_sided": two_sided, "dist": dist,
            "rows": [{"spec": s, "shift": [[mean_std(v) for v in row] for row in shift[s]], "rmse": [[mean_std(v) for v in row] for row in rmse[s]]} for s in specs]}


# ---------------------------------------------------------------- Zielgröße bei Schiefe und schweren Rändern

def population_location(dist: str, specs, size: int, seeds) -> dict[str, list[float]]:
    """Lage jedes Schätzers an der Population (Näherung): der Schätzer auf einer sehr großen Stichprobe, über die Seeds gemittelt. Mittel und Median kommen aus der Formel;
    Hodges–Lehmann als Median der Mittel unabhängiger Paare (halbe Stichprobe gegen halbe Stichprobe)."""
    mean_exact, median_exact = DIST_INFO[dist][2], DIST_INFO[dist][3]
    out = {}
    for s in specs:
        if s == "mean":
            out[s] = [mean_exact, 0.0]
            continue
        if s == "median":
            out[s] = [median_exact, 0.0]
            continue
        vals = []
        for seed in seeds:
            big = base_batch(dist, 1, size, Stream(cell_seed(seed, "pop", dist)))
            if s == "hl":
                half = size // 2
                vals.append(float(np.median((big[0, :half] + big[0, half:2 * half]) / 2.0)))
            else:
                vals.append(R.scalar(R.estimate(s, big)))
        out[s] = mean_std(vals)
    return out


def target_cell(dist: str, n: int, specs, runs: int, seeds, delta: float = 0.05) -> dict:
    """Je Schätzer auf sauberen Daten der Verteilung: Streuung SD(T), Bias gegen den Mittelwert, RMSE gegen den Mittelwert und gegen den Median der Verteilung."""
    mu, med = DIST_INFO[dist][2], DIST_INFO[dist][3]
    sd = {s: [] for s in specs}
    bias = {s: [] for s in specs}
    rmse_mu = {s: [] for s in specs}
    rmse_med = {s: [] for s in specs}
    for seed in seeds:
        x = base_batch(dist, runs, n, Stream(cell_seed(seed, "m4", dist, n)))
        for s, t in _estimates(specs, x, None, delta).items():
            sd[s].append(float(t.std()))
            bias[s].append(float(t.mean() - mu))
            rmse_mu[s].append(float(math.sqrt(np.mean((t - mu) ** 2))))
            rmse_med[s].append(float(math.sqrt(np.mean((t - med) ** 2))))
    return {"dist": dist, "n": n, "runs_per_seed": runs, "mean": mu, "median": med,
            "rows": [{"spec": s, "sd": mean_std(sd[s]), "bias": mean_std(bias[s]), "rmse_mean": mean_std(rmse_mu[s]), "rmse_median": mean_std(rmse_med[s])} for s in specs]}


def tail_cell(dist: str, n: int, specs, runs: int, seeds, delta: float = 0.05) -> dict:
    """RMSE jedes Schätzers um 0 auf sauberen Daten einer symmetrischen Verteilung, geteilt durch den RMSE des Mittels (gepaart). Unter 1: besser als das Mittel."""
    ratio = {s: [] for s in specs}
    for seed in seeds:
        x = base_batch(dist, runs, n, Stream(cell_seed(seed, "m3", dist, n)))
        rm_mean = float(math.sqrt(np.mean(x.mean(axis=1) ** 2)))
        for s, t in _estimates(specs, x, None, delta).items():
            ratio[s].append(float(math.sqrt(np.mean(t ** 2))) / rm_mean)
    return {"dist": dist, "n": n, "runs_per_seed": runs, "rows": [{"spec": s, "ratio": mean_std(ratio[s])} for s in specs]}


# ---------------------------------------------------------------- Fallen

def cut_table(ns, alphas) -> list[dict]:
    """Die tatsächliche Abschneidezahl ⌊α·n⌋ je Seite für ein Raster (die Treppe)."""
    return [{"n": n, "alpha": a, "cut": R.n_cut(n, a)} for n in ns for a in alphas]


def float_edge_pairs() -> dict:
    """Wie oft weicht `int(α·n)` von ⌊α·n + 10⁻⁹⌋ ab? Raster α = k/100 (k = 1 … 45), n = 5 … 1000. Dazu die Beispielpaare (α, n, int(α·n), eigene Zahl)."""
    total, bad, examples = 0, 0, []
    for k in range(1, 46):
        a = k / 100.0
        for n in range(5, 1001):
            total += 1
            plain, own = int(a * n), int(math.floor(a * n + R.CUT_TOLERANCE))
            if plain != own:
                bad += 1
                if len(examples) < 4:
                    examples.append([a, n, plain, own])
    return {"total": total, "differ": bad, "examples": examples}


def zero_scale_cell(kind: str, n: int, runs: int, seeds) -> dict:
    """Diskrete Daten: Anteil der Stichproben mit MAD = 0 und die Lage der Schätzer (Mittel über die Läufe). kind: "poisson:<λ>" oder "zip:<Nullanteil>:<λ>"."""
    parts = kind.split(":")
    specs = ("mean", "median", "huber:1.345", "hl", "trim:0.1")
    zero_share, locs = [], {s: [] for s in specs}
    for seed in seeds:
        stream = Stream(cell_seed(seed, "m6", kind, n))
        if parts[0] == "poisson":
            x = draw("poisson", float(parts[1]), stream, runs * n).reshape(runs, n)
        else:
            x = zero_inflated_batch(runs, n, float(parts[1]), float(parts[2]), stream)
        zero_share.append(float(np.mean(R.mad_scale(x) <= 0)))
        for s, t in _estimates(specs, x, None, 0.05).items():
            locs[s].append(float(t.mean()))
    return {"kind": kind, "n": n, "runs_per_seed": runs, "zero_scale_share": mean_std(zero_share), "location": {s: mean_std(v) for s, v in locs.items()}}


def huber_iterations(cs, n: int, runs: int, seeds) -> list[dict]:
    """Größte Zahl der Schritte der Huber-Iteration im Stapel (sauberes Normal), je c, über die Seeds gemittelt."""
    out = []
    for c in cs:
        used = []
        for seed in seeds:
            x = base_batch("Normal", runs, n, Stream(cell_seed(seed, "m6it", n)))
            used.append(R.huber_location(x, c, None, return_iterations=True)[1])
        out.append({"c": c, "max_iterations": mean_std(used)})
    return out
