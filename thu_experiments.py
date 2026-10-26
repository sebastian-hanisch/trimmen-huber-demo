"""Die vorgerechneten Messreihen der Experimente: Zusammenstellung der Zellen aus thu_curves.py und thu_kappen.py (tools/make_results.py → data/thu_results.json).

Jeder Teil (`PARTS[name]()`) läuft für sich und liefert ein JSON-fähiges Dict; mehrere Teile können parallel laufen. Läufe je Seed stehen in der Zelle (`runs_per_seed`),
die Seeds in thu_constants.SEEDS."""

from __future__ import annotations

import hashlib
from pathlib import Path

import thu_constants as C
import thu_curves as K
import thu_kappen as KP

ROOT = Path(__file__).resolve().parent
SOURCES = ("thu_rng.py", "thu_scenario.py", "thu_robust.py", "thu_private.py", "thu_neighbors.py", "thu_curves.py", "thu_kappen.py", "thu_experiments.py", "tools/make_results.py")

M1_RUNS = {10: 4000, 30: 4000, 100: 2000, 300: 1000}
M2_RUNS = {20: 1000, 100: 500, 300: 250}
M2_N = tuple(M2_RUNS)
FINE_SPECS = ("median", "trim:0.1", "trim:0.25", "winsor:0.25", "hl")
TAIL_RUNS = {10: 4000, 30: 4000, 100: 3000, 300: 1000}
STAIR_NS = (10, 20, 30, 50, 100)
STAIR_ALPHAS = tuple(round(0.05 * k, 2) for k in range(1, 10))
ZERO_KINDS = ("poisson:0.5", "zip:0.6:5", "poisson:1", "poisson:3")
POPULATION_SIZE = 1_000_000
KAPPEN_RUNS = {100: 4000, 1000: 4000, 10000: 1000}
CATONI_SPECS = tuple(f"catoni:{m:g}" for m in C.CATONI_MULTS)


def source_hash() -> str:
    """Prüfsumme der Quelldateien der Messreihen (Zeilenenden normalisiert): stimmt sie nicht mit der in der JSON-Datei überein, ist die Datei veraltet."""
    h = hashlib.sha256()
    for name in SOURCES:
        text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(name.encode("utf-8") + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


def skew_runs(dist: str, n: int) -> int:
    """Läufe je Seed für die Zielzellen: bei Lognormal(σ = 2) doppelt so viele (der Mittelwert dort ist selbst stark verrauscht)."""
    base = 2000 if n <= 100 else 1000
    return 2 * base if "σ = 2" in dist else base


def part_m1(n: int):
    return lambda: K.efficiency_bias_cell(C.FAMILY_SPECS, n, M1_RUNS[n], C.SEEDS)


def part_m2(n: int):
    return lambda: K.shift_cell(C.CURVE_SPECS, n, C.EPS_GRID, C.D_GRID, M2_RUNS[n], C.SEEDS)


def part_m2fine():
    return {"n100": K.shift_cell(FINE_SPECS, 100, C.EPS_FINE, (1000.0,), 300, C.SEEDS),
            "n300": K.shift_cell(("hl", "trim:0.25"), 300, C.EPS_FINE, (1000.0,), 100, C.SEEDS)}


def part_m2b():
    return K.shift_cell(C.CURVE_SPECS, 100, (0.1, 0.2, 0.3), C.D_CURVE, 500, C.SEEDS)


def part_m2c():
    return K.shift_cell(C.CURVE_SPECS, 100, (0.1, 0.2), (10.0,), 1000, C.SEEDS, two_sided=True)


def part_m3():
    return [K.tail_cell(dist, n, C.CURVE_SPECS, TAIL_RUNS[n], C.SEEDS) for dist in C.TAIL_DISTS for n in C.N_RASTER]


def part_m4(dist: str):
    return lambda: {"dist": dist, "population": K.population_location(dist, C.TARGET_SPECS, POPULATION_SIZE, C.SEEDS),
                    "cells": [K.target_cell(dist, n, C.TARGET_SPECS, skew_runs(dist, n), C.SEEDS) for n in C.N_RASTER]}


def part_m5():
    specs = ("mean", "median", "huber:1.345") + CATONI_SPECS
    return {"efficiency": K.efficiency_bias_cell(specs, 100, 2000, C.SEEDS, contamination=((0.1, 5.0), (0.1, 1000.0), (0.2, 1000.0))),
            "distance": [K.shift_cell(("huber:1.345", "catoni:1", "catoni:8"), n, (0.1,), C.D_CURVE + (1e9,), 500 if n < 300 else 250, C.SEEDS) for n in (20, 100, 300)],
            "t3": K.tail_cell("t(3)", 100, ("huber:1.345",) + CATONI_SPECS, 3000, C.SEEDS)}


def part_m6():
    return {"stairs": K.cut_table(STAIR_NS, STAIR_ALPHAS), "float_edge": K.float_edge_pairs(),
            "zero_scale": [K.zero_scale_cell(kind, n, 1000, C.SEEDS) for kind in ZERO_KINDS for n in (30, 100)],
            "scale_known": {"mad": K.efficiency_bias_cell(("huber:1.345",), 100, 2000, C.SEEDS, contamination=((0.2, 5.0), (0.3, 1000.0)), scale=None),
                            "known": K.efficiency_bias_cell(("huber:1.345",), 100, 2000, C.SEEDS, contamination=((0.2, 5.0), (0.3, 1000.0)), scale=1.0)},
            "iterations": K.huber_iterations((0.5, 1.345, 3.0), 100, 2000, C.SEEDS)}


def part_m7(index: int):
    spec = C.KAPPEN_SPECS[index]
    return lambda: [KP.ucurve_cell(spec, n, KAPPEN_RUNS[n], C.SEEDS, C.KAPPEN_EPS, KP.c_grid()) for n in C.KAPPEN_N]


def part_m8():
    return {"scaling": KP.scaling_table(C.KAPPEN_SCALING_SPECS, C.KAPPEN_NS_SCALING, C.KAPPEN_EPS),
            "wrong_c": [KP.wrong_c_costs(spec, 1000, eps) for spec in C.KAPPEN_SPECS for eps in (1.0, 0.1)]}


def part_m9():
    return {"enumeration": KP.enumeration_summary(), "winsor": KP.winsor_sensitivity_cell(100, 1, 10, C.SEEDS)}


PARTS = {
    **{f"m1_n{n}": part_m1(n) for n in C.N_RASTER},
    **{f"m2_n{n}": part_m2(n) for n in M2_N},
    "m2fine": part_m2fine, "m2b": part_m2b, "m2c": part_m2c, "m3": part_m3,
    **{f"m4_{i}": part_m4(dist) for i, dist in enumerate(C.SKEW_DISTS)},
    "m5": part_m5, "m6": part_m6,
    **{f"m7_{i}": part_m7(i) for i in range(len(C.KAPPEN_SPECS))},
    "m8": part_m8, "m9": part_m9,
}
