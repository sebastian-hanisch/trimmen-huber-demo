"""Vorgerechnete Messreihen (data/thu_results.json, erzeugt von tools/make_results.py aus thu_experiments.py): Laden und Abfragen.

Die Datei trägt die Prüfsumme der Quelldateien; `is_current()` vergleicht sie mit dem heutigen Stand (tests/test_results.py).
Jeder Wert einer Zelle ist [Mittelwert, Standardabweichung über die Seeds]; die Abfragen geben meist nur den Mittelwert zurück."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import thu_constants as C

PATH = Path(os.environ.get("THU_RESULTS_PATH") or Path(__file__).resolve().parent / "data" / C.RESULTS_FILE)


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def meta() -> dict:
    return load()["meta"]


def is_current() -> bool:
    import thu_experiments as X
    return meta()["erzeuger_hash"] == X.source_hash()


def _row(cell: dict, spec: str) -> dict:
    (row,) = [r for r in cell["rows"] if r["spec"] == spec]
    return row


# --- Effizienz und Verzerrung (E1) ---------------------------------------------------------------------------------------------------------

def efficiency_cell(n: int) -> dict:
    return load()[f"m1_n{n}"]


def efficiency(n: int, spec: str) -> float:
    return _row(efficiency_cell(n), spec)["eff"][0]


def efficiency_shift(n: int, spec: str, j: int = 0) -> float:
    """Verschiebung bei der j-ten Kontamination der Zelle (0: ε = 10 %, d = 5; 1: ε = 20 %, d = 5)."""
    return _row(efficiency_cell(n), spec)["shifts"][j][0]


def efficiency_std(n: int, spec: str) -> float:
    return _row(efficiency_cell(n), spec)["eff"][1]


# --- Verschiebung über dem Ausfallanteil (E2) und dem Abstand (E3) ------------------------------------------------------------------------

def shift_cell(n: int) -> dict:
    return load()[f"m2_n{n}"]


def shift_series(cell: dict, spec: str, d: float, key: str = "shift") -> list[float]:
    """Mittelwerte über die Anteile ε der Zelle bei Abstand d."""
    row = _row(cell, spec)
    return [v[0] for v in row[key][cell["d"].index(d)]]


def shift_value(n: int, spec: str, eps: float, d: float) -> float:
    cell = shift_cell(n)
    return shift_series(cell, spec, d)[cell["eps"].index(eps)]


def shift_std_max(cell: dict, key: str = "shift") -> float:
    return max(v[1] for r in cell["rows"] for row in r[key] for v in row)


def fine_cell(n: int) -> dict:
    return load()["m2fine"][f"n{n}"]


def fine_value(n: int, spec: str, eps: float, d: float = 1000.0) -> float:
    cell = fine_cell(n)
    return shift_series(cell, spec, d)[cell["eps"].index(eps)]


def distance_cell() -> dict:
    return load()["m2b"]


def distance_series(spec: str, eps: float, key: str = "shift") -> list[float]:
    """Verschiebung über den Abständen d (n = 100) bei Ausfallanteil ε."""
    cell = distance_cell()
    row = _row(cell, spec)
    j = cell["eps"].index(eps)
    return [row[key][a][j][0] for a in range(len(cell["d"]))]


def two_sided_cell() -> dict:
    return load()["m2c"]


def two_sided_value(spec: str, eps: float, key: str = "rmse") -> float:
    cell = two_sided_cell()
    return _row(cell, spec)[key][0][cell["eps"].index(eps)][0]


# --- Ränder (E5) und Zielgröße (E4) --------------------------------------------------------------------------------------------------------

def tail_cells() -> list[dict]:
    return load()["m3"]


def tail_ratio(dist: str, n: int, spec: str) -> float:
    (cell,) = [c for c in tail_cells() if c["dist"] == dist and c["n"] == n]
    return _row(cell, spec)["ratio"][0]


def skew_part(index: int) -> dict:
    return load()[f"m4_{index}"]


def skew_part_of(dist: str) -> dict:
    (part,) = [skew_part(i) for i in range(len(C.SKEW_DISTS)) if skew_part(i)["dist"] == dist]
    return part


def population_location(dist: str, spec: str) -> float:
    return skew_part_of(dist)["population"][spec][0]


def target_cell(dist: str, n: int) -> dict:
    (cell,) = [c for c in skew_part_of(dist)["cells"] if c["n"] == n]
    return cell


def target_value(dist: str, n: int, spec: str, key: str) -> float:
    """key: sd, bias, rmse_mean, rmse_median."""
    return _row(target_cell(dist, n), spec)[key][0]


# --- Catoni (E7) ----------------------------------------------------------------------------------------------------------------------------

def catoni_part() -> dict:
    return load()["m5"]


def catoni_efficiency(spec: str) -> float:
    return _row(catoni_part()["efficiency"], spec)["eff"][0]


def catoni_shift(spec: str, j: int) -> float:
    """j: 0 → ε = 10 %, d = 5; 1 → ε = 10 %, d = 1000; 2 → ε = 20 %, d = 1000 (n = 100)."""
    return _row(catoni_part()["efficiency"], spec)["shifts"][j][0]


def catoni_distance_cell(n: int) -> dict:
    (cell,) = [c for c in catoni_part()["distance"] if c["n"] == n]
    return cell


def catoni_distance_series(n: int, spec: str) -> list[float]:
    cell = catoni_distance_cell(n)
    return [row[0][0] for row in _row(cell, spec)["shift"]]          # ein Anteil (ε = 10 %), verschachtelt [d][ε]


def catoni_t3_ratio(spec: str) -> float:
    return _row(catoni_part()["t3"], spec)["ratio"][0]


# --- Fallen (E6) -----------------------------------------------------------------------------------------------------------------------------

def pitfalls() -> dict:
    return load()["m6"]


def cut(n: int, alpha: float) -> int:
    (row,) = [r for r in pitfalls()["stairs"] if r["n"] == n and abs(r["alpha"] - alpha) < 1e-9]
    return row["cut"]


def zero_scale_cell(kind: str, n: int) -> dict:
    (cell,) = [c for c in pitfalls()["zero_scale"] if c["kind"] == kind and c["n"] == n]
    return cell


def scale_known_row(which: str) -> dict:
    return _row(pitfalls()["scale_known"][which], "huber:1.345")


def iterations(c: float) -> float:
    (row,) = [r for r in pitfalls()["iterations"] if r["c"] == c]
    return row["max_iterations"][0]


# --- Kappen und privates Mittel (E8) --------------------------------------------------------------------------------------------------------

def ucurve_cells(index: int) -> list[dict]:
    return load()[f"m7_{index}"]


def ucurve_cell(index: int, n: int) -> dict:
    (cell,) = [c for c in ucurve_cells(index) if c["n"] == n]
    return cell


def ucurve_eps(index: int, n: int, eps: float) -> dict:
    (row,) = [r for r in ucurve_cell(index, n)["eps"] if r["eps"] == eps]
    return row


def scaling_rows() -> list[dict]:
    return load()["m8"]["scaling"]


def scaling_row(spec: tuple, n: int, eps: float) -> dict:
    (row,) = [r for r in scaling_rows() if tuple(r["spec"]) == tuple(spec) and r["n"] == n and r["eps"] == eps]
    return row


def wrong_c_rows() -> list[dict]:
    return load()["m8"]["wrong_c"]


def wrong_c(spec: tuple, eps: float) -> dict:
    (row,) = [r for r in wrong_c_rows() if tuple(r["spec"]) == tuple(spec) and r["eps"] == eps]
    return row


def enumeration() -> dict:
    return load()["m9"]["enumeration"]


def winsor_sensitivity() -> dict:
    return load()["m9"]["winsor"]
