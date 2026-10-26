"""Kappen und privates Mittel: Mittel gekappter Werte, Laplace-Mechanismus, geschlossene Fehlerzerlegung, Kappen = Winsorisieren.

Vereinbarung (die Demo prüft sie in thu_neighbors.py durch Aufzählung): n ist öffentlich. Zwei Datensätze sind **benachbart**, wenn genau ein Wert ersetzt ist. Ein Mechanismus M
ist ε-differentiell privat, wenn für alle benachbarten D, D′ und alle Ausgaben y gilt: log f(y | D) − log f(y | D′) ≤ ε (f ist die Dichte der Ausgabe). Das Kappungsintervall
[lo, hi] ist öffentlich, die Empfindlichkeit des Mittels gekappter Werte ist (hi − lo)/n, das Laplace-Rauschen hat die Skala b = (hi − lo)/(n·ε).
Das ist Mathematik auf kleinen Gittern und simulierten Daten, kein Sicherheitsnachweis: Gleitkommazahlen und Zufallszahlen sind hier nicht gehärtet."""

from __future__ import annotations

import math

import numpy as np

_erfc = np.frompyfunc(math.erfc, 1, 1)
C_GRID_LOG = (0.3, 1e7, 700)        # Gitter für die Suche nach C*: von 0,3 bis 10⁷, 700 Punkte (logarithmisch)


def phi(z) -> np.ndarray:
    """Verteilungsfunktion der Standardnormalverteilung."""
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * np.asarray(_erfc(-z / math.sqrt(2.0)), dtype=np.float64)


def clip_mean(x, lo: float, hi: float):
    """Mittel der auf [lo, hi] gekappten Werte (Zeile für Zeile bei einem Stapel)."""
    return np.clip(np.asarray(x, dtype=np.float64), lo, hi).mean(axis=-1)


def sensitivity(lo: float, hi: float, n: int) -> float:
    """Größte Änderung des Mittels gekappter Werte, wenn ein Wert ersetzt wird: (hi − lo)/n. Bei [0, C] ist das C/n, bei [−C, C] ist es 2C/n."""
    return (hi - lo) / n


def laplace_scale(lo: float, hi: float, n: int, eps: float) -> float:
    """Skala b des Laplace-Rauschens: Empfindlichkeit durch ε_DP (muss positiv sein)."""
    if eps <= 0:
        raise ValueError("das Budget ε_DP muss positiv sein")
    return sensitivity(lo, hi, n) / eps


def release(x, lo: float, hi: float, eps: float, noise):
    """Veröffentlichtes Mittel: gekapptes Mittel plus b · Laplace(0; 1)-Rauschen (`noise`, vom Aufrufer gezogen, damit der Zufallsstrom sichtbar bleibt)."""
    n = np.asarray(x).shape[-1]
    return clip_mean(x, lo, hi) + laplace_scale(lo, hi, n, eps) * np.asarray(noise, dtype=np.float64)


# --- Geschlossene Fehlerzerlegung für Lognormal und Pareto ---------------------------------------------------------------------------------
# Spezifikation: ("lognormal", σ) mit Median 1, oder ("pareto", α) mit x_min = 1 und α ≠ 2; Kappung auf [0, C].

def pop_mean(spec: tuple[str, float]) -> float:
    kind, p = spec
    if kind == "lognormal":
        return math.exp(p * p / 2.0)
    if kind == "pareto":
        if p <= 1.0:
            raise ValueError("Pareto braucht α > 1 für einen endlichen Mittelwert")
        return p / (p - 1.0)
    raise ValueError(f"unbekannte Verteilung: {kind}")


def clip_moments(spec: tuple[str, float], c) -> tuple[np.ndarray, np.ndarray]:
    """Exakte E[min(X, C)] und E[min(X, C)²] (Vektor C)."""
    kind, p = spec
    c = np.atleast_1d(np.asarray(c, dtype=np.float64))
    if kind == "lognormal":
        z = np.log(c) / p
        m1 = math.exp(p * p / 2.0) * phi(z - p) + c * (1.0 - phi(z))
        m2 = math.exp(2.0 * p * p) * phi(z - 2.0 * p) + c * c * (1.0 - phi(z))
        return m1, m2
    if kind == "pareto":
        if abs(p - 2.0) < 1e-12:
            raise ValueError("α = 2 ist nicht vorgesehen (Formel hat dort einen Logarithmus)")
        cc = np.maximum(c, 1.0)
        m1 = p / (p - 1.0) - cc ** (1.0 - p) / (p - 1.0)
        m2 = p / (2.0 - p) * (cc ** (2.0 - p) - 1.0) + cc ** (2.0 - p)
        below = c < 1.0                       # unter dem kleinsten Wert 1 ist jeder Wert gekappt
        return np.where(below, c, m1), np.where(below, c * c, m2)
    raise ValueError(f"unbekannte Verteilung: {kind}")


def exact_mse(spec: tuple[str, float], c, n: int, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(MSE, Verzerrung², Stichprobenvarianz, Rauschen) des privaten Mittels gegen das Populationsmittel, Kappung auf [0, C]:
    MSE = (E[min(X, C)] − μ)² + Var[min(X, C)]/n + 2·b², b = C/(n·ε) (die Varianz des Laplace-Rauschens ist 2·b²)."""
    c = np.atleast_1d(np.asarray(c, dtype=np.float64))
    m1, m2 = clip_moments(spec, c)
    bias2 = (m1 - pop_mean(spec)) ** 2
    samp = (m2 - m1 ** 2) / n
    noise = 2.0 * (c / (n * eps)) ** 2
    return bias2 + samp + noise, bias2, samp, noise


def c_star(spec: tuple[str, float], n: int, eps: float) -> float:
    """Die Kappungsgrenze mit dem kleinsten MSE: Suche auf einem logarithmischen Gitter, dann parabolische Verfeinerung in ln C."""
    lo, hi, pts = C_GRID_LOG
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), pts))
    mse = exact_mse(spec, grid, n, eps)[0]
    k = int(np.argmin(mse))
    if 0 < k < pts - 1:
        a, b, _ = np.polyfit(np.log(grid[k - 1:k + 2]), mse[k - 1:k + 2], 2)
        if a > 0:
            return float(math.exp(-b / (2.0 * a)))
    return float(grid[k])


# --- Kappen ist Winsorisieren --------------------------------------------------------------------------------------------------------------

def winsor_bounds(x, k: int) -> tuple[float, float]:
    """Die Schwellen des Winsorisierens mit k Werten je Seite: der (k+1)-kleinste und der (k+1)-größte Wert."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    return float(xs[k]), float(xs[len(xs) - 1 - k])


def clip_at_data_bounds_mean(x, k: int) -> float:
    """Mittel nach Kappen auf die Schwellen aus den Daten: dasselbe wie das winsorisierte Mittel."""
    lo, hi = winsor_bounds(x, k)
    return float(clip_mean(x, lo, hi))


def replacement_sensitivity_ratio(x, k: int, candidates) -> float:
    """Größte Änderung des winsorisierten Mittels (Schwellen aus den Daten), wenn EIN Wert durch einen Kandidaten ersetzt wird, geteilt durch (hi − lo)/n mit den
    Schwellen des ursprünglichen Datensatzes. Ein Wert über 1 heißt: Rauschen auf Basis von (hi − lo)/n unterschätzt die Empfindlichkeit."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    lo, hi = winsor_bounds(x, k)
    base = clip_at_data_bounds_mean(x, k)
    worst = 0.0
    for i in range(n):
        for v in candidates:
            if v == x[i]:
                continue
            y = x.copy()
            y[i] = v
            worst = max(worst, abs(clip_at_data_bounds_mean(y, k) - base))
    if hi == lo:                      # alle Schwellen gleich: (hi − lo)/n ist 0, jede Änderung ist dann unendlich viel größer
        return 0.0 if worst == 0.0 else math.inf
    return worst / ((hi - lo) / n)
