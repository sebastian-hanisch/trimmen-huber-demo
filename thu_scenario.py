"""Szenario: Lieferzeit-Abweichungen (Stunden gegenüber Plan) einer Sendung, mit wählbarer Verteilung und Lieferausfällen als Ausreißern.

Alle Zufallszahlen stammen aus dem SplitMix64-Strom von `thu_rng.py` (wörtliche Kopie); nichts aus `numpy.random`, damit jede Reihe auf jeder Plattform und in jeder
Python-Version gleich ist. SplitMix64 ist ein Zähler-Generator: der k-te Wert des Stroms hängt nur vom Seed und von k ab, ein ganzer Block lässt sich also auf einmal mit
numpy berechnen (`mix_words`); tests/test_scenario.py prüft ihn Wert für Wert gegen die Python-Klasse. Normalwerte nach Box-Muller; log, cos und sin kommen aus numpy,
die letzte Binärstelle kann je Plattform abweichen.

Einheit: Stunden Abweichung vom Plan; der Kern ist auf Streuung 1 skaliert. Ein Ausfall ist eine Sendung, deren Wert durch d + N(0, 1) ersetzt wird."""

from __future__ import annotations

import math
import zlib

import numpy as np

from thu_rng import MASK, SplitMix64  # noqa: F401  (SplitMix64: Gegenprobe in den Tests)

GAMMA = 0x9E3779B97F4A7C15
_C1 = 0xBF58476D1CE4E5B9
_C2 = 0x94D049BB133111EB
TWO_M53 = 2.0 ** -53

# Verteilungen der Sendungen (Kern): Name → (Art, Parameter, Mittelwert, Median, symmetrisch um 0)
DIST_INFO = {
    "Normal": ("normal", 0.0, 0.0, 0.0, True),
    "Laplace": ("laplace", 0.0, 0.0, 0.0, True),
    "t(5)": ("t", 5.0, 0.0, 0.0, True),
    "t(3)": ("t", 3.0, 0.0, 0.0, True),
    "Lognormal (σ = 0,5)": ("lognormal", 0.5, math.exp(0.5 * 0.5 / 2.0), 1.0, False),
    "Lognormal (σ = 1)": ("lognormal", 1.0, math.exp(0.5), 1.0, False),
    "Lognormal (σ = 2)": ("lognormal", 2.0, math.exp(2.0), 1.0, False),
    "Exponential": ("exponential", 1.0, 1.0, math.log(2.0), False),
    "Diskrete Tage (Poisson 0,5)": ("poisson", 0.5, 0.5, 0.0, False),
}
DISTRIBUTIONS = tuple(DIST_INFO)
SYMMETRIC = tuple(name for name, info in DIST_INFO.items() if info[4])


def mix_words(seed: int, start: int, count: int) -> np.ndarray:
    """Die Werte Nr. start … start + count − 1 (ab 0 gezählt) des SplitMix64-Stroms zum Seed, als uint64-Feld."""
    k = np.arange(start + 1, start + 1 + count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = k * np.uint64(GAMMA) + np.uint64(seed & MASK)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(_C1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(_C2)
        return z ^ (z >> np.uint64(31))


def cell_seed(base: int, *tags) -> int:
    """Stabiler Seed einer Zelle aus dem Basis-Seed und beliebig vielen Marken (Zahlen oder Texte)."""
    value = int(base)
    for tag in tags:
        part = zlib.crc32(tag.encode("utf-8")) if isinstance(tag, str) else int(round(float(tag) * 1000))
        value = (value * 1_000_003 + part) % (1 << 61)
    return value


class Stream:
    """Zufallsstrom mit Position: jeder Aufruf zieht die nächsten Werte des SplitMix64-Stroms zum Seed (als Block)."""

    def __init__(self, seed: int):
        self.seed = int(seed) & MASK
        self.pos = 0

    def words(self, count: int) -> np.ndarray:
        w = mix_words(self.seed, self.pos, count)
        self.pos += count
        return w

    def uniform(self, count: int) -> np.ndarray:
        """Gleichverteilt in [0, 1), 53 Bit."""
        return (self.words(count) >> np.uint64(11)).astype(np.float64) * TWO_M53

    def uniform_open(self, count: int) -> np.ndarray:
        """Gleichverteilt in (0, 1), nie 0 oder 1 (für Logarithmen und Umkehrfunktionen)."""
        return ((self.words(count) >> np.uint64(11)).astype(np.float64) + 0.5) * TWO_M53

    def normal(self, count: int) -> np.ndarray:
        """Standardnormal nach Box-Muller: aus zwei Strom-Werten (u1, u2) die zwei Werte r·cos(2π u2), r·sin(2π u2), r = √(−2 ln u1). Bei ungerader Zahl entfällt der letzte."""
        pairs = (count + 1) // 2
        w = self.words(2 * pairs)
        u1 = ((w[0::2] >> np.uint64(11)).astype(np.float64) + 0.5) * TWO_M53
        u2 = ((w[1::2] >> np.uint64(11)).astype(np.float64) + 0.5) * TWO_M53
        r = np.sqrt(-2.0 * np.log(u1))
        out = np.empty(2 * pairs)
        out[0::2] = r * np.cos(2.0 * math.pi * u2)
        out[1::2] = r * np.sin(2.0 * math.pi * u2)
        return out[:count]


def poisson_table(lam: float) -> np.ndarray:
    """Verteilungsfunktion F(0), F(1), … der Poisson-Verteilung bis weit hinter den Rand; letzter Eintrag exakt 1."""
    kmax = int(lam + 12.0 * math.sqrt(lam) + 40)
    pmf = np.empty(kmax + 1)
    pmf[0] = math.exp(-lam)
    for k in range(1, kmax + 1):
        pmf[k] = pmf[k - 1] * lam / k
    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0
    return cdf


def draw(kind: str, param: float, stream: Stream, count: int) -> np.ndarray:
    """`count` Werte in Standardform: normal(0; 1), laplace(0; 1), t mit `param` Freiheitsgraden, lognormal(0; param) mit Median 1, exponential(1), poisson(param), pareto(1; param)."""
    if kind == "normal":
        return stream.normal(count)
    if kind == "laplace":
        d = stream.uniform_open(count) - 0.5
        return -np.sign(d) * np.log(1.0 - 2.0 * np.abs(d))
    if kind == "t":
        nu = int(param)
        z = stream.normal(count * (nu + 1)).reshape(count, nu + 1)
        return z[:, 0] / np.sqrt((z[:, 1:] ** 2).sum(axis=1) / nu)
    if kind == "lognormal":
        return np.exp(param * stream.normal(count))
    if kind == "exponential":
        return -np.log(stream.uniform_open(count))
    if kind == "poisson":
        return np.searchsorted(poisson_table(param), stream.uniform(count), side="right").astype(np.float64)
    if kind == "pareto":
        return stream.uniform_open(count) ** (-1.0 / param)
    raise ValueError(f"unbekannte Verteilung: {kind}")


def base_batch(dist: str, runs: int, n: int, stream: Stream) -> np.ndarray:
    """Stapel (runs, n) von Sendungen der Verteilung `dist` (Name aus DIST_INFO)."""
    kind, param = DIST_INFO[dist][:2]
    return draw(kind, param, stream, runs * n).reshape(runs, n)


def n_outliers(n: int, eps: float) -> int:
    """Genau m = round(ε·n) Ausfälle (Rundung nach oben bei .5), nie mehr als n."""
    return min(n, max(0, int(math.floor(eps * n + 0.5))))


class Contamination:
    """Ein Satz Zufallszahlen für Ausfälle: der Kern (runs, n), je Sendung ein Normalwert z und eine Gleichverteilung u für das Vorzeichen. `at(m, d, zweiseitig)` ersetzt in jeder
    Zeile die ersten m Werte durch Ausfälle ±d + z. Derselbe Satz dient allen Anteilen und Abständen einer Zelle (gepaarte Vergleiche, glatte Kurven); die Reihenfolge der
    Spalten ist ohne Bedeutung, alle Schätzer hängen nur an der Menge der Werte."""

    def __init__(self, dist: str, runs: int, n: int, stream: Stream):
        self.base = base_batch(dist, runs, n, stream)
        self.z = stream.normal(runs * n).reshape(runs, n)
        self.u = stream.uniform(runs * n).reshape(runs, n)

    def at(self, m: int, d: float, two_sided: bool = False) -> np.ndarray:
        out = self.base.copy()
        if m > 0:
            sign = np.where(self.u[:, :m] < 0.5, -1.0, 1.0) if two_sided else 1.0
            out[:, :m] = sign * float(d) + self.z[:, :m]
        return out


def contaminated_batch(dist: str, runs: int, n: int, eps: float, d: float, two_sided: bool, stream: Stream) -> tuple[np.ndarray, int]:
    """(Stapel, m): Kern aus `dist`, m = round(ε·n) Ausfälle der Höhe d."""
    m = n_outliers(n, eps)
    return Contamination(dist, runs, n, stream).at(m, d, two_sided), m


def zero_inflated_batch(runs: int, n: int, zero_share: float, lam: float, stream: Stream) -> np.ndarray:
    """Diskrete Tage mit vielen Nullen: jeder Wert ist mit Wahrscheinlichkeit `zero_share` 0, sonst Poisson(lam)."""
    is_zero = stream.uniform(runs * n).reshape(runs, n) < zero_share
    counts = draw("poisson", lam, stream, runs * n).reshape(runs, n)
    return np.where(is_zero, 0.0, counts)


def one_sample(dist: str, n: int, eps: float, d: float, two_sided: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Eine Stichprobe (x, ist_ausfall): x der Länge n, ist_ausfall markiert die m Ausfälle. Die Reihenfolge ist durch einen festen Zufallsstrom gemischt."""
    stream = Stream(cell_seed(seed, "stichprobe", dist))
    x, m = contaminated_batch(dist, 1, n, eps, d, two_sided, stream)
    flag = np.zeros(n, dtype=bool)
    flag[:m] = True
    order = np.argsort(stream.uniform(n), kind="stable")
    return x[0][order], flag[order]


def population(dist: str) -> tuple[float, float]:
    """(Mittelwert, Median) des Kerns."""
    info = DIST_INFO[dist]
    return info[2], info[3]
