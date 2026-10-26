"""Die Definition durch Aufzählung prüfen: alle benachbarten Datensätze (genau ein Wert ersetzt) auf einem kleinen Gitter.

Für einen Mechanismus mit Laplace-Rauschen um einen Wert m(D) und fester Skala b ist das größte log-Verhältnis über alle Ausgaben y gleich |m(D) − m(D′)|/b. Hängt die Skala
selbst von den Daten ab (b(D) ≠ b(D′)), steht an der Stelle y die Differenz log(b′/b) + |y − m′|/b′ − |y − m|/b. Die Aufzählung ist exakt für das kleine Gitter und
belegt nichts über Gleitkommazahlen oder Zufallszahlengeneratoren."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import thu_private as P


def neighbors(data: tuple, domain) -> Iterator[tuple]:
    """Alle Datensätze, die sich von `data` in genau einem Wert unterscheiden (Werte aus `domain`)."""
    for i in range(len(data)):
        for v in domain:
            if v != data[i]:
                other = list(data)
                other[i] = v
                yield tuple(other)


def _clip_mean(values, lo: float, hi: float) -> float:
    return float(P.clip_mean(values, lo, hi))


def max_log_ratio_laplace(n: int, domain, lo: float, hi: float, eps: float, variant: str = "korrekt") -> tuple[float, int]:
    """(größtes log-Verhältnis, Zahl der Paare) für das Mittel gekappter Werte mit Laplace-Rauschen.

    variant: "korrekt" (Skala (hi − lo)/(n·ε)), "empfindlichkeit_n_plus_1" (Skala (hi − lo)/((n + 1)·ε), ein absichtlicher Fehler),
    "ohne_kappung" (Skala wie korrekt, aber das Mittel der ungekappten Werte, ein absichtlicher Fehler)."""
    if variant not in ("korrekt", "empfindlichkeit_n_plus_1", "ohne_kappung"):
        raise ValueError(f"unbekannte Variante: {variant}")
    b = P.laplace_scale(lo, hi, n + 1 if variant == "empfindlichkeit_n_plus_1" else n, eps)

    def mean_of(values):
        return float(sum(values)) / len(values) if variant == "ohne_kappung" else _clip_mean(values, lo, hi)

    worst, pairs = 0.0, 0
    for data in itertools.product(domain, repeat=n):
        m = mean_of(data)
        for other in neighbors(data, domain):
            worst = max(worst, abs(m - mean_of(other)) / b)
            pairs += 1
    return worst, pairs


def max_log_ratio_data_dependent_c(n: int, domain, eps: float, y: float, rank: int = 2) -> tuple[float, tuple, tuple]:
    """Das Gegenbeispiel: C wird aus den Daten gewählt (der Wert mit Rang `rank` in der Sortierung, ab 0), gekappt wird auf [0, C], das Rauschen hat die Skala C/(n·ε),
    und nur y wird veröffentlicht (C nicht). Liefert (größtes log-Verhältnis an der Stelle y, D, D′) über alle benachbarten Paare."""
    worst, best_pair = -math.inf, ((), ())
    for data in itertools.product(domain, repeat=n):
        c = sorted(data)[rank]
        m, b = _clip_mean(data, 0.0, c), P.laplace_scale(0.0, c, n, eps)
        for other in neighbors(data, domain):
            c2 = sorted(other)[rank]
            m2, b2 = _clip_mean(other, 0.0, c2), P.laplace_scale(0.0, c2, n, eps)
            ratio = math.log(b2 / b) + abs(y - m2) / b2 - abs(y - m) / b
            if ratio > worst:
                worst, best_pair = ratio, (data, other)
    return worst, best_pair[0], best_pair[1]
