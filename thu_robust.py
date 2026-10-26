"""Die Lageschätzer: Mittel, Median, getrimmtes und winsorisiertes Mittel, Huber, Hodges–Lehmann, Catoni. Reiner numpy-Kern, jede Funktion eine Regel.

Alle Schätzer arbeiten zeilenweise auf einem Stapel `x` der Form (R, n) und liefern R Werte (ein Vektor der Länge n ist ein Stapel mit einer Zeile; `estimate` und die
Einzelfunktionen geben dann ein Feld der Länge 1 zurück, die Hilfen `scalar` nehmen das Element). Skala: MAD / 0,6745 (konsistent für die Standardabweichung der Normalverteilung).

Die Catoni-Funktion und die Wahl von α stehen hier nach der Definition, die der Autor aus dem Gedächtnis kennt; die Originalarbeit (Catoni 2012) wurde NICHT eingesehen."""

from __future__ import annotations

import math

import numpy as np

MAD_K = 0.6744897501960817          # Φ⁻¹(0,75): MAD / MAD_K schätzt die Standardabweichung bei Normalverteilung
CUT_TOLERANCE = 1e-9                # ⌊α·n + 1e-9⌋: fängt Gleitkomma-Kanten wie 0,29 · 100 = 28,999999999999996 ab
HUBER_MAXIT = 500
HUBER_TOL = 1e-10                   # Abbruch, wenn sich die Lage um weniger als diesen Bruchteil der Skala ändert
CATONI_BISECTIONS = 100
HL_BUDGET = 4_000_000               # so viele Paarmittel höchstens gleichzeitig im Speicher


def rows(x) -> np.ndarray:
    """Eingabe als Stapel (R, n) in float64; ein Vektor wird zu einer Zeile."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("erwartet einen Vektor oder einen Stapel der Form (R, n) mit n >= 1")
    return x


def n_cut(n: int, alpha: float) -> int:
    """Wie viele Werte je Seite abgeschnitten werden: ⌊α·n⌋ mit Toleranz, höchstens so viele, dass mindestens ein Wert bleibt. Eine Treppe in α: bei n = 10 ist 5 % noch 0."""
    k = int(math.floor(alpha * n + CUT_TOLERANCE))
    return max(0, min(k, (n - 1) // 2))


def mean(x) -> np.ndarray:
    return rows(x).mean(axis=1)


def median(x) -> np.ndarray:
    return np.median(rows(x), axis=1)


def trimmed_mean(x, alpha: float) -> np.ndarray:
    """Mittel nach Abschneiden der k kleinsten und k größten Werte, k = n_cut(n, alpha)."""
    x = rows(x)
    n = x.shape[1]
    k = n_cut(n, alpha)
    return np.sort(x, axis=1)[:, k:n - k].mean(axis=1)


def winsorized_mean(x, alpha: float) -> np.ndarray:
    """Mittel nach Ersetzen der k kleinsten durch den (k+1)-kleinsten und der k größten durch den (k+1)-größten Wert, k = n_cut(n, alpha)."""
    x = rows(x)
    n = x.shape[1]
    k = n_cut(n, alpha)
    xs = np.sort(x, axis=1)
    return np.clip(x, xs[:, k][:, None], xs[:, n - 1 - k][:, None]).mean(axis=1)


def mad_scale(x) -> np.ndarray:
    """MAD / 0,6745: der Median der Abstände zum Median, auf die Normalverteilung skaliert. 0, wenn mehr als die Hälfte der Werte gleich ist."""
    x = rows(x)
    med = np.median(x, axis=1)
    return np.median(np.abs(x - med[:, None]), axis=1) / MAD_K


def _scale_vector(x: np.ndarray, scale) -> np.ndarray:
    """Skala je Zeile: None = MAD-Schätzung, sonst eine feste Zahl."""
    if scale is None:
        return mad_scale(x)
    return np.full(x.shape[0], float(scale))


def huber_location(x, c: float = 1.345, scale=None, return_iterations: bool = False):
    """Huber-Lageschätzer: Nullstelle von Σ ψ_c((x_i − μ)/s) mit ψ_c(r) = clip(r, −c, c), gerechnet als gewichtete Iteration (Start: Median).

    Skala s: None = MAD/0,6745 je Zeile, sonst eine feste Zahl ("bekannt"). Ist s = 0 (mehr als die Hälfte der Werte gleich), fällt das Ergebnis auf den Median zurück
    (Grenzfall s → 0). Ist die Wurzel nicht in HUBER_MAXIT Schritten erreicht, bleibt der letzte Stand."""
    x = rows(x)
    s = _scale_vector(x, scale)
    zero = s <= 0
    s_safe = np.where(zero, 1.0, s)
    med = np.median(x, axis=1)
    mu = med.copy()
    used = 0
    for it in range(1, HUBER_MAXIT + 1):
        r = (x - mu[:, None]) / s_safe[:, None]
        w = c / np.maximum(np.abs(r), c)          # = min(1, c/|r|), ohne Division durch 0
        mu_new = np.where(zero, med, (w * x).sum(axis=1) / w.sum(axis=1))
        step = float(np.max(np.abs(mu_new - mu) / s_safe))
        mu = mu_new
        used = it
        if step < HUBER_TOL:
            break
    return (mu, used) if return_iterations else mu


def walsh_averages(v) -> np.ndarray:
    """Alle Paarmittel (v_i + v_j)/2 mit i <= j eines Vektors, n(n+1)/2 Stück."""
    v = np.asarray(v, dtype=np.float64)
    i, j = np.triu_indices(len(v))
    return (v[i] + v[j]) / 2.0


def hodges_lehmann(x, budget: int = HL_BUDGET) -> np.ndarray:
    """Median aller Walsh-Mittel (i <= j), exakt; zeilenweise in Stücken, damit der Speicher unter `budget` Paarmitteln bleibt."""
    x = rows(x)
    n = x.shape[1]
    i, j = np.triu_indices(n)
    out = np.empty(x.shape[0])
    step = max(1, budget // len(i))
    for a in range(0, x.shape[0], step):
        blk = x[a:a + step]
        out[a:a + step] = np.median((blk[:, i] + blk[:, j]) / 2.0, axis=1)
    return out


def catoni_psi(t) -> np.ndarray:
    """ψ(t) = log(1 + t + t²/2) für t >= 0, antisymmetrisch fortgesetzt (nach der Definition aus dem Gedächtnis, Quelle nicht eingesehen). ψ(1) = log 2,5."""
    t = np.asarray(t, dtype=np.float64)
    a = np.abs(t)
    return np.sign(t) * np.log1p(a + a * a / 2.0)


def catoni_alpha(n: int, v, delta: float = 0.05) -> np.ndarray:
    """α = √(2L / (n·v·(1 + 2L/(n − 2L)))) mit L = log(1/δ) und der Varianz v (Skalar oder je Zeile); verlangt n > 2L (Definition aus dem Gedächtnis)."""
    big_l = math.log(1.0 / delta)
    if n <= 2.0 * big_l:
        raise ValueError(f"n muss größer als 2·log(1/δ) = {2 * big_l:.2f} sein")
    v = np.asarray(v, dtype=np.float64)
    return np.sqrt(2.0 * big_l / (n * v * (1.0 + 2.0 * big_l / (n - 2.0 * big_l))))


def catoni_root(x, alpha) -> np.ndarray:
    """Nullstelle von F(θ) = Σ ψ(α (x_i − θ)) durch Bisektion auf [min, max] (F fällt monoton in θ). alpha: Skalar oder je Zeile."""
    x = rows(x)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (x.shape[0],))
    lo = x.min(axis=1).copy()
    hi = x.max(axis=1).copy()
    for _ in range(CATONI_BISECTIONS):
        mid = (lo + hi) / 2.0
        f = catoni_psi(alpha[:, None] * (x - mid[:, None])).sum(axis=1)
        pos = f > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return (lo + hi) / 2.0


def catoni_location(x, delta: float = 0.05, mult: float = 1.0, variance: str = "mad") -> np.ndarray:
    """Catoni-Schätzer mit α = mult · catoni_alpha(n, v, δ). Die Varianz v kommt aus der MAD-Skala ("mad", robust) oder der Stichprobenvarianz ("stichprobe");
    ist sie 0, gilt v = 1."""
    x = rows(x)
    n = x.shape[1]
    if variance == "mad":
        v = mad_scale(x) ** 2
    elif variance == "stichprobe":
        v = x.var(axis=1, ddof=1) if n > 1 else np.zeros(x.shape[0])
    else:
        raise ValueError(f"unbekannte Varianzangabe: {variance}")
    v = np.where(v <= 0, 1.0, v)
    return catoni_root(x, mult * catoni_alpha(n, v, delta))


# --- Auswahl über Namen -------------------------------------------------------------------------------------------------------------------

def parse_spec(spec: str) -> tuple[str, float | None]:
    """"mean", "median", "hl" oder "<art>:<zahl>" mit art = trim, winsor, huber, catoni."""
    if ":" in spec:
        kind, value = spec.split(":", 1)
        if kind not in ("trim", "winsor", "huber", "catoni"):
            raise ValueError(f"unbekannter Schätzer: {spec}")
        return kind, float(value)
    if spec not in ("mean", "median", "hl"):
        raise ValueError(f"unbekannter Schätzer: {spec}")
    return spec, None


def estimate(spec: str, x, scale=None, delta: float = 0.05) -> np.ndarray:
    """Schätzer nach Namen auf einen Stapel angewandt. `scale` gilt für Huber (None = MAD), `delta` für Catoni (Varianz immer aus der MAD)."""
    kind, p = parse_spec(spec)
    if kind == "mean":
        return mean(x)
    if kind == "median":
        return median(x)
    if kind == "trim":
        return trimmed_mean(x, p)
    if kind == "winsor":
        return winsorized_mean(x, p)
    if kind == "huber":
        return huber_location(x, p, scale)
    if kind == "hl":
        return hodges_lehmann(x)
    return catoni_location(x, delta, p)


def _pct(p: float) -> str:
    return f"{100 * p:g}".replace(".", ",")


def label(spec: str) -> str:
    """Anzeigename: "Trimmen 10 %", "Huber (c = 1,345)", "Catoni (λ × 8)"."""
    kind, p = parse_spec(spec)
    if kind == "mean":
        return "Mittel"
    if kind == "median":
        return "Median"
    if kind == "hl":
        return "Hodges–Lehmann"
    if kind == "trim":
        return f"Trimmen {_pct(p)} %"
    if kind == "winsor":
        return f"Winsorisieren {_pct(p)} %"
    if kind == "huber":
        return f"Huber (c = {p:g})".replace(".", ",")
    return f"Catoni (λ × {p:g})".replace(".", ",")


def scalar(values) -> float:
    """Das einzige Element eines Ergebnisses für eine einzelne Stichprobe."""
    return float(np.asarray(values).reshape(-1)[0])
