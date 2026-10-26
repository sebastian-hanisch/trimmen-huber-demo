"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Raster der Messreihen und Presets.

Einheit: Lieferzeit-Abweichung einer Sendung in Stunden gegenüber Plan, der Kern ist auf Streuung 1 skaliert. Ein Ausfall ist eine Sendung, deren Wert durch d + N(0, 1) ersetzt wird.
Buchstaben: α ist der Trimm- bzw. Winsor-Anteil je Seite, ε der Anteil der Ausfälle, d ihr Abstand, c die Huber-Schranke, λ das Gewicht des Catoni-Schätzers (ein Vielfaches des
Wertes aus der Definition), δ das Konfidenzniveau dieser Wahl. Im Abschnitt „Kappen und privates Mittel“ heißt das Datenschutz-Budget ε_DP und die Kappungsgrenze C."""

from __future__ import annotations

from thu_scenario import DISTRIBUTIONS

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
SIDES = ("einseitig (alle Ausfälle verspätet)", "zweiseitig (Vorzeichen zufällig)")
SCALES = ("MAD (aus den Daten)", "bekannt (Skala 1)")
D_OPTIONS = (3.0, 5.0, 10.0, 30.0, 100.0, 1000.0, 10000.0, 1000000.0)
HUBER_C_OPTIONS = (0.5, 1.0, 1.345, 2.0, 3.0)
CATONI_MULT_OPTIONS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DELTA_OPTIONS = (0.2, 0.1, 0.05, 0.01)

N_MIN, N_MAX, DEFAULT_N = 10, 3000, 100        # Hodges–Lehmann live O(n²): bei n = 3000 4,5 Mio. Paarmittel (gemessen 182 ms), alles andere O(n log n) oder O(n · Iterationen)
EPS_MIN, EPS_MAX, DEFAULT_EPS = 0, 45, 10      # Ausfallanteil in Prozent
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA = 0, 45, 10   # Trimm-/Winsor-Anteil je Seite in Prozent
SEED_MAX, DEFAULT_SEED = 999999, 7
DEFAULT_D = 5.0
DEFAULT_C = 1.345
DEFAULT_MULT = 1.0
DEFAULT_DELTA = 0.05

# --- Rechenbudget ---------------------------------------------------------------------------------------------------------------------------
LIVE_CURVE_MAX_N = 300          # Nachmessen der Verschiebungskurve: n höchstens 300
LIVE_CURVE_MAX_RUNS = 300       # und höchstens 300 Läufe je Seed
LIVE_HL_BUDGET = 30_000_000     # Läufe · n² für Hodges–Lehmann im Nachmessen höchstens (Läufe · n(n+1)/2 Paarmittel pro Seed ist die Größe, die zählt)
KAPPEN_LIVE_RUNS = 300          # Läufe je Seed im Nachmessen der U-Kurve

# --- Messreihen (vorgerechnet in data/thu_results.json, erzeugt von tools/make_results.py) ---------------------------------------------------
SEEDS = (11, 12, 13, 14)        # vier feste Zufalls-Seeds je Zelle: nie ein einzelner Seed als Befund
RESULTS_FILE = "thu_results.json"
N_RASTER = (10, 30, 100, 300)
EPS_GRID = tuple(round(0.05 * k, 2) for k in range(0, 10))        # 0 bis 45 % in 5er-Schritten
EPS_FINE = tuple(round(0.04 + 0.01 * k, 2) for k in range(0, 43))   # 4 bis 46 % in 1er-Schritten
D_GRID = (3.0, 5.0, 10.0, 1000.0)
D_CURVE = (3.0, 5.0, 10.0, 30.0, 100.0, 1000.0, 1000000.0)
TRIM_ALPHAS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45)
HUBER_CS = (0.25, 0.5, 0.75, 1.0, 1.345, 2.0, 3.0, 5.0)
CATONI_MULTS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
FAMILY_SPECS = (("mean", "median", "hl")
                + tuple(f"trim:{a:g}" for a in TRIM_ALPHAS) + tuple(f"winsor:{a:g}" for a in TRIM_ALPHAS)
                + tuple(f"huber:{c:g}" for c in HUBER_CS) + tuple(f"catoni:{m:g}" for m in CATONI_MULTS))
CURVE_SPECS = ("mean", "median", "trim:0.1", "trim:0.25", "trim:0.4", "winsor:0.1", "winsor:0.25", "huber:0.5", "huber:1.345", "huber:3", "hl", "catoni:1", "catoni:8")
MAIN_SPECS = ("mean", "median", "trim:0.1", "winsor:0.1", "huber:1.345", "hl", "catoni:1")
TAIL_DISTS = ("Normal", "Laplace", "t(5)", "t(3)")
SKEW_DISTS = ("Lognormal (σ = 0,5)", "Lognormal (σ = 1)", "Lognormal (σ = 2)", "Exponential")
TARGET_SPECS = ("mean", "median", "trim:0.1", "trim:0.25", "winsor:0.1", "huber:0.5", "huber:1.345", "hl", "catoni:1")
KAPPEN_SPECS = (("lognormal", 1.0), ("pareto", 1.5), ("pareto", 3.0))
KAPPEN_SCALING_SPECS = (("lognormal", 0.5), ("lognormal", 1.0), ("lognormal", 1.5), ("pareto", 1.5), ("pareto", 3.0))
KAPPEN_N = (100, 1000, 10000)
KAPPEN_EPS = (0.1, 0.5, 1.0, 5.0)
KAPPEN_NS_SCALING = (100, 300, 1000, 3000, 10000, 100000, 1000000)

# --- Farben -------------------------------------------------------------------------------------------------------------------------------
COLOR_POINT = "rgba(80,80,80,0.55)"
COLOR_OUTLIER = "#d62728"
COLOR_LINE = "#4c78a8"
COLOR_LINE2 = "#e07b00"
COLOR_TARGET = "#2F6B65"
SPEC_COLORS = {
    "mean": "#7f7f7f", "median": "#2F6B65", "trim": "#4c78a8", "winsor": "#9467bd", "huber": "#e07b00", "hl": "#d62728", "catoni": "#8c564b",
}

# --- Presets (Werte für jeden Regler; Werte in tests/test_presets.py auf ihr Verhalten geprüft) -------------------------------------------------
_BASE = {
    "dist": "Normal", "n": DEFAULT_N, "eps": DEFAULT_EPS, "d": DEFAULT_D, "side": SIDES[0], "alpha": DEFAULT_ALPHA, "c": DEFAULT_C, "scale": SCALES[0],
    "mult": DEFAULT_MULT, "delta": DEFAULT_DELTA, "seed": DEFAULT_SEED,
}


def _preset(**changes):
    unknown = set(changes) - set(_BASE)
    assert not unknown, unknown
    return {**_BASE, **changes}


PRESETS = {
    "10 % Ausfall (Standard)": _preset(),
    "Sauber": _preset(eps=0),
    "Anteil über der Trimmgrenze": _preset(eps=15, d=1000.0),
    "Hodges–Lehmann an der Grenze": _preset(eps=30, d=1000.0),
    "Schiefe Lieferzeiten": _preset(dist="Lognormal (σ = 1)", eps=0),
    "Diskrete Tage (Skala 0)": _preset(dist="Diskrete Tage (Poisson 0,5)", eps=0),
    "Kleines n (Treppe)": _preset(n=10, alpha=5),
}
PRESET_HELP = {
    "10 % Ausfall (Standard)": "100 Sendungen aus dem Kern N(0, 1), 10 davon durch Ausfälle im Abstand 5 ersetzt. Das Mittel liegt sichtbar neben dem Kern; Median, Trimmen, Huber und Hodges–Lehmann liegen nah beieinander. "
                               "Welche davon besser sind, hängt vom Preis auf sauberen Daten ab (E1).",
    "Sauber": "Keine Ausfälle, 100 Sendungen aus dem Kern. Auf sauberen Normaldaten ist das Mittel am genauesten; was die robusten Schätzer dafür kosten, zeigt E1.",
    "Anteil über der Trimmgrenze": "15 Ausfälle im Abstand 1000, getrimmt wird nur zu je 10 % (α = 10 %). Der Ausfallanteil liegt über dem Trimm-Anteil: das getrimmte und das winsorisierte Mittel schneiden nicht alle Ausfälle ab "
                                    "(Klippe, E2); Median und Huber bleiben beim Kern.",
    "Hodges–Lehmann an der Grenze": "30 Ausfälle im Abstand 1000. Der Hodges–Lehmann-Schätzer springt in der Nähe dieses Anteils auf einen Wert weit weg vom Kern (E2 zeigt die Lage der Klippe), Median und Huber (MAD) halten.",
    "Schiefe Lieferzeiten": "Lognormal (σ = 1) ohne Ausfälle: Mittelwert und Median des Kerns sind verschieden, und jeder Schätzer zielt irgendwo dazwischen. Wohin, zeigt E4.",
    "Diskrete Tage (Skala 0)": "Ganze Tage mit mehr als der Hälfte Nullen (Poisson 0,5): Der MAD ist fast immer 0, Huber fällt auf den Median zurück und das Mittel liegt davon deutlich entfernt (E6).",
    "Kleines n (Treppe)": "10 Sendungen, α = 5 %: ⌊0,05 · 10⌋ = 0 Werte je Seite, das getrimmte Mittel ist das Mittel. Dasselbe Bild zeigt E6 als Treppe über α und n.",
}
assert DISTRIBUTIONS and len(set(PRESETS)) == len(PRESETS)
