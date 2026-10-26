"""PDF-Export (fpdf2, Kernschriften): Einstellungen, die Lagen aller Schätzer für die Stichprobe und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import thu_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich", "→": "->",
            "ε": "eps", "α": "alpha", "λ": "lambda", "δ": "delta", "σ": "sigma", "×": "x"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def estimate_rows(a: V.Analysis) -> list[tuple[str, float, float]]:
    """(Name, Lage, Abstand zum Median des Kerns) je Schätzer."""
    return [(label, value, abs(value - a.target_median)) for label, value in zip(a.labels, a.values)]


def build_pdf(a: V.Analysis, verdicts: list[tuple[str, str]]) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Getrimmt, Hodges-Lehmann, Huber: wie viel Median braucht man?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Verteilung {s.dist}, n = {s.n}, Ausfallanteil {s.eps} %, Abstand d = {s.d:g}, {s.side}, Trimm-/Winsor-Anteil {s.alpha} % (abgeschnitten je Seite: {a.cut}), "
                                f"Huber c = {s.c:g} ({s.scale}), Catoni lambda = {s.mult:g} x Definition (delta = {s.delta:g}), Zufalls-Seed {s.seed}. Ziel: Median des Kerns = {V.de(a.target_median, 3)}, "
                                f"Mittelwert des Kerns = {V.de(a.target_mean, 3)}."))
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Lagen der Schätzer in dieser Stichprobe", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for label, value, dev in estimate_rows(a):
        pdf.cell(0, 5.5, _latin(f"{label}: {V.de(value, 3)} (Abstand zum Ziel {V.de(dev, 3)})"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text))
        pdf.ln(1)
    pdf.set_y(pdf.get_y() + 3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Eine einzelne Stichprobe zeigt keine Eigenschaft eines Schätzers; die Messreihen der App (vier feste Zufalls-Seeds je Zelle) tun es. Lieferzeit-Abweichung in Stunden, Kern auf Streuung 1 skaliert."))
    return bytes(pdf.output())
