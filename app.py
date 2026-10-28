"""Getrimmt, Hodges–Lehmann, Huber: wie viel Median braucht man? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der Reihe „Mittelwerte und Lagemaße“, mit einem Abschnitt „Kappen und privates Mittel“. Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import thu_constants as C
import thu_curves as K
import thu_evaluation as V
import thu_kappen as KP
import thu_neighbors as N
import thu_private as P
import thu_results as R
import thu_robust as RB
from thu_layout import dunkles_design, kopf_css
from thu_pdf_export import build_pdf
from thu_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from thu_visualization import (
    build_catoni_distance_chart,
    build_distance_chart,
    build_enumeration_chart,
    build_fine_chart,
    build_pareto_chart,
    build_rmse_by_n_chart,
    build_sample_chart,
    build_scaling_chart,
    build_shift_chart,
    build_stairs_chart,
    build_tail_chart,
    build_target_chart,
    build_ucurve_chart,
    build_wrong_c_chart,
)

st.set_page_config(page_title="Getrimmt, Hodges–Lehmann, Huber – Sebastian Hanisch", layout="wide")

SKEW_PICK = C.SKEW_DISTS
KAPPEN_NAMES = tuple(V.spec_name(s) for s in C.KAPPEN_SPECS)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return V.analyse(settings)


@st.cache_data(show_spinner=False)
def _shift_live(n, d, two_sided, dist, runs):
    return K.shift_cell(C.CURVE_SPECS, n, C.EPS_GRID, (d,), runs, C.SEEDS, two_sided=two_sided, dist=dist)


@st.cache_data(show_spinner=False)
def _ucurve_live(spec, n, runs):
    return KP.ucurve_cell(spec, n, runs, C.SEEDS, C.KAPPEN_EPS, KP.c_grid())


@st.cache_data(show_spinner=False)
def _data_c_live(y):
    return N.max_log_ratio_data_dependent_c(4, range(1, 7), 1.0, y)


def _verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def _fmt_table(rows, **formats):
    st.dataframe(rows, hide_index=True, width="stretch", column_config={k: st.column_config.NumberColumn(format=v) for k, v in formats.items()})


def _de3(x):
    return V.de(x, 3)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in thu_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📊 Getrimmt, Hodges–Lehmann, Huber: wie viel Median braucht man?")
eff_median = R.efficiency(100, "median")
st.markdown(
    f"""
**Stück 4 der Reihe „Mittelwerte und Lagemaße“.** Eine Sendung kommt Stunden früher oder später als geplant an; zwischen den normalen Sendungen stecken **Lieferausfälle**, deren **Anteil ε** und **Abstand d** Sie wählen.
Das **Mittel** folgt jedem Ausfall, der **Median** ignoriert ihn, kostet aber auf sauberen Normaldaten gut ein Drittel der Genauigkeit (Effizienz {_de3(eff_median)} bei n = 100, Experiment E1).
Dazwischen liegen das **getrimmte und das winsorisierte Mittel**, der **Hodges–Lehmann-Schätzer** und der **Huber-Schätzer** (dazu **Catoni** als Vergleich). Die Demo zeigt, **was jede Familie auf sauberen Daten kostet**,
**wie ihre Verschiebung mit dem Ausfallanteil wächst** (bei Trimmen und Hodges–Lehmann als **Klippe**, bei Huber und Median stetig), **worauf ein Schätzer bei schiefen Daten überhaupt zielt** und **welche Fallen** lauern
(Treppe bei kleinem n, Skala 0, Gleitkomma-Kante). Ein eigener Abschnitt zeigt am Ende, was **Kappen** (= Winsorisieren auf eine feste Grenze) für ein **privates Mittel** bedeutet; er ist Lehrmaterial über Definitionen und Fehler
und **kein Sicherheitsnachweis**.

Abgrenzung: `permutation-test-demo` winsorisiert nur als Vorbehandlung eines Tests, `forecast-combination-demo` nutzt das getrimmte Mittel als eine Kombinationsregel, `kmeans-demo` vergleicht Mittelwert und Medoid,
`robust-filter-demo` gewichtet Messvarianzen nach Huber im Kalman-Filter, `elliptic-envelope-demo` schätzt robust in mehreren Dimensionen (MCD). Der Lageschätzer in **einer** Dimension ist dort nicht der Gegenstand.
Weitere Stücke der Reihe, die hier nur beim Namen vorkommen: Mittelwert und Median, Median-of-Means, Lp-Mitte, Weiszfeld.

**Buchstaben.** **ε** ist der Anteil der Ausfälle, **d** ihr Abstand, **α** der Trimm- bzw. Winsor-Anteil je Seite, **c** die Huber-Schranke, **λ** das Gewicht des Catoni-Schätzers (ein Vielfaches des Wertes aus der Definition).
Im Abschnitt „Kappen und privates Mittel“ heißt das Datenschutz-Budget **ε_DP** und die Kappungsgrenze **C**; dort gibt es keine Ausfälle.

Weiter unten: sieben Experimente (**🔬 E1 bis E7**), der Abschnitt **🔐 Kappen und privates Mittel**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktionieren die Schätzer", expanded=True):
    st.markdown(
        r"""
1. **Mittel und Median** sind die beiden Enden: das Mittel gewichtet jeden Wert gleich, der Median nur den mittleren.
2. **Getrimmtes Mittel:** Sortieren, an beiden Enden $k = \lfloor \alpha n \rfloor$ Werte wegwerfen, den Rest mitteln. **Winsorisiertes Mittel:** statt wegwerfen die Randwerte durch den nächsten behaltenen ersetzen. Beide haben eine **Treppe** in $\alpha$ (E6).
3. **Huber:** wie das Mittel, aber ein Wert, der mehr als $c$ Skaleneinheiten vom Schätzer entfernt liegt, zieht nur noch mit begrenzter Kraft ($\psi_c$ kappt den Rest). Die Skala kommt aus den Daten (MAD) oder ist bekannt.
4. **Hodges–Lehmann:** der Median aller Paarmittel $(x_i + x_j)/2$ (Walsh-Mittel). Exakt gerechnet, ohne Stichprobe der Paare.
5. **Catoni:** wie Huber ein $\psi$-Schätzer, mit logarithmisch wachsendem $\psi$ und einem Gewicht $\lambda$ aus Stichprobenumfang und Varianz. Die Definition steht hier **aus dem Gedächtnis**, die Originalarbeit wurde **nicht eingesehen**.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("📦 Sendungen")
    dist = st.selectbox("Verteilung des Kerns", C.DISTRIBUTIONS, key="dist_select",
                        help="Wie die normalen Sendungen streuen. Normal, Laplace und t sind symmetrisch um 0; Lognormal und Exponential sind schief (rechter Tail); „Diskrete Tage“ sind ganze Zahlen mit vielen Nullen.")
    n = st.slider("Sendungen n", *bounds("n_slider"), key="n_slider", help="Stichprobenumfang. Hodges–Lehmann rechnet alle n(n+1)/2 Paarmittel exakt.")
    eps = st.slider("Anteil ε der Ausfälle (%)", *bounds("eps_slider"), key="eps_slider",
                    help="Genau m = ε·n (gerundet) Sendungen werden durch Ausfälle ersetzt. Die Klippe von Trimmen und Winsorisieren liegt bei ε > α.")
    d = st.select_slider("Abstand d der Ausfälle", C.D_OPTIONS, key="d_select", format_func=V.fmt_g,
                         help="Ein Ausfall ist d + N(0, 1) Stunden verspätet (Kern: Streuung 1).")
    side = st.selectbox("Richtung der Ausfälle", C.SIDES, key="side_select",
                        help="Einseitig: alle Ausfälle verspätet, der ungünstigste Fall für die Lage. Zweiseitig: Vorzeichen je Ausfall zufällig, die Verschiebung hebt sich im Mittel auf, die Streuung nicht.")
    st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe generieren", width="stretch", on_click=randomize_seed)
    st.header("🔧 Schätzer")
    alpha = st.slider("Trimm-/Winsor-Anteil α je Seite (%)", *bounds("alpha_slider"), key="alpha_slider",
                      help="Abgeschnitten werden ⌊α·n⌋ Werte je Seite (Treppe, E6). Bei n < 20 ist 5 % noch 0 Werte, also das Mittel.")
    c_huber = st.select_slider("Huber-Schranke c", C.HUBER_C_OPTIONS, key="c_select", format_func=V.fmt_g,
                               help="Ab c Skaleneinheiten Abstand zählt ein Wert nur noch mit begrenzter Kraft. Großes c: wie das Mittel, kleines c: wie der Median.")
    scale = st.selectbox("Skala für Huber", C.SCALES, key="scale_select",
                         help="MAD: Skala aus den Daten (bei mehr als der Hälfte gleichen Werten 0, dann fällt Huber auf den Median zurück, E6). Bekannt: feste Skala 1.")
    mult = st.select_slider("Catoni-Gewicht λ (Vielfaches der Definition)", C.CATONI_MULT_OPTIONS, key="mult_select", format_func=V.fmt_g,
                            help="1 = der Wert aus der Definition (aus dem Gedächtnis, Quelle nicht eingesehen). Größere Vielfache machen Catoni robuster (E7).")
    delta = st.select_slider("Catoni-Niveau δ", C.DELTA_OPTIONS, key="delta_select", format_func=V.fmt_g, help="Geht in die Wahl von λ ein; verlangt n > 2·log(1/δ).")

settings = V.Settings(dist, int(n), int(eps), float(d), side, int(alpha), float(c_huber), scale, float(mult), float(delta), int(st.session_state["seed_input"]))
sync_query_params({k: st.session_state[k] for k in ("dist_select", "n_slider", "eps_slider", "d_select", "side_select", "alpha_slider", "c_select", "scale_select", "mult_select", "delta_select", "seed_input")})

with st.spinner("Rechne …"):
    t0 = time.perf_counter()
    a = _analysis(settings)
    elapsed = time.perf_counter() - t0

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: die Lage aller Schätzer in einer Stichprobe")
cols = st.columns(4)
cols[0].metric("Abgeschnitten je Seite", f"{a.cut} Werte", delta=f"bei n = {settings.n}, α = {settings.alpha} %", delta_color="off")
cols[1].metric("MAD-Skala", _de3(a.mad), delta="0: Huber = Median" if a.mad <= 0 else "aus den Daten", delta_color="off")
cols[2].metric("Catoni-Skala α_C", _de3(a.catoni_alpha), delta=f"λ = {V.fmt_g(settings.mult)} × Definition", delta_color="off")
cols[3].metric("Ausfälle in der Stichprobe", f"{int(a.is_outlier.sum())}", delta=f"von {settings.n}", delta_color="off")
st.plotly_chart(build_sample_chart(a.x, a.is_outlier, list(a.labels), list(a.specs), list(a.values), a.target_mean, a.target_median), width="stretch", key="main_chart")
st.caption("Punkte: die Sendungen (rot: Ausfälle), Rauten: die Lage jedes Schätzers. Gestrichelt: der Median des Kerns (das Ziel dieser Ansicht), gepunktet: der Mittelwert des Kerns, wenn er anders ist. "
           "Liegt ein Wert außerhalb des Bildes, steht am Rand ein Pfeil, der Zahlenwert in der Beschriftung bzw. beim Darüberfahren.")
_fmt_table([{"Schätzer": lab, "Lage": v, "Abstand zum Ziel": abs(v - a.target_median)} for lab, v in zip(a.labels, a.values)], Lage="%.3f", **{"Abstand zum Ziel": "%.3f"})

pdf_key = repr(settings)
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf"] = (pdf_key, build_pdf(a, [V.verdict_main(a), V.verdict_stairs(settings.n, settings.alpha)]))
if st.session_state.get("pdf", (None, None))[0] == pdf_key:
    st.download_button("⬇️ PDF herunterladen", data=st.session_state["pdf"][1], file_name="trimmen-huber.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viel Median braucht man, und worauf zielt das Ergebnis?")
_verdict(V.verdict_main(a))
_verdict(V.verdict_stairs(settings.n, settings.alpha))
st.caption(f"Rechenzeit der Analyse dieser Stichprobe: {V.de(elapsed, 2)} s. Eine einzelne Stichprobe beweist nichts; die Experimente rechnen viele.")

with st.expander("🧰 Die Schätzer und ihre Parameter erklärt"):
    st.markdown(
        r"""
| Verfahren | Was es tut | Wer setzt an | Was schiefgehen kann |
|---|---|---|---|
| **Mittel** | $\frac1n\sum x_i$ | – | Ein Ausfall im Abstand $d$ verschiebt es um $d/n$ (E2, E3). |
| **Median** | mittlerer Wert der sortierten Stichprobe | – | Auf sauberen Normaldaten deutlich ungenauer als das Mittel (E1); bei Schiefe zielt er auf den Median, nicht auf den Mittelwert (E4). |
| **Trimmen $\alpha$** | $k = \lfloor \alpha n \rfloor$ Werte je Seite weglassen, Rest mitteln | $\alpha$ (Nutzer) | Treppe in $\alpha$ (bei $n = 10$ ist 5 % noch 0 Werte); **Klippe** bei Ausfallanteil $\varepsilon > \alpha$ (E2). |
| **Winsorisieren $\alpha$** | $k$ Randwerte je Seite durch den nächsten behaltenen ersetzen | $\alpha$ (Nutzer) | Klippe bei $\varepsilon > \alpha$; liegt auf der Effizienz-gegen-Verzerrung-Kurve schlechter als Huber (E1); kann bei zweiseitigen Ausfällen schlechter als das Mittel sein (E2). |
| **Huber $c$** | gewichtetes Mittel, Gewicht $\min(1, c/\lvert r \rvert)$ mit $r = (x - \mu)/s$ | $c$ (Nutzer), Skala $s$ (aus den Daten oder bekannt) | Skala 0 bei diskreten Daten: fällt auf den Median zurück (E6); die MAD-Schätzung kostet bei hohem Anteil (E6). |
| **Hodges–Lehmann** | Median aller Walsh-Mittel $(x_i+x_j)/2$, $i \le j$ | – (kein Parameter) | **Klippe** zwischen 29 und 30 % Ausfallanteil (n = 100, d = 1000), danach bei großem Abstand schlechter als das Mittel (E2); Aufwand $O(n^2)$. |
| **Catoni** | Nullstelle von $\sum \psi(\lambda\,(x_i - \theta))$ mit logarithmisch wachsendem $\psi$ | $\lambda$ (Vielfaches der Definition), $\delta$ | Mit dem Wert aus der Definition liegt er bei kleinem Abstand der Ausfälle nahe beim Mittel und wächst mit $d$ nur noch langsam (asymptotisch logarithmisch, E7); sein eigentlicher Zweck (Fehlerquantil bei Niveau $\delta$) ist hier nicht gemessen. |
        """
    )

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.caption(f"Die Messreihen der Experimente sind vorgerechnet (vier feste Zufalls-Seeds je Zelle, Stand {R.meta()['stand']}); die Knöpfe „Nachmessen“ rechnen mit Ihren Einstellungen neu.")

# E1 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E1: Was kostet die Robustheit auf sauberen Daten?")
st.markdown("**Effizienz** = MSE des Mittels geteilt durch MSE des Schätzers, an denselben sauberen Normaldaten (Mittel = 1, größer ist besser). **Verschiebung** = Mittelwert des Schätzers bei Ausfällen (Ziel 0, kleiner ist besser). "
            "Jede Familie hat einen Regler (α, c, λ) und liegt auf einer Linie; die Kreise markieren die Reglerwerte der Seitenleiste.")
e1c1, e1c2 = st.columns(2)
e1_n = e1c1.radio("Stichprobenumfang n", C.N_RASTER, index=2, horizontal=True, key="e1_n")
e1_j = e1c2.radio("Ausfälle", (0, 1), format_func=lambda j: ("ε = 10 %, d = 5", "ε = 20 %, d = 5")[j], horizontal=True, key="e1_j")
cell1 = R.efficiency_cell(int(e1_n))
st.plotly_chart(build_pareto_chart(cell1, int(e1_j), {"trim": a.specs[2], "winsor": a.specs[3], "huber": a.specs[4], "catoni": a.specs[6]}), width="stretch", key="e1_chart")
st.caption(f"Läufe: {cell1['runs_per_seed']} je Seed, vier Seeds. Die Reglerwerte der Seitenleiste, die nicht im Raster der Messung liegen, sind nicht markiert.")
table_specs = ("mean", "median", "trim:0.05", "trim:0.1", "trim:0.25", "winsor:0.1", "huber:0.5", "huber:1.345", "huber:3", "hl", "catoni:1")
_fmt_table([{"Schätzer": RB.label(s), "Effizienz": R.efficiency(int(e1_n), s), "± Std": R.efficiency_std(int(e1_n), s), "Verschiebung (ε = 10 %, d = 5)": R.efficiency_shift(int(e1_n), s, 0),
             "Verschiebung (ε = 20 %, d = 5)": R.efficiency_shift(int(e1_n), s, 1)} for s in table_specs],
           Effizienz="%.3f", **{"± Std": "%.3f", "Verschiebung (ε = 10 %, d = 5)": "%.3f", "Verschiebung (ε = 20 %, d = 5)": "%.3f"})
_verdict(V.verdict_e1({s: (R.efficiency(int(e1_n), s), R.efficiency_shift(int(e1_n), s, 0)) for s in ("mean", "median", "huber:1.345", "hl", "trim:0.1")}, int(e1_n)))
st.markdown("**Hängt die Effizienz an n?** Kaum:")
_fmt_table([{"Schätzer": RB.label(s), **{f"n = {k}": R.efficiency(k, s) for k in C.N_RASTER}} for s in ("median", "hl", "huber:1.345", "trim:0.1")], **{f"n = {k}": "%.3f" for k in C.N_RASTER})

# E2 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E2: Verschiebung über dem Ausfallanteil: Klippe oder Anstieg?")
st.markdown("Die Verschiebung jedes Schätzers (Mittelwert über die Läufe, Ziel 0) bei wachsendem Anteil ε von Ausfällen. Die Achse der Verschiebung ist logarithmisch: eine Klippe ist ein Sprung um Größenordnungen.")
e2c1, e2c2 = st.columns(2)
e2_n = e2c1.radio("Stichprobenumfang n", (20, 100, 300), index=1, horizontal=True, key="e2_n")
e2_d = e2c2.select_slider("Abstand d", C.D_GRID, value=1000.0, key="e2_d", format_func=V.fmt_g)
cell2 = R.shift_cell(int(e2_n))
st.plotly_chart(build_shift_chart(cell2, float(e2_d)), width="stretch", key="e2_chart")
st.caption(f"{cell2['runs_per_seed']} Läufe je Seed, vier Seeds, einseitige Ausfälle. Streuung der Verschiebung zwischen den Seeds: höchstens {V.de(R.shift_std_max(cell2), 3)} Stunden.")
series2 = {s: R.shift_series(cell2, s, float(e2_d)) for s in C.CURVE_SPECS}
_verdict(V.verdict_e2(series2, cell2["eps"], RB.label))
st.markdown("**Feines Raster** (n = 100, d = 1000, Schritte von 1 Prozentpunkt, 300 Läufe je Seed): Wo genau liegen die Klippen?")
fine_specs = ("median", "trim:0.1", "trim:0.25", "winsor:0.25", "hl")
st.plotly_chart(build_fine_chart(R.fine_cell(100), list(fine_specs)), width="stretch", key="e2_fine")
fine_cell = R.fine_cell(100)
_verdict(V.verdict_hl_cliff({s: R.shift_series(fine_cell, s, 1000.0) for s in fine_specs}, fine_cell["eps"], RB.label))
fine300 = R.fine_cell(300)
_verdict(V.verdict_hl_cliff({s: R.shift_series(fine300, s, 1000.0) for s in ("hl", "trim:0.25")}, fine300["eps"], RB.label))
st.markdown("**Zweiseitige Ausfälle** (Vorzeichen je Ausfall zufällig, d = 10, n = 100, 1000 Läufe je Seed): Die Verschiebung hebt sich im Mittel auf, die Streuung nicht. RMSE um 0:")
two = R.two_sided_cell()
_fmt_table([{"Schätzer": RB.label(s), **{f"ε = {V.pct(e, 0)} %": R.two_sided_value(s, e) for e in two["eps"]}} for s in C.CURVE_SPECS],
           **{f"ε = {V.pct(e, 0)} %": "%.3f" for e in two["eps"]})
st.caption(f"Das Mittel hat dabei nach der Rechnung RMSE √((1 + ε·d²)/n): {V.de(np.sqrt((1 + 0.1 * 100) / 100), 3)} bei ε = 10 % und {V.de(np.sqrt((1 + 0.2 * 100) / 100), 3)} bei ε = 20 %.")
live_n = int(min(settings.n, C.LIVE_CURVE_MAX_N))
live_dist = settings.dist if settings.dist in C.TAIL_DISTS else "Normal"
live_runs = int(max(20, min(C.LIVE_CURVE_MAX_RUNS, C.LIVE_HL_BUDGET // max(1, 4 * live_n * live_n))))
live_two = settings.side == C.SIDES[1]
if st.button(f"Nachmessen: n = {live_n}, d = {V.fmt_g(settings.d)}, Kern {live_dist}, {'zweiseitig' if live_two else 'einseitig'} ({live_runs} Läufe je Seed)", key="e2_live"):
    t0 = time.perf_counter()
    with st.spinner(f"Rechne vier Seeds, zehn Anteile, {len(C.CURVE_SPECS)} Schätzer (n = {live_n}) …"):
        res = _shift_live(live_n, float(settings.d), live_two, live_dist, live_runs)
    st.session_state["e2_result"] = (repr(settings), res, time.perf_counter() - t0)
res = st.session_state.get("e2_result")
if res is not None and res[0] == repr(settings):
    st.plotly_chart(build_shift_chart(res[1], float(settings.d)), width="stretch", key="e2_live_chart")
    st.caption(f"Nachgemessen in {V.de(res[2], 1)} s (Rechenzeit; bei wiederholtem Aufruf mit denselben Werten aus dem Zwischenspeicher).")

# E3 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E3: Wächst die Verschiebung mit dem Abstand?")
st.markdown("Bei festem Anteil ε wird der Abstand d der Ausfälle von 3 bis 10⁶ Stunden vergrößert (n = 100, 500 Läufe je Seed). Beschränkte Schätzer bleiben stehen, das Mittel wächst proportional.")
e3_eps = st.radio("Anteil ε", (0.1, 0.2, 0.3), horizontal=True, key="e3_eps", format_func=lambda e: f"{V.pct(e, 0)} %")
dcell = R.distance_cell()
st.plotly_chart(build_distance_chart(dcell, float(e3_eps)), width="stretch", key="e3_chart")
d_specs = ("mean", "median", "trim:0.1", "winsor:0.1", "huber:1.345", "hl", "catoni:1", "catoni:8")
_fmt_table([{"Schätzer": RB.label(s), **{f"d = {V.fmt_g(dd)}": v for dd, v in zip(dcell["d"], R.distance_series(s, float(e3_eps)))}} for s in d_specs], **{f"d = {V.fmt_g(dd)}": "%.3f" for dd in dcell["d"]})
_verdict(V.verdict_e3({s: R.distance_series(s, float(e3_eps)) for s in d_specs}, list(dcell["d"])))

# E4 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E4: Worauf zielt der Schätzer, wenn die Daten schief sind?")
st.markdown("Bei schiefen Lieferzeiten haben Mittelwert und Median verschiedene Werte, und jeder Schätzer mit festem Gewicht (Trimmen, Winsorisieren, Huber, Hodges–Lehmann, Median) zielt irgendwo dazwischen; Catoni mit dem λ der Definition geht mit wachsendem n gegen den Mittelwert. Der **Bias** gegen den Mittelwert **schrumpft nicht mit n**, die Streuung schon; "
            "deshalb gewinnt das Mittel ab einem n, wenn das Ziel der Mittelwert ist. Balken: die Lage des Schätzers an der Population (n → ∞, aus 4 × 1 Mio. Werten) und die Streuung bei dem gewählten n.")
e4c1, e4c2 = st.columns(2)
e4_dist = e4c1.selectbox("Verteilung", SKEW_PICK, index=1, key="e4_dist")
e4_n = e4c2.radio("Stichprobenumfang n", C.N_RASTER, index=2, horizontal=True, key="e4_n")
part4 = R.skew_part_of(e4_dist)
pop = {s: v[0] for s, v in part4["population"].items()}
st.plotly_chart(build_target_chart(R.target_cell(e4_dist, int(e4_n)), part4["population"]), width="stretch", key="e4_chart")
rmse_series = {s: [R.target_value(e4_dist, k, s, "rmse_mean") for k in C.N_RASTER] for s in C.TARGET_SPECS}
st.plotly_chart(build_rmse_by_n_chart(e4_dist, list(C.N_RASTER), rmse_series), width="stretch", key="e4_rmse")
_fmt_table([{"Schätzer": RB.label(s), "Lage (Population)": pop[s], "Bias gegen den Mittelwert": R.target_value(e4_dist, int(e4_n), s, "bias"), "Streuung SD(T)": R.target_value(e4_dist, int(e4_n), s, "sd"),
             "RMSE gegen den Mittelwert": R.target_value(e4_dist, int(e4_n), s, "rmse_mean"), "RMSE gegen den Median": R.target_value(e4_dist, int(e4_n), s, "rmse_median")} for s in C.TARGET_SPECS],
           **{c: "%.3f" for c in ("Lage (Population)", "Bias gegen den Mittelwert", "Streuung SD(T)", "RMSE gegen den Mittelwert", "RMSE gegen den Median")})
st.caption(f"{R.target_cell(e4_dist, int(e4_n))['runs_per_seed']} Läufe je Seed, vier Seeds. Bei Lognormal (σ = 2) ist selbst der Mittelwert-Schätzer stark verrauscht, und seine simulierte Streuung liegt unter der wahren: Zahlen dort nur als Größenordnung lesen, Vergleiche mit dem Mittel sind dort nicht belastbar.")
_verdict(V.verdict_e4(e4_dist, list(C.N_RASTER), rmse_series, pop, part4["cells"][0]["mean"], part4["cells"][0]["median"], [R.target_value(e4_dist, k, "trim:0.1", "bias") for k in C.N_RASTER]))

# E5 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E5: Und bei anderen Rändern?")
st.markdown("Saubere Daten (keine Ausfälle), aber schwerere Ränder als die Normalverteilung. Verhältnis des RMSE zum RMSE des Mittels (unter 1: besser als das Mittel). Kein Schätzer gewinnt überall.")
e5_n = st.radio("Stichprobenumfang n", C.N_RASTER, index=2, horizontal=True, key="e5_n")
cells5 = [c for c in R.tail_cells() if c["n"] == int(e5_n)]
st.plotly_chart(build_tail_chart(cells5, int(e5_n), ["median", "trim:0.1", "trim:0.25", "winsor:0.1", "huber:1.345", "hl"]), width="stretch", key="e5_chart")
ratios5 = {c["dist"]: {r["spec"]: r["ratio"][0] for r in c["rows"]} for c in cells5}
_verdict(V.verdict_e5(ratios5))

# E6 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E6: Fallen")
st.markdown("**Die Treppe.** Die Zahl der abgeschnittenen Werte ist $\\lfloor \\alpha n \\rfloor$ je Seite: bei kleinem $n$ springt sie in großen Schritten, und kleine $\\alpha$ bewirken gar nichts.")
st.plotly_chart(build_stairs_chart([10, 20, 30, 50, 100], [round(0.01 * k, 2) for k in range(0, 46)]), width="stretch", key="e6_stairs")
_fmt_table([{"n": k, **{f"α = {V.pct(al, 0)} %": R.cut(k, al) for al in (0.05, 0.1, 0.25, 0.45)}} for k in (10, 20, 30, 50, 100)], **{f"α = {V.pct(al, 0)} %": "%d" for al in (0.05, 0.1, 0.25, 0.45)})
fe = R.pitfalls()["float_edge"]
ex = fe["examples"][0]
st.markdown(f"**Die Gleitkomma-Kante.** `int(α·n)` weicht von der Rechnung mit Toleranz in **{fe['differ']} von {fe['total']}** Paaren ab (α = k/100 für k = 1 bis 45, n = 5 bis 1000), zum Beispiel α = {V.fmt_g(ex[0])} und n = {ex[1]}: "
            f"`int({V.fmt_g(ex[0])} * {ex[1]})` ist {ex[2]}, gemeint sind {ex[3]} Werte je Seite. Diese Demo rechnet ⌊α·n + 10⁻⁹⌋; `scipy.stats.trim_mean` schneidet in solchen Fällen einen Wert weniger ab.")
st.markdown("**Skala 0 bei diskreten Daten.** Der MAD ist 0, wenn mehr als die Hälfte der Werte gleich ist; Huber fällt dann auf den Median zurück und ignoriert alle Werte außer dem mittleren. Anteil der Stichproben mit MAD = 0 und mittlere Lage (1000 Läufe je Seed):")
zs_rows = []
for kind, name, mean_value in (("poisson:0.5", "Poisson(0,5)", 0.5), ("zip:0.6:5", "60 % Nullen + Poisson(5)", 0.4 * 5), ("poisson:1", "Poisson(1)", 1.0), ("poisson:3", "Poisson(3)", 3.0)):
    for k in (30, 100):
        cz = R.zero_scale_cell(kind, k)
        zs_rows.append({"Verteilung": name, "n": k, "Anteil MAD = 0 (%)": 100 * cz["zero_scale_share"][0], "Mittelwert": mean_value, "Median": cz["location"]["median"][0], "Huber (c = 1,345)": cz["location"]["huber:1.345"][0],
                        "Hodges–Lehmann": cz["location"]["hl"][0], "Trimmen 10 %": cz["location"]["trim:0.1"][0]})
_fmt_table(zs_rows, **{"Anteil MAD = 0 (%)": "%.1f %%", "Mittelwert": "%.3f", "Median": "%.3f", "Huber (c = 1,345)": "%.3f", "Hodges–Lehmann": "%.3f", "Trimmen 10 %": "%.3f"})
mad_row, known_row = R.scale_known_row("mad"), R.scale_known_row("known")
st.markdown(f"**Bekannte Skala gegen MAD.** Huber (c = 1,345) mit Skala 1 statt MAD: Effizienz {_de3(known_row['eff'][0])} gegen {_de3(mad_row['eff'][0])}; Verschiebung bei ε = 20 %, d = 5: {_de3(known_row['shifts'][0][0])} gegen {_de3(mad_row['shifts'][0][0])}; "
            f"bei ε = 30 %, d = 1000: {_de3(known_row['shifts'][1][0])} gegen {_de3(mad_row['shifts'][1][0])}. Die MAD-Schätzung kostet mit steigendem Anteil.")
st.markdown(f"**Rechnung.** Huber braucht eine Iteration: größte Zahl der Schritte im Stapel von 2000 Stichproben (n = 100, Mittel über Seeds) {V.de(R.iterations(0.5), 1)} bei c = 0,5, {V.de(R.iterations(1.345), 1)} bei c = 1,345, "
            f"{V.de(R.iterations(3.0), 1)} bei c = 3. Hodges–Lehmann ist exakt: bei n = {settings.n} sind es {settings.n * (settings.n + 1) // 2} Paarmittel.")

# E7 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E7: Catoni und das Gewicht λ")
st.markdown("Catoni nach der Definition aus dem Gedächtnis (Quelle nicht eingesehen): Mit dem Wert λ × 1 liegt die Verschiebung bei kleinem Abstand nahe beim Mittel (Tabelle, Spalte d = 5) und wächst mit dem Abstand weiter, "
            "aber sehr langsam (asymptotisch logarithmisch) statt proportional; erst ein Vielfaches λ macht ihn robuster, und er bleibt auch dann nicht beschränkt. "
            "Sein eigentlicher Zweck, ein enges Fehlerquantil bei Niveau δ unter endlicher Varianz, ist hier **nicht gemessen** (das gehört zu Median-of-Means). n = 100, 2000 Läufe je Seed.")
ce = R.catoni_part()["efficiency"]
_fmt_table([{"Schätzer": RB.label(s), "Effizienz": R.catoni_efficiency(s), "Verschiebung (ε = 10 %, d = 5)": R.catoni_shift(s, 0), "Verschiebung (ε = 10 %, d = 1000)": R.catoni_shift(s, 1),
             "Verschiebung (ε = 20 %, d = 1000)": R.catoni_shift(s, 2), "RMSE/Mittel auf t(3)": (R.catoni_t3_ratio(s) if s != "mean" and s != "median" else float("nan"))}
            for s in [r["spec"] for r in ce["rows"]]],
           Effizienz="%.3f", **{c: "%.3f" for c in ("Verschiebung (ε = 10 %, d = 5)", "Verschiebung (ε = 10 %, d = 1000)", "Verschiebung (ε = 20 %, d = 1000)", "RMSE/Mittel auf t(3)")})
st.plotly_chart(build_catoni_distance_chart(R.catoni_part()["distance"]), width="stretch", key="e7_chart")
st.caption("Verschiebung über dem Abstand bei ε = 10 %; die Linien für n = 20 und n = 300 lassen sich in der Legende einblenden. Catoni wächst mit d (und mit n, weil λ mit 1/√n fällt), Huber bleibt stehen.")
_verdict(V.verdict_catoni({s: (R.catoni_efficiency(s), R.catoni_shift(s, 0), R.catoni_shift(s, 1)) for s in ("mean", "huber:1.345", "catoni:1", "catoni:32")}))

st.markdown("---")

# --- Kappen und privates Mittel --------------------------------------------------------------------------------------------------------------

st.header("🔐 Kappen und privates Mittel")
st.warning("**Was das nicht ist.** Dieser Abschnitt zeigt Mathematik auf kleinen Gittern und simulierten Daten: eine Definition, einen Mechanismus, seinen Fehler und ein Gegenbeispiel. Er ist **kein Sicherheitsnachweis** und kein "
           "Werkzeug, um echte Daten zu veröffentlichen: Gleitkommazahlen und Zufallszahlen sind hier nicht gehärtet, mehrere Veröffentlichungen verbrauchen Budget (hier nicht gerechnet), die Daten sind simuliert.")
st.markdown(
    r"""
**Die Idee.** Soll eine Kennzahl (hier: die mittlere Lieferzeit je Auftrag eines Lieferanten) mit Partnern geteilt werden, ohne dass ein einzelner Auftrag herauslesbar ist, kappt man jeden Wert auf eine **öffentliche** Grenze $C$ (das ist Winsorisieren
mit fester Grenze), bildet das Mittel und addiert Laplace-Rauschen. Je kleiner $C$, desto weniger Rauschen, aber desto größer die Verzerrung durch das Kappen: ein Kompromiss, der eine U-Kurve ergibt.

**Definition.** Der Stichprobenumfang $n$ ist öffentlich. Zwei Datensätze sind **benachbart**, wenn genau ein Wert ersetzt ist. Ein Mechanismus ist **$\varepsilon_{\mathrm{DP}}$-differentiell privat**, wenn für alle benachbarten Datensätze $D, D'$ und alle Ausgaben $y$
gilt $\log f(y \mid D) - \log f(y \mid D') \le \varepsilon_{\mathrm{DP}}$. Das Mittel gekappter Werte auf $[0, C]$ ändert sich beim Ersetzen eines Werts um höchstens $C/n$; mit Laplace-Rauschen der Skala $b = C/(n\,\varepsilon_{\mathrm{DP}})$ ist die Schranke eingehalten (E8e prüft das durch Aufzählung).
"""
)
k1, k2, k3 = st.columns(3)
kp_name = k1.selectbox("Verteilung der Aufträge", KAPPEN_NAMES, key="kp_spec")
kp_n = k2.radio("Aufträge n", C.KAPPEN_N, index=1, horizontal=True, key="kp_n")
kp_eps = k3.radio("Budget ε_DP", C.KAPPEN_EPS, index=2, horizontal=True, key="kp_eps", format_func=V.fmt_g)
kp_index = KAPPEN_NAMES.index(kp_name)
kp_spec = C.KAPPEN_SPECS[kp_index]
kp_cell = R.ucurve_cell(kp_index, int(kp_n))
kp_cgrid = kp_cell["cgrid"]
kp_c = st.select_slider("Kappungsgrenze C", [round(c, 2) for c in kp_cgrid], value=round(kp_cgrid[12], 2), key="kp_c", format_func=lambda c: V.de(c, 2))
kp_parts = P.exact_mse(kp_spec, kp_cgrid, int(kp_n), float(kp_eps))
kp_row = R.ucurve_eps(kp_index, int(kp_n), float(kp_eps))
kp_k = min(range(len(kp_cgrid)), key=lambda i: abs(kp_cgrid[i] - kp_c))

# E8a ------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E8a: Die U-Kurve: Kappen gegen Rauschen")
st.markdown(f"Fehler des privaten Mittels gegen das Populationsmittel ({V.de(kp_cell['mu'], 4)}). Die schwarze Linie ist die **geschlossene Zerlegung** MSE = Verzerrung² + Varianz/n + 2b²; die Punkte sind die Simulation "
            f"({kp_cell['runs_per_seed']} Datensätze je Seed, vier Seeds, Streuung als Balken).")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("RMSE bei diesem C", V.de(float(np.sqrt(kp_parts[0][kp_k])), 4), delta="Formel", delta_color="off")
mc2.metric("Verzerrung (Kappung)", V.de(float(np.sqrt(kp_parts[1][kp_k])), 4), delta="Wurzel des Anteils", delta_color="off")
mc3.metric("Stichprobenfehler", V.de(float(np.sqrt(kp_parts[2][kp_k])), 4), delta="Wurzel des Anteils", delta_color="off")
mc4.metric("Rauschen", V.de(float(np.sqrt(kp_parts[3][kp_k])), 4), delta=f"{V.pct(float(kp_parts[3][kp_k] / kp_parts[0][kp_k]), 0)} % der MSE", delta_color="off")
st.plotly_chart(build_ucurve_chart(kp_row, kp_cgrid, kp_parts, kp_row["rmse"], float(kp_cgrid[kp_k])), width="stretch", key="e8a_chart")
k_best = int(np.argmin(kp_parts[0]))
_verdict(V.verdict_ucurve(kp_cgrid[k_best], float(np.sqrt(kp_parts[0][k_best])), kp_cell["sample_rmse"][0], float(kp_parts[3][k_best] / kp_parts[0][k_best])))
st.caption(f"Simulation gegen Formel: RMSE²/MSE-Formel hat über das Raster den Median {V.de(kp_row['ratio_median'], 3)} (kleinster {V.de(kp_row['ratio_min'], 3)}, größter {V.de(kp_row['ratio_max'], 3)}); "
           "bei sehr großem C und schweren Rändern streut die Simulation, weil seltene große Werte mit 4 × Läufen nicht sicher vorkommen.")
if st.button(f"Nachmessen: {kp_name}, n = {int(kp_n)} ({C.KAPPEN_LIVE_RUNS} Datensätze je Seed)", key="e8a_live"):
    t0 = time.perf_counter()
    with st.spinner("Rechne vier Seeds, 25 Grenzen, vier Budgets …"):
        live = _ucurve_live(kp_spec, int(kp_n), C.KAPPEN_LIVE_RUNS)
    st.session_state["e8a_result"] = (repr((kp_spec, int(kp_n), float(kp_eps))), live, time.perf_counter() - t0)
res = st.session_state.get("e8a_result")
if res is not None and res[0] == repr((kp_spec, int(kp_n), float(kp_eps))):
    live_row = next(r for r in res[1]["eps"] if r["eps"] == float(kp_eps))
    st.plotly_chart(build_ucurve_chart(live_row, res[1]["cgrid"], kp_parts, live_row["rmse"], float(kp_cgrid[kp_k])), width="stretch", key="e8a_live_chart")
    st.caption(f"Nachgemessen mit {C.KAPPEN_LIVE_RUNS} Datensätzen je Seed in {V.de(res[2], 1)} s.")

# E8b ------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E8b: Wohin wandert das Minimum?")
st.markdown("Die beste Kappungsgrenze C* der geschlossenen Zerlegung für wachsendes n. Bei Pareto-Rändern folgt sie einem Potenzgesetz (Herleitung: Verzerrung ∼ C^(1−α), Rauschen ∼ C/(n·ε_DP), also C* ∼ (n·ε_DP)^(1/α)); bei Lognormal nicht.")
sc_eps = st.radio("Budget ε_DP", C.KAPPEN_EPS, index=2, horizontal=True, key="sc_eps", format_func=V.fmt_g)
st.plotly_chart(build_scaling_chart(R.scaling_rows(), list(C.KAPPEN_SCALING_SPECS), float(sc_eps)), width="stretch", key="e8b_chart")
slopes, exponents = {}, {}
for spec in C.KAPPEN_SCALING_SPECS:
    per_eps = []
    for e in C.KAPPEN_EPS:
        per_eps.append(V.local_slopes(C.KAPPEN_NS_SCALING, [R.scaling_row(spec, k, e)["c_star"] for k in C.KAPPEN_NS_SCALING]))
    slopes[V.spec_name(spec)] = [float(np.mean(col)) for col in zip(*per_eps)]
    exponents[V.spec_name(spec)] = 1.0 / spec[1] if spec[0] == "pareto" else None
_verdict(V.verdict_scaling(slopes, exponents))
c_same = [P.c_star(("lognormal", 1.0), k, 1000.0 / k) for k in (100, 1000, 10000)]
st.caption("Steigung = d ln C*/d ln n zwischen benachbarten n des Rasters 100 bis 10⁶, gemittelt über die vier Budgets. C* hängt nicht nur am Produkt n·ε_DP: Für Lognormal (σ = 1) und n·ε_DP = 1000 ist "
           f"C* = {V.de(c_same[0], 1)} bei n = 100 (ε_DP = 10), {V.de(c_same[1], 1)} bei n = 1000 (ε_DP = 1) und {V.de(c_same[2], 1)} bei n = 10 000 (ε_DP = 0,1).")

# E8c ------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E8c: Was kostet ein falsches C?")
st.markdown("RMSE bei der Grenze f · C* geteilt durch den RMSE bei C* (geschlossene Zerlegung, n = 1000). Wer C raten muss, kennt C* nicht.")
st.plotly_chart(build_wrong_c_chart(R.wrong_c_rows()), width="stretch", key="e8c_chart")
wc_sel = st.radio("Beispiel", ("ε_DP = 1", "ε_DP = 0,1"), horizontal=True, key="e8c_sel")
_verdict(V.verdict_wrong_c(R.wrong_c(("lognormal", 1.0), 1.0 if wc_sel == "ε_DP = 1" else 0.1)["rows"]))
st.caption("Lognormal (σ = 1), n = 1000.")

# E8d ------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E8d: C aus den Daten gewählt: wo bricht die Garantie?")
st.markdown("Naheliegend ist, C aus den Daten zu wählen (zum Beispiel den dritt-kleinsten von vier Werten) und nur das Rauschen darauf zu skalieren. Dann hängt die **Skala selbst von den Daten ab**, und ein einziger Wert kann sie verändern. "
            "Aufzählung aller benachbarten Datensätze (vier Werte aus 1 bis 6, ε_DP = 1): das größte log-Verhältnis über alle Paare an der Stelle $y$:")
dc = R.enumeration()["data_c"]
st.plotly_chart(build_enumeration_chart(dc), width="stretch", key="e8d_chart")
_fmt_table([{"y": r["y"], "größtes log-Verhältnis": r["worst"], "Datensatz D": str(tuple(r["data"])), "Nachbar D′": str(tuple(r["neighbor"]))} for r in dc], y="%.0f", **{"größtes log-Verhältnis": "%.2f"})
_verdict(V.verdict_data_c({r["y"]: r["worst"] for r in dc}, 1.0))
y_pick = st.select_slider("Selbst nachrechnen: Stelle y", [3.0, 10.0, 100.0, 1000.0], value=100.0, key="e8d_y", format_func=V.fmt_g)
if st.button("Aufzählung für dieses y jetzt rechnen", key="e8d_live"):
    t0 = time.perf_counter()
    with st.spinner("Zähle 1296 Datensätze mit je 20 Nachbarn auf …"):
        out = _data_c_live(float(y_pick))
    st.session_state["e8d_result"] = (float(y_pick), out, time.perf_counter() - t0)
res = st.session_state.get("e8d_result")
if res is not None and res[0] == float(y_pick):
    st.info(f"Nachgerechnet ({V.de(res[2], 1)} s): bei y = {V.fmt_g(res[0])} ist das größte log-Verhältnis {V.de(res[1][0], 2)}, erreicht bei D = {tuple(res[1][1])} und D′ = {tuple(res[1][2])}.")

# E8e ------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E8e: Die Empfindlichkeit durch Aufzählung geprüft")
st.markdown("Werte 0 bis 6, Kappung auf [0, 3] (also Empfindlichkeit 3/n), alle benachbarten Datensätze. Das größte log-Verhältnis darf das Budget ε_DP nicht übersteigen. Zwei **absichtliche Fehler** zeigen, dass die Prüfung etwas fängt: "
            "eine Skala mit n + 1 statt n im Nenner und ein Mittel ohne Kappung.")
enum = R.enumeration()
_fmt_table([{"n": r["n"], "ε_DP": r["eps"], "Paare": r["pairs"], "korrekt": r["korrekt"], "Fehler: Empfindlichkeit mit n + 1": r["empfindlichkeit_n_plus_1"], "Fehler: Kappung vergessen": r["ohne_kappung"]} for r in enum["laplace"]],
           n="%d", **{"ε_DP": "%.1f", "korrekt": "%.4f", "Fehler: Empfindlichkeit mit n + 1": "%.4f", "Fehler: Kappung vergessen": "%.4f", "Paare": "%d"})
sym = enum["symmetric"]
st.success(f"Die korrekte Skala hält ε_DP genau ein (größtes log-Verhältnis = ε_DP, die Schranke ist scharf); beide Fehler überschreiten sie. "
           f"Bei beidseitigem Kappen auf [−C, C] ist die Empfindlichkeit 2C/n: Aufzählung mit n = {sym['n']}, Werte −3 bis 3, Kappung [−2, 2]: Empfindlichkeit {V.de(sym['sensitivity'], 4)}, größtes log-Verhältnis {V.de(sym['worst'], 4)} bei ε_DP = {V.fmt_g(sym['eps'])}.")

# E8f ------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E8f: Kappen ist Winsorisieren")
st.markdown("Kappt man auf die Schwellen aus den Daten (den (k+1)-kleinsten und den (k+1)-größten Wert), ist das Mittel genau das winsorisierte Mittel dieser Demo. Beispiel (1, 2, 3, 4, 100) mit k = 1: "
            f"Schwellen {P.winsor_bounds((1, 2, 3, 4, 100), 1)}, gekapptes Mittel {V.de(P.clip_at_data_bounds_mean((1, 2, 3, 4, 100), 1), 1)}, winsorisiertes Mittel {V.de(RB.scalar(RB.winsorized_mean([1, 2, 3, 4, 100], 0.2)), 1)}.")
ws = R.winsor_sensitivity()
st.markdown(f"Der Unterschied liegt in der **Empfindlichkeit**: bei festen, öffentlichen Schwellen ist sie $(\\text{{hi}} - \\text{{lo}})/n$; bei Schwellen aus den Daten wandern die Schwellen beim Ersetzen eines Werts mit. "
            f"Gemessen an {ws['datasets_per_seed']} lognormalen Datensätzen je Seed (n = {ws['n']}, k = {ws['k']} Wert je Seite, alle {ws['n']} Positionen, 13 Ersatzwerte):")
_verdict(V.verdict_winsor_sensitivity(ws["mean_ratio"][0], ws["min_ratio"], ws["largest_ratio"]))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ausfälle unabhängig vom Kern, genau m = ε·n Stück, ein fester Abstand d (± Rauschen)** | Echte Ausfälle häufen sich in Wellen und haben verschiedene Höhen; die Klippen und ihre Lage gelten für dieses Modell. | – |
| **Eine Dimension** | Für mehrere Dimensionen braucht es andere Schätzer (zum Beispiel MCD). | `elliptic-envelope-demo` (Streuung, Richtung) |
| **Bekannte Verteilungen des Kerns** | Alle Messreihen laufen auf den neun Verteilungen dieser Demo; echte Lieferzeiten können anders aussehen. | – |
| **Keine Konfidenzintervalle, keine Tests** | Hier steht nur die Lage; Yuen-Test und Zwei-Stichproben-Hodges–Lehmann sind nicht gebaut. | – |
| **Skala aus dem MAD** | Bei diskreten Daten ist sie oft 0, dann fällt Huber auf den Median zurück (E6). | Nutzer: Skala bekannt wählen |
| **Catoni nach der Definition aus dem Gedächtnis** | Die Originalarbeit wurde nicht eingesehen; die Garantie bei Niveau δ ist nicht gemessen. | Median-of-Means (Stück der Reihe) |
| **Durchschlagspunkte nur gemessen** | Die Lage der Klippen (bei ε > α, bei Hodges–Lehmann zwischen 29 und 30 %) ist eine Messung an vier Seeds, kein Satz aus der Literatur. | – |
| **Vier Seeds je Zelle** | Die Streuung der Tabellen ist die zwischen vier festen Seeds, kein Konfidenzintervall. | – |
| **Privates Mittel: n öffentlich, Ersetzen-Nachbarschaft, öffentliche Grenze** | Ist n geheim, ist C aus den Daten gewählt oder gibt es mehrere Veröffentlichungen, gilt die Garantie nicht (E8d); Gleitkomma und Zufallszahlen sind nicht gehärtet. | – |
| **Privates Mittel: Daten simuliert** | Aussagen gelten für Lognormal und Pareto dieser Demo, nicht für echte Aufträge. | – |
    """
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist eine Stichprobe $x_1, \dots, x_n$ von Lieferzeit-Abweichungen (Stunden gegenüber Plan), darunter $m = \lfloor \varepsilon n + 1/2 \rfloor$ Ausfälle. Gesucht ist ein Schätzer $T(x)$ der Lage des Kerns.

| Symbol | Bedeutung |
|---|---|
| $x_{(1)} \le \dots \le x_{(n)}$ | die sortierte Stichprobe |
| $\varepsilon$, $d$ | Anteil und Abstand der Ausfälle |
| $\alpha$, $k$ | Trimm-/Winsor-Anteil je Seite und die Zahl abgeschnittener Werte |
| $s$ | Skala (MAD/0,6745 oder bekannt) |
| $c$, $\psi_c$ | Huber-Schranke und Huber-Funktion |
| $\lambda$, $\delta$ | Catoni-Gewicht und -Niveau |
| $C$, $\varepsilon_{\mathrm{DP}}$ | Kappungsgrenze und Datenschutz-Budget |

**Abgeschnittene Werte.** Mit einer Toleranz gegen Gleitkomma-Kanten:

$$k = \left\lfloor \alpha n + 10^{-9} \right\rfloor.$$

**Getrimmtes und winsorisiertes Mittel.**

$$T_{\mathrm{trim}} = \frac{1}{n - 2k}\sum_{i=k+1}^{n-k} x_{(i)}.$$

$$T_{\mathrm{wins}} = \frac{1}{n}\left(k\,x_{(k+1)} + \sum_{i=k+1}^{n-k} x_{(i)} + k\,x_{(n-k)}\right).$$

**Skala.**

$$s = \frac{\operatorname{median}_i \left|x_i - \operatorname{median}(x)\right|}{0{,}6745}.$$

**Huber.** $\psi_c(r) = \max(-c, \min(c, r))$; der Schätzer ist die Nullstelle

$$\sum_{i=1}^{n} \psi_c\!\left(\frac{x_i - \mu}{s}\right) = 0,$$

gerechnet als gewichtetes Mittel mit Gewichten $w_i = \min(1, c / |r_i|)$, $r_i = (x_i - \mu)/s$, ab dem Median iteriert. Ist $s = 0$, gilt $\mu = \operatorname{median}(x)$.

**Hodges–Lehmann.**

$$T_{\mathrm{HL}} = \operatorname{median}\left\{ \frac{x_i + x_j}{2} : 1 \le i \le j \le n \right\}.$$

**Catoni** (Definition aus dem Gedächtnis, Quelle nicht eingesehen). Für $t \ge 0$ ist $\psi(t) = \log(1 + t + t^2/2)$, für $t < 0$ gilt $\psi(t) = -\psi(-t)$. Der Schätzer ist die Nullstelle

$$\sum_{i=1}^{n} \psi\big(\alpha_{\mathrm{C}}\,(x_i - \theta)\big) = 0 \quad\text{mit}\quad \alpha_{\mathrm{C}} = \lambda\sqrt{\frac{2L}{n\,v\left(1 + \frac{2L}{n - 2L}\right)}}, \quad L = \log\frac{1}{\delta},$$

wobei $v$ die Varianz ist (hier $s^2$ aus dem MAD) und $\lambda = 1$ den Wert der Definition bedeutet. Es gilt $n > 2L$.

**Effizienz und Verschiebung.** Auf sauberen Normaldaten und bei Ausfällen (Ziel 0):

$$\mathrm{Eff}(T) = \frac{\operatorname{MSE}(\bar x)}{\operatorname{MSE}(T)}, \qquad \mathrm{Verschiebung}(T) = \mathbb{E}\,[T].$$

**Mittel bei Ausfällen.** Mit $m$ einseitigen Ausfällen im Abstand $d$ ist die Verschiebung des Mittels

$$\mathbb{E}\,[\bar x] = \frac{m}{n}\,d = \varepsilon\,d.$$

**Privates Mittel.** Kappen auf $[0, C]$ und Laplace-Rauschen:

$$\hat\mu = \frac{1}{n}\sum_{i=1}^{n} \min(x_i, C) + \mathrm{Lap}(b), \qquad b = \frac{C}{n\,\varepsilon_{\mathrm{DP}}}.$$

**Differentielle Privatheit.** Für alle benachbarten $D, D'$ (ein Wert ersetzt) und alle Ausgaben $y$:

$$\log \frac{f(y \mid D)}{f(y \mid D')} \le \varepsilon_{\mathrm{DP}}.$$

**Empfindlichkeit.** Beim Kappen auf $[\mathrm{lo}, \mathrm{hi}]$ ändert ein ersetzter Wert das Mittel um höchstens

$$\Delta = \frac{\mathrm{hi} - \mathrm{lo}}{n},$$

also $C/n$ bei $[0, C]$ und $2C/n$ bei $[-C, C]$.

**Zerlegung des Fehlers.** Gegen das Populationsmittel $\mu$, mit $X_C = \min(X, C)$:

$$\operatorname{MSE}(C) = \big(\mathbb{E}\,X_C - \mu\big)^2 + \frac{\operatorname{Var}\,X_C}{n} + 2\,b^2.$$

**Skalierung der besten Grenze.** Bei Pareto-Rändern mit Index $\alpha_P$ ist die Verzerrung $\sim C^{1-\alpha_P}$ und das Rauschen $\sim C/(n\,\varepsilon_{\mathrm{DP}})$; das Gleichgewicht gibt

$$C^{*} \sim (n\,\varepsilon_{\mathrm{DP}})^{1/\alpha_P}.$$

**Wird $C$ aus den Daten gewählt**, hängt die Skala $b(D)$ von den Daten ab, und an der Stelle $y$ ist

$$\log \frac{f(y \mid D)}{f(y \mid D')} = \log\frac{b(D')}{b(D)} + \frac{|y - m(D')|}{b(D')} - \frac{|y - m(D)|}{b(D)}$$

unbeschränkt in $y$ (E8d).

**Literatur.** Keine Quelle wurde gelesen: Hodges–Lehmann, Huber, Catoni (2012) und die Arbeiten zur differentiellen Privatheit sind **nicht eingesehen**. Alle Aussagen dieser Demo beruhen auf eigener Rechnung und auf Messungen; die Schätzer sind gegen
`scipy` (`trim_mean`, `winsorize`, `brentq`, `minimize_scalar`, `quad`), gegen die Aufzählung aller Paarmittel und aller benachbarten Datensätze geprüft.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
