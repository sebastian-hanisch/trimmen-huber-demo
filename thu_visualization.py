"""Plotly-Figuren: alle Achsen fest (fixedrange), Beschriftungen als Annotationen, deutsche Achsentexte."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import thu_constants as C
import thu_robust as R

HEIGHT = 420


def _color(spec: str) -> str:
    return C.SPEC_COLORS[R.parse_spec(spec)[0]]


def _dash(spec: str) -> str:
    """Mehrere Kurven derselben Familie unterscheiden sich im Strich."""
    kind, p = R.parse_spec(spec)
    if kind in ("trim", "winsor"):
        return {0.1: "solid", 0.25: "dash", 0.4: "dot"}.get(p, "solid")
    if kind == "huber":
        return {0.5: "dash", 1.345: "solid", 3.0: "dot"}.get(p, "solid")
    if kind == "catoni":
        return "solid" if p == 1.0 else "dash"
    return "solid"


def _finish(fig: go.Figure, x_title: str, y_title: str, *, log_x: bool = False, log_y: bool = False, height: int = HEIGHT, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=30, b=60), showlegend=legend, legend=dict(orientation="h", y=-0.22), hovermode="closest")
    fig.update_xaxes(title=x_title, fixedrange=True, type="log" if log_x else None)       # None: automatisch (Balken haben Text auf der x-Achse)
    fig.update_yaxes(title=y_title, fixedrange=True, type="log" if log_y else None)
    return fig


def _pct_tick(eps_list) -> list[float]:
    return [100 * e for e in eps_list]


# --- Hauptansicht: eine Stichprobe und die Lagen aller Schätzer ----------------------------------------------------------------------------

def _view_range(x: np.ndarray, flag: np.ndarray, values: list[float]) -> tuple[float, float]:
    """Sichtbereich: die Nicht-Ausfälle, die Schätzer und Ausfälle, die höchstens drei Spannweiten des Kerns entfernt liegen; alles Weitere wird am Rand als Pfeil angedeutet."""
    core = x[~flag] if (~flag).any() else x
    lo, hi = float(core.min()), float(core.max())
    span = max(hi - lo, 1.0)
    for v in list(values) + [float(v) for v in x[flag]]:
        if lo - 3.0 * span <= v <= hi + 3.0 * span:
            lo, hi = min(lo, v), max(hi, v)
    pad = 0.08 * (hi - lo) + 0.2
    return lo - pad, hi + pad


def build_sample_chart(x, flag, labels: list[str], specs: list[str], values: list[float], target_mean: float, target_median: float) -> go.Figure:
    x = np.asarray(x, dtype=np.float64)
    flag = np.asarray(flag, dtype=bool)
    lo, hi = _view_range(x, flag, values)
    rows = len(specs)
    jitter = ((np.arange(len(x)) * 0.6180339887) % 1.0) * 0.8 - 0.4
    fig = go.Figure()
    for name, mask, color in (("Sendungen", ~flag, C.COLOR_POINT), ("Ausfälle", flag, C.COLOR_OUTLIER)):
        xv = x[mask]
        inside = (xv >= lo) & (xv <= hi)
        fig.add_trace(go.Scatter(x=xv[inside], y=(rows + 0.9 + jitter[mask][inside] * 0.5), mode="markers", name=name, marker=dict(color=color, size=6), hovertemplate="%{x:.3f}<extra>" + name + "</extra>"))
        if (~inside).any():
            fig.add_trace(go.Scatter(x=np.where(xv[~inside] > hi, hi, lo), y=(rows + 0.9 + jitter[mask][~inside] * 0.5), mode="markers", name=f"{name} außerhalb des Bildes",
                                     marker=dict(color=color, size=8, symbol="arrow-right" if (xv[~inside] > hi).any() else "arrow-left"),
                                     text=[f"{v:.3g}" for v in xv[~inside]], hovertemplate="%{text}<extra>außerhalb des Bildes</extra>"))
    for i, (label, spec, v) in enumerate(zip(labels, specs, values)):
        inside = lo <= v <= hi
        fig.add_trace(go.Scatter(x=[v if inside else (hi if v > hi else lo)], y=[rows - i], mode="markers+text", name=label, showlegend=False,
                                 marker=dict(color=_color(spec), size=13, symbol="diamond" if inside else ("arrow-right" if v > hi else "arrow-left")),
                                 text=[f"{label}: {v:.3g}"], textposition="middle left" if (v - lo) > 0.6 * (hi - lo) else "middle right", hovertemplate=f"{label}: {v:.4f}<extra></extra>"))
    for value, name, dash in ((target_median, "Median des Kerns", "dash"), (target_mean, "Mittelwert des Kerns", "dot")):
        if lo <= value <= hi and (name == "Median des Kerns" or abs(target_mean - target_median) > 1e-9):
            fig.add_vline(x=value, line=dict(color=C.COLOR_TARGET, dash=dash, width=1.5))
            fig.add_annotation(x=value, y=0.35, text=name, showarrow=False, font=dict(size=10, color=C.COLOR_TARGET), yanchor="top")
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=[0.0, rows + 1.6], showticklabels=False, showgrid=False)
    return _finish(fig, "Abweichung vom Plan (Stunden)", "", height=380)


# --- E1 Effizienz gegen Verzerrung -----------------------------------------------------------------------------------------------------------

def build_pareto_chart(cell: dict, contamination_index: int, chosen: dict[str, str]) -> go.Figure:
    """x: Effizienz auf sauberen Daten (Mittel = 1), y: Verschiebung bei Ausfällen; je Familie eine Linie über ihren Regler, die gewählten Reglerstellungen groß."""
    rows = {r["spec"]: r for r in cell["rows"]}
    fig = go.Figure()
    for kind, name in (("trim", "Trimmen"), ("winsor", "Winsorisieren"), ("huber", "Huber"), ("catoni", "Catoni")):
        specs = [s for s in rows if R.parse_spec(s)[0] == kind]
        fig.add_trace(go.Scatter(x=[rows[s]["eff"][0] for s in specs], y=[rows[s]["shifts"][contamination_index][0] for s in specs], mode="lines+markers", name=name,
                                 line=dict(color=C.SPEC_COLORS[kind]), text=[R.label(s) for s in specs], hovertemplate="%{text}<br>Effizienz %{x:.3f}<br>Verschiebung %{y:.3f}<extra></extra>"))
    for spec in ("mean", "median", "hl"):
        fig.add_trace(go.Scatter(x=[rows[spec]["eff"][0]], y=[rows[spec]["shifts"][contamination_index][0]], mode="markers+text", name=R.label(spec), text=[R.label(spec)],
                                 textposition="top center", marker=dict(color=_color(spec), size=12, symbol="diamond"), hovertemplate=f"{R.label(spec)}<br>Effizienz %{{x:.3f}}<br>Verschiebung %{{y:.3f}}<extra></extra>"))
    for spec in chosen.values():
        if spec in rows:
            fig.add_trace(go.Scatter(x=[rows[spec]["eff"][0]], y=[rows[spec]["shifts"][contamination_index][0]], mode="markers", name="gewählt: " + R.label(spec), showlegend=False,
                                     marker=dict(color=_color(spec), size=17, symbol="circle-open", line=dict(width=3))))
    return _finish(fig, "Effizienz auf sauberen Daten (Mittel = 1, größer ist besser)", "Verschiebung bei Ausfällen (kleiner ist besser)")


# --- E2 Verschiebung über dem Anteil, E3 über dem Abstand --------------------------------------------------------------------------------

def build_shift_chart(cell: dict, d: float, specs: list[str] | None = None, key: str = "shift") -> go.Figure:
    specs = specs or [r["spec"] for r in cell["rows"]]
    rows = {r["spec"]: r for r in cell["rows"]}
    a = cell["d"].index(d)
    fig = go.Figure()
    keep = [i for i, e in enumerate(cell["eps"]) if e > 0]            # ε = 0 hat Verschiebung 0 und gehört nicht auf die logarithmische Achse
    for spec in specs:
        ys = [max(rows[spec][key][a][i][0], 1e-3) for i in keep]
        fig.add_trace(go.Scatter(x=[100 * cell["eps"][i] for i in keep], y=ys, mode="lines+markers", name=R.label(spec), line=dict(color=_color(spec), dash=_dash(spec)), marker=dict(size=5),
                                 hovertemplate=f"{R.label(spec)}<br>ε = %{{x:.0f}} %<br>%{{y:.3f}}<extra></extra>"))
    return _finish(fig, "Anteil ε der Ausfälle (%)", "Verschiebung des Schätzers (Stunden, logarithmisch)" if key == "shift" else "RMSE um 0 (Stunden, logarithmisch)", log_y=True)


def build_distance_chart(cell: dict, eps: float, specs: list[str] | None = None) -> go.Figure:
    specs = specs or [r["spec"] for r in cell["rows"]]
    rows = {r["spec"]: r for r in cell["rows"]}
    j = cell["eps"].index(eps)
    fig = go.Figure()
    for spec in specs:
        ys = [max(rows[spec]["shift"][a][j][0], 1e-3) for a in range(len(cell["d"]))]
        fig.add_trace(go.Scatter(x=cell["d"], y=ys, mode="lines+markers", name=R.label(spec), line=dict(color=_color(spec), dash=_dash(spec)), marker=dict(size=5),
                                 hovertemplate=f"{R.label(spec)}<br>d = %{{x:g}}<br>%{{y:.3f}}<extra></extra>"))
    return _finish(fig, "Abstand d der Ausfälle (Stunden, logarithmisch)", "Verschiebung des Schätzers (Stunden, logarithmisch)", log_x=True, log_y=True)


def build_fine_chart(cell: dict, specs: list[str], d: float = 1000.0) -> go.Figure:
    rows = {r["spec"]: r for r in cell["rows"]}
    a = cell["d"].index(d)
    fig = go.Figure()
    for spec in specs:
        fig.add_trace(go.Scatter(x=_pct_tick(cell["eps"]), y=[max(v[0], 1e-3) for v in rows[spec]["shift"][a]], mode="lines+markers", name=R.label(spec),
                                 line=dict(color=_color(spec), dash=_dash(spec)), marker=dict(size=4)))
    return _finish(fig, "Anteil ε der Ausfälle (%)", "Verschiebung (Stunden, logarithmisch)", log_y=True)


# --- E4 Zielgröße, E5 Ränder -----------------------------------------------------------------------------------------------------------------

def build_target_chart(cell: dict, population: dict) -> go.Figure:
    """Je Schätzer: Lage an der Population (Strich) gegen Mittelwert und Median der Verteilung, dazu die Streuung SD(T) als Balken."""
    specs = [r["spec"] for r in cell["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[R.label(s) for s in specs], y=[population[s][0] for s in specs], name="Lage an der Population (n → ∞)", marker_color=C.COLOR_LINE))
    fig.add_trace(go.Bar(x=[R.label(s) for s in specs], y=[r["sd"][0] for r in cell["rows"]], name=f"Streuung SD(T) bei n = {cell['n']}", marker_color=C.COLOR_LINE2))
    fig.add_hline(y=cell["mean"], line=dict(color="#7f7f7f", dash="dot"), annotation_text="Mittelwert der Verteilung", annotation_position="top left")
    fig.add_hline(y=cell["median"], line=dict(color=C.COLOR_TARGET, dash="dash"), annotation_text="Median der Verteilung", annotation_position="bottom left")
    fig.update_layout(barmode="group")
    return _finish(fig, "", "Stunden")


def build_rmse_by_n_chart(dist: str, ns: list[int], series: dict[str, list[float]]) -> go.Figure:
    """RMSE gegen den Mittelwert der Verteilung über n (log-log): das Mittel gewinnt, wo seine Linie unter den anderen liegt."""
    fig = go.Figure()
    for spec, ys in series.items():
        fig.add_trace(go.Scatter(x=ns, y=ys, mode="lines+markers", name=R.label(spec), line=dict(color=_color(spec), dash=_dash(spec)), marker=dict(size=5)))
    return _finish(fig, "Stichprobenumfang n (logarithmisch)", "RMSE gegen den Mittelwert (logarithmisch)", log_x=True, log_y=True)


def build_tail_chart(cells: list[dict], n: int, specs: list[str]) -> go.Figure:
    fig = go.Figure()
    for spec in specs:
        ys = [next(r["ratio"][0] for r in c["rows"] if r["spec"] == spec) for c in cells]
        fig.add_trace(go.Bar(x=[c["dist"] for c in cells], y=ys, name=R.label(spec), marker_color=_color(spec)))
    fig.add_hline(y=1.0, line=dict(color="#555", dash="dot"), annotation_text="Mittel = 1", annotation_position="top left")
    fig.update_layout(barmode="group")
    return _finish(fig, f"Verteilung der sauberen Daten (n = {n})", "RMSE geteilt durch RMSE des Mittels (unter 1: besser)")


# --- E6 Fallen, E7 Catoni --------------------------------------------------------------------------------------------------------------------

def build_stairs_chart(ns: list[int], alphas: list[float]) -> go.Figure:
    fig = go.Figure()
    for n in ns:
        fig.add_trace(go.Scatter(x=[100 * a for a in alphas], y=[R.n_cut(n, a) for a in alphas], mode="lines+markers", line_shape="hv", name=f"n = {n}"))
    return _finish(fig, "Trimm-Anteil α je Seite (%)", "Tatsächlich abgeschnittene Werte je Seite")


def build_catoni_distance_chart(cells: list[dict]) -> go.Figure:
    """Verschiebung über dem Abstand für Huber und Catoni (ε = 10 %), eine Linie je n und Schätzer."""
    fig = go.Figure()
    for cell in cells:
        for r in cell["rows"]:
            spec = r["spec"]
            fig.add_trace(go.Scatter(x=cell["d"], y=[max(row[0][0], 1e-3) for row in r["shift"]], mode="lines+markers", name=f"{R.label(spec)}, n = {cell['n']}",
                                     line=dict(color=_color(spec), dash="solid" if cell["n"] == 100 else ("dot" if cell["n"] == 20 else "dash")), marker=dict(size=4),
                                     visible=True if cell["n"] == 100 else "legendonly"))
    return _finish(fig, "Abstand d der Ausfälle (Stunden, logarithmisch)", "Verschiebung (Stunden, logarithmisch)", log_x=True, log_y=True)


# --- E8 Kappen und privates Mittel -----------------------------------------------------------------------------------------------------------

def build_ucurve_chart(cell_eps: dict, cgrid: list[float], mse_parts: tuple, sim_rmse=None, chosen_c: float | None = None) -> go.Figure:
    """U-Kurve des privaten Mittels: RMSE aus der geschlossenen Zerlegung (Linie), die Anteile Verzerrung, Stichprobe und Rauschen (gestrichelt), Simulation (Punkte).
    Anteile unter einem Fünfzigstel des kleinsten RMSE werden nicht gezeichnet (die Verzerrung geht bei großem C gegen 0 und würde die logarithmische Achse sprengen)."""
    mse, bias2, samp, noise = mse_parts
    cg = [float(c) for c in cgrid]
    rmse = np.sqrt(np.asarray(mse))
    floor = float(rmse.min()) / 50.0
    fig = go.Figure()
    for name, vals, color in (("Verzerrung (Kappung)", bias2, "#4c78a8"), ("Stichprobenfehler", samp, "#2F6B65"), ("Rauschen", noise, C.COLOR_OUTLIER)):
        root = np.sqrt(np.maximum(np.asarray(vals), 0.0))
        fig.add_trace(go.Scatter(x=cg, y=[float(v) if v >= floor else None for v in root], mode="lines", name=name + " (Wurzel des Anteils)", line=dict(color=color, dash="dash", width=1.5)))
    fig.add_trace(go.Scatter(x=cg, y=rmse, mode="lines", name="RMSE (Formel)", line=dict(color="#222", width=3)))
    if sim_rmse is not None:
        fig.add_trace(go.Scatter(x=cg, y=[v[0] for v in sim_rmse], error_y=dict(type="data", array=[v[1] for v in sim_rmse], visible=True), mode="markers", name="RMSE (Simulation)",
                                 marker=dict(color=C.COLOR_LINE2, size=7)))
    k = int(np.argmin(mse))
    fig.add_trace(go.Scatter(x=[cg[k]], y=[float(rmse[k])], mode="markers", name="Minimum des Rasters", marker=dict(symbol="star", size=15, color="#222")))
    top = float(rmse.max())
    if chosen_c is not None:
        fig.add_trace(go.Scatter(x=[chosen_c, chosen_c], y=[floor, top], mode="lines", name="gewähltes C", line=dict(color=C.COLOR_TARGET, dash="dot", width=2)))
    fig = _finish(fig, "Kappungsgrenze C (logarithmisch)", "Fehler gegen das Populationsmittel (logarithmisch)", log_x=True, log_y=True)
    fig.update_yaxes(range=[math.log10(floor), math.log10(top) + 0.1])
    fig.update_xaxes(range=[math.log10(min(cg)) - 0.05, math.log10(max(cg)) + 0.05])
    return fig


def build_scaling_chart(rows: list[dict], specs: list[tuple], eps: float) -> go.Figure:
    fig = go.Figure()
    for spec in specs:
        sel = sorted((r for r in rows if tuple(r["spec"]) == tuple(spec) and r["eps"] == eps), key=lambda r: r["n"])
        name = f"Lognormal (σ = {spec[1]:g})" if spec[0] == "lognormal" else f"Pareto (α = {spec[1]:g})"
        fig.add_trace(go.Scatter(x=[r["n"] for r in sel], y=[r["c_star"] for r in sel], mode="lines+markers", name=name.replace(".", ","), marker=dict(size=5),
                                 line=dict(dash="dash" if spec[0] == "pareto" else "solid")))
    return _finish(fig, "Stichprobenumfang n (logarithmisch)", "Beste Kappungsgrenze C* (logarithmisch)", log_x=True, log_y=True)


def build_wrong_c_chart(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    factors = [r["factor"] for r in rows[0]["rows"]]
    for row in rows:
        name = (f"Lognormal (σ = {row['spec'][1]:g})" if row["spec"][0] == "lognormal" else f"Pareto (α = {row['spec'][1]:g})").replace(".", ",") + f", ε_DP = {row['eps']:g}".replace(".", ",")
        fig.add_trace(go.Bar(x=[f"C* · {f:g}".replace(".", ",") for f in factors], y=[r["rmse_ratio"] for r in row["rows"]], name=name))
    fig.update_layout(barmode="group")
    return _finish(fig, "Gewählte Kappungsgrenze", "RMSE geteilt durch den RMSE bei C*")


def build_enumeration_chart(data_c: list[dict]) -> go.Figure:
    ys = [r["y"] for r in data_c]
    fig = go.Figure(go.Scatter(x=ys, y=[r["worst"] for r in data_c], mode="lines+markers", name="C aus den Daten: größtes log-Verhältnis", marker=dict(size=8), line=dict(color=C.COLOR_OUTLIER)))
    fig.add_trace(go.Scatter(x=[min(ys), max(ys)], y=[1.0, 1.0], mode="lines", name="ε_DP = 1 (zugesagte Schranke)", line=dict(color=C.COLOR_TARGET, dash="dash")))
    return _finish(fig, "Ausgabe y (logarithmisch)", "größtes log-Verhältnis über alle benachbarten Paare (logarithmisch)", log_x=True, log_y=True)
