"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Regler an seine Grenzen und prüft,
dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key= können zufällig identischen Inhalt rendern und
kollidieren) und `StreamlitAPIException` bei Reglern mit berechneten Grenzen (min == max)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import thu_constants as C
import thu_kappen as KP

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    at = _fresh_app()
    assert len(at.get("plotly_chart")) >= 14


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("key,options", [
    ("dist_select", C.DISTRIBUTIONS), ("side_select", C.SIDES), ("scale_select", C.SCALES),
])
def test_every_option_of_the_selectboxes_does_not_raise(key, options):
    for value in options:
        at = _fresh_app()
        at.sidebar.selectbox(key=key).set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (key, value, [str(e) for e in at.exception])


@pytest.mark.parametrize("key,options", [
    ("d_select", C.D_OPTIONS), ("c_select", C.HUBER_C_OPTIONS), ("mult_select", C.CATONI_MULT_OPTIONS), ("delta_select", C.DELTA_OPTIONS),
])
def test_every_option_of_the_select_sliders_does_not_raise(key, options):
    for value in (options[0], options[-1]):
        at = _fresh_app()
        at.sidebar.select_slider(key=key).set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_corner_combinations_of_all_regler_do_not_raise():
    """Alle Minima und alle Maxima zugleich, dazu das kleinste n mit dem kleinsten Niveau und der größten Catoni-Skala sowie das größte n mit Skala 0."""
    corners = [
        {"dist_select": C.DISTRIBUTIONS[0], "n_slider": C.N_MIN, "eps_slider": C.EPS_MIN, "d_select": C.D_OPTIONS[0], "alpha_slider": C.ALPHA_MIN, "c_select": C.HUBER_C_OPTIONS[0],
         "mult_select": C.CATONI_MULT_OPTIONS[0], "delta_select": C.DELTA_OPTIONS[0]},
        {"dist_select": C.DISTRIBUTIONS[4], "n_slider": C.N_MAX, "eps_slider": C.EPS_MAX, "d_select": C.D_OPTIONS[-1], "alpha_slider": C.ALPHA_MAX, "c_select": C.HUBER_C_OPTIONS[-1],
         "mult_select": C.CATONI_MULT_OPTIONS[-1], "delta_select": C.DELTA_OPTIONS[-1], "side_select": C.SIDES[1]},
        {"n_slider": C.N_MIN, "delta_select": 0.01, "eps_slider": C.EPS_MAX, "alpha_slider": C.ALPHA_MAX},
        {"dist_select": C.DISTRIBUTIONS[-1], "n_slider": 1000, "eps_slider": 0, "scale_select": C.SCALES[1]},
    ]
    for corner in corners:
        at = _fresh_app()
        for key, value in corner.items():
            widget = at.sidebar.slider(key=key) if key in ("n_slider", "eps_slider", "alpha_slider") else at.sidebar.select_slider(key=key) if key in ("d_select", "c_select", "mult_select", "delta_select") else at.sidebar.selectbox(key=key)
            widget.set_value(value)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (corner, [str(e) for e in at.exception])


def test_the_experiment_selectboxes_do_not_raise():
    at = _fresh_app()
    for dist in C.SKEW_DISTS:
        at.selectbox(key="e4_dist").set_value(dist).run(timeout=TIMEOUT)
        assert not at.exception, (dist, [str(e) for e in at.exception])
    for name in ("Lognormal (σ = 1)", "Pareto (α = 1,5)", "Pareto (α = 3)"):
        at.selectbox(key="kp_spec").set_value(name).run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])


def test_kappen_section_every_n_and_budget_and_the_c_ends():
    for n in C.KAPPEN_N:
        for eps in (C.KAPPEN_EPS[0], C.KAPPEN_EPS[-1]):
            at = _fresh_app()
            at.radio(key="kp_n").set_value(n)
            at.radio(key="kp_eps").set_value(eps).run(timeout=TIMEOUT)
            assert not at.exception, (n, eps, [str(e) for e in at.exception])
            grid = KP.c_grid()
            for value in (round(grid[0], 2), round(grid[-1], 2)):
                at.select_slider(key="kp_c").set_value(value).run(timeout=TIMEOUT)
                assert not at.exception, (n, eps, value, [str(e) for e in at.exception])


def test_experiment_radios_every_option():
    for key in ("e1_n", "e1_j", "e2_n", "e3_eps", "e4_n", "e5_n", "sc_eps", "e8c_sel"):
        at = _fresh_app()
        for option in list(at.radio(key=key).options):
            at.radio(key=key).set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (key, option, [str(e) for e in at.exception])


def test_e2_live_button_and_pdf_do_not_raise_and_show_a_chart():
    at = _fresh_app()
    live = next(b for b in at.button if b.key == "e2_live")
    live.click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Nachgemessen in" in c.value for c in at.caption)
    at = _fresh_app()
    next(b for b in at.button if b.key == "pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["pdf"][1][:5] == b"%PDF-"
