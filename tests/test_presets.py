"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und die Hilfetexte beschreiben, was das Preset wirklich tut."""

import thu_constants as C
import thu_evaluation as ev
import thu_presets as P
import thu_robust as R


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds_and_options():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.EPS_MIN <= preset["eps"] <= C.EPS_MAX and C.ALPHA_MIN <= preset["alpha"] <= C.ALPHA_MAX and 0 <= preset["seed"] <= C.SEED_MAX
        assert preset["d"] in C.D_OPTIONS and preset["c"] in C.HUBER_C_OPTIONS and preset["mult"] in C.CATONI_MULT_OPTIONS and preset["delta"] in C.DELTA_OPTIONS
        assert preset["side"] in C.SIDES and preset["scale"] in C.SCALES and preset["dist"] in C.DISTRIBUTIONS


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["10 % Ausfall (Standard)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert {key: P.SETTING_SPECS[state].default for key, state in P.PRESET_KEYS.items()} == dict(C._BASE)


def test_every_preset_runs_through_the_analysis():
    for name, preset in C.PRESETS.items():
        a = ev.analyse(ev.Settings(**preset))
        assert len(a.values) == 7 and all(v == v for v in a.values), name


def test_the_help_texts_match_what_the_presets_do():
    s = {name: ev.Settings(**p) for name, p in C.PRESETS.items()}
    assert s["Sauber"].eps == 0
    over = s["Anteil über der Trimmgrenze"]
    assert over.eps == 15 and over.alpha == 10 and over.eps > over.alpha and over.d == 1000.0 and "15 Ausfälle im Abstand 1000" in C.PRESET_HELP["Anteil über der Trimmgrenze"]
    assert s["Hodges–Lehmann an der Grenze"].eps == 30 and s["Hodges–Lehmann an der Grenze"].d == 1000.0 and "30 Ausfälle im Abstand 1000" in C.PRESET_HELP["Hodges–Lehmann an der Grenze"]
    assert s["Schiefe Lieferzeiten"].dist == "Lognormal (σ = 1)" and s["Schiefe Lieferzeiten"].eps == 0
    small = s["Kleines n (Treppe)"]
    assert small.n == 10 and small.alpha == 5 and R.n_cut(small.n, small.alpha / 100) == 0 and "α = 5 %" in C.PRESET_HELP["Kleines n (Treppe)"]
    std = s["10 % Ausfall (Standard)"]
    assert std.n == 100 and std.eps == 10 and std.d == 5.0 and "10 davon" in C.PRESET_HELP["10 % Ausfall (Standard)"]


def test_the_discrete_preset_has_scale_zero_and_huber_equals_the_median():
    a = ev.analyse(ev.Settings(**C.PRESETS["Diskrete Tage (Skala 0)"]))
    assert a.mad == 0.0 and a.values[4] == a.values[1]                 # Huber = Median
    assert a.values[0] > a.values[1]                                   # das Mittel liegt über dem Median


def test_the_clip_presets_behave_as_described():
    a = ev.analyse(ev.Settings(**C.PRESETS["Anteil über der Trimmgrenze"]))
    mean_, median_, trim, wins, huber, hl, _ = a.values
    assert trim > 50 and wins > 50 and mean_ > 100 and abs(median_) < 1 and abs(huber) < 1                  # Trimmen 10 % schneidet 10 von 15 Ausfällen ab, 5 bleiben (Verschiebung 50)
    b = ev.analyse(ev.Settings(**C.PRESETS["Hodges–Lehmann an der Grenze"]))
    assert b.values[5] > 100 and abs(b.values[4]) < 2 and abs(b.values[1]) < 2
