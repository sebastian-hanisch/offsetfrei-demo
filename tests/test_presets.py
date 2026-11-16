"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset ändert etwas; Permalink-Begrenzung."""

import math


import off_constants as C
import off_evaluation as E
import off_presets as P


def _settings(preset: dict) -> E.Settings:
    return E.Settings(preset["strecke"], preset["R"], preset["d"], preset["r"], preset["rho"], preset["b"], E.umax_from_choice(preset["umax"]))


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, p in C.PRESETS.items():
        assert p["strecke"] in C.STRECKEN and p["R"] in C.R_OPTIONS and p["umax"] in C.UMAX_OPTIONS, name
        assert C.D_MIN <= p["d"] <= C.D_MAX and C.SOLL_MIN <= p["r"] <= C.SOLL_MAX and C.RHO_MIN <= p["rho"] <= C.RHO_MAX and C.B_MIN <= p["b"] <= C.B_MAX, name


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert _settings(C.PRESETS[first]) == E.Settings()
    for key, state_key in P.PRESET_KEYS.items():
        assert C.PRESETS[first][key] == P.SETTING_SPECS[state_key].default


def test_setting_specs_cover_every_preset_key_and_every_regler():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)
    assert set(P.SIDEBAR_KEYS) <= set(P.SETTING_SPECS)
    urls = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(urls) == len(set(urls))                                    # jede Adressen-Variable nur einmal


def test_no_two_presets_are_identical():
    unterscheidbar = {tuple(sorted(p.items(), key=lambda kv: kv[0])) for p in C.PRESETS.values()}
    assert len(unterscheidbar) == len(C.PRESETS)


def test_each_preset_changes_something_against_the_default():
    """Ein Preset ohne Effekt wäre ein totes Szenario: gegen das Standard-Preset ändert jedes mindestens eine Einstellung."""
    first = next(iter(C.PRESETS))
    for name, p in list(C.PRESETS.items())[1:]:
        assert any(p[k] != C.PRESETS[first][k] for k in p), name


def test_permalink_clamping_and_choices():
    f = P.SETTING_SPECS["d_slider"]
    assert P.clamp_value(f, 99.0) == C.D_MAX and P.clamp_value(f, -99.0) == C.D_MIN
    r = P.SETTING_SPECS["R_select"]
    assert P.clamp_value(r, 0.1) == 0.1 and P.clamp_value(r, 0.12345) is None
    u = P.SETTING_SPECS["umax_select"]
    assert u.caster("aus") == "aus" and u.caster("1.5") == 1.5 and P.clamp_value(u, "aus") == "aus" and P.clamp_value(u, 1.5) == 1.5 and P.clamp_value(u, 7.0) is None
    assert P.SETTING_SPECS["strecke_select"].caster("wagen") == "wagen"


def test_umax_from_choice():
    assert E.umax_from_choice("aus") is None and E.umax_from_choice(1.5) == 1.5


def test_option_lists_have_no_duplicates_and_are_sorted():
    for opts in (C.R_OPTIONS, C.DELTA_OPTIONS, C.QI_OPTIONS, C.OMEGA_OPTIONS, C.UMAX_OPTIONS[1:]):
        assert list(opts) == sorted(set(opts))
    assert all(math.isfinite(v) for v in C.R_OPTIONS)
