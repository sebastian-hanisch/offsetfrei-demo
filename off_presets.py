"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button: Schnellstart in der Seitenleiste, das aktive Preset ist hervorgehoben.

Ein Klick auf ein Preset setzt die Regler des Presets und alle übrigen auf ihren Standardwert (`apply_preset`). Die allgemeinen Regler (Strecke, R, d, Sollwert, Pol, Messbias, Stellgrenze, Zufalls-Seed) stehen in der Seitenleiste; die sechs Regler der Experimente
(wahre Verstärkung, Zufallsstrecken, falsch gesetzte Konstante, Integralgewicht, Rampe, Frequenz) stehen im Hauptbereich beim Experiment. Alle zusammen stehen in
SETTING_SPECS: Ändert man einen von ihnen, ist kein Preset mehr hervorgehoben (tests/test_layout.py prüft das für jeden)."""

import math
import random
from dataclasses import dataclass

import streamlit as st

import off_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


def _umax(text):
    return C.UMAX_AUS if str(text) == C.UMAX_AUS else float(text)


SETTING_SPECS = {
    "strecke_select": SettingSpec("strecke", str, C.DEFAULT_STRECKE, choices=C.STRECKEN),
    "R_select": SettingSpec("R", float, C.DEFAULT_R, choices=C.R_OPTIONS),
    "d_slider": SettingSpec("d", float, C.DEFAULT_D, C.D_MIN, C.D_MAX),
    "r_slider": SettingSpec("r", float, C.DEFAULT_SOLL, C.SOLL_MIN, C.SOLL_MAX),
    "rho_slider": SettingSpec("rho", float, C.DEFAULT_RHO, C.RHO_MIN, C.RHO_MAX),
    "b_slider": SettingSpec("b", float, C.DEFAULT_B, C.B_MIN, C.B_MAX),
    "umax_select": SettingSpec("umax", _umax, C.DEFAULT_UMAX, choices=C.UMAX_OPTIONS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "g_slider": SettingSpec("g", float, C.DEFAULT_G, C.G_MIN, C.G_MAX),
    "delta_select": SettingSpec("delta", float, C.DEFAULT_DELTA, choices=C.DELTA_OPTIONS),
    "dcal_slider": SettingSpec("dcal", float, C.DEFAULT_DCAL, C.DCAL_MIN, C.DCAL_MAX),
    "qi_select": SettingSpec("qi", float, C.DEFAULT_QI, choices=C.QI_OPTIONS),
    "rampe_slider": SettingSpec("s", float, C.DEFAULT_RAMPE, C.RAMPE_MIN, C.RAMPE_MAX),
    "omega_select": SettingSpec("omega", float, C.DEFAULT_OMEGA, choices=C.OMEGA_OPTIONS),
}
PRESET_KEYS = {"strecke": "strecke_select", "R": "R_select", "d": "d_slider", "r": "r_slider", "rho": "rho_slider", "b": "b_slider", "umax": "umax_select"}
SIDEBAR_KEYS = ("strecke_select", "R_select", "d_slider", "r_slider", "rho_slider", "b_slider", "umax_select", "seed_input")


def _same(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return math.isclose(float(a), float(b), abs_tol=1e-9)


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clamp_value(spec: SettingSpec, value):
    """Wert auf die Grenzen bzw. die Auswahl des Reglers begrenzen; None, wenn er nicht übernommen werden kann (Auswahl ohne Treffer)."""
    if spec.choices is not None:
        return next((c for c in spec.choices if _same(value, c)), None)
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            value = clamp_value(spec, value)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Das Szenario laden: die Regler des Presets bekommen seine Werte, alle übrigen (Zufalls-Seed und die Regler der Experimente) ihren Standardwert."""
    key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for state_key, spec in SETTING_SPECS.items():
        key = key_of.get(state_key)
        st.session_state[state_key] = C.PRESETS[name][key] if key in C.PRESETS[name] else spec.default


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS, dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(_same(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default)) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
