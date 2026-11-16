"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

from off_presets import SETTING_SPECS

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


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


# ---- Ergänzungen der Demo: Auswahlfelder, Eckfälle, Permalink -----------------------------------------------------------------------------

def test_every_select_at_its_first_and_last_option_does_not_raise():
    at0 = _fresh_app()
    arten = {"selectbox": at0.selectbox, "select_slider": at0.select_slider}
    for art, widgets in arten.items():
        for w in widgets:
            optionen = list(SETTING_SPECS[w.key].choices)                          # die Rohwerte (w.options enthält die formatierten Texte)
            for pick in (optionen[0], optionen[-1]):
                at = _fresh_app()
                getattr(at, art)(key=w.key).set_value(pick).run(timeout=TIMEOUT)
                assert not at.exception, (w.key, pick, [str(e) for e in at.exception])


def test_every_strecke_runs():
    from off_constants import STRECKEN

    for kind in STRECKEN:
        at = _fresh_app()
        at.selectbox(key="strecke_select").set_value(kind).run(timeout=TIMEOUT)
        assert not at.exception, (kind, [str(e) for e in at.exception])
        assert len(at.warning) + len(at.info) + len(at.success) >= 3             # die drei Kernfragen-Verdikte stehen da


def _alle(at, rand):
    for s in at.slider:
        v = s.min if rand == "min" else s.max
        s.set_value(int(v) if isinstance(s.value, int) else v)
    for w in list(at.select_slider) + list(at.selectbox):
        o = list(SETTING_SPECS[w.key].choices)
        w.set_value(o[0] if rand == "min" else o[-1])
    for n in at.number_input:
        n.set_value(n.min if rand == "min" else n.max)
    return at


@pytest.mark.parametrize("rand", ["min", "max"])
def test_all_controls_at_min_or_all_at_max_does_not_raise(rand):
    at = _fresh_app()
    _alle(at, rand).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_permalink_is_clamped_and_applied():
    at = AppTest.from_file(APP_PATH)
    at.query_params["d"] = "99"
    at.query_params["R"] = "0.1"
    at.query_params["strecke"] = "skalar1"
    at.query_params["umax"] = "1.5"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="d_slider").value == 2.0 and at.select_slider(key="R_select").value == 0.1
    assert at.selectbox(key="strecke_select").value == "skalar1" and at.select_slider(key="umax_select").value == 1.5


def test_permalink_with_invalid_values_falls_back_to_defaults():
    at = AppTest.from_file(APP_PATH)
    at.query_params["d"] = "nan"
    at.query_params["R"] = "0.1234"
    at.query_params["umax"] = "viel"
    at.query_params["strecke"] = "kugel"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="d_slider").value == 1.0 and at.select_slider(key="R_select").value == 1.0
    assert at.select_slider(key="umax_select").value == "aus" and at.selectbox(key="strecke_select").value == "wagen"


def test_main_texts_name_the_price_next_to_the_removal():
    """Der Preis steht in derselben Zeile wie die Beseitigung (Z2): im Standard-Lauf nennt die Verdikt-Zeile der Kernfrage 2 beides."""
    at = _fresh_app()
    texte = [w.value for w in at.warning] + [w.value for w in at.info]
    zeile = next(t for t in texte if t.startswith("Das Störungsmodell beseitigt den Dauerfehler"))
    assert "9,02" in zeile and "2,38" in zeile and "kostet" in zeile
