"""Offsetfrei regeln: die Störung mitschätzen und was es kostet - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Wagen (Doppelintegrator) und ein Skalar als Handbeispiel, eine konstante Störkraft am Eingang, eine Zustandsrückführung aus der Riccati-Gleichung (LQR) und ein
Störungsmodell mit Beobachter. Primäransicht: der Störsprung mit und ohne Störungsmodell; dazu fünf Experimente (Dauerfehler und Sollwert, Beobachterpole,
Modellfehler, zwei Wege, Grenzen). Aufbau nach DEMO-PLAYBOOK §2.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import off_constants as C
import off_dynamics as K
import off_evaluation as E
from off_layout import dunkles_design, kopf_css
from off_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from off_visualization import (
    build_beobachter,
    build_dauerfehler_R,
    build_frequenzgang,
    build_gebiet,
    build_ort,
    build_rampe,
    build_rausch,
    build_schaetzung,
    build_stellgrenze,
    build_zwei_wege,
)

st.set_page_config(page_title="Offsetfrei regeln – Sebastian Hanisch", layout="wide")

NEIGHBOUR = "https://github.com/sebastian-hanisch"
WAGEN_HINWEIS = "Dieses Experiment rechnet immer am Wagen (mit R und ρ der Seitenleiste), auch wenn dort der Skalar gewählt ist."


def _delta(x: float) -> str:
    """Differenz mit Dezimalkomma und ASCII-Minus (Streamlit liest nur dieses als negativ und wählt Pfeil und Farbe danach)."""
    return E.de(x, 3, sign=True).replace("−", "-")


@st.cache_data(show_spinner=False)
def _ergebnis(settings):
    return E.ergebnis(settings)


@st.cache_data(show_spinner=False)
def _trennung(kind, R, rho):
    return E.trennung(K.entwurf(kind, R, rho))


@st.cache_data(show_spinner=False)
def _hand():
    return E.hand_skalar()


@st.cache_data(show_spinner=False)
def _dauer():
    return E.dauerfehler_tabelle()


@st.cache_data(show_spinner=False)
def _soll():
    return E.sollwert_tabelle()


@st.cache_data(show_spinner=False)
def _messreihen():
    return E.load_messreihen()


@st.cache_data(show_spinner="Rechne die Tabelle der Beobachterpole …")
def _beobachter(R):
    return E.beobachter_tabelle(R)


@st.cache_data(show_spinner=False)
def _kalman(R):
    return E.kalman_zeilen(R)


@st.cache_data(show_spinner=False)
def _rausch(R, rhos, seed):
    return E.rausch_laeufe(R, rhos, seed)


@st.cache_data(show_spinner=False)
def _modell(R, rho, g):
    return E.modell_g(R, rho, g)


@st.cache_data(show_spinner=False)
def _gebiet(R):
    return E.gebiet_ueber_rho(R)


@st.cache_data(show_spinner=False)
def _zufall(delta, n, seed, R, rho):
    return E.zufallsstrecken(delta, n, seed, R, rho)


@st.cache_data(show_spinner=False)
def _konstante(R, d, dcal):
    return E.konstante_falsch(R, d, dcal)


@st.cache_data(show_spinner=False)
def _zwei(R, rho, qi):
    return E.zwei_wege(R, rho, qi)


@st.cache_data(show_spinner=False)
def _integral_tabelle(R):
    return [K.integral_zeile("wagen", R, qi) for qi in C.QI_OPTIONS]


@st.cache_data(show_spinner=False)
def _rampe(R, rho, s, w):
    return E.rampe_sinus(R, rho, s, w)


@st.cache_data(show_spinner=False)
def _rampe_verlauf(R, rho, s):
    return E.rampe_verlauf(R, rho, s)


@st.cache_data(show_spinner=False)
def _stell(R, rho):
    return E.stellgrenze_tabelle(R, rho)


@st.cache_data(show_spinner=False)
def _rang():
    return E.rang_tabelle()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in off_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Offsetfrei regeln: die Störung mitschätzen und was es kostet")
st.markdown(
    f"""
Ein Wagen soll an einem Ort bleiben, und eine **konstante Störkraft** schiebt ihn. Die Zustandsrückführung u = −K x aus der Riccati-Gleichung hält ihn nicht am Ziel: Es bleibt ein **Dauerfehler** von d/K₁ (am Wagen bei R = 1: 2,369·d).
Dieses Stück der Reihe „Regelung (von LQR bis MPC)“ zeigt, wie ein **Störungsmodell mit Beobachter** den Fehler beseitigt, und fragt, **was die Beseitigung kostet** (Spitze, Frequenzgang, Reserve) und wo sie nicht hilft (Rampe, Messbias, Stellgrenze).
Vorgänger in der Reihe: [`plan-stoerung-demo`]({NEIGHBOUR}/plan-stoerung-demo) misst den Dauerfehler als Zahl, [`lqr-riccati-demo`]({NEIGHBOUR}/lqr-riccati-demo) liefert den Gewinn K.
Alles gilt **in diesem Modell** (linear, eine Störung am Eingang, Ort gemessen) und ist eigene Nachrechnung.

Weiter unten: die **Kernfragen**, **fünf Experimente**, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Regelung ohne Dauerfehler", expanded=True):
    st.markdown(
        """
1. **Die Rückführung:** Aus der Riccati-Gleichung (Q = I, Stellgewicht R) folgt K; das Gesetz u = −K x ist stabil, kennt aber die Störung d nicht. Im Beharrungszustand verlangt der Wagen v = 0 und K₁·p = d, also **p = d/K₁**.
2. **Das Störungsmodell:** Die Störung wird als Zustand mit d⁺ = d angehängt, und ein **Beobachter** schätzt (x, d) aus dem gemessenen Ort. Das Gesetz u = −K (x̂ − x_s) − d̂ hebt die geschätzte Störung auf.
3. **Warum der Fehler verschwindet:** Im Beharrungszustand verlangt die d̂-Zeile des Beobachters eine Innovation von null, also y = ŷ; dann bleibt für den wahren Ort nur p = 0. Die wahre Strecke geht nicht ein, nur ihre Stabilität (Beweisskizze in der README).
4. **Der Preis:** Der Beobachter braucht Zeit, bis er d kennt. In dieser Zeit schiebt die Störung den Wagen weiter als das reine LQR (Spitze 9,02 gegen 2,38 bei R = 1 und Beobachterpolen 3 × 0,5), und das stabile Gebiet der Streckenverstärkung wird schmaler.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Strecke und Regler**")
    strecke = st.selectbox("Strecke", C.STRECKEN, key="strecke_select", format_func=lambda k: K.KIND_LABEL[k],
                           help="Wagen: Doppelintegrator (Ort und Geschwindigkeit, Ort gemessen). Skalar: x⁺ = a x + u + d, Handbeispiel mit Q = 1.")
    R = st.select_slider("Stellgewicht R", options=C.R_OPTIONS, key="R_select", format_func=lambda v: E.de(v, 3).rstrip("0").rstrip(","),
                         help="Gewicht des Eingriffs in den Kosten x'x + R u²; kleineres R heißt stärkerer Eingriff.")
    rho = st.slider("Beobachterpol ρ", *bounds("rho_slider"), key="rho_slider", step=C.RHO_STEP, format="%.2f",
                    help="Wagen: dreifacher Pol bei ρ. Skalar: Pole ρ und ρ/2. Kleiner heißt schneller, 0 wäre ein Deadbeat-Beobachter.")
    st.markdown("**Störung und Ziel**")
    d = st.slider("Konstante Störkraft d", *bounds("d_slider"), key="d_slider", step=C.D_STEP, format="%.1f",
                  help="Eine Kraft am Eingang, die ab k = 0 konstant wirkt (Störsprung aus der Ruhe).")
    r = st.slider("Sollwert r des Ortes", *bounds("r_slider"), key="r_slider", step=C.SOLL_STEP, format="%.1f",
                  help="Der Wagen startet in Ruhe im Sollwert; die Kennzahlen messen die Abweichung davon.")
    st.markdown("**Messung und Stellglied**")
    b = st.slider("Messbias b", *bounds("b_slider"), key="b_slider", step=C.B_STEP, format="%.1f",
                  help="Der Ortssensor zeigt p + b. Offsetfrei heißt: die gemessene Größe erreicht das Ziel.")
    umax = st.select_slider("Stellgrenze |u| ≤", options=C.UMAX_OPTIONS, key="umax_select",
                            format_func=lambda v: "aus" if v == C.UMAX_AUS else E.de(v, 1),
                            help="Die Stellgröße wird abgeschnitten; der Beobachter bekommt die angewendete Stellgröße. aus: keine Grenze.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1,
                           help="Für das Messrauschen (Experiment B) und die Zufallsstrecken (Experiment C).")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)

settings = E.Settings(strecke, float(R), float(d), float(r), float(rho), float(b), E.umax_from_choice(umax))
res = _ergebnis(settings)
ent = res["ent"]
mit, ohne = res["mit"], res["ohne"]
n_zeigen = min(res["T"], max(40, 2 * max(res["k_mit"]["abkling"], 20)))

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: der Störsprung mit und ohne Störungsmodell")
beob_text = "3 × ρ" if strecke == "wagen" else "ρ und ρ/2"
st.caption(f"{K.KIND_LABEL[strecke]}, R = {E.de(settings.R, 3).rstrip('0').rstrip(',')}, Störsprung d = {E.de(settings.d, 1)} bei k = 0, Start in Ruhe im Sollwert r = {E.de(settings.r, 1)}; "
           f"Beobachterpole {beob_text} = {E.de(settings.rho, 2)}. Zähleinheit: Periode.")
if res["grenze"] is not None:
    p_ohne_ss, p_mit_ss = float(res["g_ohne"]["p"][-1]), float(res["g_mit"]["p"][-1])
    kenn_mit = K.kennzahlen(res["g_mit"]["p"], settings.r)
    kenn_ohne = K.kennzahlen(res["g_ohne"]["p"], settings.r)
    ort_label = f"Ort nach {C.T_GRENZE} Schritten"
else:
    p_ohne_ss, p_mit_ss = ohne["p_fest"], mit["p_fest"]
    kenn_mit, kenn_ohne = res["k_mit"], res["k_ohne"]
    ort_label = "Ort im Beharrungszustand"
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric(f"{ort_label}, ohne Störungsmodell", E.de(p_ohne_ss, 3), delta=f"{_delta(p_ohne_ss - settings.r)} gegen Ziel", delta_color="inverse",
          help="Fixpunkt des reinen LQR (u = −K x, Zustand gemessen); am Wagen und am Skalar a = 1 gleich r + d/K₁.")
m2.metric(f"{ort_label}, mit Störungsmodell", E.de(p_mit_ss, 3), delta=f"{_delta(p_mit_ss - settings.r)} gegen Ziel", delta_color="inverse",
          help="Fixpunkt des Beobachterkreises; mit Messbias b bleibt der wahre Ort um −b daneben, die gemessene Größe erreicht das Ziel.")
m3.metric("Spitze |p − r| mit Störungsmodell", E.de(kenn_mit["spitze"], 2), delta=f"{_delta(kenn_mit['spitze'] - kenn_ohne['spitze'])} gegen ohne Modell", delta_color="inverse",
          help="Größte Abweichung vom Sollwert nach dem Störsprung; Referenz: dieselbe Spitze beim reinen LQR.")
m4.metric("Abklingzeit mit Störungsmodell", f"{res['k_mit']['abkling']} Perioden" if res["grenze"] is None else "–", delta="letzte Periode über 1 % der Spitze" if res["grenze"] is None else None,
          delta_color="off", help="Mit Stellgrenze nicht definiert (der Lauf ist nicht linear).")
c1, c2 = st.columns([2, 1])
c1.plotly_chart(build_ort(res, settings, n_zeigen), width="stretch", key="ort_chart")
c2.plotly_chart(build_schaetzung(res, settings, n_zeigen), width="stretch", key="dh_chart")
if res["T"] > n_zeigen:
    st.caption(f"Gezeigt sind die ersten {n_zeigen} Perioden; die Kennzahlen stammen aus dem ganzen Lauf ({E.fmt_int(res['T'])} Perioden).")
tr = _trennung(strecke, settings.R, settings.rho)
st.caption(f"Eigenwerte von A − B K (Rückführung): {E.eigen_text(tr['eig_rueck'])}. "
           f"Trennung: das charakteristische Polynom des Beobachterkreises ist das Produkt aus dem Polynom von A − B K und dem des Beobachters (größter Koeffizientenabstand {tr['abstand']:.1e}; "
           "dreifache Pole vergleicht man über die Koeffizienten, nicht über Eigenwerte).")

with st.expander("🧮 Das Handbeispiel in Brüchen (Skalar x⁺ = x + u + d, Kosten x² + 2 u², Pole des Beobachters ½ und ¼, d = 1)"):
    h = _hand()
    st.markdown(
        f"""
**Riccati von Hand:** P₀ = {E.X.text(h['P'][0])}, P₁ = {E.X.text(h['P'][1])}, P₂ = {E.X.text(h['P'][2])}, … → P = 2 (löst P = 1 + P − P²/(2 + P)); k = P/(2 + P) = **{E.X.text(h['k'])}**.
**Dauerfehler ohne Störungsmodell:** x⁺ = x/2 + 1 gibt x = 0, 1, 3/2, 7/4, 15/8, … = 2 (1 − 2⁻ᵏ), also **d/k = 2** (Dauerfehler 2·d).
**Beobachter** für (x, d), Ort gemessen: Fehlermatrix [[1 − Lₓ, 1], [−L_d, 1]], Polynom z² − (2 − Lₓ) z + (1 − Lₓ + L_d); Pole ½ und ¼ geben **Lₓ = {E.X.text(h['Lx'])}, L_d = {E.X.text(h['Ld'])}**.
Polynom des ganzen Kreises (x, x̂, d̂): z³ {' '.join(('+ ' if c > 0 else '− ') + E.X.text(abs(c)) + ('z²' if i == 1 else 'z' if i == 2 else '') for i, c in enumerate(h['poly'][1:], start=1))} = (z − ½)(z − ½)(z − ¼); Fixpunkt (x, x̂, d̂) = ({', '.join(E.X.text(v) for v in h['fest'])}).
        """
    )
    st.dataframe(
        [{"k": int(z[0]), "x_k ohne Modell": E.X.text(h["ohne"][int(z[0])]), "x_k mit Modell": E.X.text(z[1]), "x̂_k": E.X.text(z[2]), "d̂_k": E.X.text(z[3])} for z in h["zeilen"]],
        hide_index=True, width="stretch",
    )
    st.caption("Die Spitze 2 mit Modell ist hier genau so groß wie der Dauerfehler ohne Modell: Zufall bei diesen Zahlen, keine Regel. Die Rechnung steht als Test in `tests/test_dynamics.py`.")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
rs_kern = _rampe(settings.R, settings.rho, C.DEFAULT_RAMPE, C.DEFAULT_OMEGA)
verdikte = E.kernfragen(settings, res, rs_kern)
for frage, (kind, text) in zip(("**1. Bleibt ein Fehler stehen?**", "**2. Wie kommt er weg, und was kostet das?**", "**3. Wann hilft es nicht?**"), verdikte):
    st.markdown(frage)
    getattr(st, kind)(text)

st.markdown("---")

# --- Experiment A --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment A: Dauerfehler und Sollwert")
st.caption("Das reine LQR ohne Störungsmodell, Zustand gemessen, Störkraft d = 1 am Eingang. Links der Wagen über R, rechts der Skalar a = 1 (x_ss/d = 1/k). "
           "Unter den Tabellen: der Sollwert ohne Störung. Die Reihen mit 2 000 R und 300 rationalen Gesetzen sind **vorgerechnet** (`data/off_messreihen.json`, `tools/off_make_messreihen.py`).")
wagen_rows, skalar_rows = _dauer()
mr = _messreihen()
ca, cb = st.columns(2)
ca.dataframe([{"R": E.de(w["R"], 3).rstrip("0").rstrip(","), "K₁": E.de(w["K1"], 5), "p_ss / d = 1/K₁": E.de(w["p_ss"], 5), "Spektralradius": E.de(w["radius"], 4)} for w in wagen_rows], hide_index=True, width="stretch")
cb.dataframe([{"R": E.de(w["R"], 1), "k": E.de(w["k"], 5), "x_ss / d = 1/k": E.de(w["x_ss"], 5)} for w in skalar_rows], hide_index=True, width="stretch")
st.plotly_chart(build_dauerfehler_R(wagen_rows), width="stretch", key="dauer_chart")
sw = _soll()
st.markdown("**Sollwert ohne Störung** (Zielrechnung [[I − A, −B], [C, 0]] (x_s, u_s) = (B d̂, r)):")
st.dataframe(
    [{"Fall": f"Wagen, r = {E.de(sw['r_wagen'], 1)}, u = −K (x − (r, 0))", "Ort im Beharrungszustand": E.de(sw["p_wagen"], 4), "Fehler": E.de(sw["r_wagen"] - sw["p_wagen"], 4)},
     {"Fall": f"Skalar a = ½, k = ¼, r = {E.X.text(sw['r'])}, ohne Zielwert u_s", "Ort im Beharrungszustand": E.X.text(sw["x_ohne"]), "Fehler": E.X.text(sw["fehler_ohne"])},
     {"Fall": f"Skalar a = ½, k = ¼, r = {E.X.text(sw['r'])}, mit Zielwert u_s = {E.X.text(sw['u_s'])}", "Ort im Beharrungszustand": E.X.text(sw["x_mit"]), "Fehler": E.X.text(sw["r"] - sw["x_mit"])}],
    hide_index=True, width="stretch",
)
st.caption("Der Wagen erreicht den Sollwert ohne Zielwert, weil er selbst integriert (u_s = 0); der Skalar a = ½ braucht u_s = (1 − a) r/b, sonst bleibt der Fehler r (1 − a)/(1 − a + b k).")
kind, text = E.verdict_dauerfehler_R(wagen_rows, mr.get("R"), mr.get("brueche"))
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment B --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment B: Wie schnell soll der Beobachter sein?")
st.caption(f"Störsprung d = 1 aus der Ruhe, Ziel 0, Ort gemessen, Beobachter 3 × ρ; Messrauschen weiß mit Varianz 1 auf dem Ort, exakt aus der Lyapunov-Gleichung. R = {E.de(settings.R, 3).rstrip('0').rstrip(',')} aus der Seitenleiste, Wagen. "
           f"g-Gebiet: stabile Streckenverstärkung (Entwurf für g = 1); Reserve: kleinster Betrag von 1 + L am Streckeneingang. {WAGEN_HINWEIS}")
b_rows, b_ohne = _beobachter(settings.R)
st.dataframe(
    [{"ρ": E.de(z["rho"], 2), "Spitze": E.de(z["spitze"], 2), "Schritt": str(z["schritt"]), "Abklingzeit": str(z["abkling"]), "max |p/d| (bei ω)": f"{E.de(z['peak'], 2)} ({E.de(z['wpeak'], 3)})",
      "std(p)": E.de(z["std_p"], 2), "std(u)": E.de(z["std_u"], 3), "g-Gebiet": f"{E.de(z['g_min'], 3)} bis {E.de(z['g_max'], 3)}", "min |1 + L|": E.de(z["reserve"], 3)} for z in b_rows]
    + [{"ρ": "ohne Beobachter", "Spitze": E.de(b_ohne["spitze"], 2), "Schritt": "–", "Abklingzeit": "–", "max |p/d| (bei ω)": E.de(b_ohne["peak"], 2), "std(p)": "–", "std(u)": "–",
        "g-Gebiet": f"{E.de(b_ohne['g_min'], 4)} bis {E.de(b_ohne['g_max'], 3)}", "min |1 + L|": E.de(b_ohne["reserve"], 3)}],
    hide_index=True, width="stretch",
)
st.caption(f"Ohne Beobachter bleibt der Ort bei {E.de(b_ohne['dauerfehler'], 3)}·d stehen (Dauerfehler); die Spitze {E.de(b_ohne['spitze'], 2)} ist die des reinen LQR.")
st.plotly_chart(build_beobachter(b_rows, b_ohne, settings.rho), width="stretch", key="beob_chart")
rhos_rausch = tuple(sorted({0.1, round(settings.rho, 2), 0.9}))
st.markdown(f"**Stellgröße bei Messrauschen** (Zufalls-Seed {int(seed)}, {C.T_RAUSCH} Perioden, dieselbe Rauschfolge für alle Pole):")
st.plotly_chart(build_rausch(_rausch(settings.R, rhos_rausch, int(seed))), width="stretch", key="rausch_chart")
kal = _kalman(settings.R)
st.markdown("**Kalman-Verstärkung statt Polvorgabe** (Vorhersageform, Messrauschen 1, Prozessrauschen der Geschwindigkeit 0,01 und der Störung q_d): dieselbe Familie von Kompromissen, kein eigener Befund. Filter, NEES und Q/R-Einstellung: [`kalman-filter-demo`]" + f"({NEIGHBOUR}/kalman-filter-demo).")
st.dataframe(
    [{"q_d": f"{z['qd']:g}".replace(".", ","), "Spitze": E.de(z["spitze"], 2), "Abklingzeit": str(z["abkling"]), "std(p)": E.de(z["std_p"], 3), "max |p/d|": E.de(z["peak"], 2), "g-Gebiet": f"{E.de(z['g_min'], 3)} bis {E.de(z['g_max'], 3)}"} for z in kal],
    hide_index=True, width="stretch",
)
z_schnell = next(z for z in b_rows if abs(z["rho"] - 0.1) < 1e-9)
z_langsam = next(z for z in b_rows if abs(z["rho"] - 0.9) < 1e-9)
kind, text = E.verdict_beobachter(z_schnell, z_langsam, b_ohne)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment C --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment C: Wenn das Modell nicht stimmt")
st.caption(f"Entwurf für die nominelle Strecke (Wagen, R, ρ der Seitenleiste), die wahre Strecke weicht ab. {WAGEN_HINWEIS}")
g = st.slider("Wahre Streckenverstärkung g (Entwurf für g = 1)", *bounds("g_slider"), key="g_slider", step=C.G_STEP, format="%.2f",
              help="Die wahre Strecke ist B_t = g B und B_dt = g B: der Wagen wird g-mal so stark gestellt und gestört.")
mg = _modell(settings.R, settings.rho, float(g))
ge_zeilen, ge_ohne = _gebiet(settings.R)
st.plotly_chart(build_gebiet(ge_zeilen, ge_ohne, float(g), settings.rho), width="stretch", key="gebiet_chart")
kind, text = E.verdict_modell(float(g), mg["stabil"], mg["p_ss"], mg["gebiet"], mg["gebiet_ohne"], settings.rho, mr.get("zufallsstrecken_summe"))
getattr(st, kind)(text)
if mg["p_ss_ohne"] is not None:
    st.caption(f"Zum Vergleich das reine LQR bei diesem g: stabil, Dauerfehler {E.de(mg['p_ss_ohne'], 3)}·d (nominell {E.de(1 / float(ent.K[0, 0]), 3)}·d). Das Gebiet des Balkens „ohne Beobachter“ ist breiter, nur eben mit Dauerfehler.")
st.markdown("**Zufallsstrecken** (je Eintrag von A und B ± δ, B_dt = (u(−1; 1), u(0,3; 1,7)); Entwurf nominell, Beobachter 3 × ρ):")
delta = st.select_slider("Fehler δ je Eintrag", options=C.DELTA_OPTIONS, key="delta_select", format_func=lambda v: E.de(v, 2),
                         help="Jeder Eintrag von A und B wird gleichverteilt um ±δ verschoben; die Störung wirkt über einen zufälligen Vektor B_dt.")
zs = _zufall(float(delta), C.N_ZUFALL_LIVE, int(seed), settings.R, settings.rho)
st.caption(f"Live (Zufalls-Seed {int(seed)}): {E.fmt_int(zs['stabil'])} von {E.fmt_int(zs['n'])} Zufallsstrecken stabil (Spektralradius unter 1 − 10⁻⁶); größter Betrag des Ortsfehlers unter den stabilen: {zs['max_pss']:.1e} (Rundungsrest, Dauerfehler 0). "
           + (f"**Vorgerechnet** über fünf δ mit je 3 000 Strecken: Dauerfehler 0 in {E.fmt_int(mr['zufallsstrecken_summe']['stabil'])} von {E.fmt_int(mr['zufallsstrecken_summe']['stabil'])} stabilen Fällen (größter Rest {mr['zufallsstrecken_summe']['max_pss']:.1e})." if mr else ""))
st.markdown("**Eine einmal falsch gesetzte Konstante bleibt** (statische Vorsteuerung u = −K x − d_cal, kein Beobachter, d = 1; Gegenstück zum Beobachter):")
dcal = st.slider("Gesetzte Konstante d_cal", *bounds("dcal_slider"), key="dcal_slider", step=C.DCAL_STEP, format="%.1f",
                 help="Die Vorsteuerung zieht d_cal von der Stellgröße ab; die wahre Störung ist d = 1. Ohne Beobachter bleibt der Fehler (d − d_cal)/K₁.")
kf = _konstante(settings.R, 1.0, float(dcal))
tab_dcal = [(_konstante(settings.R, 1.0, v)["p_ss"], v) for v in C.DCAL_TABELLE]
st.dataframe([{"d_cal": E.de(v, 1), "p_ss": E.de(p, 3)} for p, v in tab_dcal], hide_index=True, width="stretch")
st.info(f"d_cal = {E.de(float(dcal), 1)} statt d = 1: der Ort bleibt bei {E.de(kf['p_ss'], 3)} (= (d − d_cal)/K₁ = {E.de(kf['formel'], 3)}); eine Konstante, die einmal falsch gesetzt ist, bleibt als Fehler, "
        "das Muster des Dauerfehlers im Lager steht in [`leitung-mitrechnen-demo`]" + f"({NEIGHBOUR}/leitung-mitrechnen-demo) (dort ohne Beobachter, nur als Konstante).")
st.caption(f"**Messbias:** zeigt der Sensor p + b, ist der wahre Ort im Beharrungszustand −b daneben (mit Störungsmodell in 3 von 3 gemessenen Beobachtern, Seitenleiste: b = {E.de(settings.b, 1)} gibt {E.de(settings.r - settings.b, 2)}); "
           "die gemessene Größe erreicht das Ziel, ein Messfehler bleibt als wahrer Ortsfehler.")

st.markdown("---")

# --- Experiment D --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment D: Zwei Wege zur Offsetfreiheit")
st.caption(f"Zustand gemessen (Ort und Geschwindigkeit), Sollwert 0, Störsprung d = 1. Weg 1: Integralzustand ξ⁺ = ξ + p, LQR mit Q = diag(1, 1, q_i) und dem R der Seitenleiste. "
           f"Weg 2: Störungsbeobachter nur für d̂ (Pole ρ, 0,9 ρ, 0,8 ρ), K aus dem unerweiterten LQR. {WAGEN_HINWEIS}")
qi = st.select_slider("Gewicht q_i des Integralzustands", options=C.QI_OPTIONS, key="qi_select", format_func=lambda v: f"{v:g}".replace(".", ","),
                      help="Kleiner heißt langsamerer Integralanteil: kleinere Spitze, aber lange Abklingzeit.")
zw = _zwei(settings.R, settings.rho, float(qi))
st.plotly_chart(build_zwei_wege(zw["integral"], zw["beobachter"], 120), width="stretch", key="zwei_chart")
it = _integral_tabelle(settings.R)
st.dataframe(
    [{"q_i": f"{z['qi']:g}".replace(".", ","), "Spitze": E.de(z["spitze"], 3), "Abklingzeit": str(z["abkling"]), "max |p/d|": E.de(z["peak"], 3), "Spektralradius": E.de(z["spektralradius"], 3),
      "g-Gebiet": f"{E.de(z['g_min'], 3)} bis {E.de(z['g_max'], 3)}", "min |1 + L|": E.de(z["reserve"], 3)} for z in it],
    hide_index=True, width="stretch",
)
kind, text = E.verdict_zwei_wege(zw["integral"], zw["beobachter"], float(qi), settings.rho)
getattr(st, kind)(text)
st.caption(f"Der Integrator mit Stellgrenze und Rückrechnung steht in [`windup-pi-demo`]({NEIGHBOUR}/windup-pi-demo); hier hat keiner der beiden Wege eine Stellgrenze.")

st.markdown("---")

# --- Experiment E --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment E: Grenzen der Offsetfreiheit")
st.caption(f"Drei Fälle, in denen „offsetfrei“ nicht trägt. {WAGEN_HINWEIS}")
tab_rampe, tab_stell, tab_rang = st.tabs(["Rampe und Sinus", "Stellgrenze", "Störung am falschen Ort"])
with tab_rampe:
    s_r = st.slider("Steigung s der Rampe d = s·k", *bounds("rampe_slider"), key="rampe_slider", step=C.RAMPE_STEP, format="%.3f",
                    help="Die Störung wächst linear mit der Zeit statt konstant zu sein.")
    w_s = st.select_slider("Kreisfrequenz ω der Sinusstörung", options=C.OMEGA_OPTIONS, key="omega_select", format_func=lambda v: E.de(v, 2),
                           help="Bogenmaß je Periode; die Störung ist d = sin(ω k).")
    rp = _rampe(settings.R, settings.rho, float(s_r), float(w_s))
    pm, po = _rampe_verlauf(settings.R, settings.rho, float(s_r))
    ca, cb = st.columns(2)
    ca.plotly_chart(build_rampe(pm, po), width="stretch", key="rampe_chart")
    cb.plotly_chart(build_frequenzgang(rp["ws"], rp["kurve"], rp["kurve0"], float(w_s)), width="stretch", key="freq_chart")
    kind, text = E.verdict_rampe(float(s_r), rp["faktor"], float(rp["c0"][0]), float(rp["ohne_c1"][0]), rp["peak"], float(w_s), rp["gain"], rp["gain0"])
    getattr(st, kind)(text)
with tab_stell:
    st.caption("Stellgrenze |u| ≤ 1, Start (Ort 4, Geschwindigkeit 0), Schätzung 0; die Stellgröße wird abgeschnitten, der Beobachter bekommt die angewendete. Die Kurven für d = 1,5 verlassen das Bild.")
    sz = _stell(settings.R, settings.rho)
    st.plotly_chart(build_stellgrenze(sz), width="stretch", key="stell_chart")
    st.dataframe(
        [{"d": E.de(z["d"], 1), "ohne Modell: Ort nach 300": E.de(z["ohne"]["p_end"], 3), "mit Modell: Ort nach 300": E.de(z["mit"]["p_end"], 3), "d/K₁ (nur ohne aktive Grenze)": E.de(z["formel"], 3),
          "mit Modell: Spitze |p|": E.de(z["mit"]["spitze"], 1), "mit Modell: Ort nach 50 / 100": f"{E.de(z['mit']['p50'], 1)} / {E.de(z['mit']['p100'], 1)}"} for z in sz],
        hide_index=True, width="stretch",
    )
    kind, text = E.verdict_stellgrenze(sz)
    getattr(st, kind)(text)
with tab_rang:
    st.caption("Rangbedingung der Beobachtbarkeit des erweiterten Modells: rank [[I − A, −B_d], [C, C_d]] = n + n_d (exakt in Brüchen gerechnet). Ist sie verletzt, gibt es kein Störungsmodell für diese Messung.")
    rg = _rang()
    st.dataframe([{"Fall": t["fall"], "Rang": str(t["rang"]), "verlangt": str(t["verlangt"]), "erfüllt": "ja" if t["erfuellt"] else "nein"} for t in rg], hide_index=True, width="stretch")
    kind, text = E.verdict_rang(rg)
    getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Störung ist konstant** (d⁺ = d) | Bei einer Rampe bleibt am Wagen ein konstanter Ortsfehler von 57,6·s (R = 1, ρ = 0,5), bei einem Sinus verstärkt der Beobachterkreis p/d bis 13,1 gegen 2,37 (Experiment E) | – |
| **Die Stellgrenze lässt die Gegenkraft zu** (Betrag von d kleiner als u_max) | Ist der Betrag von d größer als u_max, läuft der Wagen weg, mit und ohne Störungsmodell (Ort 22 728 nach 300 Schritten bei d = 1,5, u_max = 1) | [`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo), [`windup-pi-demo`](https://github.com/sebastian-hanisch/windup-pi-demo) |
| **Der Sensor misst, was geregelt wird** (b = 0) | Mit Messbias b bleibt der wahre Ort bei −b: offsetfrei heißt, die gemessene Größe erreicht das Ziel | – |
| **Die Rangbedingung ist erfüllt** (eine Störung am Eingang, eine Messung) | Störung am Ausgang beim Wagen: Rang 2 von 3, das Störungsmodell ist nicht entwerfbar | – |
| **Der Entwurf ist getrennt** (K aus dem unerweiterten LQR, L aus Polvorgabe) | Der Preis steht in Spitze (9,02 gegen 2,38), Frequenzgang (13,1 gegen 2,37) und stabilem g-Gebiet (0,41 bis 1,29 gegen 0 bis 1,94); andere Auslegungen sind nicht gemessen | [`lqr-reserven-demo`](https://github.com/sebastian-hanisch/lqr-reserven-demo) |
| **Keine Verzögerung** (Rechenzeit, Totzeit) | Beobachter mit Verzug ist nicht gemessen | [`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben die Strecke $x_{k+1} = A x_k + B (u_k + d)$ mit gemessenem Ort $y_k = C x_k + b$, ein Gewicht $R$ und eine konstante Störung $d$ am Eingang. Gesucht ist ein Gesetz, das $y_k$ auf den Sollwert $r$ bringt, ohne die Störung zu kennen.

| Symbol | Bedeutung |
|---|---|
| $x = (p, v)$ | Zustand des Wagens: Ort und Geschwindigkeit; $A = \begin{pmatrix}1&1\\0&1\end{pmatrix}$, $B = \begin{pmatrix}0\\1\end{pmatrix}$, $C = (1\;\,0)$ |
| $K$, $P$ | Gewinn und Lösung der Riccati-Gleichung, $u = -Kx$ |
| $z = (x, d)$ | erweiterter Zustand mit $d_{k+1} = d_k$; $A_e = \begin{pmatrix}A&B\\0&1\end{pmatrix}$, $B_e = \begin{pmatrix}B\\0\end{pmatrix}$, $C_e = (C\;\,0)$ |
| $L$ | Beobachterverstärkung, hier nach Ackermann mit dreifachem Pol $\rho$ |
| $\hat x$, $\hat d$ | Schätzwerte von Zustand und Störung |
| $x_s$, $u_s$ | Zielwerte im Beharrungszustand aus $\begin{pmatrix}I-A&-B\\C&0\end{pmatrix}\begin{pmatrix}x_s\\u_s\end{pmatrix} = \begin{pmatrix}B\hat d\\ r\end{pmatrix}$ |

**Rückführung und Dauerfehler.** Aus $P = Q + A^\top P A - A^\top P B (R + B^\top P B)^{-1} B^\top P A$ folgt $K = (R + B^\top P B)^{-1} B^\top P A$ mit $Q = I$. Der Regelkreis $x_{k+1} = (A - BK) x_k + B d$ hat den Fixpunkt $x = (I - A + BK)^{-1} B d$; am Wagen verlangt er $v = 0$ und $K_1 p = d$, also $p_{ss} = d / K_1$.

**Störungsmodell mit Beobachter** (Vorhersageform) und Gesetz:

$$\hat z_{k+1} = A_e \hat z_k + B_e u_k + L \,(y_k - C_e \hat z_k), \qquad u_k = u_s - K(\hat x_k - x_s), \qquad u_s = -\hat d \;\text{(Wagen)}.$$

**Trennung.** Das charakteristische Polynom des Kreises aus Strecke, Beobachter und Gesetz ist $\det(zI - A + BK)\cdot\det(zI - A_e + L C_e)$.

**Offsetfreiheit.** Im Fixpunkt verlangt die $\hat d$-Zeile $L_d (y - C_e \hat z) = 0$, also $y = \hat y$ (für $L_d \neq 0$); die $\hat v$-Zeile gibt dann $\hat v = 0$, das Gesetz $K_1 \hat p = 0$, also $p = \hat p = 0$. Die wahre Strecke geht nicht ein, nur ihre Stabilität.

**Rangbedingung.** Das erweiterte Modell ist beobachtbar, wenn $(C, A)$ beobachtbar ist und $\operatorname{rank}\begin{pmatrix}I-A&-B_d\\C&C_d\end{pmatrix} = n + n_d$ gilt (höchstens so viele Störungen wie Messungen).

**Integralzustand.** $\xi_{k+1} = \xi_k + y_k - r$ und $u = -K_a (x, \xi)$ mit $K_a$ aus der Riccati-Gleichung des erweiterten Paares $(A_a, B_a)$, $Q_a = \operatorname{diag}(1, 1, q_i)$. Im Fixpunkt verlangt die $\xi$-Zeile $y = r$.

**Spitze und Abklingzeit.** Spitze $= \max_k |p_k - r|$ nach einem Störsprung aus der Ruhe, Abklingzeit $=$ letzte Periode mit $|p_k - r|$ über 1 Prozent der Spitze.

**Literatur** (nur im Wortlaut gelesen, soweit angegeben; die Gleichungen sind hier in eigenen Worten und eigene Nachrechnung in diesem Modell): Rawlings, Mayne, Diehl, *Model Predictive Control*, Kap. 1.5.2 (Störungsmodell, Rangbedingung, Lemma 1.10); Borrelli, Bemporad, Morari, *Predictive Control*, Kap. 12.7 (Sätze zur Beobachtbarkeit und zur Offsetfreiheit, Bemerkung zu Integratoren); Åström, Murray, *Feedback Systems*, Kap. 7.4 (Integralzustand). Stellenangaben nach dem Quellenbericht der Planung, die Originalstellen sind beim Bau nicht erneut geöffnet worden.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)

sync_query_params({
    "strecke_select": strecke, "R_select": settings.R, "d_slider": settings.d, "r_slider": settings.r, "rho_slider": settings.rho, "b_slider": settings.b, "umax_select": umax,
    "seed_input": int(seed), "g_slider": float(g), "delta_select": float(delta), "dcal_slider": float(dcal), "qi_select": float(qi), "rampe_slider": float(s_r), "omega_select": float(w_s),
})
