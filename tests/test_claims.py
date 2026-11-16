"""Claims: jede Zahl der README (Befundtabelle M1 bis M12, Tabelle M9, Korrekturen) und der App-Texte wird hier mit denselben Funktionen nachgerechnet wie in der App.

Voraussetzungen wie in der README: Wagen, konstante Störkraft am Eingang, Ort gemessen, K aus der DARE (Q = I), Beobachter 3 x rho (Ackermann), Störsprung aus der Ruhe, Messrauschen weiß mit
Varianz 1. Plattformabhängiges (Spitzen, Gebiete, Frequenzspitzen, Zählungen von Zufallsstrecken) steht als Band: relative Toleranz 1e-9 bzw. die gedruckten Stellen, Zählungen ± 2;
exakt nur, was in Brüchen gerechnet ist. Die Vormessungen (out_rg_v17_*.txt) sind die Erwartung; Abweichungen stehen in der README unter „Befunde und Korrekturen“."""

import re
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

import off_constants as C
import off_dynamics as K
import off_evaluation as E
import off_exakt as X
from oracle_helpers import wagen_kreis, wagen_L

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _wagen(R=1.0, rho=0.5):
    return K.entwurf("wagen", R, rho)


# ---- M1 Dauerfehler -------------------------------------------------------------------------------------------------------------------

def test_m1_dauerfehler_des_gesetzes_am_wagen():
    soll = {1.0: (2.369, 0.422), 0.1: (1.727, 0.362), 0.01: (1.630, 0.380), 0.001: (1.619, 0.382)}
    for R, (p_ss, rho_sp) in soll.items():
        ent = _wagen(R)
        Acl, Bw = K.kreis_ohne(ent)
        z = K.fixpunkt(Acl, Bw, (1.0, 0.0, 0.0))
        assert z[0] == pytest.approx(p_ss, abs=6e-4) and abs(z[1]) < 1e-12, R
        assert float(K.spektralradius(Acl)) == pytest.approx(rho_sp, abs=6e-4), R
    Af = [[F(1), F(1)], [F(-2, 5), F(-1, 5)]]
    assert X.fsolve(X.fsub(X.feye(2), Af), [0, 1]) == [F(5, 2), 0]


def test_m1_skalar_a_gleich_eins():
    for R, soll in ((0.5, 1.366), (1.0, 1.618), (2.0, 2.000), (5.0, 2.791), (20.0, 5.000)):
        assert K.dauerfehler_lqr(K.entwurf("skalar1", R, 0.5)) == pytest.approx(soll, abs=6e-4), R


def test_m1_vorgerechnete_reihen_stimmen_mit_der_neu_rechnung():
    mr = E.load_messreihen()
    neu_R = E.messreihe_R(mr["R"]["n"], mr["R"]["seed"])
    assert (mr["R"]["n"], mr["R"]["rueckgaenge"], mr["R"]["unter_phi"]) == (2000, 0, 0)
    assert neu_R["rueckgaenge"] == 0 and neu_R["unter_phi"] == 0
    assert neu_R["min"] == pytest.approx(mr["R"]["min"], rel=1e-9) and neu_R["max"] == pytest.approx(mr["R"]["max"], rel=1e-6)
    assert mr["R"]["min"] == pytest.approx(1.618034, abs=2e-6) and 105.6 < mr["R"]["max"] < 105.8             # README: 1,618034 bis 105,7
    assert mr["R"]["min"] > E.PHI - 1e-6                                                                       # nie unter 1,618033
    neu_b = E.messreihe_brueche(mr["brueche"]["n"], mr["brueche"]["seed"])
    assert mr["brueche"]["n"] == 300 and mr["brueche"]["exakt"] == 300 and neu_b == mr["brueche"]


# ---- M2 Sollwert ---------------------------------------------------------------------------------------------------------------------

def test_m2_sollwert_wagen_und_skalar():
    t = E.sollwert_tabelle()
    assert t["p_wagen"] == pytest.approx(2.5, abs=1e-12)
    assert t["x_ohne"] == F(1, 3) and t["fehler_ohne"] == F(2, 3) and t["u_s"] == F(1, 2) and t["x_mit"] == 1


# ---- M3 Störungsmodell mit Beobachter ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("R, rho", [(1.0, 0.5), (1.0, 0.8), (0.1, 0.3)])
def test_m3_fixpunkt_ist_null_null_d(R, rho):
    Acl, Bw = K.kreis(_wagen(R, rho))
    z = K.fixpunkt(Acl, Bw, (1.0, 0.0, 0.0))
    assert abs(z[0]) < 1e-9 and abs(z[1]) < 1e-9 and z[4] == pytest.approx(1.0, abs=1e-9)


def test_m3_trennung_und_dreifache_pole():
    for rho in C.RHO_TABELLE:
        assert E.trennung(_wagen(1.0, rho))["abstand"] < 1e-10, rho
    ent = _wagen(1.0, 0.5)
    ev = np.linalg.eigvals(ent.Ae - ent.L @ ent.Ce)
    assert 1e-6 < np.max(np.abs(ev - 0.5)) < 1e-4                                   # README: etwa 7·10⁻⁶ (Band, plattformabhängig)


def test_m3_fixpunkt_mit_sollwert_in_brueche():
    """Sollwert r = 1 und r = −3/2 bei d = 1: p = r exakt (Brüche, K = (2/5, 6/5), L = (3/2, 3/4, 1/8))."""
    k1, k2, (Lp, Lv, Ld) = F(2, 5), F(6, 5), (F(3, 2), F(3, 4), F(1, 8))
    Mf = [[1, 1, 0, 0, 0], [0, 1, -k1, -k2, -1], [Lp, 0, 1 - Lp, 1, 0], [Lv, 0, -Lv - k1, 1 - k2, 0], [Ld, 0, -Ld, 0, 1]]
    for r in (F(1), F(-3, 2)):
        fest = X.fsolve([[(1 if i == j else 0) - Mf[i][j] for j in range(5)] for i in range(5)], [0, 1 + k1 * r, 0, k1 * r, 0])
        assert fest[0] == r and fest[1] == 0 and fest[4] == 1


# ---- M4 Rangbedingung -----------------------------------------------------------------------------------------------------------------

def test_m4_rangbedingung():
    t = {x["fall"]: (x["rang"], x["verlangt"]) for x in E.rang_tabelle()}
    assert t == {"Wagen, Störung am Eingang": (3, 3), "Wagen, Störung am Ausgang": (2, 3), "Wagen, zwei Störungen bei einer Messung": (3, 4),
                 "Skalar a = 1, Störung am Eingang": (2, 2), "Skalar a = 1, Störung am Ausgang": (1, 2), "Skalar a = ½, Störung am Ausgang": (2, 2)}


# ---- M5 Rampe und Sinus -----------------------------------------------------------------------------------------------------------------------

def test_m5_rampe_und_sinus():
    ent = _wagen()
    Acl, Bw = K.kreis(ent)
    A0, B0 = K.kreis_ohne(ent)
    c0, c1 = K.rampe_ansatz(Acl, Bw[:, 0], 0.01)
    assert c0[0] == pytest.approx(0.5758, abs=6e-5) and c0[0] / 0.01 == pytest.approx(57.58, abs=6e-3) and abs(c1[0]) < 1e-12
    assert K.rampe_ansatz(A0, B0[:, 0], 0.01)[1][0] == pytest.approx(0.02369, abs=6e-6)
    ent_h = ent._replace(K=np.array([[0.4, 1.2]]))
    Ach, Bwh = K.kreis(ent_h)
    c0h, _ = K.rampe_ansatz(Ach, Bwh[:, 0], 0.01)
    assert c0h[0] == pytest.approx(0.59, abs=1e-9) and c0h[4] == pytest.approx(-0.06, abs=1e-9)             # 59·s und 6·s Rückstand
    tab = {0.01: (0.576, 2.369), 0.05: (2.851, 2.366), 0.1: (5.539, 2.357), 0.2: (9.892, 2.322), 0.5: (12.467, 2.074), 1.0: (4.568, 1.425), 2.0: (0.183, 0.688)}
    for w, (mit, ohne) in tab.items():
        assert K.gain_ueber_w(Acl, Bw[:, 0], [w])[0] == pytest.approx(mit, abs=6e-4), w
        assert K.gain_ueber_w(A0, B0[:, 0], [w])[0] == pytest.approx(ohne, abs=6e-4), w
    peak, wp = K.max_gain(Acl, Bw[:, 0])
    assert peak == pytest.approx(13.120, abs=6e-4) and wp == pytest.approx(0.3952, abs=6e-5)


# ---- M6 Stellgrenze -------------------------------------------------------------------------------------------------------------------------

def test_m6_stellgrenze_start_4_0():
    z = {r["d"]: r for r in E.stellgrenze_tabelle()}
    assert z[0.3]["ohne"]["p_end"] == pytest.approx(0.71076, abs=6e-6) and abs(z[0.3]["mit"]["p_end"]) < 1e-9
    assert z[0.9]["ohne"]["p_end"] == pytest.approx(2.13228, abs=6e-6) and abs(z[0.9]["mit"]["p_end"]) < 1e-9
    assert (z[1.5]["mit"]["p50"], z[1.5]["mit"]["p100"], z[1.5]["mit"]["p_end"]) == pytest.approx((665.5, 2578.0, 22728.0), abs=0.05)
    assert (z[1.5]["ohne"]["p50"], z[1.5]["ohne"]["p100"], z[1.5]["ohne"]["p_end"]) == pytest.approx((616.5, 2479.0, 22429.0), abs=0.05)
    assert z[0.3]["formel"] == pytest.approx(z[0.3]["ohne"]["p_end"], abs=6e-6)                              # d/K1 ohne aktive Grenze


def test_m6_gleichstand_und_start_in_ruhe():
    ent = _wagen()
    po, _, _ = K.lauf_grenze(ent, 1.0, 1.0, False, T=300, x0=(4.0, 0.0))
    pm, _, _ = K.lauf_grenze(ent, 1.0, 1.0, True, T=300, x0=(4.0, 0.0))
    assert po[300] == 4.0 and pm[300] == pytest.approx(303.0, abs=1e-6)
    pr, _, _ = K.lauf_grenze(ent, 0.9, 1.0, True, T=300)
    assert np.abs(pr).max() == pytest.approx(40.5, abs=1e-6) and abs(pr[-1]) < 1e-9


# ---- M7 Modellfehler ------------------------------------------------------------------------------------------------------------------------

def test_m7_gebiete_der_streckenverstaerkung():
    lo0, hi0 = K.g_gebiet_ohne(_wagen())
    assert lo0 == pytest.approx(0.0005) and hi0 == pytest.approx(1.936, abs=6e-4)
    for rho, (lo, hi) in {0.2: (0.786, 1.170), 0.5: (0.406, 1.293), 0.8: (0.274, 1.785)}.items():
        g = K.g_gebiet(_wagen(1.0, rho))
        assert g == pytest.approx((lo, hi), abs=6e-4), rho


def test_m7_vorgerechnet_g_faelle_und_zufallsstrecken():
    mr = E.load_messreihen()
    neu_g = E.messreihe_g()
    for alt, neu in zip(mr["g"], neu_g):
        assert alt["stabil"] == alt["versuche"] == neu["stabil"] == 2000
        assert alt["max_abs"] < 1e-12 and neu["max_abs"] < 1e-12
    assert max(x["max_abs"] for x in mr["g"]) < 1.2e-14                                                     # README: größter Rest 1,1e-14
    assert [z["stabil"] for z in mr["zufallsstrecken"]] == [3000, 2999, 2641, 1155, 376] and [z["n"] for z in mr["zufallsstrecken"]] == [3000] * 5
    assert mr["zufallsstrecken_summe"]["stabil"] == 10171 and mr["zufallsstrecken_summe"]["n"] == 15000 and mr["zufallsstrecken_summe"]["max_pss"] < 1.4e-14
    neu = E.messreihe_zufallsstrecken()
    for a, b in zip(mr["zufallsstrecken"], neu):
        assert abs(a["stabil"] - b["stabil"]) <= 2                                                          # Zählungen als Band (± 2, Plattform)
        assert b["max_pss"] < 1e-9


def test_m7_falsche_schrittweite_h_gibt_dasselbe_gebiet():
    ent = _wagen()
    L = wagen_L(F(1, 2))

    def stabil(h):
        M, _ = wagen_kreis(ent.K[0, 0], ent.K[0, 1], L, At=np.array([[1.0, h], [0.0, 1.0]]))
        return bool(np.max(np.abs(np.linalg.eigvals(M))) < 1.0)

    lo, hi = K.g_gebiet(ent)
    for h, erwartet in ((lo + 1e-4, True), (lo - 1e-4, False), (hi - 1e-4, True), (hi + 1e-4, False)):
        assert stabil(h) is erwartet, h
    worst = 0.0
    for h in np.linspace(lo + 1e-4, hi - 1e-4, 500):
        M, bd = wagen_kreis(ent.K[0, 0], ent.K[0, 1], L, At=np.array([[1.0, h], [0.0, 1.0]]))
        z = np.linalg.solve(np.eye(5) - M, bd)
        worst = max(worst, abs(z[0]), abs(z[1]))
    assert worst < 1e-7                                                                                     # Dauerfehler 0 in 500 Werten


# ---- M8 Falsche Konstante, Messbias, L_d = 0 --------------------------------------------------------------------------------------------------

def test_m8_falsche_konstante():
    soll = {0.0: 2.369, 0.5: 1.185, 0.9: 0.237, 1.0: 0.0, 1.1: -0.237, 1.5: -1.185}
    for dcal, p in soll.items():
        r = E.konstante_falsch(1.0, 1.0, dcal)
        assert r["p_ss"] == pytest.approx(p, abs=6e-4) and r["p_ss"] == pytest.approx(r["formel"], rel=1e-9, abs=1e-12), dcal


def test_m8_messbias_und_l_d_null():
    for rho in (0.3, 0.5, 0.8):
        Acl, Bw = K.kreis(_wagen(1.0, rho))
        assert K.fixpunkt(Acl, Bw, (0.0, 0.0, 1.0))[0] == pytest.approx(-1.0, abs=1e-9)
        z = K.fixpunkt(Acl, Bw, (0.7, 0.0, -0.4))
        assert z[0] == pytest.approx(0.4, abs=1e-9) and z[4] == pytest.approx(0.7, abs=1e-9)
    from scipy.signal import place_poles

    ent = _wagen()
    L0 = np.vstack([place_poles(ent.A.T, ent.C.T, [0.5, 0.4]).gain_matrix.T, [[0.0]]])
    Acl, Bw = K.kreis(ent._replace(L=L0))
    assert K.spektralradius(Acl) == pytest.approx(1.0, abs=1e-9)
    red = np.linalg.solve(np.eye(4) - Acl[:4, :4], Bw[:4, 0])
    assert red[0] == pytest.approx(16.509, abs=6e-4)


# ---- M9 Tabelle der Beobachterpole ---------------------------------------------------------------------------------------------------------

TABELLE_M9 = {
    # rho: (Spitze, Schritt, Abklingzeit, max|p/d|, omega, std(p), std(u), g_min, g_max, Reserve)
    0.05: (6.000, 4, 9, 8.209, 0.740, 6.749, 19.386, 0.876, 1.127, 0.1125),
    0.1: (6.000, 4, 9, 8.516, 0.701, 5.850, 15.940, 0.851, 1.146, 0.1264),
    0.2: (6.087, 5, 10, 9.239, 0.624, 4.473, 10.792, 0.786, 1.170, 0.1454),
    0.3: (6.942, 5, 12, 10.159, 0.549, 3.483, 7.271, 0.691, 1.200, 0.1660),
    0.5: (9.022, 6, 17, 13.120, 0.395, 2.154, 3.125, 0.406, 1.293, 0.2195),
    0.7: (14.469, 8, 30, 20.727, 0.234, 1.262, 1.112, 0.329, 1.510, 0.3131),
    0.8: (22.523, 11, 46, 32.076, 0.153, 0.894, 0.558, 0.274, 1.785, 0.3988),
    0.9: (55.491, 21, 94, 78.935, 0.074, 0.541, 0.204, 0.203, 2.621, 0.5563),
    0.95: (162.011, 40, 191, 230.385, 0.036, 0.352, 0.086, 0.160, 4.309, 0.7036),
}


def test_m9_tabelle_der_beobachterpole():
    zeilen, ohne = E.beobachter_tabelle(1.0)
    for z in zeilen:
        sp, sch, ab, pk, wp, sdp, sdu, gl, gh, res = TABELLE_M9[z["rho"]]
        assert z["spitze"] == pytest.approx(sp, abs=6e-4) and z["schritt"] == sch and z["abkling"] == ab, z["rho"]
        assert z["peak"] == pytest.approx(pk, abs=6e-4) and z["wpeak"] == pytest.approx(wp, abs=6e-4), z["rho"]
        assert z["std_p"] == pytest.approx(sdp, abs=6e-4) and z["std_u"] == pytest.approx(sdu, abs=6e-4), z["rho"]
        assert z["g_min"] == pytest.approx(gl, abs=6e-4) and z["g_max"] == pytest.approx(gh, abs=6e-4) and z["reserve"] == pytest.approx(res, abs=6e-5), z["rho"]
    assert ohne["spitze"] == pytest.approx(2.376, abs=6e-4) and ohne["dauerfehler"] == pytest.approx(2.369, abs=6e-4) and ohne["peak"] == pytest.approx(2.369, abs=6e-4)
    assert (ohne["g_min"], ohne["g_max"]) == pytest.approx((0.0005, 1.936), abs=6e-4) and ohne["reserve"] == pytest.approx(0.4836, abs=6e-5)


def test_m9_keine_einstellung_senkt_alles_zugleich():
    """Über das Raster: Spitze steigt mit ρ, std(u) fällt, die Abklingzeit steigt; es gibt kein ρ, das bei Spitze UND std(u) zugleich das Minimum hat (9 Zeilen)."""
    zeilen, _ = E.beobachter_tabelle(1.0)
    sp = [z["spitze"] for z in zeilen]
    su = [z["std_u"] for z in zeilen]
    assert all(a <= b for a, b in zip(sp, sp[1:])) and all(a > b for a, b in zip(su, su[1:]))
    assert int(np.argmin(sp)) != int(np.argmin(su))


def test_m9_kalman_und_zustand_gemessen():
    kal = E.kalman_zeilen(1.0)
    for z, (sp, ab) in zip(kal, [(30.35, 471), (22.29, 57), (11.92, 25), (7.18, 11)]):
        assert z["spitze"] == pytest.approx(sp, abs=6e-3) and z["abkling"] == ab, z["qd"]
    for rho, (sp, ab, pk) in {0.1: (3.000, 8, 4.245), 0.3: (3.230, 9, 4.832), 0.5: (3.910, 13, 5.672), 0.8: (6.172, 29, 8.374)}.items():
        z = K.zustand_gemessen_zeile("wagen", 1.0, rho)
        assert z["spitze"] == pytest.approx(sp, abs=6e-4) and z["abkling"] == ab and z["peak"] == pytest.approx(pk, abs=6e-4) and abs(z["p_end"]) < 1e-9


def test_m9_gegenproben_spitzenfolge_in_brueche():
    ent = _wagen()._replace(K=np.array([[0.4, 1.2]]))
    Acl, Bw = K.kreis(ent)
    Z = K.antwort(Acl, Bw, (1.0, 0.0, 0.0), np.zeros(5), 8)
    assert list(Z[1:, 0]) == pytest.approx([0, 1, 3, 6, 67 / 8, 147 / 16, 171 / 20, 5657 / 800], abs=1e-12)


# ---- M10 Kreuzungsschritte ---------------------------------------------------------------------------------------------------------------

def test_m10_wann_zahlt_sich_die_offsetfreiheit_aus():
    soll = {0.1: (7, 11, 20.3), 0.3: (8, 16, 31.1), 0.5: (12, 27, 57.6), 0.7: (20, 70, 159.0), 0.9: (77, 828, 1955.2)}
    for rho, (ab, summe, total) in soll.items():
        r = E.ergebnis(E.Settings(rho=rho))
        assert r["ab_unter"] == ab and r["ab_summe"] == summe, rho
        s = K.sprung(r["ent"], 1.0, T=C.T_LANG)
        assert float(np.abs(s["p"]).sum()) == pytest.approx(total, abs=0.06), rho


# ---- M11 Integralzustand -------------------------------------------------------------------------------------------------------------------

def test_m11_integralzustand():
    soll = {1e-4: (2.294, 466, 2.357, 0.990), 1e-3: (2.164, 151, 2.333, 0.969), 1e-2: (1.899, 51, 2.259, 0.904), 0.1: (1.494, 18, 2.076, 0.718), 1.0: (1.132, 9, 1.772, 0.481)}
    for qi, (sp, ab, pk, rad) in soll.items():
        z = K.integral_zeile("wagen", 1.0, qi)
        assert z["spitze"] == pytest.approx(sp, abs=6e-4) and z["abkling"] == ab and z["peak"] == pytest.approx(pk, abs=6e-4) and z["spektralradius"] == pytest.approx(rad, abs=6e-4), qi
        assert abs(z["p_fest"]) < 1e-9
    for qi, (gl, gh, res) in {1e-4: (0.012, 1.927, 0.481), 1.0: (0.341, 1.520, 0.342)}.items():
        z = K.integral_zeile("wagen", 1.0, qi)
        assert (z["g_min"], z["g_max"]) == pytest.approx((gl, gh), abs=1e-3) and z["reserve"] == pytest.approx(res, abs=6e-4)
    bs = [K.zustand_gemessen_zeile("wagen", 1.0, rho) for rho in (0.1, 0.3, 0.5, 0.8)]
    assert 3.0 <= min(b["spitze"] for b in bs) < 3.01 and 6.17 < max(b["spitze"] for b in bs) < 6.18 and 4.2 < min(b["peak"] for b in bs) and max(b["peak"] for b in bs) < 8.4


# ---- Presets: erwartete Verdikt-Zeile je Preset -----------------------------------------------------------------------------------------------------

def _verdikte(name):
    p = C.PRESETS[name]
    s = E.Settings(p["strecke"], p["R"], p["d"], p["r"], p["rho"], p["b"], E.umax_from_choice(p["umax"]))
    res = E.ergebnis(s)
    rs = E.rampe_sinus(s.R, s.rho, C.DEFAULT_RAMPE, C.DEFAULT_OMEGA)
    return E.kernfragen(s, res, rs)


def test_preset_standard_wagen_mit_beobachter():
    v1, v2, v3 = _verdikte("Mit Störungsbeobachter (ρ = 0,5)")
    assert v1[0] == "warning" and "bleibt der Ort bei 2,369" in v1[1] and "K₁ = 0,4221" in v1[1]
    assert v2[0] == "warning" and "kostet aber Spitze" in v2[1] and "9,02 (Schritt 6)" in v2[1] and "gegen 2,38" in v2[1] and "Faktor 3,8" in v2[1] and "Ab Schritt 12" in v2[1] and "ab Schritt 27" in v2[1]
    assert v3[0] == "info" and "57,6·s" in v3[1] and "13,12" in v3[1] and "2,37" in v3[1]


def test_preset_von_hand_meldet_keinen_preis():
    """Skalar R = 2: Spitze mit und ohne Modell gleich 2: der Text darf keinen Preis melden (Preset ohne Effekt meldet keinen Effekt)."""
    v1, v2, v3 = _verdikte("Von Hand (Skalar, d = 1)")
    assert "bleibt der Ort bei 2,000" in v1[1] and "hier d/k mit k = 0,5000" in v1[1]
    assert v2[0] == "info" and "ohne zusätzliche Spitze" in v2[1] and "kostet aber Spitze" not in v2[1] and "2,00 (Schritt 2) gegen 2,00" in v2[1] and "Faktor 1,0" in v2[1] and "Ab Schritt 4" in v2[1]


def test_preset_kleines_r():
    v1, v2, _ = _verdikte("Kleines R (R = 0,01)")
    assert "bleibt der Ort bei 1,630" in v1[1] and "K₁ = 0,6136" in v1[1]
    assert "7,83" in v2[1] and "gegen 1,63" in v2[1] and "Faktor 4,8" in v2[1] and "kostet aber Spitze" in v2[1]


def test_preset_schneller_und_langsamer_beobachter():
    _, v2, _ = _verdikte("Schneller Beobachter (ρ = 0,1)")
    assert "6,00 (Schritt 4)" in v2[1] and "Abklingzeit des Beobachterkreises: 9 Perioden" in v2[1] and "Ab Schritt 7" in v2[1] and "ab Schritt 11" in v2[1]
    _, v2, _ = _verdikte("Langsamer Beobachter (ρ = 0,9)")
    assert "55,49 (Schritt 21)" in v2[1] and "94 Perioden" in v2[1] and "Ab Schritt 77" in v2[1] and "ab Schritt 828" in v2[1] and "Faktor 23,4" in v2[1]


def test_preset_messbias():
    v1, v2, v3 = _verdikte("Messbias (b = −0,4)")
    assert "Fehler der gemessenen Größe im Beharrungszustand 0,000" in v1[1] and "bleibt der Ort bei 1,658" in v1[1]
    assert "b = −0,40" in v3[1] and "bei 0,40 (= r − b)" in v3[1] and "7,03" in v2[1]


def test_preset_stellgrenze():
    v1, v2, v3 = _verdikte("Stellgrenze (d = 1,5)")
    assert v1[0] == "warning" and "Der Ort bleibt nicht stehen" in v1[1] and "22 724,0 ohne" in v1[1] and "23 319,0 mit" in v1[1]
    assert v2[0] == "warning" and "beseitigt den Fehler hier nicht" in v2[1]
    assert v3[0] == "warning" and "Hier hilft nichts" in v3[1]


# ---- Zahlen der App-Texte -----------------------------------------------------------------------------------------------------------------------

def test_zahlen_der_app_texte_und_der_preset_hilfen():
    """Einleitung, Erklärung und Hilfen nennen 2,369·d, 9,02 gegen 2,38, 13,1 gegen 2,37, g 0,41 bis 1,29 gegen 0 bis 1,94, 57,6·s, 22 728; Preset-Hilfen 1,630, 1,618, 6,0, 55, 94."""
    r = E.ergebnis(E.Settings())
    assert r["ohne"]["p_fest"] == pytest.approx(2.369, abs=6e-4) and r["k_mit"]["spitze"] == pytest.approx(9.02, abs=6e-3) and r["k_ohne"]["spitze"] == pytest.approx(2.38, abs=6e-3)
    z, ohne = E.beobachter_tabelle(1.0)
    z5 = next(x for x in z if x["rho"] == 0.5)
    assert z5["peak"] == pytest.approx(13.1, abs=0.05) and ohne["peak"] == pytest.approx(2.37, abs=6e-3)
    assert (z5["g_min"], z5["g_max"]) == pytest.approx((0.41, 1.29), abs=6e-3) and (ohne["g_min"], ohne["g_max"]) == pytest.approx((0.0, 1.94), abs=6e-3)
    rs = E.rampe_sinus(1.0, 0.5, 0.01, 0.4)
    assert rs["faktor"] == pytest.approx(57.6, abs=0.05)
    assert E.stellgrenze_tabelle()[2]["mit"]["p_end"] == pytest.approx(22728, abs=0.5)
    assert E.dauerfehler_tabelle()[0][2]["p_ss"] == pytest.approx(1.630, abs=6e-4)
    zr = {x["rho"]: x for x in z}
    assert zr[0.1]["spitze"] == pytest.approx(6.0, abs=0.05) and zr[0.9]["spitze"] == pytest.approx(55, abs=0.5) and zr[0.9]["abkling"] == 94 and zr[0.1]["std_u"] == pytest.approx(15.9, abs=0.05)


# ---- README ---------------------------------------------------------------------------------------------------------------------------------

def test_readme_zahlen_stimmen_mit_den_vorgerechneten_reihen_ueberein():
    mr = E.load_messreihen()
    assert E.fmt_int(mr["zufallsstrecken_summe"]["stabil"]) in README
    assert " / ".join(E.fmt_int(z["stabil"]) for z in mr["zufallsstrecken"]) in README
    assert f"{mr['brueche']['exakt']} von {mr['brueche']['n']}" in README
    assert f"Zufalls-Seed {mr['brueche']['seed']}" in README and f"Zufalls-Seed {mr['R']['seed']}" in README
    assert E.fmt_int(mr["zufallsstrecken_summe"]["n"]) in README


def test_readme_testzahl_stimmt_mit_der_suite_ueberein():
    out = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "--collect-only", "-q", "tests", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                         timeout=300).stdout
    n = int(re.search(r"(\d+) tests? collected", out).group(1))
    m = re.search(r"Die Suite hat \*\*(\d+) Tests\*\*", README)
    assert m and int(m.group(1)) == n, (m and m.group(1), n)


def test_readme_hat_keine_offenen_platzhalter_ausser_dem_live_link():
    todos = [z for z in README.splitlines() if "TODO" in z]
    assert len(todos) == 1 and "nach dem Deploy" in todos[0]
