"""Kern und Handbeispiele (DEMO-PLAYBOOK §1b): jede Einheit an einem Beispiel, dessen Ergebnis von Hand (in Brüchen) feststeht.

Das Handbeispiel: Skalar x+ = x + u + d, Kosten Summe (x² + 2 u²), konstante Störung d = 1. Die Werte unten sind von Hand gerechnet (Riccati, Fixpunkte, Polynome) und
stehen als Brüche im Test, nicht als Ergebnis des Codes."""

from fractions import Fraction as F

import numpy as np
import pytest

import off_dynamics as K
import off_evaluation as E
import off_exakt as X


# ---- Handbeispiel 1 bis 4: Skalar, Brüche -----------------------------------------------------------------------------------------

def test_riccati_von_hand():
    """P_0 = 1, P_1 = 1 + 1 - 1/3 = 5/3, P_2 = 21/11; DARE P = 2, k = P/(2 + P) = 1/2, Pol 1 - k = 1/2."""
    h = E.hand_skalar()
    assert h["P"] == [F(1), F(5, 3), F(21, 11)]
    assert h["Pinf"] == 2 and h["k"] == F(1, 2)
    assert 1 + F(2) - F(2) ** 2 / (2 + F(2)) == 2                           # P = 1 + P - P²/(2 + P) bei P = 2
    ent = K.entwurf("skalar1", 2.0, 0.5)
    assert ent.K[0, 0] == pytest.approx(0.5, abs=1e-12)


def test_dauerfehler_von_hand_skalar():
    """x+ = x/2 + 1: x = 0, 1, 3/2, 7/4, ... = 2 (1 - 2^-k); Fixpunkt d/k = 2."""
    h = E.hand_skalar()
    assert h["ohne"][:5] == [F(0), F(1), F(3, 2), F(7, 4), F(15, 8)]
    assert all(h["ohne"][j] == 2 * (1 - F(1, 2 ** j)) for j in range(9))
    assert h["dauerfehler"] == 2


def test_beobachter_von_hand_skalar():
    """Fehlermatrix [[1 - Lx, 1], [-Ld, 1]]: Polynom z² - (2 - Lx) z + (1 - Lx + Ld); Pole 1/2 und 1/4 geben Lx = 5/4, Ld = 3/8 (z² - 3/4 z + 1/8)."""
    h = E.hand_skalar()
    assert (h["Lx"], h["Ld"]) == (F(5, 4), F(3, 8))
    assert [F(1), -(2 - h["Lx"]), 1 - h["Lx"] + h["Ld"]] == [F(1), F(-3, 4), F(1, 8)]
    assert X.polmul([F(1), F(-1, 2)], [F(1), F(-1, 4)]) == [F(1), F(-3, 4), F(1, 8)]


def test_verlauf_mit_beobachter_von_hand():
    """Von Hand: u_0 = 0, x_1 = 1; x_2 = 2, x^_2 = 5/4, d^_2 = 3/8; u_2 = -5/8 - 3/8 = -1, x_3 = 2 - 1 + 1 = 2. Die Folge x_k = 0, 1, 2, 2, 25/16, 69/64, ..."""
    z = E.hand_skalar()["zeilen"]
    assert [r[1] for r in z[:6]] == [F(0), F(1), F(2), F(2), F(25, 16), F(69, 64)]
    assert z[2][2] == F(5, 4) and z[2][3] == F(3, 8) and z[3][2] == F(25, 16) and z[3][3] == F(21, 32)
    assert z[8][1] == F(1025, 4096) and z[8][3] == F(32385, 32768)
    assert max(r[1] for r in z) == 2 and z[2][1] == z[3][1] == 2                 # Spitze 2 = Dauerfehler des reinen LQR (Zufall dieser Zahlen)


def test_kreispolynom_und_fixpunkt_von_hand():
    """Polynom des Kreises (x, x^, d^): z³ - 5/4 z² + 1/2 z - 1/16 = (z - 1/2)² (z - 1/4); Fixpunkt (0, 0, 1): Dauerfehler 0, d^ = d."""
    h = E.hand_skalar()
    assert h["poly"] == [F(1), F(-5, 4), F(1, 2), F(-1, 16)]
    assert h["poly"] == X.poly_from_roots([F(1, 2), F(1, 2), F(1, 4)])
    assert h["fest"] == [0, 0, 1]


def test_integralzustand_von_hand():
    """xi+ = xi + x, u = -x - xi/4: Polynom z² - z + 1/4 = (z - 1/2)², Fixpunkt (0, 4), x_k = k / 2^(k-1) = 1, 1, 3/4, 1/2, 5/16, 3/16."""
    k1, ki = F(1), F(1, 4)
    M = [[1 - k1, -ki], [F(1), F(1)]]
    assert X.charpoly(M) == [F(1), F(-1), F(1, 4)]
    assert X.fsolve([[1 - M[0][0], -M[0][1]], [-M[1][0], 1 - M[1][1]]], [1, 0]) == [0, 4]
    x, xi, xs = F(0), F(0), [F(0)]
    for _ in range(10):
        x, xi = (1 - k1) * x - ki * xi + 1, xi + x
        xs.append(x)
    assert xs[1:7] == [F(1), F(1), F(3, 4), F(1, 2), F(5, 16), F(3, 16)] and all(xs[j] == F(j, 2 ** (j - 1)) for j in range(1, 11))
    assert max(xs) == 1


def test_sollwert_von_hand():
    """x+ = x/2 + u, k = 1/4, r = 1: ohne Zielwert x_ss = 1/3 (Fehler 2/3 = r (1 - a)/(1 - a + b k)); mit u_s = (1 - a) r/b = 1/2 Fehler 0."""
    t = E.sollwert_tabelle()
    assert t["x_ohne"] == F(1, 3) and t["fehler_ohne"] == F(2, 3) == t["formel_ohne"]
    assert t["u_s"] == F(1, 2) and t["x_mit"] == 1


def test_zielrechnung_von_hand_floats():
    """[[1 - a, -b], [1, 0]] (x_s, u_s) = (b d^, r) am Skalar a = 1/2: d^ = 0: (1, 1/2); d^ = 1/4: (1, 1/4); der Wagen braucht kein u_s (a = 1)."""
    ent = K.entwurf("skalar_halb", 2.0, 0.5)
    Tx, Tu = K.zielkoeff(ent)
    assert np.allclose(Tx[:, 1], [1.0]) and Tu[0, 1] == pytest.approx(0.5)             # r = 1, d^ = 0
    xs = Tx[:, 0] * 0.25 + Tx[:, 1] * 1.0
    us = Tu[0, 0] * 0.25 + Tu[0, 1] * 1.0
    assert xs[0] == pytest.approx(1.0) and us == pytest.approx(0.25)
    wag = K.entwurf("wagen", 1.0, 0.5)
    Tx, Tu = K.zielkoeff(wag)
    assert np.allclose(Tx, [[0.0, 1.0], [0.0, 0.0]]) and np.allclose(Tu, [[-1.0, 0.0]])  # x_s = (r, 0), u_s = -d^


# ---- Handbeispiel 6 und 7: Wagen ---------------------------------------------------------------------------------------------------

def _wagen_kreis_exakt(k1, k2, L):
    """5er-Kreis (p, v, p^, v^, d^) des Wagens mit u = -K x^ - d^ in Brüchen: unabhängiger Aufbau ohne Matrix-Hilfsfunktionen des Kerns."""
    Lp, Lv, Ld = L
    zeilen = [
        [1, 1, 0, 0, 0],                                       # p+  = p + v
        [0, 1, -k1, -k2, -1],                                  # v+  = v + u + d,  u = -k1 p^ - k2 v^ - d^
        [Lp, 0, 1 - Lp, 1, 0],                                 # p^+ = p^ + v^ + Lp (p - p^)
        [Lv, 0, -Lv - k1, 1 - k2, 0],                          # v^+ = v^ + d^ + u + Lv (p - p^)
        [Ld, 0, -Ld, 0, 1],                                    # d^+ = d^ + Ld (p - p^)
    ]
    return [[F(x) for x in z] for z in zeilen]


def test_wagen_dauerfehler_von_hand():
    """K = (2/5, 6/5): Polynom z² - 4/5 z + 1/5, Dauerfehler p_ss = d/K1 = 5/2·d, v_ss = 0 (Brüche)."""
    k1, k2 = F(2, 5), F(6, 5)
    Aw = [[F(1), F(1)], [-k1, 1 - k2]]
    assert X.charpoly(Aw) == [F(1), F(-4, 5), F(1, 5)]
    assert X.fsolve(X.fsub(X.feye(2), Aw), [0, 1]) == [F(5, 2), 0]
    ent = K.entwurf("wagen", 1.0, 0.5)
    Acl, Bw = K.kreis_ohne(ent)
    Acl_k = np.array([[1.0, 1.0], [-0.4, -0.2]])
    assert np.allclose(np.linalg.solve(np.eye(2) - Acl_k, [0.0, 1.0]), [2.5, 0.0])


@pytest.mark.parametrize("R, soll", [(1.0, 2.36921), (0.1, 1.72661), (0.01, 1.62965)])
def test_lqr_dauerfehler_zweiter_rechenweg(R, soll):
    """1/K1 aus der DARE (SciPy) gegen die Zahl von V17a (fünf Stellen); der Fixpunkt des Kerns gleich 1/K1."""
    ent = K.entwurf("wagen", R, 0.5)
    assert 1.0 / ent.K[0, 0] == pytest.approx(soll, abs=6e-6)
    assert K.dauerfehler_lqr(ent) == pytest.approx(1.0 / ent.K[0, 0], rel=1e-12)


def test_wagen_beobachter_von_hand():
    """Ackermann für 3 x 1/2: L = (3/2, 3/4, 1/8); Polynom des 5er-Kreises = (z² - 4/5 z + 1/5)(z - 1/2)³; Fixpunkt (0, 0, 0, 0, 1); p_1..p_8 = 0, 1, 3, 6, 67/8, 147/16, 171/20, 5657/800."""
    Ae = X.fmat([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    Lf = X.ackermann(Ae, X.fmat([[1, 0, 0]]), [F(1, 2)] * 3)
    assert Lf == [F(3, 2), F(3, 4), F(1, 8)]
    Acf = _wagen_kreis_exakt(F(2, 5), F(6, 5), Lf)
    assert X.charpoly(Acf) == X.polmul(X.polmul([F(1), F(-4, 5), F(1, 5)], [F(1), F(-1, 2)]), X.polmul([F(1), F(-1, 2)], [F(1), F(-1, 2)]))
    bd = [F(0), F(1), F(0), F(0), F(0)]
    assert X.fsolve([[(1 if i == j else 0) - Acf[i][j] for j in range(5)] for i in range(5)], bd) == [0, 0, 0, 0, 1]
    z, ps = [F(0)] * 5, []
    for _ in range(8):
        z = [sum(Acf[i][j] * z[j] for j in range(5)) + bd[i] for i in range(5)]
        ps.append(z[0])
    assert ps == [0, 1, 3, 6, F(67, 8), F(147, 16), F(171, 20), F(5657, 800)]


def test_kern_gleich_handbeispiel_wagen_gleitkomma():
    """Der Kern (Gleitkomma, K aus der DARE ersetzt durch K = (2/5, 6/5) über eine Kopie des Entwurfs) liefert p_1..p_8 des Handbeispiels."""
    ent = K.entwurf("wagen", 1.0, 0.5)
    ent = ent._replace(K=np.array([[0.4, 1.2]]))
    Acl, Bw = K.kreis(ent)
    Z = K.antwort(Acl, Bw, (1.0, 0.0, 0.0), np.zeros(5), 8)
    assert list(Z[1:, 0]) == pytest.approx([0, 1, 3, 6, 67 / 8, 147 / 16, 171 / 20, 5657 / 800], abs=1e-12)


def test_rampe_von_hand_wagen():
    """Rampe d_k = s k: der Ort wird konstant 59 s (K = (2/5, 6/5), Beobachter 3 x 1/2, s = 1/100), d^ liegt um 6 s hinter der Rampe."""
    ent = K.entwurf("wagen", 1.0, 0.5)._replace(K=np.array([[0.4, 1.2]]))
    Acl, Bw = K.kreis(ent)
    c0, c1 = K.rampe_ansatz(Acl, Bw[:, 0], 0.01)
    assert c0[0] == pytest.approx(0.59, abs=1e-9) and c1[0] == pytest.approx(0.0, abs=1e-9)
    assert c0[4] == pytest.approx(-0.06, abs=1e-9) and c1[4] == pytest.approx(0.01, abs=1e-9)


def test_rang_von_hand():
    """Wagen, Eingang: Determinante ungleich 0; Wagen, Ausgang: 0; Skalar a = 1 Ausgang: 0; Skalar a = 1/2 Ausgang: 1/2."""
    assert X.fdet([[0, -1, 0], [0, 0, -1], [1, 0, 0]]) != 0
    assert X.fdet([[0, -1, 0], [0, 0, 0], [1, 0, 1]]) == 0
    assert X.fdet([[0, 0], [1, 1]]) == 0
    assert X.fdet([[F(1, 2), 0], [1, 1]]) == F(1, 2)
    erwartet = {"Wagen, Störung am Eingang": (3, 3), "Wagen, Störung am Ausgang": (2, 3), "Wagen, zwei Störungen bei einer Messung": (3, 4),
                "Skalar a = 1, Störung am Eingang": (2, 2), "Skalar a = 1, Störung am Ausgang": (1, 2), "Skalar a = ½, Störung am Ausgang": (2, 2)}
    for name in K.RANG_FAELLE:
        assert K.rang_bedingung(*K.rang_fall(name)) == erwartet[name], name


# ---- Einheiten des Kerns ----------------------------------------------------------------------------------------------------------

def test_strecken_und_gewinn():
    A, B, C = K.strecke("wagen")
    assert A.tolist() == [[1, 1], [0, 1]] and B.ravel().tolist() == [0, 1] and C.tolist() == [[1, 0]]
    K1 = K.lqr_gain(A, B, 1.0)
    assert K1[0] == pytest.approx([0.42208244, 1.24392885], abs=1e-7)                 # K = (0,42208; 1,24393)
    with pytest.raises(ValueError):
        K.strecke("kugel")


def test_erweitertes_modell_und_beobachter_pole():
    A, B, C = K.strecke("wagen")
    Ae, Be, Ce = K.erweitert(A, B, C)
    assert Ae.tolist() == [[1, 1, 0], [0, 1, 1], [0, 0, 1]] and Be.ravel().tolist() == [0, 1, 0] and Ce.tolist() == [[1, 0, 0]]
    assert K.beobachter_pole("wagen", 0.5) == (0.5, 0.5, 0.5) and K.beobachter_pole("skalar1", 0.5) == (0.5, 0.25)
    L = K.ackermann(Ae, Ce, (0.5, 0.5, 0.5))
    assert L.ravel() == pytest.approx([1.5, 0.75, 0.125])


def test_kennzahlen_spitze_schritt_abklingzeit():
    p = [0.0, 1.0, 3.0, 6.0, 2.0, 0.5, 0.05, 0.0]
    k = K.kennzahlen(p)
    assert k == {"spitze": 6.0, "schritt": 3, "abkling": 5}                           # letzte Periode über 1 % von 6 (= 0,06): Schritt 5 (0,5), 6 ist 0,05
    assert K.kennzahlen([0.0, 0.0, 0.0]) == {"spitze": 0.0, "schritt": 0, "abkling": 0}
    assert K.kennzahlen([5.0, 5.0, 5.0], ziel=5.0)["spitze"] == 0.0                   # Abweichung vom Ziel, nicht der Ort


def test_kreuzungsschritte():
    mit = [3.0, 1.0, 0.5, 0.1, 0.0]
    ohne = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert E.kreuzung(mit, ohne) == (3, 5)                                            # |mit| ist ab Schritt 3 kleiner; die Summe 4,6 liegt erst bei Schritt 5 unter 5
    assert E.kreuzung([2.0, 2.0], [1.0, 1.0]) == (None, None)
    assert E.kreuzung([0.5, 0.5], [1.0, 1.0]) == (1, 1)


def test_horizont_und_sprung_grundfaelle():
    ent = K.entwurf("wagen", 1.0, 0.5)
    s = K.sprung(ent, 1.0)
    assert s["p"][0] == 0.0 and s["T"] >= 100 and s["p_fest"] == pytest.approx(0.0, abs=1e-12)
    s0 = K.sprung(ent, 1.0, mit_beobachter=False)
    assert s0["p_fest"] == pytest.approx(1 / ent.K[0, 0], rel=1e-12) and s0["dh"] is None
    s_null = K.sprung(ent, 0.0)
    assert np.all(s_null["p"] == 0.0)                                                # ohne Störung bleibt der Wagen in Ruhe


def test_sollwert_wagen_ohne_zielwert():
    ent = K.entwurf("wagen", 1.0, 0.5)
    Acl, Bw = K.kreis_ohne(ent)
    z = K.fixpunkt(Acl, Bw, (0.0, 2.5, 0.0))
    assert z == pytest.approx([2.5, 0.0], abs=1e-12)                                  # p = r, v = 0 (V17a B)


def test_stellgrenze_ohne_aktive_grenze_ist_der_lineare_lauf():
    ent = K.entwurf("wagen", 1.0, 0.5)
    s = K.sprung(ent, 0.3, T=60)
    p, u, dd = K.lauf_grenze(ent, 0.3, 1e6, True, T=60)
    assert p == pytest.approx(s["p"], abs=1e-12)


def test_stellgrenze_betrag_der_stoerung_groesser_als_umax_laeuft_weg():
    """v+ = v + d + u >= v + d - u_max: bei d > u_max wächst die Geschwindigkeit in jedem Schritt um mindestens d - u_max, mit und ohne Modell."""
    ent = K.entwurf("wagen", 1.0, 0.5)
    for modell in (False, True):
        p, u, dd = K.lauf_grenze(ent, 1.5, 1.0, modell, T=100, x0=(4.0, 0.0))
        v = np.diff(p)
        assert np.all(np.diff(v) >= 0.5 - 1e-12)                                      # Beschleunigung mindestens 0,5 je Schritt
        assert p[-1] > 1000


def test_beobachter_pol_eins_ist_abgelehnt():
    with pytest.raises(ValueError):
        K.entwurf("wagen", 1.0, 1.0)
    with pytest.raises(ValueError):
        K.entwurf("wagen", 0.0, 0.5)
