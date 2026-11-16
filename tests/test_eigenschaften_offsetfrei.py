"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Beziehungen, die ohne Kenntnis des richtigen Ergebnisses gelten müssen. Je Eigenschaft 20 bis 300 Instanzen mit festem Zufalls-Seed.

Voraussetzung aller Linearitätsaussagen: das lineare Modell ohne Stellgrenze und ohne Messbias-Grenze; die Aussagen gelten für den Regelkreis, nicht für den Lauf mit Abschneiden."""

from fractions import Fraction as F

import numpy as np
import pytest

import off_dynamics as K
import off_evaluation as E
import off_exakt as X
from off_rng import SplitMix64
from oracle_helpers import wagen_kreis


def _zufall_entwuerfe(n, seed):
    rng = SplitMix64(seed)
    kinds = ("wagen", "skalar1", "skalar_halb")
    for _ in range(n):
        kind = kinds[rng.below(3)]
        R = float(np.power(10.0, -2.0 + 3.0 * rng.random()))
        rho = 0.05 + 0.9 * rng.random()
        yield K.entwurf(kind, R, rho)


def test_linearitaet_und_spiegelung_der_sprungantwort():
    """Störsprung c·d: Ort, Schätzung und Spitze skalieren mit |c|; d → −d kehrt das Vorzeichen um (30 zufällige Entwürfe, c = 2,5 und −1)."""
    for ent in _zufall_entwuerfe(30, 11):
        a = K.sprung(ent, 1.0, T=150)
        b = K.sprung(ent, 2.5, T=150)
        m = K.sprung(ent, -1.0, T=150)
        assert b["p"] == pytest.approx(2.5 * a["p"], rel=1e-9, abs=1e-12)
        assert b["dh"] == pytest.approx(2.5 * a["dh"], rel=1e-9, abs=1e-12)
        assert m["p"] == pytest.approx(-a["p"], rel=1e-9, abs=1e-12)
        assert K.kennzahlen(b["p"])["spitze"] == pytest.approx(2.5 * K.kennzahlen(a["p"])["spitze"], rel=1e-9)
        assert K.kennzahlen(m["p"])["spitze"] == pytest.approx(K.kennzahlen(a["p"])["spitze"], rel=1e-12)


def test_ueberlagerung_zweier_stoerungen_und_zeitverschiebung():
    """Linear: Antwort auf d1 + d2 gleich Summe der Antworten; eine um j Schritte verschobene Störung verschiebt die Antwort (Start in Ruhe)."""
    ent = K.entwurf("wagen", 1.0, 0.5)
    Acl, Bw = K.kreis(ent)
    rng = SplitMix64(5)
    for _ in range(20):
        d1, d2 = rng.random() * 2 - 1, rng.random() * 2 - 1
        z = np.zeros(5)
        pa = [K.antwort(Acl, Bw, (d, 0.0, 0.0), z, 60)[:, 0] for d in (d1, d2, d1 + d2)]
        assert pa[2] == pytest.approx(pa[0] + pa[1], abs=1e-12)
    j = 7
    basis = K.antwort(Acl, Bw, (1.0, 0.0, 0.0), np.zeros(5), 80)[:, 0]
    z = np.zeros(5)
    verschoben = [0.0]
    for k in range(80):
        z = Acl @ z + Bw[:, 0] * (1.0 if k >= j else 0.0)
        verschoben.append(z[0])
    assert verschoben[j:] == pytest.approx(basis[: 81 - j], abs=1e-12)


def test_offsetfreiheit_ist_unabhaengig_von_k_und_l_in_brueche():
    """300 zufällige rationale Paare (K, L) mit stabilem Kreis (L_d ≠ 0, K1 ≠ 0): der Fixpunkt des 5er-Kreises ist in Brüchen exakt (0, 0, 0, 0, 1)."""
    rng = SplitMix64(23)
    n = 0
    versuche = 0
    while n < 300 and versuche < 20000:
        versuche += 1
        k1, k2 = F(rng.below(60) + 1, 100), F(rng.below(191) + 10, 100)
        L = [F(rng.below(300) - 50, 100), F(rng.below(150) - 20, 100), F(rng.below(40) + 1, 100)]
        M, _ = wagen_kreis(float(k1), float(k2), [float(x) for x in L])       # Gleitkomma-Kreis nur zur Stabilitätsprüfung, die Aussage selbst ist exakt
        if np.max(np.abs(np.linalg.eigvals(M))) >= 1 - 1e-6:
            continue
        Lp, Lv, Ld = L
        Mf = [[F(1), F(1), F(0), F(0), F(0)], [F(0), F(1), -k1, -k2, F(-1)], [Lp, F(0), 1 - Lp, F(1), F(0)], [Lv, F(0), -Lv - k1, 1 - k2, F(0)], [Ld, F(0), -Ld, F(0), F(1)]]
        fest = X.fsolve([[(1 if i == j else 0) - Mf[i][j] for j in range(5)] for i in range(5)], [0, 1, 0, 0, 0])
        assert fest == [0, 0, 0, 0, 1], (k1, k2, L)
        n += 1
    assert n == 300


def test_p_ss_ohne_beobachter_ist_monoton_in_r_und_nie_unter_dem_goldenen_schnitt():
    """2 000 R log-gleichverteilt in [1e-8, 1e4] (SplitMix64, ein Zufalls-Seed): 1/K1 wächst mit R, kleinster Wert 1,618034, kein Wert unter dem goldenen Schnitt minus 1e-6."""
    r = E.messreihe_R(2000, 99)
    assert r["rueckgaenge"] == 0 and r["unter_phi"] == 0
    assert r["min"] == pytest.approx(E.PHI, abs=1e-6) and r["max"] > 50


def test_beobachter_ohne_stoerungsverstaerkung_hat_den_eigenwert_eins():
    """L_d = 0: der Kreis hat den Eigenwert 1 (d^ bleibt auf dem Startwert), der Ort bleibt bei einem von 0 verschiedenen Wert (Randfall der Offsetfreiheit)."""
    from scipy.signal import place_poles

    for p2 in ((0.5, 0.4), (0.3, 0.2), (0.7, 0.6)):
        ent = K.entwurf("wagen", 1.0, 0.5)
        pl = place_poles(ent.A.T, ent.C.T, list(p2)).gain_matrix.T
        L0 = np.vstack([pl, [[0.0]]])
        Acl, Bw = K.kreis(ent._replace(L=L0))
        assert K.spektralradius(Acl) == pytest.approx(1.0, abs=1e-9)
        assert np.any(np.abs(np.linalg.eigvals(Acl) - 1.0) < 1e-9)


def test_sollwert_und_stoerung_wirken_addierend_im_fixpunkt():
    """Wagen mit Beobachter: p = r, v = 0 für jeden Sollwert r und jede Störung d (Brüche: r = 1, −3/2 bei d = 1; Gleitkomma: 25 Paare)."""
    ent = K.entwurf("wagen", 1.0, 0.5)
    Acl, Bw = K.kreis(ent)
    rng = SplitMix64(31)
    for _ in range(25):
        d, r, b = rng.random() * 4 - 2, rng.random() * 10 - 5, 0.0
        z = K.fixpunkt(Acl, Bw, (d, r, b))
        assert z[0] == pytest.approx(r, abs=1e-9) and z[1] == pytest.approx(0.0, abs=1e-9) and z[4] == pytest.approx(d, abs=1e-9)
    Lf = X.ackermann(X.fmat([[1, 1, 0], [0, 1, 1], [0, 0, 1]]), X.fmat([[1, 0, 0]]), [F(1, 2)] * 3)
    k1, k2 = F(2, 5), F(6, 5)
    Mf = [[F(1), F(1), 0, 0, 0], [0, F(1), -k1, -k2, F(-1)], [Lf[0], 0, 1 - Lf[0], F(1), 0], [Lf[1], 0, -Lf[1] - k1, 1 - k2, 0], [Lf[2], 0, -Lf[2], 0, F(1)]]
    for r in (F(1), F(-3, 2)):
        rhs = [F(0), 1 + k1 * r, F(0), k1 * r, F(0)]                         # u = ... + k1 r wirkt auf v und v^
        fest = X.fsolve([[(1 if i == j else 0) - Mf[i][j] for j in range(5)] for i in range(5)], rhs)
        assert fest[0] == r and fest[1] == 0 and fest[4] == 1


def test_dauerfehler_haengt_nur_an_k1_nicht_am_beobachter():
    """Der Dauerfehler d/K1 hängt nur an K1: dasselbe K gibt dasselbe p_ss, egal wie der Beobachter gewählt ist (20 zufällige Entwürfe, Wagen)."""
    for ent in _zufall_entwuerfe(20, 7):
        if ent.kind != "wagen":
            continue
        assert K.dauerfehler_lqr(ent) == pytest.approx(1.0 / ent.K[0, 0], rel=1e-10)
        assert K.dauerfehler_lqr(ent) == pytest.approx(K.dauerfehler_lqr(K.entwurf("wagen", ent.R, 0.9)), rel=1e-12)


def test_messbias_verschiebt_nur_den_ort_um_minus_b():
    """Der Beharrungsort mit Beobachter ist r − b, unabhängig von d und ρ (20 Paare)."""
    rng = SplitMix64(41)
    for _ in range(20):
        rho = 0.1 + 0.8 * rng.random()
        ent = K.entwurf("wagen", 1.0, rho)
        Acl, Bw = K.kreis(ent)
        d, b = rng.random() * 4 - 2, rng.random() * 4 - 2
        assert K.fixpunkt(Acl, Bw, (d, 0.0, b))[0] == pytest.approx(-b, abs=1e-9)


def test_rampe_mit_beobachter_gibt_konstanten_ortsfehler_ohne_beobachter_laeuft_er_weg():
    """Für jede Steigung s skaliert der Fehler linear (p = f·s) und seine Steigung ist 0; ohne Beobachter ist die Steigung des Ortes s/K1 (20 Werte s)."""
    ent = K.entwurf("wagen", 1.0, 0.5)
    Acl, Bw = K.kreis(ent)
    A0, B0 = K.kreis_ohne(ent)
    rng = SplitMix64(51)
    f1 = K.rampe_ansatz(Acl, Bw[:, 0], 1.0)[0][0]
    for _ in range(20):
        s = rng.random() * 0.1
        c0, c1 = K.rampe_ansatz(Acl, Bw[:, 0], s)
        assert c0[0] == pytest.approx(f1 * s, rel=1e-9) and abs(c1[0]) < 1e-12
        assert K.rampe_ansatz(A0, B0[:, 0], s)[1][0] == pytest.approx(s / ent.K[0, 0], rel=1e-9)


def test_stabilitaetsgebiet_mit_beobachter_liegt_fuer_kleine_rho_innerhalb_des_gebiets_ohne_beobachter():
    """Gemessen für ρ = 0,05 bis 0,8 (die obere Schranke liegt bei ρ = 0,9 schon darüber): g-Gebiet mit Beobachter ⊂ Gebiet ohne Beobachter. Aussage nur für diese Pole."""
    ent0 = K.entwurf("wagen", 1.0, 0.5)
    lo0, hi0 = K.g_gebiet_ohne(ent0)
    for rho in (0.05, 0.2, 0.5, 0.8):
        lo, hi = K.g_gebiet(K.entwurf("wagen", 1.0, rho))
        assert lo >= lo0 and hi <= hi0, rho
    assert K.g_gebiet(K.entwurf("wagen", 1.0, 0.9))[1] > hi0


def test_gleiche_zufalls_seeds_gleiche_ausgabe():
    a = E.zufallsstrecken(0.1, 200, 77)
    b = E.zufallsstrecken(0.1, 200, 77)
    c = E.zufallsstrecken(0.1, 200, 78)
    assert a == b and a != c
    r1 = E.rausch_laeufe(1.0, (0.5,), 5, 50)[0.5]["u"]
    r2 = E.rausch_laeufe(1.0, (0.5,), 5, 50)[0.5]["u"]
    assert np.array_equal(r1, r2)


def test_integralzustand_offsetfrei_fuer_jedes_qi():
    rng = SplitMix64(61)
    for _ in range(20):
        qi = float(np.power(10.0, -4.0 + 4.0 * rng.random()))
        z = K.integral_zeile("wagen", 1.0, qi)
        assert abs(z["p_fest"]) < 1e-9 and z["spektralradius"] < 1.0
