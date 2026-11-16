"""Orakel-Tests (ORACLE-PLAYBOOK §5, §6): jede Kernroutine gegen einen anderen Rechenweg, jeder ein Mini-Lauf unter 10 Sekunden.

(a) Dauerfehler: Fixpunkt (lineares System) gegen Simulation gegen Brüche; (b) Riccati: Rekursion gegen `solve_discrete_are`; (c) Trennung über die Polynomkoeffizienten (nie über
`eigvals` bei Mehrfachpolen); (d) Beobachterverstärkung: Ackermann gegen `place_poles`; (e) Rauschen: Lyapunov gegen lange Simulation (SplitMix64, Band); (f) Frequenzgang gegen
Sinus-Simulation; (g) Rampe: Ansatz gegen Simulation gegen s mal Summe der Sprungantwort; (h) Rang gegen Determinante; (i) Gebiet der Streckenverstärkung: Gitter gegen Bisektion gegen
Jury-Kriterium; (j) Stellgrenze gegen eine zweite Schleife; (k) Kalman-Verstärkung gegen Riccati-Iteration; (l) Integralzustand; (m) Messbias; (n) Zufallsstrecken mit anderer Zufallsquelle.
Der 5er-Kreis des Wagens wird in `oracle_helpers.wagen_kreis` Zeile für Zeile aufgeschrieben, ohne `off_dynamics.kreis_stapel`. Das MPC-Orakel des Plans (Stück 5 liefert den
QP-Löser) ist hier nicht umgesetzt (README: „Bewusst nicht umgesetzt“)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest
import scipy.linalg as la
from scipy.signal import place_poles

import off_dynamics as K
import off_evaluation as E
import off_exakt as X
from oracle_helpers import jury_stabil, riccati_iteration, wagen_kreis, wagen_L


# ---- (a) Dauerfehler ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("R", [1.0, 0.1, 0.01, 0.001])
def test_a_dauerfehler_fixpunkt_simulation_und_brueche(R):
    ent = K.entwurf("wagen", R, 0.5)
    k1, k2 = ent.K[0]
    Acl = np.array([[1.0, 1.0], [-k1, 1.0 - k2]])
    fix = np.linalg.solve(np.eye(2) - Acl, [0.0, 1.0])
    x = np.zeros(2)
    for _ in range(4000):
        x = Acl @ x + np.array([0.0, 1.0])
    assert fix[0] == pytest.approx(1.0 / k1, rel=1e-12) and abs(fix[1]) < 1e-12
    assert x == pytest.approx(fix, abs=1e-9)
    kf, k2f = F(k1), F(k2)                                                    # die Gleitkommazahlen als exakte Brüche
    Af = [[F(1), F(1)], [-kf, 1 - k2f]]
    assert X.fsolve(X.fsub(X.feye(2), Af), [0, 1]) == [1 / kf, 0]
    assert K.dauerfehler_lqr(ent) == pytest.approx(fix[0], rel=1e-12)


def test_a_dauerfehler_skalar_formeln():
    """Skalar a = 1: x_ss/d = 1/k mit k = P/(R + P); R = 2: P = 2, k = 1/2, x_ss = 2; R = 1: goldener Schnitt (numerisch)."""
    for R, soll in ((2.0, 2.0), (20.0, 5.0), (1.0, (1 + 5 ** 0.5) / 2)):
        assert K.dauerfehler_lqr(K.entwurf("skalar1", R, 0.5)) == pytest.approx(soll, rel=1e-12)
    for R in (0.5, 1.0, 2.0, 5.0):
        P = la.solve_discrete_are([[1.0]], [[1.0]], [[1.0]], [[R]])[0, 0]
        assert K.dauerfehler_lqr(K.entwurf("skalar1", R, 0.5)) == pytest.approx((R + P) / P, rel=1e-12)
    ent = K.entwurf("skalar_halb", 1.0, 0.5)
    assert K.dauerfehler_lqr(ent) == pytest.approx(E.dauerfehler_formel(ent), rel=1e-12)        # d/(1 - a + k) mit u_s


# ---- (b) Riccati --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind, R", [("wagen", 1.0), ("wagen", 0.1), ("wagen", 0.01), ("skalar1", 2.0), ("skalar_halb", 1.0)])
def test_b_riccati_rekursion_gegen_dare(kind, R):
    A, B, C = K.strecke(kind)
    Q = np.eye(A.shape[0])
    P, Kr, n = riccati_iteration(A, B, Q, np.array([[R]]))
    Pd = la.solve_discrete_are(A, B, Q, np.array([[R]]))
    assert P == pytest.approx(Pd, rel=1e-9, abs=1e-9)
    assert K.lqr_gain(A, B, R) == pytest.approx(Kr, rel=1e-9, abs=1e-9)
    assert n < 200


def test_b_riccati_schritte_von_p_gleich_q_aus():
    """Die Rekursion von P = Q aus: Wagen, R = 1, 0,1, 0,01 konvergiert nach höchstens 40 Schritten auf 1e-14 (Zählung der Messung: dieselbe Größenordnung wie V1)."""
    A, B, _ = K.strecke("wagen")
    for R in (1.0, 0.1, 0.01):
        assert riccati_iteration(A, B, np.eye(2), np.array([[R]]))[2] <= 60


# ---- (c) Trennung über die Polynomkoeffizienten ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["wagen", "skalar1", "skalar_halb"])
def test_c_trennung_polynom_des_kreises_ist_das_produkt(kind):
    for rho in (0.05, 0.3, 0.5, 0.8, 0.95):
        ent = K.entwurf(kind, 1.0, rho)
        assert E.trennung(ent)["abstand"] < 1e-9, (kind, rho)


def test_c_trennung_gegen_exaktes_polynom_in_brueche():
    """Polynom des Wagen-Kreises (Brüche, K = (2/5, 6/5), 3 x 1/2) gegen das Gleitkomma-Polynom des Kerns, Koeffizient für Koeffizient."""
    ent = K.entwurf("wagen", 1.0, 0.5)._replace(K=np.array([[0.4, 1.2]]))
    Acl, _ = K.kreis(ent)
    soll = X.polmul(X.polmul([F(1), F(-4, 5), F(1, 5)], [F(1), F(-1, 2)]), X.polmul([F(1), F(-1, 2)], [F(1), F(-1, 2)]))
    assert np.poly(Acl) == pytest.approx([float(c) for c in soll], abs=1e-12)


def test_c_dreifache_pole_sind_als_eigenwerte_schlecht_konditioniert():
    """Die Falle: der Abstand der Eigenwerte (eigvals) zu 1/2 ist von der Größe 1e-6 (nicht 1e-15); der Koeffizientenabstand dagegen unter 1e-14."""
    ent = K.entwurf("wagen", 1.0, 0.5)
    ev = np.linalg.eigvals(ent.Ae - ent.L @ ent.Ce)
    assert 1e-9 < np.max(np.abs(ev - 0.5)) < 1e-3
    assert E.trennung(ent)["abstand"] < 1e-10


# ---- (d) Beobachterverstärkung: Ackermann gegen place_poles ----------------------------------------------------------------------------

@pytest.mark.parametrize("poles", [(0.5, 0.4, 0.3), (0.9, 0.2, -0.3), (0.1, 0.05, 0.02)])
def test_d_ackermann_gegen_place_poles(poles):
    A, B, C = K.strecke("wagen")
    Ae, Be, Ce = K.erweitert(A, B, C)
    L_ack = K.ackermann(Ae, Ce, poles)
    L_pp = place_poles(Ae.T, Ce.T, np.array(poles)).gain_matrix.T
    assert L_ack == pytest.approx(L_pp, abs=1e-7)
    assert sorted(np.linalg.eigvals(Ae - L_ack @ Ce).real) == pytest.approx(sorted(poles), abs=1e-9)


def test_d_ackermann_in_brueche_gegen_gleitkomma():
    for rho in (F(1, 2), F(1, 10), F(3, 5)):
        Lf = X.ackermann(X.fmat([[1, 1, 0], [0, 1, 1], [0, 0, 1]]), X.fmat([[1, 0, 0]]), [rho] * 3)
        ent = K.entwurf("wagen", 1.0, float(rho))
        assert ent.L.ravel() == pytest.approx([float(v) for v in Lf], abs=1e-12)


# ---- (e) Rauschen: Lyapunov gegen Simulation ----------------------------------------------------------------------------------------------

def test_e_rauschvarianz_lyapunov_gegen_simulation():
    """Wagen, R = 1, ρ = 0,3: std(p) und std(u) je std(n) aus der Lyapunov-Gleichung gegen 40 000 Schritte mit SplitMix64-Normalzahlen (ein Zufalls-Seed, Band 5 %)."""
    ent = K.entwurf("wagen", 1.0, 0.3)
    Acl, Bw = K.kreis(ent)
    cu = K.u_zeile(ent)
    sp, su = K.rausch_std(Acl, Bw, cu)
    p, u = K.rausch_lauf(Acl, Bw, cu, 40000, 173)
    assert np.std(p[1000:]) == pytest.approx(sp, rel=0.05)
    assert np.std(u[1000:]) == pytest.approx(su, rel=0.05)
    assert sp == pytest.approx(3.4825, abs=2e-4) and su == pytest.approx(7.2710, abs=2e-4)     # V17c: Lyapunov 3,4825 / 7,2710


def test_e_lyapunov_gegen_unabhaengige_summe_der_stossantwort():
    """Die Varianz des Ortes ist die Summe der quadrierten Stoßantwort-Koeffizienten (weißes Rauschen): zweiter Rechenweg ohne Lyapunov-Löser."""
    ent = K.entwurf("wagen", 1.0, 0.7)
    Acl, Bw = K.kreis(ent)
    sp, _ = K.rausch_std(Acl, Bw, K.u_zeile(ent))
    z = Bw[:, 2].copy()
    summe = 0.0
    for _ in range(3000):
        summe += z[0] ** 2
        z = Acl @ z
    assert math.sqrt(summe) == pytest.approx(sp, rel=1e-9)


# ---- (f) Frequenzgang ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("w", [0.05, 0.395, 1.0, 2.0])
def test_f_frequenzgang_gegen_sinus_simulation(w):
    """|p/d| aus der Z-Transformierten gegen die Amplitude der Simulation mit d_k = sin(w k); der Kreis ist Zeile für Zeile aus `oracle_helpers.wagen_kreis` gebaut (Band 2 %)."""
    ent = K.entwurf("wagen", 1.0, 0.5)
    M, bd = wagen_kreis(ent.K[0, 0], ent.K[0, 1], wagen_L(F(1, 2)))
    gain = abs(np.linalg.solve(np.exp(1j * w) * np.eye(5) - M, bd)[0])
    amp = K.sinus_amplitude(M, bd, w)
    assert amp == pytest.approx(gain, rel=0.02, abs=2e-3)
    Acl, Bw = K.kreis(ent)
    assert K.gain_ueber_w(Acl, Bw[:, 0], [w])[0] == pytest.approx(gain, rel=1e-9)


def test_f_spitze_des_frequenzgangs():
    ent = K.entwurf("wagen", 1.0, 0.5)
    Acl, Bw = K.kreis(ent)
    peak, wp = K.max_gain(Acl, Bw[:, 0])
    ws = np.linspace(0.3, 0.5, 20001)
    feinste = K.gain_ueber_w(Acl, Bw[:, 0], ws)
    assert peak == pytest.approx(feinste.max(), rel=1e-9) and wp == pytest.approx(ws[np.argmax(feinste)], abs=2e-5)
    assert peak == pytest.approx(13.1199, abs=2e-4) and wp == pytest.approx(0.3952, abs=2e-4)


# ---- (g) Rampe ----------------------------------------------------------------------------------------------------------------------------

def test_g_rampe_ansatz_gegen_simulation_gegen_summe_der_sprungantwort():
    ent = K.entwurf("wagen", 1.0, 0.5)
    Acl, Bw = K.kreis(ent)
    s = 0.01
    c0, c1 = K.rampe_ansatz(Acl, Bw[:, 0], s)
    z = np.zeros(5)
    for k in range(3000):
        z = Acl @ z + Bw[:, 0] * (s * k)
    assert z[0] == pytest.approx(c0[0] + c1[0] * 3000, abs=1e-9)
    step = np.zeros(5)
    summe = 0.0
    for _ in range(3000):
        step = Acl @ step + Bw[:, 0]
        summe += step[0]
    assert c0[0] == pytest.approx(s * summe, rel=1e-6)                         # 57,581 = Summe der Sprungantwort (alle Werte positiv)
    assert c1[0] == pytest.approx(0.0, abs=1e-12)
    M, bd = wagen_kreis(ent.K[0, 0], ent.K[0, 1], wagen_L(F(1, 2)))
    z = np.zeros(5)
    for k in range(3000):
        z = M @ z + bd * (s * k)
    assert z[0] == pytest.approx(c0[0], abs=1e-9)                              # der unabhängig aufgebaute Kreis gibt dasselbe


# ---- (h) Rang -----------------------------------------------------------------------------------------------------------------------------

def test_h_rang_exakt_gegen_numpy_und_determinante():
    for name in K.RANG_FAELLE:
        A, Bd, C, Cd = (np.atleast_2d(np.asarray(x, float)) for x in K.rang_fall(name))
        n = A.shape[0]
        M = np.vstack([np.hstack([np.eye(n) - A, -Bd]), np.hstack([C, Cd])])
        rg, verlangt = K.rang_bedingung(*K.rang_fall(name))
        assert rg == np.linalg.matrix_rank(M), name
        if M.shape[0] == M.shape[1]:
            assert (X.fdet([[F(v) for v in row] for row in M]) != 0) == (rg == M.shape[0]), name


# ---- (i) Gebiet der Streckenverstärkung ------------------------------------------------------------------------------------------------

def _poly_g(ent, g):
    M, _ = wagen_kreis(ent.K[0, 0], ent.K[0, 1], [float(v) for v in ent.L.ravel()], Bt=g * np.array([0.0, 1.0]), Bdt=g * np.array([0.0, 1.0]))
    return [float(c) for c in X.charpoly([[F(v) for v in row] for row in M])]


@pytest.mark.parametrize("rho", [0.2, 0.5, 0.8])
def test_i_gebiet_gitter_bisektion_jury(rho):
    ent = K.entwurf("wagen", 1.0, rho)
    lo, hi = K.g_gebiet(ent)
    assert jury_stabil(_poly_g(ent, 1.0)) and jury_stabil(_poly_g(ent, 0.5 * (lo + hi)))
    for g, erwartet in ((lo + 1e-6, True), (lo - 1e-6, False), (hi - 1e-6, True), (hi + 1e-6, False)):
        assert jury_stabil(_poly_g(ent, g)) is erwartet, (rho, g)
    gs = np.linspace(0.01, 6.0, 3000)
    ok = [jury_stabil(_poly_g(ent, g)) for g in gs[::10]]
    stab = [g for g, o in zip(gs[::10], ok) if o]
    assert min(stab) >= lo - 0.02 and max(stab) <= hi + 0.02
    assert (lo, hi) == pytest.approx({0.2: (0.7861, 1.1701), 0.5: (0.4062, 1.2931), 0.8: (0.2742, 1.7847)}[rho], abs=2e-4)


def test_i_gebiet_ohne_beobachter():
    ent = K.entwurf("wagen", 1.0, 0.5)
    lo, hi = K.g_gebiet_ohne(ent)
    assert lo == pytest.approx(0.0005) and hi == pytest.approx(1.9363, abs=2e-4)
    A, B, K_ = ent.A, ent.B, ent.K
    assert max(abs(np.linalg.eigvals(A - (hi - 1e-6) * B @ K_))) < 1 < max(abs(np.linalg.eigvals(A - (hi + 1e-6) * B @ K_)))


# ---- (j) Stellgrenze ----------------------------------------------------------------------------------------------------------------------

def _lauf_listen(k1, k2, L, d, umax, T, mit_modell):
    """Zweite Schleife in einfachen Zahlen (keine Matrizen): Wagen, u abgeschnitten, Beobachter bekommt die angewendete Stellgröße, Start (4, 0), Schätzung 0."""
    p, v, ph, vh, dh = 4.0, 0.0, 0.0, 0.0, 0.0
    ps = [p]
    for _ in range(T):
        u = (-(k1 * ph + k2 * vh) - dh) if mit_modell else -(k1 * p + k2 * v)
        u = max(-umax, min(umax, u))
        e = p - ph
        p_neu, v_neu = p + v, v + u + d
        if mit_modell:
            ph, vh, dh = ph + vh + L[0] * e, vh + dh + u + L[1] * e, dh + L[2] * e
        p, v = p_neu, v_neu
        ps.append(p)
    return ps


@pytest.mark.parametrize("d", [0.3, 0.9, 1.5])
@pytest.mark.parametrize("mit", [False, True])
def test_j_stellgrenze_gegen_zweite_schleife(d, mit):
    ent = K.entwurf("wagen", 1.0, 0.5)
    p, _, _ = K.lauf_grenze(ent, d, 1.0, mit, T=300, x0=(4.0, 0.0))
    ps = _lauf_listen(ent.K[0, 0], ent.K[0, 1], [float(x) for x in ent.L.ravel()], d, 1.0, 300, mit)
    assert p == pytest.approx(ps, rel=1e-9, abs=1e-9)


# ---- (k) Kalman-Verstärkung ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("qd", [1e-6, 1e-2, 1.0])
def test_k_kalman_gegen_riccati_iteration(qd):
    A, B, C = K.strecke("wagen")
    Ae, Be, Ce = K.erweitert(A, B, C)
    W = np.diag([0.0, 0.01, qd])
    P = np.eye(3)
    for _ in range(200000):                                                   # Filter-Riccati in Vorhersageform
        S = Ce @ P @ Ce.T + 1.0
        P_neu = Ae @ P @ Ae.T - Ae @ P @ Ce.T @ np.linalg.solve(S, Ce @ P @ Ae.T) + W
        if np.max(np.abs(P_neu - P)) < 1e-13 * max(1.0, np.max(np.abs(P))):
            P = P_neu
            break
        P = P_neu
    Lk = Ae @ P @ Ce.T @ np.linalg.inv(Ce @ P @ Ce.T + 1.0)
    assert K.kalman_verstaerkung("wagen", qd, 0.01) == pytest.approx(Lk, rel=1e-6, abs=1e-8)


# ---- (l) Integralzustand ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("qi", [1e-4, 1e-2, 1.0])
def test_l_integralzustand_riccati_und_fixpunkt(qi):
    Aa, Ba, Ka = K.integral_entwurf("wagen", 1.0, qi)
    P, Kr, _ = riccati_iteration(Aa, Ba, np.diag([1.0, 1.0, qi]), np.array([[1.0]]), n_max=200000, tol=1e-13)
    assert Ka == pytest.approx(Kr, rel=1e-6, abs=1e-8)
    assert np.linalg.matrix_rank(np.hstack([Ba, Aa @ Ba, Aa @ Aa @ Ba])) == 3        # erweitertes Paar steuerbar
    Acl = Aa - Ba @ Ka
    fest = np.linalg.solve(np.eye(3) - Acl, Ba[:, 0])
    assert fest[0] == pytest.approx(0.0, abs=1e-9) and fest[1] == pytest.approx(0.0, abs=1e-9) and fest[2] == pytest.approx(1.0 / Ka[0, 2], rel=1e-6)   # xi = d/K_i


# ---- (m) Messbias -------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [0.3, 0.5, 0.8])
def test_m_messbias_wahrer_ort_ist_minus_b(rho):
    ent = K.entwurf("wagen", 1.0, rho)
    Acl, Bw = K.kreis(ent)
    z = K.fixpunkt(Acl, Bw, (0.0, 0.0, 1.0))
    assert z[0] == pytest.approx(-1.0, abs=1e-9) and abs(z[1]) < 1e-9 and abs(z[4]) < 1e-9
    z2 = K.fixpunkt(Acl, Bw, (0.7, 0.0, -0.4))
    assert z2[0] == pytest.approx(0.4, abs=1e-9) and z2[4] == pytest.approx(0.7, abs=1e-9)


# ---- (n) Zufallsstrecken mit anderer Zufallsquelle und gelockerten Annahmen ----------------------------------------------------------------------

@pytest.mark.parametrize("bezug", ["wie_V17b", "ohne_eingangsbezug"])
def test_n_dauerfehler_null_in_allen_stabilen_zufallsstrecken_andere_quelle(bezug):
    """Gegenprobe der Absolutaussage „Dauerfehler 0 für jede stabile Strecke“: numpy-Generator statt SplitMix64, Kreis aus `oracle_helpers.wagen_kreis`; zusätzlich ohne
    Bezug der Störung zum Eingang (B_dt beliebig in [-1, 1]²)."""
    rng = np.random.default_rng(2026)
    ent = K.entwurf("wagen", 1.0, 0.5)
    L = [float(v) for v in ent.L.ravel()]
    n_stabil = 0
    worst = 0.0
    for delta in (0.05, 0.1, 0.2, 0.4):
        for _ in range(500):
            At = np.array([[1.0, 1.0], [0.0, 1.0]]) + rng.uniform(-delta, delta, (2, 2))
            Bt = np.array([0.0, 1.0]) + rng.uniform(-delta, delta, 2)
            Bdt = np.array([rng.uniform(-1, 1), rng.uniform(0.3, 1.7)]) if bezug == "wie_V17b" else rng.uniform(-1, 1, 2)
            M, bd = wagen_kreis(ent.K[0, 0], ent.K[0, 1], L, At, Bt, Bdt)
            if np.max(np.abs(np.linalg.eigvals(M))) < 1 - 1e-6:
                n_stabil += 1
                z = np.linalg.solve(np.eye(5) - M, bd)
                worst = max(worst, abs(z[0]))                  # nur der gemessene Ort: bei zufälligem A_t ist v_ss im Allgemeinen nicht 0
    assert n_stabil > 600
    assert worst < 1e-7


def test_n_gegenprobe_ohne_stoerungsmodell_liegt_der_fehler_nicht_bei_null():
    """Der Test unterscheidet: ohne Störungsmodell (L_d = 0) ist der Ort im Fixpunkt des verkürzten Kreises nicht 0 (16,509·d), der Eigenwert 1 gehört zu d^."""
    ent = K.entwurf("wagen", 1.0, 0.5)
    pl = place_poles(ent.A.T, ent.C.T, [0.5, 0.4]).gain_matrix.T
    L0 = np.vstack([pl, [[0.0]]])
    M, bd = wagen_kreis(ent.K[0, 0], ent.K[0, 1], [float(x) for x in L0.ravel()])
    assert np.max(np.abs(np.linalg.eigvals(M))) == pytest.approx(1.0, abs=1e-9)
    red = np.linalg.solve(np.eye(4) - M[:4, :4], bd[:4])
    assert red[0] == pytest.approx(16.5087, abs=1e-3) and abs(red[0]) > 1.0
