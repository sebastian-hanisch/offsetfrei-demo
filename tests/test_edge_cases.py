"""Randfälle (DEMO-PLAYBOOK §12): die Ränder der Regler, die Entartungen und die Schwellen. Jeder Test benennt den Randfall, den er prüft."""

import numpy as np
import pytest

import off_constants as C
import off_dynamics as K
import off_evaluation as E


@pytest.mark.parametrize("rho", [0.0, 0.01, 0.05, 0.1])
def test_schneller_beobachter_hat_die_spitze_sechs(rho):
    """ρ → 0 (Deadbeat im Rahmen der Gleitkommagenauigkeit): Spitze 6,0000 bei Schritt 4 für alle ρ bis 0,1 (R = 1, Wagen); die Abklingzeit ist 8 bis 9."""
    e = K.entwurf("wagen", 1.0, rho)
    k = K.kennzahlen(K.sprung(e, 1.0)["p"])
    assert k["spitze"] == pytest.approx(6.0, abs=1e-3) and k["schritt"] == 4 and k["abkling"] in (8, 9)


def test_langsamer_beobachter_rho_095():
    z = K.beobachter_zeile("wagen", 1.0, 0.95)
    assert z["spitze"] == pytest.approx(162.0108, abs=2e-3) and z["schritt"] == 40 and z["abkling"] == 191


def test_rho_eins_ist_abgelehnt_und_rho_null_erlaubt():
    with pytest.raises(ValueError):
        K.entwurf("wagen", 1.0, 1.0)
    with pytest.raises(ValueError):
        K.entwurf("wagen", 1.0, -0.1)
    assert K.entwurf("wagen", 1.0, 0.0).L.shape == (3, 1)


@pytest.mark.parametrize("R", [1e-8, 1e4])
def test_extreme_stellgewichte(R):
    """R → 10⁻⁸ und R = 10⁴: der Dauerfehler bleibt 1/K1 (1,618034 bzw. 107,3), der Beobachterkreis ist stabil."""
    e = K.entwurf("wagen", R, 0.5)
    assert K.dauerfehler_lqr(e) == pytest.approx(1.0 / e.K[0, 0], rel=1e-10)
    assert K.dauerfehler_lqr(e) == pytest.approx({1e-8: 1.618034, 1e4: 107.349}[R], abs=2e-3)
    s = K.sprung(e, 1.0)
    assert s["rho"] < 1.0 and abs(s["p_fest"]) < 1e-9


def test_keine_stoerung_keine_kennzahlen_und_kein_nenner_null():
    r = E.ergebnis(E.Settings(d=0.0))
    assert r["k_mit"] == {"spitze": 0.0, "schritt": 0, "abkling": 0} and r["ab_unter"] is None and r["ab_summe"] is None
    assert E.verdict_preis(0.0, 0.0, 0, 0.0, 0.0, 0, None, None)[0] == "info"


@pytest.mark.parametrize("kind", C.STRECKEN)
def test_jede_strecke_mit_sollwert_stoerung_und_bias(kind):
    """Mit Beobachter: der wahre Ort geht auf r − b, ohne Beobachter auf r + d·Formel (Wagen, Skalar a = 1: 1/K1; Skalar a = ½: 1/(1 − a + k))."""
    s = E.Settings(kind, 1.0, 0.7, 2.0, 0.5, -0.4)
    r = E.ergebnis(s)
    assert r["mit"]["p_fest"] == pytest.approx(2.0 + 0.4, abs=1e-9)
    assert r["ohne"]["p_fest"] == pytest.approx(2.0 + 0.7 * r["formel"], abs=1e-9)
    assert np.all(np.isfinite(r["mit"]["p"])) and np.all(np.isfinite(r["ohne"]["p"]))


def test_messbias_abklingzeit_misst_gegen_den_beharrungswert():
    """Mit Messbias geht der Ort nicht auf das Ziel, sondern auf r − b: die Abklingzeit misst gegen diesen Wert, nicht gegen das Ziel (sonst stünde dort die Länge des Laufs)."""
    r = E.ergebnis(E.Settings(d=0.7, b=-0.4))
    assert r["k_mit"]["abkling"] < r["T"] and r["mit"]["p_fest"] == pytest.approx(0.4, abs=1e-9)


def test_skalar_handbeispiel_spitze_gleich_dauerfehler():
    """Skalar, R = 2, d = 1, Pole ½ und ¼: Spitze mit Modell 2 (Schritt 2) und ohne Modell 2 (Dauerfehler), Faktor 1: der Preis ist dort nicht sichtbar."""
    r = E.ergebnis(E.Settings("skalar1", 2.0, 1.0, 0.0, 0.5))
    assert r["k_mit"]["spitze"] == pytest.approx(2.0, abs=1e-12) and r["k_mit"]["schritt"] == 2 and r["ohne"]["p_fest"] == pytest.approx(2.0)
    assert r["mit"]["p"][:6] == pytest.approx([0, 1, 2, 2, 25 / 16, 69 / 64], abs=1e-12)


def test_stellgrenze_gleichstand_d_gleich_umax():
    """|d| = u_max: v+ >= v, die Geschwindigkeit kann nicht mehr abgebaut werden. Start (4; 0): ohne Modell bleibt der Wagen bei 4,0 (u = −1 hebt d = 1 genau auf), mit Modell läuft er
    mit der Geschwindigkeit 1 weg (Ort 303 nach 300 Schritten, R = 1, ρ = 0,5). Gleichstand an der Schwelle, beide Läufe stehen in der README."""
    e = K.entwurf("wagen", 1.0, 0.5)
    p0, u0, _ = K.lauf_grenze(e, 1.0, 1.0, False, T=300, x0=(4.0, 0.0))
    p1, u1, dd = K.lauf_grenze(e, 1.0, 1.0, True, T=300, x0=(4.0, 0.0))
    assert np.all(p0 == 4.0) and np.all(u0 == -1.0)
    assert p1[300] == pytest.approx(303.0, abs=1e-9) and p1[100] == pytest.approx(103.0, abs=1e-9) and dd[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(np.diff(p1)) >= -1e-12)                                    # nie langsamer als zuvor


def test_stellgrenze_unter_der_schwelle_und_start_in_ruhe():
    """d = 0,9 < u_max = 1 aus der Ruhe: mit Modell geht der Ort auf 0, aber die Spitze ist 40,5 (vermutlich verlängert das Abschneiden den Anlauf; nicht einzeln gemessen); ohne Modell 2,132."""
    e = K.entwurf("wagen", 1.0, 0.5)
    pm, _, _ = K.lauf_grenze(e, 0.9, 1.0, True, T=300)
    po, _, _ = K.lauf_grenze(e, 0.9, 1.0, False, T=300)
    assert abs(pm[-1]) < 1e-9 and np.abs(pm).max() == pytest.approx(40.5, abs=1e-6) and po[-1] == pytest.approx(0.9 / e.K[0, 0], abs=1e-4)


def test_stoerung_am_ausgang_beim_wagen_hat_rang_zwei_von_drei():
    """Wagen, Störung am Ausgang (B_d = 0, C_d = 1): Rang 2 von 3, die Rangbedingung ist verletzt, ein Beobachter ist nicht entwerfbar (die App zeigt dort keine Zahl)."""
    rg, verlangt = K.rang_bedingung(*K.rang_fall("Wagen, Störung am Ausgang"))
    assert (rg, verlangt) == (2, 3)
    tab = E.rang_tabelle()
    assert [t["erfuellt"] for t in tab] == [True, False, False, True, False, True]


def test_umax_option_aus_und_floats():
    assert C.UMAX_OPTIONS[0] == "aus" and all(isinstance(v, float) for v in C.UMAX_OPTIONS[1:])
    assert min(C.UMAX_OPTIONS[1:]) > 0


def test_horizont_grenzen():
    e = K.entwurf("wagen", 1.0, 0.95)
    assert 100 <= K.horizont(K.kreis(e)[0]) <= 6000
    assert K.horizont(np.array([[1.0, 0.0], [0.0, 0.5]])) == 100                      # Spektralradius 1: Vorgabe


def test_zufallsstrecken_ohne_stabile_gibt_null_statt_fehler():
    z = E.zufallsstrecken(0.4, 5, 1)
    assert z["n"] == 5 and 0 <= z["stabil"] <= 5 and z["max_pss"] >= 0.0
    assert E.zufallsstrecken(0.02, 1, 3)["n"] == 1


def test_g_gebiet_nicht_stabil_bei_g_eins_gibt_nan():
    """Ist der Entwurf selbst nicht stabil (hier künstlich: L = 0), meldet das Gebiet NaN statt einer Zahl."""
    e = K.entwurf("wagen", 1.0, 0.5, L=np.zeros((3, 1)))
    lo, hi = K.g_gebiet(e)
    assert np.isnan(lo) and np.isnan(hi)


def test_dauerfehler_tabelle_hat_vier_und_fuenf_zeilen():
    w, s = E.dauerfehler_tabelle()
    assert [r["R"] for r in w] == list(C.R_TABELLE) and [r["R"] for r in s] == list(C.R_SKALAR)
    assert all(a["p_ss"] < b["p_ss"] for a, b in zip(w[1:], w[:-1]))                  # kleineres R, kleinerer Fehler
