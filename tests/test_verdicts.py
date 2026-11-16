"""Verdikte an von Hand gesetzten Werten: jede Schwelle der Art und jede Zahl im Text steht hier (reine Funktionen in `off_evaluation.py`)."""

import pytest

import off_constants as C
import off_dynamics as K
import off_evaluation as E


def test_de_und_fmt_int():
    assert E.de(2.3692, 3) == "2,369" and E.de(-1.5, 2) == "−1,50" and E.de(0.5, 1, sign=True) == "+0,5" and E.de(-0.25, 2, sign=True) == "−0,25"
    assert E.fmt_int(10163) == "10 163" and E.fmt_int(999) == "999"


# ---- Kernfrage 1 --------------------------------------------------------------------------------------------------------------------------

def test_dauerfehler_ohne_stoerung_ist_info():
    art, text = E.verdict_dauerfehler(0.0, 0.0, 0.0, 0.0, "irrelevant")
    assert art == "info" and "kein Fehler stehen" in text and "d = 0" in text


def test_dauerfehler_nennt_ort_fehler_und_formel():
    art, text = E.verdict_dauerfehler(1.0, 0.0, 2.3692, 0.0, "am Wagen gleich d/K₁ mit K₁ = 0,4221")
    assert art == "warning"
    assert "Bei d = 1,00 bleibt der Ort bei 2,369" in text and "+2,369 neben dem Ziel r = 0,00" in text and "K₁ = 0,4221" in text
    assert "Fehler der gemessenen Größe im Beharrungszustand 0,000" in text


def test_dauerfehler_mit_sollwert_und_negativer_stoerung():
    art, text = E.verdict_dauerfehler(-0.5, 2.0, 0.8, 0.0, "x")
    assert art == "warning" and "Bei d = −0,50" in text and "−1,200 neben dem Ziel r = 2,00" in text


# ---- Kernfrage 2 --------------------------------------------------------------------------------------------------------------------------

def test_preis_steht_in_derselben_zeile_wie_die_beseitigung():
    art, text = E.verdict_preis(1.0, 9.0217, 6, 2.3764, 2.3692, 17, 12, 27)
    assert art == "warning"
    assert text.startswith("Das Störungsmodell beseitigt den Dauerfehler, die Beseitigung kostet aber Spitze")
    assert "9,02 (Schritt 6)" in text and "gegen 2,38 beim reinen LQR" in text and "bei 2,37 stehen bleibt" in text and "Faktor 3,8" in text
    assert "Abklingzeit des Beobachterkreises: 17 Perioden" in text and "Ab Schritt 12" in text and "die kumulierte Summe ab Schritt 27" in text


def test_preis_schwelle_faktor_eineinhalb_ist_inklusive():
    assert E.verdict_preis(1.0, 3.0, 2, 2.0, 1.0, 5, 1, 1)[0] == "warning"            # genau 1,5
    assert E.verdict_preis(1.0, 2.9, 2, 2.0, 1.0, 5, 1, 1)[0] == "info"               # 1,45


def test_preis_ohne_kreuzung_innerhalb_der_laenge():
    art, text = E.verdict_preis(1.0, 9.0, 6, 2.0, 2.0, 17, None, None)
    assert f"Innerhalb von {E.fmt_int(C.T_LANG)} Schritten" in text and "nicht dauerhaft unter" in text
    art, text = E.verdict_preis(1.0, 9.0, 6, 2.0, 2.0, 17, 12, None)
    assert "Ab Schritt 12" in text and "innerhalb von 6 000 Schritten nicht" in text


def test_preis_varianten_je_nach_faktor():
    assert "kostet aber Spitze" in E.verdict_preis(1.0, 3.0, 2, 2.0, 1.0, 5, 1, 1)[1]
    art, text = E.verdict_preis(1.0, 2.2, 2, 2.0, 1.0, 5, 1, 1)
    assert art == "info" and "kostet hier nur wenig Spitze" in text and "Faktor 1,1" in text
    art, text = E.verdict_preis(1.0, 2.0, 2, 2.0, 1.0, 5, 1, 1)
    assert art == "info" and "ohne zusätzliche Spitze" in text and "Faktor 1,0" in text and "kostet" not in text.replace("kostet hier", "")
    art, text = E.verdict_preis(1.0, 1.5, 2, 2.0, 1.0, 5, 1, 1)                      # Spitze mit Modell kleiner: auch kein Preis
    assert "ohne zusätzliche Spitze" in text


def test_grenze_texte_kernfrage_1_und_2():
    art, text = E.verdict_dauerfehler_grenze(1.5, 1.0, 22724.0, 23319.0)
    assert art == "warning" and "|d| = 1,50 ist größer als die Stellgrenze 1,00" in text and "bei 22 724,0 ohne und bei 23 319,0 mit Störungsmodell" in text and "Nach 300 Schritten" in text
    art, text = E.verdict_preis_grenze(0.9, 1.0, 2.1323, 0.0, 4.5, 40.5)
    assert art == "info" and "beseitigt den Dauerfehler auch mit Stellgrenze" in text and "u_max = 1,00, |d| = 0,90" in text and "0,000 gegen 2,132" in text and "40,50 gegen 4,50" in text
    assert "nicht vollständig" in E.verdict_preis_grenze(0.9, 1.0, 2.1, 0.5, 4.5, 40.5)[1]


def test_formel_text_je_strecke():
    assert E.formel_text("wagen", 0.4221) == "am Wagen gleich d/K₁ mit K₁ = 0,4221"
    assert E.formel_text("skalar1", 0.5) == "hier d/k mit k = 0,5000" and "a = ½" in E.formel_text("skalar_halb", 0.8)


def test_de_hat_kein_minus_null_und_tausendertrenner():
    assert E.de(-1e-17, 3) == "0,000" and E.de(-0.0004, 3) == "0,000" and E.de(-0.0006, 3) == "−0,001" and E.de(-1e-17, 2, sign=True) == "+0,00"
    assert E.de_tsd(22728.0) == "22 728,0" and E.de_tsd(-1234.56, 2) == "−1 234,56" and E.de_tsd(0.04) == "0,0"


def test_preis_ohne_stoerung_nichts_zu_bezahlen():
    assert E.verdict_preis(0.0, 0.0, 0, 0.0, 0.0, 0, None, None) == ("info", "Ohne Störung gibt es nichts zu beseitigen und nichts zu bezahlen.")
    assert E.verdict_preis(1.0, 0.0, 0, 0.0, 0.0, 0, None, None)[0] == "info"         # Spitze ohne Modell 0: kein Nenner null


# ---- Kernfrage 3 --------------------------------------------------------------------------------------------------------------------------

def test_grenzen_reihenfolge_und_schwellen():
    args = dict(rampe_faktor=57.6, peak=13.12, wpeak=0.395, peak0=2.37)
    art, text = E.verdict_grenzen(1.5, 1.0, 0.0, 0.0, **args)
    assert art == "warning" and "Hier hilft nichts" in text and "|d| = 1,50" in text and "Stellgrenze 1,00" in text
    art, text = E.verdict_grenzen(1.0, 1.0, 0.0, 0.0, **args)                         # Gleichstand an der Schwelle |d| = u_max
    assert art == "warning" and "Gleichstand an der Schwelle" in text and "u_max = 1,00" in text
    art, text = E.verdict_grenzen(0.9, 1.0, 0.0, 0.0, **args)                         # unter der Schwelle: die Grenzen der Offsetfreiheit
    assert art == "info" and "57,6·s" in text and "13,12" in text and "ω = 0,395" in text and "2,37" in text
    art, text = E.verdict_grenzen(-1.5, 1.0, 0.0, 0.0, **args)                        # Betrag
    assert art == "warning" and "Hier hilft nichts" in text
    art, text = E.verdict_grenzen(0.7, None, -0.4, 0.4, **args)                       # Messbias
    assert art == "info" and "b = −0,40" in text and "bei 0,40 (= r − b)" in text
    art, text = E.verdict_grenzen(1.5, None, 0.0, 0.0, **args)                        # ohne Stellgrenze gibt es keine Schwelle
    assert "Hier hilft nichts" not in text


# ---- Experimente --------------------------------------------------------------------------------------------------------------------------

def test_dauerfehler_R_text_nennt_werte_und_gegenproben():
    rows = [{"R": 1.0, "p_ss": 2.36921}, {"R": 0.1, "p_ss": 1.72661}, {"R": 0.01, "p_ss": 1.62965}, {"R": 0.001, "p_ss": 1.61920}]
    mr = {"n": 2000, "min": 1.618034, "unter_phi": 0}
    br = {"n": 300, "exakt": 300}
    art, text = E.verdict_dauerfehler_R(rows, mr, br)
    assert art == "info" and "2,369 / 1,727 / 1,630 / 1,619 für R = 1 / 0,1 / 0,01 / 0,001" in text
    assert "In 300 von 300 zufälligen stabilen rationalen Gesetzen" in text and "unter 2 000 zufälligen R" in text and "1,618034" in text
    assert "Vorgerechnet" not in E.verdict_dauerfehler_R(rows, None, None)[1]


def test_beobachter_text_nennt_beide_pole_und_den_vergleich():
    zs = {"rho": 0.1, "spitze": 6.0, "std_u": 15.94}
    zl = {"rho": 0.9, "spitze": 55.4914, "std_u": 0.204, "abkling": 94}
    art, text = E.verdict_beobachter(zs, zl, {"spitze": 2.3764, "dauerfehler": 2.3692})
    assert "ρ = 0,10: Spitze 6,0" in text and "rauscht mit 15,9 je Standardabweichung" in text and "ρ = 0,90: Rauschen 0,20" in text
    assert "Spitze 55,5 und Abklingzeit 94" in text and "ohne Beobachter: Spitze 2,38 mit Dauerfehler 2,37" in text


def test_modell_text_stabil_und_instabil():
    art, text = E.verdict_modell(0.8, True, 1e-15, (0.4062, 1.2931), (0.0005, 1.9363), 0.5, {"stabil": 10171})
    assert art == "success" and "g = 0,80: stabil, Dauerfehler im Fixpunkt 0,000" in text and "von 0,406 bis 1,293" in text and "0,0005 bis 1,936" in text
    assert "0 in 10 171 von 10 171 stabilen Zufallsstrecken" in text
    art, text = E.verdict_modell(0.3, False, None, (0.4062, 1.2931), (0.0005, 1.9363), 0.5, None)
    assert art == "warning" and "**instabil**" in text and "zwischen g = 0,406 und 1,293" in text and "Vorgerechnet" not in text


def test_zwei_wege_text_ist_ein_entwurfsvergleich_kein_satz():
    art, text = E.verdict_zwei_wege({"spitze": 1.4941, "abkling": 18, "peak": 2.0763}, {"spitze": 3.9096, "abkling": 13, "peak": 5.6717}, 0.1, 0.5)
    assert "q_i = 0,1" in text and "Spitze 1,49, Abklingzeit 18, max|p/d| 2,08" in text and "Spitze 3,91, Abklingzeit 13, max|p/d| 5,67" in text
    assert "Pole 0,50, 0,450, 0,400" in text and "kein Satz über Beobachter und Integralzustand" in text


def test_rampe_und_stellgrenze_und_rang_texte():
    art, text = E.verdict_rampe(0.01, 57.58, 0.5758, 0.02369, 13.12, 0.4, 13.1, 2.18)
    assert "Rampe s = 0,010" in text and "0,576 (= 57,6·s)" in text and "Steigung 0,0237" in text and "|p/d| = 13,10 mit, 2,18 ohne" in text and "Spitze des Beobachterkreises ist 13,12" in text
    zeilen = [{"d": 0.3, "ohne": {"p_end": 0.71076}, "mit": {"p_end": 4.7e-16}}, {"d": 0.9, "ohne": {"p_end": 2.13228}, "mit": {"p_end": 0.0}},
              {"d": 1.5, "ohne": {"p_end": 22429.0}, "mit": {"p_end": 22728.0}}]
    art, text = E.verdict_stellgrenze(zeilen)
    assert art == "warning" and "d = 0,3: ohne Modell 0,711, mit Modell 0,000" in text and "d = 0,9: ohne 2,132, mit 0,000" in text and "mit Modell 22 728,0, ohne 22 429,0" in text
    art, text = E.verdict_rang([{"fall": "A", "erfuellt": True}, {"fall": "B", "erfuellt": False}, {"fall": "C", "erfuellt": False}])
    assert "in 1 von 3 Fällen erfüllt" in text and "nicht erfüllt: B; C" in text


def test_eigen_text_und_trennung():
    import numpy as np

    assert E.eigen_text(np.array([0.378 + 0.188j, 0.378 - 0.188j])) == "0,378 ± 0,188 i"
    assert E.eigen_text(np.array([0.5, 0.25])) == "0,250, 0,500"
    t = E.trennung(K.entwurf("wagen", 1.0, 0.5))
    assert t["abstand"] < 1e-12 and len(t["eig_rueck"]) == 2


def test_dauerfehler_formel_stimmt_fuer_alle_strecken():
    for kind in C.STRECKEN:
        ent = K.entwurf(kind, 1.0, 0.5)
        assert E.dauerfehler_formel(ent) == pytest.approx(K.dauerfehler_lqr(ent), rel=1e-12)
