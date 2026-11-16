"""Auswertungen der Experimente, vorgerechnete Messreihen und Verdikte als reine Funktionen (Art, Text).

Die App ruft `getattr(st, art)(text)`; `tests/test_verdicts.py` prüft jede Schwelle und jede Zahl im Text an handgesetzten Werten.
Rechnerische Grundlagen stehen in `off_dynamics.py` (Gleitkomma) und `off_exakt.py` (Brüche)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

import off_constants as C
import off_dynamics as K
import off_exakt as X
from off_rng import SplitMix64

PHI = (1.0 + math.sqrt(5.0)) / 2.0     # goldener Schnitt: numerisch gleich dem Grenzwert von 1/K1 für R gegen 0 (nicht hergeleitet)


def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma und Minuszeichen (Anzeige); ein auf null gerundeter Wert hat kein Minuszeichen."""
    if round(x, nd) == 0:
        x = 0.0
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def de_tsd(x: float, nd: int = 1) -> str:
    """Wie `de`, mit Leerzeichen als Tausendertrenner (22728,0 -> 22 728,0)."""
    s = f"{(0.0 if round(x, nd) == 0 else x):,.{nd}f}"
    return s.replace(",", " ").replace(".", ",").replace("-", "−")


def fmt_int(n: int) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner (10163 -> 10 163)."""
    return f"{int(n):,}".replace(",", " ")


# ---------------------------------------------------------------------------------------------------------------------------------
# Einstellungen und Ergebnis (Primäransicht)
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    strecke: str = C.DEFAULT_STRECKE
    R: float = C.DEFAULT_R
    d: float = C.DEFAULT_D
    r: float = C.DEFAULT_SOLL
    rho: float = C.DEFAULT_RHO
    b: float = C.DEFAULT_B
    umax: float | None = None           # None: keine Stellgrenze


def umax_from_choice(choice) -> float | None:
    return None if choice == C.UMAX_AUS else float(choice)


def trennung(ent: K.Entwurf) -> dict:
    """Trennungsprinzip: charakteristisches Polynom des Kreises gegen das Produkt aus dem Polynom von A - B K und dem des Beobachters (größter Koeffizientenabstand).
    Verglichen wird über die Koeffizienten, nicht über Eigenwerte (dreifache Pole sind als Eigenwerte schlecht konditioniert)."""
    Acl, _ = K.kreis(ent)
    pol_kreis = np.poly(Acl)
    pol_soll = np.polymul(np.poly(ent.A - ent.B @ ent.K), np.poly(ent.Ae - ent.L @ ent.Ce))
    return {"abstand": float(np.max(np.abs(pol_kreis - pol_soll))), "eig_rueck": np.linalg.eigvals(ent.A - ent.B @ ent.K)}


def eigen_text(eigs) -> str:
    """Eigenwerte als Text: reelle einzeln, komplexe Paare als „a ± b i“ (nur das Paar mit positivem Imaginärteil)."""
    teile = []
    for v in sorted(eigs, key=lambda c: (c.real, c.imag)):
        if abs(v.imag) < 1e-9:
            teile.append(de(float(v.real), 3))
        elif v.imag > 0:
            teile.append(f"{de(float(v.real), 3)} ± {de(float(v.imag), 3)} i")
    return ", ".join(teile)


def dauerfehler_formel(ent: K.Entwurf) -> float:
    """Geschlossene Formel des Dauerfehlers ohne Störungsmodell je Einheit d: Wagen und Skalar a = 1: 1/K1; Skalar a < 1: 1/(1 - a + k) (Ziel r, Zielwert u_s)."""
    if ent.kind == "skalar_halb":
        return 1.0 / (1.0 - float(ent.A[0, 0]) + float(ent.K[0, 0]))
    return 1.0 / float(ent.K[0, 0])


def kreuzung(dev_mit, dev_ohne):
    """Erster Schritt, ab dem |p - r| mit Störungsmodell dauerhaft unter dem ohne liegt, und erster Schritt, ab dem die kumulierte Summe dauerhaft darunter liegt.
    Beide Felder beginnen bei Schritt 1; None, wenn es innerhalb der Länge nicht eintritt."""
    dev_mit, dev_ohne = np.asarray(dev_mit, float), np.asarray(dev_ohne, float)

    def ab(klein):
        schlecht = np.nonzero(~klein)[0]
        if schlecht.size == 0:
            return 1
        i = int(schlecht[-1])
        return None if i == len(klein) - 1 else i + 2

    return ab(dev_mit < dev_ohne), ab(np.cumsum(dev_mit) < np.cumsum(dev_ohne))


def ergebnis(s: Settings) -> dict:
    """Die Primäransicht: Störsprung d (und Messbias b) bei k = 0, Start in Ruhe im Sollwert r; ohne und mit Störungsmodell."""
    ent = K.entwurf(s.strecke, s.R, s.rho)
    T = max(K.horizont(K.kreis(ent)[0]), 100)
    mit = K.sprung(ent, s.d, s.r, s.b, True, T=T)
    ohne = K.sprung(ent, s.d, s.r, s.b, False, T=T)
    lang_mit = K.sprung(ent, s.d, s.r, s.b, True, T=C.T_LANG)["p"]
    lang_ohne = K.sprung(ent, s.d, s.r, s.b, False, T=C.T_LANG)["p"]
    ab_unter, ab_summe = kreuzung(np.abs(lang_mit[1:] - s.r), np.abs(lang_ohne[1:] - s.r))
    k_mit = K.kennzahlen(mit["p"], s.r, mit["p_fest"])
    k_ohne = K.kennzahlen(ohne["p"], s.r, ohne["p_fest"])
    out = {"ent": ent, "mit": mit, "ohne": ohne, "k_mit": k_mit, "k_ohne": k_ohne, "ab_unter": ab_unter, "ab_summe": ab_summe, "T": T,
           "formel": dauerfehler_formel(ent), "grenze": None}
    if s.umax is not None:
        x0 = K.zielkoeff(ent)[0][:, 1] * s.r
        for name, modell in (("g_mit", True), ("g_ohne", False)):
            p, u, dd = K.lauf_grenze(ent, s.d, s.umax, modell, T=C.T_GRENZE, x0=x0, r=s.r, b=s.b)
            out[name] = {"p": p, "u": u, "dh": dd}
        out["grenze"] = s.umax
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel in Brüchen (Skalar, R = 2, d = 1)
# ---------------------------------------------------------------------------------------------------------------------------------

def hand_skalar(schritte: int = 8) -> dict:
    """Das Handbeispiel in Brüchen: Skalar x+ = x + u + d, Kosten x² + 2 u², Beobachterpole 1/2 und 1/4, Störsprung d = 1 aus der Ruhe."""
    a = b = F(1)
    Q, R = F(1), F(2)
    P = [Q]
    for _ in range(2):
        P.append(Q + a * a * P[-1] - (a * P[-1] * b) ** 2 / (R + b * b * P[-1]))
    Pinf = F(2)
    k = a * b * Pinf / (R + b * b * Pinf)
    ohne = [F(0)]
    for _ in range(schritte):
        ohne.append((1 - k) * ohne[-1] + 1)
    Ae = X.fmat([[1, 1], [0, 1]])
    Ce = X.fmat([[1, 0]])
    Lx, Ld = X.ackermann(Ae, Ce, [F(1, 2), F(1, 4)])
    x = xh = dh = F(0)
    zeilen = [(F(0), x, xh, dh)]
    for j in range(schritte):
        u = -k * xh - dh
        x, xh, dh = x + u + 1, xh + u + dh + Lx * (x - xh), dh + Ld * (x - xh)
        zeilen.append((F(j + 1), x, xh, dh))
    Mk = [[F(1), -k, F(-1)], [Lx, 1 - Lx - k, F(0)], [Ld, -Ld, F(1)]]
    poly = X.charpoly(Mk)
    fest = X.fsolve([[(1 if i == j else 0) - Mk[i][j] for j in range(3)] for i in range(3)], [1, 0, 0])
    return {"P": P, "Pinf": Pinf, "k": k, "dauerfehler": 1 / k, "ohne": ohne, "Lx": Lx, "Ld": Ld, "zeilen": zeilen, "poly": poly, "fest": fest}


# ---------------------------------------------------------------------------------------------------------------------------------
# Experimente (Hauptbereich)
# ---------------------------------------------------------------------------------------------------------------------------------

def dauerfehler_tabelle():
    """p_ss/d des reinen LQR (Wagen, Ziel 0) für R = 1, 0,1, 0,01, 0,001 und am Skalar a = 1 für R = 0,5 bis 20 (x_ss/d = 1/k)."""
    wagen = []
    for R in C.R_TABELLE:
        ent = K.entwurf("wagen", R, 0.5)
        wagen.append({"R": R, "K1": float(ent.K[0, 0]), "p_ss": K.dauerfehler_lqr(ent), "radius": float(K.spektralradius(K.kreis_ohne(ent)[0]))})
    skalar = []
    for R in C.R_SKALAR:
        ent = K.entwurf("skalar1", R, 0.5)
        skalar.append({"R": R, "k": float(ent.K[0, 0]), "x_ss": K.dauerfehler_lqr(ent)})
    return wagen, skalar


def sollwert_tabelle(r_wagen: float = 2.5, r_skalar: float = 1.0):
    """Sollwert ohne Störung: Wagen (Fixpunkt p = r, weil er selbst integriert) und Skalar a = 1/2 mit k = 1/4 (kein LQR) ohne und mit Zielwert u_s, in Brüchen."""
    ent = K.entwurf("wagen", 1.0, 0.5)
    Acl, Bw = K.kreis_ohne(ent)
    p_wagen = float(K.fixpunkt(Acl, Bw, (0.0, r_wagen, 0.0))[0])
    a, bb, k = F(1, 2), F(1), F(1, 4)
    r = F(r_skalar).limit_denominator(1000)
    x_ohne = bb * k * r / (1 - a + bb * k)
    u_s = (1 - a) * r / bb
    x_mit = (bb * (k * r + u_s)) / (1 - a + bb * k)
    return {"r_wagen": r_wagen, "p_wagen": p_wagen, "r": r, "x_ohne": x_ohne, "x_mit": x_mit, "u_s": u_s, "fehler_ohne": r - x_ohne,
            "formel_ohne": r * (1 - a) / (1 - a + bb * k)}


def beobachter_tabelle(R: float):
    """Zeilen des Experiments B (Wagen, R, Beobachter 3 x rho) und die Vergleichszeile ohne Beobachter."""
    zeilen = [K.beobachter_zeile("wagen", R, rho) for rho in C.RHO_TABELLE]
    return zeilen, K.ohne_zeile("wagen", R)


def kalman_zeilen(R: float):
    zeilen = []
    for qd in C.QD_KALMAN:
        Lk = K.kalman_verstaerkung("wagen", qd, C.KALMAN_V)
        z = K.beobachter_zeile("wagen", R, None, L=Lk)
        z["qd"] = qd
        zeilen.append(z)
    return zeilen


def rausch_laeufe(R: float, rhos, seed: int, T: int = C.T_RAUSCH):
    """Stellgröße bei weißem Messrauschen (Zufalls-Seed, SplitMix64), dieselbe Rauschfolge für alle Pole `rhos` (Wagen)."""
    out = {}
    for rho in rhos:
        ent = K.entwurf("wagen", R, rho)
        Acl, Bw = K.kreis(ent)
        p, u = K.rausch_lauf(Acl, Bw, K.u_zeile(ent), T, seed)
        out[rho] = {"p": p, "u": u}
    return out


def zwei_wege(R: float, rho: float, qi: float, T: int = 120):
    """Experiment D (Wagen, Zustand gemessen, Sollwert 0, Störsprung d = 1): Integralzustand gegen Störungsbeobachter (Pole rho, 0,9 rho, 0,8 rho)."""
    integral = K.integral_zeile("wagen", R, qi)
    beob = K.zustand_gemessen_zeile("wagen", R, rho)
    return {"integral": integral, "beobachter": beob, "T": T}


def modell_g(R: float, rho: float, g: float):
    """Experiment C: wahre Streckenverstärkung g (B_t = g B, B_dt = g B) bei Entwurf für g = 1: stabil? Fixpunkt je Einheit d, Gebiete mit und ohne Beobachter."""
    ent = K.entwurf("wagen", R, rho)
    Acl, Bw = K.kreis_g(ent, [g])
    rad = float(K.spektralradius(Acl)[0])
    stabil = rad < 1.0
    fest = K.fixpunkt(Acl[0], Bw[0], (1.0, 0.0, 0.0)) if stabil else None
    # ohne Beobachter: u = -K x mit wahrer Strecke g B
    Acl0 = ent.A - g * ent.B @ ent.K
    rad0 = float(K.spektralradius(Acl0))
    fest0 = float(np.linalg.solve(np.eye(2) - Acl0, g * ent.B[:, 0])[0]) if rad0 < 1.0 else None
    return {"radius": rad, "stabil": stabil, "p_ss": None if fest is None else float(fest[0]), "radius_ohne": rad0, "stabil_ohne": rad0 < 1.0, "p_ss_ohne": fest0,
            "gebiet": K.g_gebiet(ent), "gebiet_ohne": K.g_gebiet_ohne(ent)}


def gebiet_ueber_rho(R: float):
    """Gebiet der stabilen Streckenverstärkung für jedes rho der Tabelle (Wagen), ohne Beobachter als Vergleich."""
    return [{"rho": rho, "gebiet": K.g_gebiet(K.entwurf("wagen", R, rho))} for rho in C.RHO_TABELLE], K.g_gebiet_ohne(K.entwurf("wagen", R, 0.5))


def zufallsstrecken(delta: float, n: int, seed: int, R: float = 1.0, rho: float = 0.5):
    """Zufallsstrecken: je Eintrag von A und B ± delta, B_dt = (u(-1, 1), u(0,3 bis 1,7)); Entwurf nominell (Wagen). Rückgabe: Zahl der stabilen (Spektralradius unter 1 - 1e-6), größter |p_ss|."""
    ent = K.entwurf("wagen", R, rho)
    rng = SplitMix64(seed)
    u = np.array([rng.random() for _ in range(n * 8)]).reshape(n, 8)
    At = ent.A[None] + delta * (2.0 * u[:, 0:4].reshape(n, 2, 2) - 1.0)
    Bt = ent.B[None] + delta * (2.0 * u[:, 4:6].reshape(n, 2, 1) - 1.0)
    Bdt = np.stack([-1.0 + 2.0 * u[:, 6], 0.3 + 1.4 * u[:, 7]], axis=1)[:, :, None]
    Acl, Bw = K.kreis_stapel(ent, At=At, Bt=Bt, Bdt=Bdt)
    stabil = K.spektralradius(Acl) < 1.0 - 1e-6
    if not stabil.any():
        return {"n": n, "stabil": 0, "max_pss": 0.0}
    nz = Acl.shape[1]
    A_s, b_s = Acl[stabil], Bw[stabil][:, :, 0]
    z = np.linalg.solve(np.eye(nz)[None] - A_s, b_s[:, :, None])[:, :, 0]
    return {"n": n, "stabil": int(stabil.sum()), "max_pss": float(np.abs(z[:, 0]).max())}


def konstante_falsch(R: float, d: float, dcal: float):
    """Eine einmal falsch gesetzte Konstante: statische Vorsteuerung u = -K x - d_cal ohne Beobachter. p_ss = (d - d_cal)/K1 (Wagen); gerechnet und per Formel."""
    ent = K.entwurf("wagen", R, 0.5)
    Acl, _ = K.kreis_ohne(ent)
    p = float(np.linalg.solve(np.eye(2) - Acl, ent.B[:, 0] * (d - dcal))[0])
    return {"p_ss": p, "formel": (d - dcal) / float(ent.K[0, 0])}


def rampe_sinus(R: float, rho: float, s: float, w: float):
    """Experiment E (a): Rampe d_k = s k und Sinus d_k = sin(w k) am Wagen: mit und ohne Beobachter."""
    ent = K.entwurf("wagen", R, rho)
    Acl, Bw = K.kreis(ent)
    A0, B0 = K.kreis_ohne(ent)
    c0, c1 = K.rampe_ansatz(Acl, Bw[:, 0], s)
    o0, o1 = K.rampe_ansatz(A0, B0[:, 0], s)
    peak, wpeak = K.max_gain(Acl, Bw[:, 0])
    peak0, wpeak0 = K.max_gain(A0, B0[:, 0])
    gain = float(K.gain_ueber_w(Acl, Bw[:, 0], [w])[0])
    gain0 = float(K.gain_ueber_w(A0, B0[:, 0], [w])[0])
    ws = K.W_GRID[::12]
    faktor = float(K.rampe_ansatz(Acl, Bw[:, 0], 1.0)[0][0])          # Ortsfehler je Einheit Steigung (unabhängig von s)
    return {"c0": c0, "c1": c1, "ohne_c0": o0, "ohne_c1": o1, "faktor": faktor, "peak": peak, "wpeak": wpeak, "peak0": peak0, "wpeak0": wpeak0,
            "gain": gain, "gain0": gain0, "ws": ws, "kurve": K.gain_ueber_w(Acl, Bw[:, 0], ws), "kurve0": K.gain_ueber_w(A0, B0[:, 0], ws), "K1": float(ent.K[0, 0])}


def rampe_verlauf(R: float, rho: float, s: float, T: int = 150):
    """Ort bei der Rampe d_k = s k aus der Ruhe, mit und ohne Beobachter (Simulation)."""
    ent = K.entwurf("wagen", R, rho)
    Acl, Bw = K.kreis(ent)
    A0, B0 = K.kreis_ohne(ent)
    z = np.zeros(Acl.shape[0])
    x = np.zeros(2)
    pm, po = [0.0], [0.0]
    for k in range(T):
        z = Acl @ z + Bw[:, 0] * (s * k)
        x = A0 @ x + B0[:, 0] * (s * k)
        pm.append(float(z[0]))
        po.append(float(x[0]))
    return np.array(pm), np.array(po)


def stellgrenze_tabelle(R: float = 1.0, rho: float = 0.5):
    """Experiment E (b): Stellgrenze |u| <= 1, Start (4, 0), d = 0,3 / 0,9 / 1,5, ohne und mit Störungsmodell (Abschneiden, angewendete Stellgröße im Beobachter)."""
    ent = K.entwurf("wagen", R, rho)
    zeilen = []
    for d in C.STELL_D:
        row = {"d": d}
        for name, modell in (("ohne", False), ("mit", True)):
            p, u, dd = K.lauf_grenze(ent, d, C.STELL_UMAX, modell, T=C.T_GRENZE, x0=C.STELL_START)
            row[name] = {"p": p, "p50": float(p[50]), "p100": float(p[100]), "p_end": float(p[-1]), "spitze": float(np.abs(p).max()), "dh_end": float(dd[-1]), "n_gesaettigt": int(np.sum(np.abs(u) >= C.STELL_UMAX - 1e-12))}
        row["formel"] = d / float(ent.K[0, 0])
        zeilen.append(row)
    return zeilen


def rang_tabelle():
    return [{"fall": name, "rang": rg, "verlangt": ver, "erfuellt": rg == ver} for name in K.RANG_FAELLE for rg, ver in [K.rang_bedingung(*K.rang_fall(name))]]


# ---------------------------------------------------------------------------------------------------------------------------------
# Vorgerechnete Messreihen (data/off_messreihen.json, erzeugt von tools/off_make_messreihen.py)
# ---------------------------------------------------------------------------------------------------------------------------------

def messreihe_R(n: int = 2000, seed: int = 174, lo: float = -8.0, hi: float = 4.0) -> dict:
    """n Werte R log-gleichverteilt in [10^lo, 10^hi] (Wagen, Q = I): 1/K1 wächst mit R und liegt nie unter dem goldenen Schnitt."""
    rng = SplitMix64(seed)
    exps = np.array([lo + (hi - lo) * rng.random() for _ in range(n)])
    Rs = np.sort(np.power(10.0, exps))
    A, B, _ = K.strecke("wagen")
    vals = np.array([1.0 / K.lqr_gain(A, B, R)[0, 0] for R in Rs])
    return {"n": n, "R_lo": float(Rs[0]), "R_hi": float(Rs[-1]), "min": float(vals.min()), "max": float(vals.max()), "rueckgaenge": int(np.sum(np.diff(vals) < -1e-9)),
            "unter_phi": int(np.sum(vals < PHI - 1e-6)), "phi": PHI, "seed": seed}


def messreihe_brueche(n: int = 300, seed: int = 17) -> dict:
    """n zufällige stabile rationale Gesetze K = (k1, k2) am Wagen: Fixpunkt in Brüchen gleich (d/k1, 0)?"""
    rng = SplitMix64(seed)
    ok = versuche = 0
    while versuche < n:
        k1 = F(rng.below(60) + 1, 100)
        k2 = F(rng.below(191) + 10, 100)
        Acl = np.array([[1.0, 1.0], [-float(k1), 1.0 - float(k2)]])
        if np.abs(np.linalg.eigvals(Acl)).max() >= 1.0 - 1e-9:
            continue
        versuche += 1
        Af = X.fmat([[1, 1], [-k1, 1 - k2]])
        xs = X.fsolve(X.fsub(X.feye(2), Af), [F(0), F(1)])
        ok += int(xs[0] == 1 / k1 and xs[1] == 0)
    return {"n": versuche, "exakt": ok, "seed": seed}


def messreihe_g(n: int = 1000, seed: int = 172, rhos=(0.2, 0.5, 0.8)) -> list[dict]:
    """Je Beobachter n zufällige g im stabilen Gebiet, je mit B_dt = g B und B_dt = B: Dauerfehler 0 in allen stabilen Fällen?"""
    out = []
    rng = SplitMix64(seed)
    for rho in rhos:
        ent = K.entwurf("wagen", 1.0, rho)
        lo, hi = K.g_gebiet(ent)
        gs = np.array([lo + 1e-6 + (hi - lo - 2e-6) * rng.random() for _ in range(n)])
        Bt = gs[:, None, None] * ent.B[None]
        worst, stabil = 0.0, 0
        for Bdt in (Bt, np.broadcast_to(ent.B[None], Bt.shape)):
            Acl, Bw = K.kreis_stapel(ent, Bt=Bt, Bdt=Bdt)
            st = K.spektralradius(Acl) < 1.0
            stabil += int(st.sum())
            z = np.linalg.solve(np.eye(Acl.shape[1])[None] - Acl[st], Bw[st][:, :, 0:1])[:, :, 0]
            worst = max(worst, float(np.abs(z[:, :2]).max()))
        out.append({"rho": rho, "gebiet": [float(lo), float(hi)], "stabil": stabil, "versuche": 2 * n, "max_abs": worst})
    return out


def messreihe_zufallsstrecken(deltas=C.DELTA_OPTIONS, n: int = 3000, seed: int = 175) -> list[dict]:
    out = []
    for i, delta in enumerate(deltas):
        z = zufallsstrecken(delta, n, seed + 1000 * i)
        z["delta"] = delta
        out.append(z)
    return out


def messreihen() -> dict:
    zs = messreihe_zufallsstrecken()
    return {"R": messreihe_R(), "brueche": messreihe_brueche(), "g": messreihe_g(), "zufallsstrecken": zs,
            "zufallsstrecken_summe": {"n": sum(z["n"] for z in zs), "stabil": sum(z["stabil"] for z in zs), "max_pss": max(z["max_pss"] for z in zs)}}


def load_messreihen() -> dict:
    """Die vorgerechneten Reihen (`data/off_messreihen.json`); leeres Wörterbuch, wenn die Datei fehlt."""
    try:
        return json.loads(C.MESSREIHEN_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------------------------------------------------------------
# Verdikte (reine Funktionen, Art und Text)
# ---------------------------------------------------------------------------------------------------------------------------------

def verdict_dauerfehler(d: float, r: float, p_ohne: float, p_mit_gemessen_fehler: float, formel_text: str) -> tuple[str, str]:
    """Kernfrage 1: Bleibt ein Fehler stehen? `p_ohne`: Ort im Beharrungszustand ohne Störungsmodell, `p_mit_gemessen_fehler`: Fehler der gemessenen Größe mit Störungsmodell."""
    if abs(d) < 1e-12:
        return "info", "Ohne Störung (d = 0) bleibt kein Fehler stehen: Das Gesetz u = −K x hält den Ort am Ziel. Der Dauerfehler entsteht erst durch eine konstante Störung."
    fehler = p_ohne - r
    return "warning", (f"**Ja, ohne Störungsmodell.** Bei d = {de(d, 2)} bleibt der Ort bei {de(p_ohne, 3)}, also {de(fehler, 3, sign=True)} neben dem Ziel r = {de(r, 2)} ({formel_text}). "
                       f"Mit Störungsmodell und Beobachter ist der Fehler der gemessenen Größe im Beharrungszustand {de(p_mit_gemessen_fehler, 3)}.")


def verdict_preis(d: float, spitze_mit: float, schritt_mit: int, spitze_ohne: float, fehler_ohne: float, abkling: int, ab_unter, ab_summe, horizont: int = C.T_LANG) -> tuple[str, str]:
    """Kernfrage 2: Wie kommt er weg, und was kostet das? Der Preis steht in derselben Zeile wie die Beseitigung; ein Preset ohne Preis meldet keinen Preis."""
    if abs(d) < 1e-12 or spitze_ohne <= 0:
        return "info", "Ohne Störung gibt es nichts zu beseitigen und nichts zu bezahlen."
    faktor = spitze_mit / spitze_ohne
    zahlen = (f"Nach dem Störsprung steigt die Abweichung vom Ziel auf {de(spitze_mit, 2)} (Schritt {schritt_mit}) gegen {de(spitze_ohne, 2)} beim reinen LQR, "
              f"das bei {de(abs(fehler_ohne), 2)} stehen bleibt (Faktor {de(faktor, 1)}). Abklingzeit des Beobachterkreises: {abkling} Perioden.")
    if faktor >= 1.5:
        kopf = "Das Störungsmodell beseitigt den Dauerfehler, die Beseitigung kostet aber Spitze: " + zahlen
    elif faktor > 1.0 + 1e-9:
        kopf = "Das Störungsmodell beseitigt den Dauerfehler und kostet hier nur wenig Spitze: " + zahlen
    else:
        kopf = "Das Störungsmodell beseitigt den Dauerfehler ohne zusätzliche Spitze: " + zahlen
    if ab_unter is None:
        fuss = f" Innerhalb von {fmt_int(horizont)} Schritten liegt die Abweichung nicht dauerhaft unter der ohne Störungsmodell."
    else:
        fuss = f" Ab Schritt {ab_unter} liegt die Abweichung dauerhaft darunter"
        fuss += f", die kumulierte Summe ab Schritt {ab_summe}." if ab_summe is not None else f", die kumulierte Summe innerhalb von {fmt_int(horizont)} Schritten nicht."
    return ("warning" if faktor >= 1.5 else "info"), kopf + fuss


def verdict_dauerfehler_grenze(d: float, umax: float, p_ohne: float, p_mit: float, T: int = C.T_GRENZE) -> tuple[str, str]:
    """Kernfrage 1 und 2 mit Stellgrenze, wenn |d| > u_max: der Ort bleibt nicht stehen."""
    return "warning", (f"**Der Ort bleibt nicht stehen:** Die Störkraft |d| = {de(abs(d), 2)} ist größer als die Stellgrenze {de(umax, 2)}. Nach {T} Schritten liegt der Wagen bei {de_tsd(p_ohne)} ohne und "
                       f"bei {de_tsd(p_mit)} mit Störungsmodell; ein Dauerfehler ist dort nicht definiert, der Ort wächst ohne Schranke.")


def verdict_preis_grenze(d: float, umax: float, p_ohne: float, p_mit: float, spitze_ohne: float, spitze_mit: float, T: int = C.T_GRENZE) -> tuple[str, str]:
    """Kernfrage 2 mit Stellgrenze, wenn |d| < u_max (Start in Ruhe im Sollwert, Abschneiden): Fehler nach T Schritten und Spitze mit und ohne Störungsmodell."""
    kopf = "Das Störungsmodell beseitigt den Dauerfehler auch mit Stellgrenze" if abs(p_mit) < 1e-6 else "Das Störungsmodell beseitigt den Dauerfehler mit Stellgrenze nicht vollständig"
    return "info", (f"{kopf} (u_max = {de(umax, 2)}, |d| = {de(abs(d), 2)}): Abweichung vom Ziel nach {T} Schritten {de(abs(p_mit), 3)} gegen {de(abs(p_ohne), 3)} ohne Störungsmodell; "
                    f"die Spitze ist {de(spitze_mit, 2)} gegen {de(spitze_ohne, 2)} beim reinen LQR (vermutlich verlängert das Abschneiden den Anlauf; die Ursache ist nicht einzeln gemessen).")


def formel_text(strecke: str, k: float) -> str:
    """Satz zur Formel des Dauerfehlers je Strecke (Verdikt der Kernfrage 1)."""
    if strecke == "wagen":
        return f"am Wagen gleich d/K₁ mit K₁ = {de(k, 4)}"
    if strecke == "skalar1":
        return f"hier d/k mit k = {de(k, 4)}"
    return f"hier d/(1 − a + k) mit a = ½ und k = {de(k, 4)}"


def kernfragen(s: Settings, res: dict, rs: dict) -> list[tuple[str, str]]:
    """Die drei Verdikte der Kernfragen (Bleibt ein Fehler stehen? Wie kommt er weg, und was kostet das? Wann hilft es nicht?) aus dem Ergebnis `ergebnis(s)` und der Rampen-/Sinus-Messung `rs`."""
    ent, mit, ohne = res["ent"], res["mit"], res["ohne"]
    k = float(ent.K[0, 0])
    grenze = res["grenze"] is not None
    if grenze:
        po, pm = float(res["g_ohne"]["p"][-1]), float(res["g_mit"]["p"][-1])
        weg = abs(s.d) > s.umax + 1e-12
        schwelle = abs(s.d) >= s.umax - 1e-12
    else:
        po, pm = ohne["p_fest"], mit["p_fest"]
        weg = schwelle = False
    if weg:
        v1 = verdict_dauerfehler_grenze(s.d, s.umax, po, pm)
    else:
        gemessen = (pm + s.b - s.r) if np.isfinite(pm) else float("nan")
        v1 = verdict_dauerfehler(s.d, s.r, po, gemessen if not grenze else pm - s.r, formel_text(s.strecke, k))
    if grenze and schwelle:
        v2 = ("warning", f"Das Störungsmodell beseitigt den Fehler hier nicht: |d| ist nicht kleiner als die Stellgrenze {de(s.umax, 2)} (Kernfrage 3).")
    elif grenze:
        v2 = verdict_preis_grenze(s.d, s.umax, po - s.r, pm - s.r, K.kennzahlen(res["g_ohne"]["p"], s.r)["spitze"], K.kennzahlen(res["g_mit"]["p"], s.r)["spitze"])
    else:
        v2 = verdict_preis(s.d, res["k_mit"]["spitze"], res["k_mit"]["schritt"], res["k_ohne"]["spitze"], ohne["p_fest"] - s.r, res["k_mit"]["abkling"], res["ab_unter"], res["ab_summe"])
    v3 = verdict_grenzen(s.d, s.umax, s.b, s.r - s.b, rs["faktor"], rs["peak"], rs["wpeak"], rs["peak0"])
    return [v1, v2, v3]


def verdict_grenzen(d: float, umax: float | None, b: float, p_bias: float, rampe_faktor: float, peak: float, wpeak: float, peak0: float) -> tuple[str, str]:
    """Kernfrage 3: Wann hilft es nicht? Reihenfolge: Stellgrenze, Messbias, sonst die Grenzen der Offsetfreiheit (Rampe, Sinus)."""
    if umax is not None and abs(d) > umax + 1e-12:
        return "warning", (f"**Hier hilft nichts:** Die Störkraft |d| = {de(abs(d), 2)} ist größer als die Stellgrenze {de(umax, 2)}. Wegen v⁺ ≥ v + |d| − u_max läuft der Wagen mit jeder Regel weg, "
                           "mit und ohne Störungsmodell (im Lauf oben, der Ort wächst ohne Schranke).")
    if umax is not None and abs(abs(d) - umax) <= 1e-12:
        return "warning", (f"**Gleichstand an der Schwelle:** |d| = u_max = {de(umax, 2)}. Die Gegenkraft reicht im Beharrungszustand genau, ohne Reserve; wegen v⁺ ≥ v + |d| − u_max kann eine einmal "
                           "entstandene Geschwindigkeit nicht mehr abgebaut werden, ob der Wagen ankommt, hängt vom Anlauf ab (Lauf oben und Experiment E).")
    if abs(b) > 1e-12:
        return "info", (f"**Offsetfrei heißt: die gemessene Größe erreicht das Ziel.** Der Sensor zeigt um b = {de(b, 2)} daneben, also bleibt der wahre Ort bei {de(p_bias, 2)} (= r − b); "
                        "das Störungsmodell kann einen Messfehler nicht von der Lage des Wagens unterscheiden.")
    return "info", (f"Die Offsetfreiheit gilt für **konstante** Störungen: Bei einer Rampe d = s·k bleibt am Wagen ein konstanter Ortsfehler von {de(rampe_faktor, 1)}·s, "
                    f"bei einem Sinus erreicht |p/d| mit Beobachter {de(peak, 2)} bei ω = {de(wpeak, 3)} (reines LQR höchstens {de(peak0, 2)}); sie gilt nur, wenn die Stellgrenze die Gegenkraft zulässt (|d| < u_max).")


def verdict_dauerfehler_R(rows: list[dict], messreihe: dict, brueche: dict) -> tuple[str, str]:
    """Experiment A: Dauerfehler p_ss/d am Wagen über R und die vorgerechneten Gegenproben."""
    werte = " / ".join(de(r["p_ss"], 3) for r in rows)
    rs = " / ".join(de(r["R"], 3).rstrip("0").rstrip(",") for r in rows)
    text = f"Wagen, d = 1: p_ss/d = {werte} für R = {rs}; kleineres R drückt ihn, aber nicht beliebig weit."
    if messreihe and brueche:
        text += (f" Vorgerechnet: In {fmt_int(brueche['exakt'])} von {fmt_int(brueche['n'])} zufälligen stabilen rationalen Gesetzen ist der Fixpunkt in Brüchen exakt (d/K₁, 0); "
                 f"unter {fmt_int(messreihe['n'])} zufälligen R (10⁻⁸ bis 10⁴) liegt der kleinste Wert von 1/K₁ bei {de(messreihe['min'], 6)} und keiner unter {de(PHI - 1e-6, 6)} "
                 f"(numerisch gleich dem goldenen Schnitt {de(PHI, 6)}, nicht hergeleitet).")
    return "info", text


def verdict_beobachter(z_schnell: dict, z_langsam: dict, ohne: dict) -> tuple[str, str]:
    """Experiment B: schneller gegen langsamer Beobachter (Zeilen aus `K.beobachter_zeile`), kein Satz über andere Entwürfe."""
    return "info", (f"ρ = {de(z_schnell['rho'], 2)}: Spitze {de(z_schnell['spitze'], 1)}, aber die Stellgröße rauscht mit {de(z_schnell['std_u'], 1)} je Standardabweichung des Messrauschens. "
                    f"ρ = {de(z_langsam['rho'], 2)}: Rauschen {de(z_langsam['std_u'], 2)}, aber Spitze {de(z_langsam['spitze'], 1)} und Abklingzeit {z_langsam['abkling']}. "
                    f"Zum Vergleich ohne Beobachter: Spitze {de(ohne['spitze'], 2)} mit Dauerfehler {de(ohne['dauerfehler'], 2)}. Keine Einstellung senkt in diesen Zeilen alles zugleich.")


def verdict_modell(g: float, stabil: bool, p_ss, gebiet: tuple, gebiet_ohne: tuple, rho: float, summe: dict | None) -> tuple[str, str]:
    """Experiment C: wahre Verstärkung g."""
    if not stabil:
        text = f"g = {de(g, 2)}: der Kreis ist **instabil**; das stabile Gebiet liegt bei Beobachter 3 × {de(rho, 2)} zwischen g = {de(gebiet[0], 3)} und {de(gebiet[1], 3)} (reines LQR: {de(gebiet_ohne[0], 4)} bis {de(gebiet_ohne[1], 3)}). Ein Dauerfehler ist dort nicht definiert."
        return "warning", text
    text = (f"g = {de(g, 2)}: stabil, Dauerfehler im Fixpunkt {de(abs(p_ss) if abs(p_ss) > 1e-9 else 0.0, 3)}; das stabile Gebiet ist bei Beobachter 3 × {de(rho, 2)} g von {de(gebiet[0], 3)} bis {de(gebiet[1], 3)} "
            f"(reines LQR: {de(gebiet_ohne[0], 4)} bis {de(gebiet_ohne[1], 3)}).")
    if summe:
        text += f" Vorgerechnet: Dauerfehler 0 in {fmt_int(summe['stabil'])} von {fmt_int(summe['stabil'])} stabilen Zufallsstrecken."
    return "success", text


def verdict_zwei_wege(integral: dict, beob: dict, qi: float, rho: float) -> tuple[str, str]:
    """Experiment D: ein Entwurf je Familie, Zustand gemessen; kein Satz über beide Verfahren allgemein."""
    return "info", (f"Zustand gemessen, bei diesen beiden Entwürfen: Integralzustand (q_i = {f'{qi:g}'.replace('.', ',')}) Spitze {de(integral['spitze'], 2)}, Abklingzeit {integral['abkling']}, max|p/d| {de(integral['peak'], 2)}; "
                    f"Störungsbeobachter (Pole {de(rho, 2)}, {de(0.9 * rho, 3)}, {de(0.8 * rho, 3)}) Spitze {de(beob['spitze'], 2)}, Abklingzeit {beob['abkling']}, max|p/d| {de(beob['peak'], 2)}. "
                    "Das ist ein Unterschied der Entwurfsstruktur in diesen Zahlen, kein Satz über Beobachter und Integralzustand.")


def verdict_rampe(s: float, faktor: float, c0: float, steigung_ohne: float, peak: float, w: float, gain: float, gain0: float) -> tuple[str, str]:
    """Experiment E (a): Rampe und Sinus."""
    return "info", (f"Rampe s = {de(s, 3)}: mit Beobachter wird der Ort **konstant**, aber nicht 0: {de(c0, 3)} (= {de(faktor, 1)}·s); ohne Beobachter läuft er mit der Steigung {de(steigung_ohne, 4)} je Schritt weg. "
                    f"Sinus ω = {de(w, 2)}: |p/d| = {de(gain, 2)} mit, {de(gain0, 2)} ohne Beobachter; die Spitze des Beobachterkreises ist {de(peak, 2)}.")


def verdict_stellgrenze(zeilen: list[dict]) -> tuple[str, str]:
    """Experiment E (b): u_max = 1, Start (4; 0)."""
    z = {r["d"]: r for r in zeilen}
    a, b, c = z[0.3], z[0.9], z[1.5]
    return "warning", (f"u_max = {de(C.STELL_UMAX, 1)}, Start (4; 0), Ort nach {C.T_GRENZE} Schritten: d = 0,3: ohne Modell {de(a['ohne']['p_end'], 3)}, mit Modell {de(abs(a['mit']['p_end']), 3)}; "
                       f"d = 0,9: ohne {de(b['ohne']['p_end'], 3)}, mit {de(abs(b['mit']['p_end']), 3)}; d = 1,5 (größer als u_max): mit Modell {de_tsd(c['mit']['p_end'])}, ohne {de_tsd(c['ohne']['p_end'])} "
                       "(der Wagen läuft weg, mit und ohne Modell).")


def verdict_rang(tabelle: list[dict]) -> tuple[str, str]:
    """Experiment E (c): Rangbedingung."""
    nein = [t["fall"] for t in tabelle if not t["erfuellt"]]
    ja = len(tabelle) - len(nein)
    return "info", (f"Die Rangbedingung ist in {ja} von {len(tabelle)} Fällen erfüllt; nicht erfüllt: {'; '.join(nein)}. "
                    "Beim Wagen mit Störung am Ausgang ist das Störungsmodell mit einer Ortsmessung nicht entwerfbar.")
