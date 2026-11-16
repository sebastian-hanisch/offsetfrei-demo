"""Kern: Regelkreis aus Zustandsrückführung (LQR), Störungsmodell mit Beobachter und Integralzustand, am Wagen und am Skalar.

Alles in diesem Modul ist eigene Nachrechnung in diesem Modell: Strecke x+ = A x + B (u + d), gemessen y = C x + b, Ziel y = r. Der Wagen (Doppelintegrator,
Ts = 1, x = (Ort, Geschwindigkeit), Ort gemessen) und der Skalar (a = 1 oder a = 1/2) sind die Strecken. Der Regelkreis hat den Zustand z = (x, x_hat, d_hat)
und die Eingänge w = (d, r, b): d die wahre Störung (konstant oder eine Folge), r der Sollwert, b ein Messbias. Matrizen sind numpy-Felder; die Stapel-
Funktionen (`kreis_stapel`) bauen viele Regelkreise auf einmal (Raster der Streckenverstärkung, Zufallsstrecken). Die exakte Bruchrechnung steht in `off_exakt.py`.

Zähleinheit überall: Periode. Konvention: u = -K x, Kosten sum(x'Qx + u'Ru) mit Q = I (am Skalar Q = 1) und ohne Faktor 1/2.
Die Herkunft der Gleichungen (Störungsmodell, Rangbedingung, Integralzustand) steht in der README; nichts ist aus einem Buch übernommen."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar
from scipy.signal import place_poles

from off_rng import SplitMix64

KINDS = ("wagen", "skalar1", "skalar_halb")
KIND_LABEL = {"wagen": "Wagen (Doppelintegrator)", "skalar1": "Skalar a = 1", "skalar_halb": "Skalar a = ½"}


def strecke(kind: str):
    """Strecke (A, B, C) als numpy-Felder: x+ = A x + B (u + d), y = C x."""
    if kind == "wagen":
        return np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([[0.0], [1.0]]), np.array([[1.0, 0.0]])
    if kind == "skalar1":
        return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
    if kind == "skalar_halb":
        return np.array([[0.5]]), np.array([[1.0]]), np.array([[1.0]])
    raise ValueError(f"unbekannte Strecke: {kind}")


def lqr_gain(A, B, R: float, Q=None):
    """K aus der diskreten Riccati-Gleichung (SciPy); u = -K x. Q = I, wenn nichts anderes steht."""
    n = A.shape[0]
    Q = np.eye(n) if Q is None else Q
    Rm = np.array([[float(R)]])
    P = la.solve_discrete_are(A, B, Q, Rm)
    return np.linalg.solve(Rm + B.T @ P @ B, B.T @ P @ A)


def erweitert(A, B, C):
    """Erweitertes Modell z = (x, d) mit d+ = d: Ae, Be, Ce."""
    n = A.shape[0]
    Ae = np.zeros((n + 1, n + 1))
    Ae[:n, :n] = A
    Ae[:n, n:] = B
    Ae[n, n] = 1.0
    Be = np.vstack([B, [[0.0]]])
    Ce = np.hstack([C, np.zeros((C.shape[0], 1))])
    return Ae, Be, Ce


def ackermann(Ae, Ce, poles):
    """Beobachterverstärkung L (n x 1) mit den Polen `poles` für eine Messung: L = phi(Ae) O^-1 e_n (Ackermann)."""
    n = Ae.shape[0]
    O = np.vstack([Ce @ np.linalg.matrix_power(Ae, i) for i in range(n)])
    phi = np.eye(n)
    for p in poles:
        phi = phi @ (Ae - p * np.eye(n))
    return phi @ np.linalg.solve(O, np.eye(n)[:, [n - 1]])


def beobachter_pole(kind: str, rho: float):
    """Wagen: dreifacher Pol rho. Skalar: Pole rho und rho/2 (wie im Handbeispiel von Hand)."""
    return (rho, rho, rho) if kind == "wagen" else (rho, rho / 2.0)


class Entwurf(NamedTuple):
    kind: str
    R: float
    rho: float
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    K: np.ndarray       # (1, n)
    Ae: np.ndarray
    Be: np.ndarray
    Ce: np.ndarray
    L: np.ndarray       # (n+1, 1)


def entwurf(kind: str, R: float, rho: float, L=None) -> Entwurf:
    """LQR-Gewinn und Beobachterverstärkung (Ackermann, Pole nach `beobachter_pole`); `L` ersetzt die Ackermann-Verstärkung."""
    A, B, C = strecke(kind)
    Ae, Be, Ce = erweitert(A, B, C)
    if L is None and not 0.0 <= rho < 1.0:
        raise ValueError("Der Beobachterpol rho muss in [0, 1) liegen: bei rho = 1 hat der Beobachterkreis den Eigenwert 1")
    if R <= 0:
        raise ValueError("Das Stellgewicht R muss positiv sein")
    Lg = ackermann(Ae, Ce, beobachter_pole(kind, rho)) if L is None else np.asarray(L, float).reshape(-1, 1)
    return Entwurf(kind, float(R), float(rho), A, B, C, lqr_gain(A, B, R), Ae, Be, Ce, Lg)


def zielkoeff(ent: Entwurf):
    """Stationäre Zielrechnung: [[I - A, -B], [C, 0]] (x_s, u_s) = (B d_hat, r). Rückgabe Tx (n x 2) und Tu (1 x 2), Spalten (d_hat, r)."""
    A, B, C = ent.A, ent.B, ent.C
    n = A.shape[0]
    M = np.block([[np.eye(n) - A, -B], [C, np.zeros((1, 1))]])
    rhs = np.zeros((n + 1, 2))
    rhs[:n, 0] = B[:, 0]
    rhs[n, 1] = 1.0
    sol = np.linalg.solve(M, rhs)
    return sol[:n], sol[n:]


def gesetz(ent: Entwurf, mit_us: bool = True):
    """Koeffizienten des Gesetzes u = -K x_hat + g_dh * d_hat + g_r * r (aus u = u_s - K (x_hat - x_s)). Ohne `mit_us` fehlt der Zielwert u_s des Sollwerts."""
    Tx, Tu = zielkoeff(ent)
    g_dh = float(Tu[0, 0] + (ent.K @ Tx[:, 0])[0])
    g_r = float((Tu[0, 1] if mit_us else 0.0) + (ent.K @ Tx[:, 1])[0])
    return g_dh, g_r


def _s3(x, last):
    x = np.asarray(x, float)
    return x[None] if x.ndim == last else x


def kreis_stapel(ent: Entwurf, At=None, Bt=None, Bdt=None, L=None, Cm=None, mit_us: bool = True):
    """Regelkreise (Beobachter + Gesetz) als Stapel. At (N,n,n), Bt (N,n,1), Bdt (N,n,1): wahre Strecke (Vorgabe: die Entwurfsstrecke); L (n+1, m), Cm (m, n): Messung
    und Verstärkung (Vorgabe: die des Entwurfs). Rückgabe Acl (N, nz, nz) und Bw (N, nz, 3) mit den Spalten d, r, b; nz = 2 n + 1, z = (x, x_hat, d_hat)."""
    A, B, K = ent.A, ent.B, ent.K
    n = A.shape[0]
    At = _s3(A if At is None else At, 2)
    Bt = _s3(B if Bt is None else Bt, 2)
    Bdt = _s3(B if Bdt is None else Bdt, 2)
    Cm = ent.C if Cm is None else Cm
    Lg = ent.L if L is None else np.asarray(L, float)
    N = max(At.shape[0], Bt.shape[0], Bdt.shape[0])
    g_dh, g_r = gesetz(ent, mit_us)
    Lx, Ld = Lg[:n], Lg[n:]
    nz = 2 * n + 1
    Acl = np.zeros((N, nz, nz))
    Acl[:, :n, :n] = At
    Acl[:, :n, n:2 * n] = -(Bt @ K)
    Acl[:, :n, 2 * n:] = Bt * g_dh
    Acl[:, n:2 * n, :n] = Lx @ Cm
    Acl[:, n:2 * n, n:2 * n] = A - B @ K - Lx @ Cm
    Acl[:, n:2 * n, 2 * n:] = B * (1.0 + g_dh)
    Acl[:, 2 * n, :n] = Ld @ Cm
    Acl[:, 2 * n, n:2 * n] = -(Ld @ Cm)
    Acl[:, 2 * n, 2 * n] = 1.0
    Bw = np.zeros((N, nz, 3))
    Bw[:, :n, 0] = Bdt[:, :, 0]
    Bw[:, :n, 1] = Bt[:, :, 0] * g_r
    Bw[:, n:2 * n, 1] = B[:, 0] * g_r
    Bw[:, n:2 * n, 2] = Lx[:, 0]
    Bw[:, 2 * n, 2] = Ld[0, 0]
    return Acl, Bw


def kreis(ent: Entwurf, **kw):
    """Ein Regelkreis: Acl (nz x nz), Bw (nz x 3); Argumente wie `kreis_stapel`, aber ohne Stapel."""
    Acl, Bw = kreis_stapel(ent, **kw)
    return Acl[0], Bw[0]


def kreis_ohne(ent: Entwurf, At=None, Bt=None, Bdt=None, mit_us: bool = True):
    """Reines LQR ohne Störungsmodell, Zustand gemessen: u = -K x + g_r r (d_hat = 0). Acl (n x n), Bw (n x 3), Spalten d, r, b (b ohne Wirkung)."""
    A, B, K = ent.A, ent.B, ent.K
    n = A.shape[0]
    At = A if At is None else At
    Bt = B if Bt is None else Bt
    Bdt = B if Bdt is None else Bdt
    _, g_r = gesetz(ent, mit_us)
    Bw = np.zeros((n, 3))
    Bw[:, 0] = Bdt[:, 0]
    Bw[:, 1] = Bt[:, 0] * g_r
    return At - Bt @ K, Bw


def fixpunkt(Acl, Bw, w):
    """Beharrungszustand z* = (I - Acl)^-1 Bw w (Gleitkomma); setzt einen stabilen Kreis voraus."""
    nz = Acl.shape[0]
    return np.linalg.solve(np.eye(nz) - Acl, Bw @ np.asarray(w, float))


def spektralradius(M) -> float | np.ndarray:
    """Größter Eigenwertbetrag; für einen Stapel (N,n,n) ein Feld der Länge N."""
    return np.abs(np.linalg.eigvals(M)).max(axis=-1)


def antwort(Acl, Bw, w, z0, T: int):
    """Z[0..T] mit z_{k+1} = Acl z_k + Bw w (Eingang w konstant ab k = 0)."""
    nz = Acl.shape[0]
    Z = np.empty((T + 1, nz))
    Z[0] = z0
    bw = Bw @ np.asarray(w, float)
    for k in range(T):
        Z[k + 1] = Acl @ Z[k] + bw
    return Z


def horizont(Acl, kleinster=100, groesster=6000) -> int:
    """Zahl der Schritte, nach der ein stabiler Kreis auf etwa 1e-7 abgeklungen ist (mit Sicherheit für Mehrfachpole); sonst `kleinster`."""
    rho = float(spektralradius(Acl))
    if rho >= 1.0 - 1e-9 or rho <= 0.0:
        return kleinster
    return int(min(groesster, max(kleinster, 1.5 * math.log(1e-7) / math.log(rho) + 30)))


def kennzahlen(p, ziel: float = 0.0, fest: float | None = None):
    """Spitze = größter Betrag |p_k - ziel| für k >= 1, Schritt der Spitze, Abklingzeit = letzte Periode, in der |p_k - fest| über 1 % der Spitze liegt.
    `fest` ist der Beharrungswert (Vorgabe: das Ziel); mit Messbias oder Dauerfehler ist er vom Ziel verschieden."""
    p = np.asarray(p, float)
    dev = np.abs(p[1:] - ziel)
    spitze = float(dev.max()) if dev.size else 0.0
    if spitze <= 0.0:
        return {"spitze": 0.0, "schritt": 0, "abkling": 0}
    ref = ziel if fest is None or not np.isfinite(fest) else fest
    ks = np.nonzero(np.abs(p[1:] - ref) > 0.01 * spitze)[0]
    return {"spitze": spitze, "schritt": int(np.argmax(dev)) + 1, "abkling": int(ks[-1]) + 1 if ks.size else 0}


def ruhelage(Acl, Bw, r: float = 0.0):
    """Ruhelage im Sollwert ohne Störung und Messbias: Beharrungszustand für w = (0, r, 0)."""
    return fixpunkt(Acl, Bw, (0.0, r, 0.0))


def sprung(ent: Entwurf, d: float, r: float = 0.0, b: float = 0.0, mit_beobachter: bool = True, mit_us: bool = True, T: int | None = None):
    """Störsprung d (und Messbias b) bei k = 0, Start in Ruhe im Sollwert r. Rückgabe: Dict mit Ort p[0..T], Schätzung d_hat (nur mit Beobachter), Beharrungsort."""
    if mit_beobachter:
        Acl, Bw = kreis(ent, mit_us=mit_us)
        idx_d = 2 * ent.A.shape[0]
    else:
        Acl, Bw = kreis_ohne(ent, mit_us=mit_us)
        idx_d = None
    rho = float(spektralradius(Acl))
    T = horizont(Acl) if T is None else T
    z0 = ruhelage(Acl, Bw, r)
    Z = antwort(Acl, Bw, (d, r, b), z0, T)
    p = Z[:, 0]
    fest = fixpunkt(Acl, Bw, (d, r, b))[0] if rho < 1.0 - 1e-9 else float("nan")
    return {"p": p, "dh": Z[:, idx_d] if idx_d is not None else None, "p_fest": float(fest), "rho": rho, "T": T, "Z": Z}


def dauerfehler_lqr(ent: Entwurf) -> float:
    """Ort im Beharrungszustand je Einheit d ohne Störungsmodell (Wagen, Ziel 0): Fixpunkt von x+ = (A - B K) x + B; gleich 1/K1."""
    Acl, Bw = kreis_ohne(ent)
    return float(fixpunkt(Acl, Bw, (1.0, 0.0, 0.0))[0])


def u_zeile(ent: Entwurf, mit_us: bool = True):
    """Zeile cu mit u = cu z + g_r r für den Zustand z = (x, x_hat, d_hat)."""
    n = ent.A.shape[0]
    g_dh, _ = gesetz(ent, mit_us)
    cu = np.zeros(2 * n + 1)
    cu[n:2 * n] = -ent.K[0]
    cu[2 * n] = g_dh
    return cu


# ---------------------------------------------------------------------------------------------------------------------------------
# Frequenzgang, Rauschen, Reserve, Gebiet der Streckenverstärkung
# ---------------------------------------------------------------------------------------------------------------------------------

W_GRID = np.logspace(-3, math.log10(math.pi), 6000)


def gain_ueber_w(Acl, bd, ws, idx: int = 0):
    """|G(e^{jw})| des Ausgangs z[idx] auf den Eingang bd, G = (e^{jw} I - Acl)^-1 bd, für ein Feld von Frequenzen."""
    ws = np.asarray(ws, float)
    nz = Acl.shape[0]
    M = np.exp(1j * ws)[:, None, None] * np.eye(nz)[None] - Acl[None]
    sol = np.linalg.solve(M, np.broadcast_to(np.asarray(bd, complex).reshape(1, nz, 1), (len(ws), nz, 1)))
    return np.abs(sol[:, idx, 0])


def max_gain(Acl, bd, idx: int = 0):
    """Spitze des Frequenzgangs |p/d| über w in (0, pi]: Gitter mit 6000 Punkten, dann feinere Suche um das Maximum. Rückgabe (Spitze, w)."""
    g = gain_ueber_w(Acl, bd, W_GRID, idx)
    i = int(np.argmax(g))
    lo, hi = W_GRID[max(i - 1, 0)], W_GRID[min(i + 1, len(W_GRID) - 1)]
    res = minimize_scalar(lambda w: -float(gain_ueber_w(Acl, bd, [w], idx)[0]), bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    if -res.fun > g[i]:
        return float(-res.fun), float(res.x)
    return float(g[i]), float(W_GRID[i])


def rausch_std(Acl, Bw, cu):
    """Standardabweichung von Ort und Stellgröße je Standardabweichung des weißen Messrauschens (auf y), exakt aus der Lyapunov-Gleichung. Rückgabe (std_p, std_u)."""
    bn = Bw[:, 2:3]
    Sigma = la.solve_discrete_lyapunov(Acl, bn @ bn.T)
    return float(math.sqrt(max(Sigma[0, 0], 0.0))), float(math.sqrt(max(cu @ Sigma @ cu, 0.0)))


def rausch_lauf(Acl, Bw, cu, T: int, seed: int, z0=None):
    """Eine Simulation mit weißem Messrauschen (SplitMix64, Standardnormal), Start z0 (Vorgabe 0). Rückgabe: Ort p und Stellgröße u (Länge T)."""
    rng = SplitMix64(seed)
    nz = Acl.shape[0]
    bn = Bw[:, 2]
    z = np.zeros(nz) if z0 is None else np.array(z0, float)
    p = np.empty(T)
    u = np.empty(T)
    for k in range(T):
        z = Acl @ z + bn * rng.normal()
        p[k] = z[0]
        u[k] = cu @ z
    return p, u


def kreis_g(ent: Entwurf, gs, L=None, Cm=None):
    """Stapel für die wahre Streckenverstärkung g: B_t = g B und B_dt = g B (Entwurf für g = 1)."""
    gs = np.asarray(gs, float)
    Bt = gs[:, None, None] * ent.B[None]
    return kreis_stapel(ent, Bt=Bt, Bdt=Bt, L=L, Cm=Cm)


def _stabil_g(ent, g, L=None):
    Acl, _ = kreis_g(ent, [g], L)
    return float(spektralradius(Acl)[0]) < 1.0


def _kante(f, lo, hi, n=60):
    """Bisektion: f(lo) ist wahr, f(hi) falsch; Rückgabe der Grenze (die wahre Seite)."""
    for _ in range(n):
        m = 0.5 * (lo + hi)
        if f(m):
            lo = m
        else:
            hi = m
    return lo


def g_gebiet(ent: Entwurf, L=None, hi=8.0, n=4000):
    """Zusammenhängendes Gebiet der Streckenverstärkung g um g = 1, in dem der Kreis stabil ist: (g_min, g_max); Gitter mit n Punkten, Kanten per Bisektion.
    Ohne Beobachter (L = None und `ent.L` nicht verwendet) siehe `g_gebiet_ohne`."""
    gs = np.linspace(0.0005, hi, n)
    Acl, _ = kreis_g(ent, gs, L)
    ok = spektralradius(Acl) < 1.0
    return _gebiet_aus_maske(gs, ok, lambda g: _stabil_g(ent, g, L))


def _gebiet_aus_maske(gs, ok, f):
    i1 = int(np.argmin(np.abs(gs - 1.0)))
    if not ok[i1]:
        return (float("nan"), float("nan"))
    a = i1
    while a > 0 and ok[a - 1]:
        a -= 1
    b = i1
    while b < len(gs) - 1 and ok[b + 1]:
        b += 1
    lo = gs[a] if a == 0 else _kante(f, gs[a], gs[a - 1])
    hi = gs[b] if b == len(gs) - 1 else _kante(f, gs[b], gs[b + 1])
    return (float(lo), float(hi))


def g_gebiet_ohne(ent: Entwurf, hi=8.0, n=4000):
    """Dasselbe für das reine LQR (u = -K x, Zustand gemessen), wahre Strecke g B."""
    gs = np.linspace(0.0005, hi, n)
    Acl = ent.A[None] - gs[:, None, None] * (ent.B @ ent.K)[None]
    ok = spektralradius(Acl) < 1.0
    return _gebiet_aus_maske(gs, ok, lambda g: float(spektralradius(ent.A - g * ent.B @ ent.K)) < 1.0)


def _rueckfuehr_min(gv):
    """Kleinster Betrag von 1 + L(e^{jw}) über das feine Gitter und eine Nachsuche um das Minimum."""
    g = gv(W_GRID[::3])
    i = int(np.argmin(g))
    ws = W_GRID[::3]
    lo, hi = ws[max(i - 1, 0)], ws[min(i + 1, len(ws) - 1)]
    res = minimize_scalar(lambda w: float(gv(np.array([w]))[0]), bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    return float(min(g[i], res.fun))


def reserve(ent: Entwurf, L=None):
    """min |1 + L(e^{jw})| am Streckeneingang mit Beobachter: Regler Cc(z) = Kz (zI - Ac)^-1 L, Strecke P(z) = C (zI - A)^-1 B, Ac = Ae - L Ce - Be Kz."""
    A, B, C, Ae, Be, Ce = ent.A, ent.B, ent.C, ent.Ae, ent.Be, ent.Ce
    Lg = ent.L if L is None else np.asarray(L, float)
    n = A.shape[0]
    g_dh, _ = gesetz(ent)
    Kz = np.hstack([ent.K, [[-g_dh]]])
    Ac = Ae - Lg @ Ce - Be @ Kz

    def gv(ws):
        z = np.exp(1j * ws)
        Pz = (C @ np.linalg.solve(z[:, None, None] * np.eye(n)[None] - A[None], np.broadcast_to(B, (len(ws), n, 1))))[:, 0, 0]
        Cc = (Kz @ np.linalg.solve(z[:, None, None] * np.eye(n + 1)[None] - Ac[None], np.broadcast_to(Lg, (len(ws), n + 1, 1))))[:, 0, 0]
        return np.abs(1.0 + Cc * Pz)

    return _rueckfuehr_min(gv)


def reserve_ohne(ent: Entwurf):
    """min |1 + K (zI - A)^-1 B| für das reine LQR mit gemessenem Zustand (Wagen: 0,4836 bei R = 1)."""
    A, B, K = ent.A, ent.B, ent.K
    n = A.shape[0]

    def gv(ws):
        z = np.exp(1j * ws)
        return np.abs(1.0 + (K @ np.linalg.solve(z[:, None, None] * np.eye(n)[None] - A[None], np.broadcast_to(B, (len(ws), n, 1))))[:, 0, 0])

    return _rueckfuehr_min(gv)


# ---------------------------------------------------------------------------------------------------------------------------------
# Beobachter-Tabelle, Kalman-Verstärkung, Zustand gemessen
# ---------------------------------------------------------------------------------------------------------------------------------

def beobachter_zeile(kind: str, R: float, rho: float | None = None, L=None):
    """Kennzahlen eines Beobachterkreises (Störsprung d = 1 aus der Ruhe, Ziel 0, Ort gemessen): Spitze, Schritt, Abklingzeit, max|p/d| (bei w), Rauschen std(p) und std(u)
    je std(n), Gebiet der Streckenverstärkung und kleinster Betrag der Rückführdifferenz."""
    ent = entwurf(kind, R, 0.5 if rho is None else rho, L=L)
    Acl, Bw = kreis(ent)
    s = sprung(ent, 1.0)
    kz = kennzahlen(s["p"])
    peak, wpeak = max_gain(Acl, Bw[:, 0])
    sp, su = rausch_std(Acl, Bw, u_zeile(ent))
    gmin, gmax = g_gebiet(ent)
    return {"rho": rho, "spitze": kz["spitze"], "schritt": kz["schritt"], "abkling": kz["abkling"], "peak": peak, "wpeak": wpeak,
            "std_p": sp, "std_u": su, "g_min": gmin, "g_max": gmax, "reserve": reserve(ent), "spektralradius": float(spektralradius(Acl))}


def ohne_zeile(kind: str, R: float):
    """Dieselben Kennzahlen für das reine LQR ohne Störungsmodell (Zustand gemessen): Dauerfehler je Einheit d, Spitze, max|p/d|, Reserve, g-Gebiet."""
    ent = entwurf(kind, R, 0.5)
    Acl, Bw = kreis_ohne(ent)
    s = sprung(ent, 1.0, mit_beobachter=False)
    kz = kennzahlen(s["p"])
    peak, wpeak = max_gain(Acl, Bw[:, 0])
    gmin, gmax = g_gebiet_ohne(ent)
    return {"dauerfehler": s["p_fest"], "spitze": kz["spitze"], "peak": peak, "wpeak": wpeak, "g_min": gmin, "g_max": gmax, "reserve": reserve_ohne(ent)}


def kalman_verstaerkung(kind: str, qd: float, v_rausch: float = 0.01):
    """Kalman-Verstärkung in Vorhersageform für das erweiterte Modell (Ort gemessen), Messrauschen 1, Prozessrauschen v: `v_rausch`, d: `qd` (nur Wagen)."""
    A, B, C = strecke(kind)
    Ae, Be, Ce = erweitert(A, B, C)
    n = A.shape[0]
    W = np.zeros((n + 1, n + 1))
    if n == 2:
        W[1, 1] = v_rausch
    W[n, n] = qd
    Pk = la.solve_discrete_are(Ae.T, Ce.T, W, np.array([[1.0]]))
    return Ae @ Pk @ Ce.T @ np.linalg.inv(Ce @ Pk @ Ce.T + 1.0)


def zustand_gemessen(kind: str, R: float, rho: float):
    """Beobachter nur für d_hat (Ort und Geschwindigkeit werden gemessen): Verstärkung (n+1) x n mit den Polen rho, 0,9 rho, 0,8 rho (place_poles)."""
    ent = entwurf(kind, R, rho)
    n = ent.A.shape[0]
    Cm = np.eye(n)
    Cem = np.hstack([Cm, np.zeros((n, 1))])
    poles = np.array([rho, 0.9 * rho, 0.8 * rho][: n + 1])
    Lm = place_poles(ent.Ae.T, Cem.T, poles).gain_matrix.T
    return ent, Cm, Lm


def zustand_gemessen_zeile(kind: str, R: float, rho: float):
    ent, Cm, Lm = zustand_gemessen(kind, R, rho)
    Acl, Bw = kreis(ent, L=Lm, Cm=Cm)
    T = horizont(Acl)
    z0 = np.zeros(Acl.shape[0])
    p = antwort(Acl, Bw, (1.0, 0.0, 0.0), z0, T)[:, 0]
    kz = kennzahlen(p)
    peak, wpeak = max_gain(Acl, Bw[:, 0])
    return {"rho": rho, "spitze": kz["spitze"], "abkling": kz["abkling"], "peak": peak, "p_end": float(p[-1]), "p": p}


# ---------------------------------------------------------------------------------------------------------------------------------
# Integralzustand als zweiter Weg
# ---------------------------------------------------------------------------------------------------------------------------------

def integral_entwurf(kind: str, R: float, qi: float):
    """Erweitertes Paar (x, xi) mit xi+ = xi + y (Sollwert 0), LQR mit Q = diag(1, ..., 1, qi) und dem R der Auslegung. Rückgabe Aa, Ba, Ka."""
    A, B, C = strecke(kind)
    n = A.shape[0]
    Aa = np.zeros((n + 1, n + 1))
    Aa[:n, :n] = A
    Aa[n, :n] = C[0]
    Aa[n, n] = 1.0
    Ba = np.vstack([B, [[0.0]]])
    Q = np.diag([1.0] * n + [float(qi)])
    return Aa, Ba, lqr_gain(Aa, Ba, R, Q)


def integral_zeile(kind: str, R: float, qi: float):
    """Kennzahlen des Integralzustands: Spitze, Abklingzeit, max|p/d|, Spektralradius, g-Gebiet (B_t = g B), kleinster Betrag der Rückführdifferenz, Fixpunkt (Ort, xi)."""
    Aa, Ba, Ka = integral_entwurf(kind, R, qi)
    Acl = Aa - Ba @ Ka
    T = horizont(Acl, groesster=8000)
    Z = antwort(Acl, np.hstack([Ba, np.zeros((Ba.shape[0], 2))]), (1.0, 0.0, 0.0), np.zeros(Aa.shape[0]), T)
    kz = kennzahlen(Z[:, 0])
    peak, wpeak = max_gain(Acl, Ba[:, 0])
    gs = np.linspace(0.0005, 8.0, 4000)
    stab = spektralradius(Aa[None] - gs[:, None, None] * (Ba @ Ka)[None]) < 1.0
    gmin, gmax = _gebiet_aus_maske(gs, stab, lambda g: float(spektralradius(Aa - g * Ba @ Ka)) < 1.0)
    n = Aa.shape[0]

    def gv(ws):
        z = np.exp(1j * ws)
        return np.abs(1.0 + (Ka @ np.linalg.solve(z[:, None, None] * np.eye(n)[None] - Aa[None], np.broadcast_to(Ba, (len(ws), n, 1))))[:, 0, 0])

    fest = np.linalg.solve(np.eye(n) - Acl, Ba[:, 0])
    return {"qi": qi, "K": Ka[0], "spitze": kz["spitze"], "schritt": kz["schritt"], "abkling": kz["abkling"], "peak": peak, "wpeak": wpeak,
            "spektralradius": float(spektralradius(Acl)), "g_min": gmin, "g_max": gmax, "reserve": _rueckfuehr_min(gv), "p_fest": float(fest[0]), "xi_fest": float(fest[-1]),
            "p": Z[:, 0]}


# ---------------------------------------------------------------------------------------------------------------------------------
# Grenzen: Rampe, Sinus, Stellgrenze, Rangbedingung
# ---------------------------------------------------------------------------------------------------------------------------------

def rampe_ansatz(Acl, bd, s: float):
    """Rampe d_k = s k: Ansatz z_k = c0 + c1 k. Rückgabe (c0, c1); setzt einen stabilen Kreis voraus. Der Ort ist p_k = c0[0] + c1[0] k."""
    nz = Acl.shape[0]
    I = np.eye(nz)
    c1 = np.linalg.solve(I - Acl, np.asarray(bd, float) * s)
    c0 = np.linalg.solve(I - Acl, -c1)
    return c0, c1


def sinus_amplitude(Acl, bd, w: float, T: int | None = None, idx: int = 0):
    """Eingeschwungene Amplitude von z[idx] bei der Störung d_k = sin(w k), per Simulation (Gegenprobe zum Frequenzgang): größter Betrag über die letzten drei Perioden."""
    nz = Acl.shape[0]
    per = max(int(round(2 * math.pi / w)), 20)
    T = int(T if T is not None else max(12000, 600 * per // 10))
    z = np.zeros(nz)
    bd = np.asarray(bd, float)
    amp = 0.0
    for k in range(T):
        z = Acl @ z + bd * math.sin(w * k)
        if k >= T - 3 * per:
            amp = max(amp, abs(z[idx]))
    return amp


def lauf_grenze(ent: Entwurf, d: float, umax: float | None, mit_modell: bool, T: int = 300, x0=None, r: float = 0.0, b: float = 0.0):
    """Nichtlinearer Lauf mit Stellgrenze |u| <= umax (Abschneiden). Beobachter und Strecke bekommen die ANGEWENDETE Stellgröße. Start: Strecke x0, Schätzung 0.
    Rückgabe p, angewendetes u, d_hat (Längen T + 1 bzw. T)."""
    A, B, C, K = ent.A, ent.B, ent.C, ent.K
    n = A.shape[0]
    g_dh, g_r = gesetz(ent)
    Lx, Ld = ent.L[:n], ent.L[n:]
    x = np.zeros(n) if x0 is None else np.array(x0, float)
    xh = np.zeros(n)
    dh = 0.0
    p = np.empty(T + 1)
    uu = np.empty(T)
    dd = np.empty(T + 1)
    p[0], dd[0] = x[0], dh
    for k in range(T):
        if mit_modell:
            u = float(-K[0] @ xh + g_dh * dh + g_r * r)
        else:
            u = float(-K[0] @ x + g_r * r)
        if umax is not None:
            u = min(max(u, -umax), umax)
        y = float(C[0] @ x + b)
        e = y - float(C[0] @ xh)
        x_neu = A @ x + B[:, 0] * (u + d)
        if mit_modell:
            xh = A @ xh + B[:, 0] * (u + dh) + Lx[:, 0] * e
            dh = dh + float(Ld[0, 0]) * e
        x = x_neu
        p[k + 1], uu[k], dd[k + 1] = x[0], u, dh
    return p, uu, dd


def rang_bedingung(A, Bd, C, Cd):
    """Rang der Matrix [[I - A, -Bd], [C, Cd]] (exakt in Brüchen) gegen n + n_d (n_d = Zahl der Störungen). Rückgabe (Rang, verlangt)."""
    from fractions import Fraction

    from off_exakt import frank

    A, Bd, C, Cd = (np.atleast_2d(np.asarray(x, float)) for x in (A, Bd, C, Cd))
    n = A.shape[0]
    top = np.hstack([np.eye(n) - A, -Bd])
    bot = np.hstack([C, Cd])
    M = np.vstack([top, bot])
    return frank([[Fraction(float(v)).limit_denominator(10**6) for v in row] for row in M]), n + Bd.shape[1]


def rang_fall(name: str):
    """Die Fälle der Rangbedingung als (A, Bd, C, Cd): Störung am Eingang oder am Ausgang, Wagen und Skalar."""
    Aw, Bw_, Cw = strecke("wagen")
    z11 = np.zeros((1, 1))
    one = np.ones((1, 1))
    fall = {
        "Wagen, Störung am Eingang": (Aw, Bw_, Cw, z11),
        "Wagen, Störung am Ausgang": (Aw, np.zeros((2, 1)), Cw, one),
        "Wagen, zwei Störungen bei einer Messung": (Aw, np.hstack([Bw_, np.zeros((2, 1))]), Cw, np.array([[0.0, 1.0]])),
        "Skalar a = 1, Störung am Eingang": (np.array([[1.0]]), one, one, z11),
        "Skalar a = 1, Störung am Ausgang": (np.array([[1.0]]), z11, one, one),
        "Skalar a = ½, Störung am Ausgang": (np.array([[0.5]]), z11, one, one),
    }
    return fall[name]


RANG_FAELLE = ("Wagen, Störung am Eingang", "Wagen, Störung am Ausgang", "Wagen, zwei Störungen bei einer Messung",
               "Skalar a = 1, Störung am Eingang", "Skalar a = 1, Störung am Ausgang", "Skalar a = ½, Störung am Ausgang")
