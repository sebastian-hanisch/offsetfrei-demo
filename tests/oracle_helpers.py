"""Hilfen der Orakel-Tests: ein zweiter Rechenweg, der nichts aus `off_dynamics.py` benutzt (nur NumPy, SciPy und Brüche).

- `riccati_iteration`: die Riccati-Rekursion von P = Q aus, bis sie sich nicht mehr ändert (gegen `scipy.linalg.solve_discrete_are`).
- `wagen_kreis`: der 5er-Kreis (p, v, p^, v^, d^) des Wagens mit u = -K x^ - d^ Zeile für Zeile aufgeschrieben, mit wahrer Strecke (A_t, B_t, B_dt) und dem
  Beobachter des Entwurfs (L aus der Ackermann-Formel in Brüchen, `off_exakt.ackermann`).
- `jury_stabil`: Jury-Kriterium für ein Polynom (Koeffizienten höchster Grad zuerst): alle Wurzeln im Einheitskreis?"""

import numpy as np

import off_exakt as X


def riccati_iteration(A, B, Q, R, n_max=5000, tol=1e-14):
    """Rekursion P <- Q + A'PA - A'PB (R + B'PB)^-1 B'PA von P = Q aus; Rückgabe (P, K, Zahl der Schritte)."""
    P = Q.copy()
    for k in range(1, n_max + 1):
        S = R + B.T @ P @ B
        P_neu = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(S, B.T @ P @ A)
        if np.max(np.abs(P_neu - P)) < tol:
            P = P_neu
            break
        P = P_neu
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K, k


def wagen_L(rho_bruch):
    """Ackermann-Verstärkung für 3 x rho (Brüche) als Gleitkomma-Liste."""
    Ae = X.fmat([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    return [float(v) for v in X.ackermann(Ae, X.fmat([[1, 0, 0]]), [rho_bruch] * 3)]


def wagen_kreis(k1, k2, L, At=None, Bt=None, Bdt=None):
    """5er-Kreis (p, v, p^, v^, d^) des Wagens mit u = -k1 p^ - k2 v^ - d^; Rückgabe (Acl, bd)."""
    At = np.array([[1.0, 1.0], [0.0, 1.0]]) if At is None else At
    Bt = np.array([0.0, 1.0]) if Bt is None else np.ravel(Bt)
    Bdt = np.array([0.0, 1.0]) if Bdt is None else np.ravel(Bdt)
    Lp, Lv, Ld = L
    M = np.zeros((5, 5))
    M[0, 0], M[0, 1] = At[0, 0], At[0, 1]                       # Strecke: x+ = At x + Bt u + Bdt d
    M[1, 0], M[1, 1] = At[1, 0], At[1, 1]
    for i in (0, 1):
        M[i, 2], M[i, 3], M[i, 4] = -Bt[i] * k1, -Bt[i] * k2, -Bt[i] * 1.0
    M[2, :] = [Lp, 0.0, 1.0 - Lp, 1.0, 0.0]                      # p^+ = p^ + v^ + Lp (p - p^)
    M[3, :] = [Lv, 0.0, -Lv - k1, 1.0 - k2, 0.0]                 # v^+ = v^ + d^ + u + Lv (p - p^), u = -k1 p^ - k2 v^ - d^
    M[4, :] = [Ld, 0.0, -Ld, 0.0, 1.0]                           # d^+ = d^ + Ld (p - p^)
    bd = np.array([Bdt[0], Bdt[1], 0.0, 0.0, 0.0])
    return M, bd


def jury_stabil(coeffs, tol=0.0):
    """Jury-Kriterium: True, wenn alle Nullstellen von a0 z^n + ... + an (coeffs = [a0, ..., an]) streng im Einheitskreis liegen.
    Bedingungen: P(1) > 0, (-1)^n P(-1) > 0, |a_n| < a_0 und je Zeile der Jury-Tabelle |b_0| > |b_letzter|."""
    a = [float(c) for c in coeffs]
    if a[0] < 0:
        a = [-c for c in a]
    n = len(a) - 1
    p1 = sum(a)
    pm1 = sum(c * (-1) ** (n - i) for i, c in enumerate(a))
    if p1 <= tol or (-1) ** n * pm1 <= tol or abs(a[-1]) >= a[0] - tol:
        return False
    row = a
    while len(row) > 3:
        m = len(row)
        b = [row[0] * row[j] - row[m - 1] * row[m - 1 - j] for j in range(m - 1)]
        if abs(b[0]) <= abs(b[-1]) + tol:
            return False
        row = b
    return True
