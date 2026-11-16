"""Exakte Bruchrechnung (fractions) für das Handbeispiel und die Gegenproben: Gleichungssysteme, Determinante, Rang, charakteristisches Polynom, Ackermann.

Nur Standardbibliothek. Matrizen sind Listen von Listen aus Fraction. Die Funktionen sind klein gehalten, damit man sie neben der Gleitkomma-Rechnung
von `off_dynamics.py` als zweiten Rechenweg lesen kann (Orakel-Tests und Handbeispiel in `tests/`)."""

from fractions import Fraction as F


def fmat(rows):
    return [[F(x) for x in r] for r in rows]


def feye(n):
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def fmul(X, Y):
    return [[sum(X[i][k] * Y[k][j] for k in range(len(Y))) for j in range(len(Y[0]))] for i in range(len(X))]


def fadd(X, Y):
    return [[X[i][j] + Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def fsub(X, Y):
    return [[X[i][j] - Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def fscale(X, c):
    return [[c * v for v in r] for r in X]


def fpow(X, k):
    R = feye(len(X))
    for _ in range(k):
        R = fmul(R, X)
    return R


def fmatvec(X, v):
    return [sum(X[i][j] * v[j] for j in range(len(v))) for i in range(len(X))]


def fsolve(X, b):
    """Gauß mit Brüchen: löst X z = b für eine quadratische, nicht singuläre Matrix; ValueError bei singulärer Matrix."""
    n = len(X)
    M = [list(map(F, X[i])) + [F(b[i])] for i in range(n)]
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None:
            raise ValueError("Matrix ist singulär")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [v / piv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [M[r][j] - f * M[c][j] for j in range(n + 1)]
    return [M[i][n] for i in range(n)]


def fdet(X):
    n = len(X)
    M = [list(map(F, r)) for r in X]
    d = F(1)
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None:
            return F(0)
        if p != c:
            M[c], M[p] = M[p], M[c]
            d = -d
        d *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [M[r][j] - f * M[c][j] for j in range(n)]
    return d


def frank(X):
    """Rang einer beliebigen Bruchmatrix (Gauß-Verfahren)."""
    M = [list(map(F, r)) for r in X]
    rows, cols = len(M), len(M[0])
    rank = 0
    for c in range(cols):
        p = next((r for r in range(rank, rows) if M[r][c] != 0), None)
        if p is None:
            continue
        M[rank], M[p] = M[p], M[rank]
        for r in range(rank + 1, rows):
            f = M[r][c] / M[rank][c]
            M[r] = [M[r][j] - f * M[rank][j] for j in range(cols)]
        rank += 1
    return rank


def charpoly(M):
    """Faddeev-LeVerrier mit Brüchen: Koeffizienten [1, c_{n-1}, ..., c_0] von det(zI - M)."""
    n = len(M)
    c = [F(1)]
    Mk = feye(n)
    for k in range(1, n + 1):
        AM = fmul(M, Mk)
        ck = -sum(AM[i][i] for i in range(n)) / k
        c.append(ck)
        Mk = [[AM[i][j] + (ck if i == j else 0) for j in range(n)] for i in range(n)]
    return c


def polmul(p, q):
    r = [F(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def poly_from_roots(roots):
    p = [F(1)]
    for rt in roots:
        p = polmul(p, [F(1), -F(rt)])
    return p


def ackermann(Ae, Ce, poles):
    """Beobachterverstärkung L (Liste) mit den Polen `poles` für ein Paar (Ae, Ce) mit einer Messung: L = phi(Ae) O^-1 e_n (Ackermann), in Brüchen."""
    n = len(Ae)
    rows, cur = [], Ce
    for _ in range(n):
        rows.append(cur[0])
        cur = fmul(cur, Ae)
    w = fsolve(rows, [F(0)] * (n - 1) + [F(1)])
    Phi = feye(n)
    for p in poles:
        Phi = fmul(Phi, fsub(Ae, fscale(feye(n), F(p))))
    return [sum(Phi[i][k] * w[k] for k in range(n)) for i in range(n)]


def text(x, max_den=10**6) -> str:
    """Bruch als Text: ganze Zahl ohne Nenner, sonst Zähler/Nenner."""
    x = F(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
