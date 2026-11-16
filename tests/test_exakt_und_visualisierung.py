"""Bruchrechnung (`off_exakt.py`) gegen NumPy und die Abbildungen (`off_visualization.py`): alle Achsen gesperrt, jede Figur baut sich aus echten Auswertungen."""

from fractions import Fraction as F

import numpy as np
import pytest

import off_constants as C
import off_dynamics as K
import off_evaluation as E
import off_exakt as X
import off_visualization as V
from off_rng import SplitMix64


def _zufallsmatrix(n, seed, grenze=4):
    g = SplitMix64(seed)
    return [[F(g.below(2 * grenze + 1) - grenze) for _ in range(n)] for _ in range(n)]


def test_charpoly_gegen_numpy_und_cayley_hamilton():
    for seed in range(20):
        M = _zufallsmatrix(4, seed)
        cp = X.charpoly(M)
        assert [float(c) for c in cp] == pytest.approx(np.poly(np.array([[float(v) for v in r] for r in M])), abs=1e-8)
        acc = [[F(0)] * 4 for _ in range(4)]
        for c in cp:                                                       # Horner: p(M) = 0
            acc = X.fadd(X.fmul(acc, M), X.fscale(X.feye(4), c))
        assert all(v == 0 for r in acc for v in r)


def test_rang_determinante_und_gleichungssystem_gegen_numpy():
    for seed in range(40):
        M = _zufallsmatrix(4, 100 + seed, grenze=2)
        Mf = np.array([[float(v) for v in r] for r in M])
        assert X.frank(M) == np.linalg.matrix_rank(Mf)
        d = X.fdet(M)
        assert float(d) == pytest.approx(np.linalg.det(Mf), abs=1e-9)
        if d != 0:
            b = [F(1), F(2), F(-1), F(3)]
            x = X.fsolve(M, b)
            assert X.fmatvec(M, x) == b
        else:
            with pytest.raises(ValueError):
                X.fsolve(M, [1, 2, -1, 3])


def test_text_und_poly_aus_nullstellen():
    assert X.text(F(5, 4)) == "5/4" and X.text(F(8, 4)) == "2" and X.text(F(-3, 8)) == "-3/8"
    assert X.poly_from_roots([F(1, 2), F(1, 4)]) == [F(1), F(-3, 4), F(1, 8)]
    assert X.frank([[1, 2], [2, 4]]) == 1 and X.frank([[0, 0], [0, 0]]) == 0


def _figuren():
    s = E.Settings()
    res = E.ergebnis(s)
    zeilen, ohne = E.beobachter_tabelle(1.0)
    figuren = [
        V.build_ort(res, s, 60),
        V.build_schaetzung(res, s, 60),
        V.build_dauerfehler_R(E.dauerfehler_tabelle()[0]),
        V.build_beobachter(zeilen, ohne, 0.5),
        V.build_rausch(E.rausch_laeufe(1.0, (0.1, 0.5), 3, 30)),
        V.build_gebiet(E.gebiet_ueber_rho(1.0)[0], K.g_gebiet_ohne(K.entwurf("wagen", 1.0, 0.5)), 0.8, 0.5),
        V.build_zwei_wege(K.integral_zeile("wagen", 1.0, 0.1), K.zustand_gemessen_zeile("wagen", 1.0, 0.5), 60),
        V.build_rampe(*E.rampe_verlauf(1.0, 0.5, 0.01)),
        V.build_stellgrenze(E.stellgrenze_tabelle()),
    ]
    rs = E.rampe_sinus(1.0, 0.5, 0.01, 0.4)
    figuren.append(V.build_frequenzgang(rs["ws"], rs["kurve"], rs["kurve0"], 0.4))
    sg = E.Settings(d=1.5, umax=1.0, b=-0.4)
    figuren.append(V.build_ort(E.ergebnis(sg), sg, 60))
    figuren.append(V.build_schaetzung(E.ergebnis(sg), sg, 60))
    return figuren


def test_alle_figuren_haben_gesperrte_achsen_und_daten():
    for fig in _figuren():
        assert len(fig.data) >= 1
        assert fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def test_ort_figur_zeigt_ohne_und_mit_modell_und_das_ziel():
    s = E.Settings(r=2.0)
    fig = V.build_ort(E.ergebnis(s), s, 50)
    namen = [t.name for t in fig.data]
    assert any("ohne Störungsmodell" in n for n in namen) and any("mit Störungsmodell" in n for n in namen)
    assert len(fig.data[0].x) == 50 and any(abs(sh.y0 - 2.0) < 1e-12 for sh in fig.layout.shapes)
    sb = E.Settings(b=-0.4, d=0.7)
    assert any("gemessen" in (t.name or "") for t in V.build_ort(E.ergebnis(sb), sb, 50).data)


def test_farben_sind_definiert():
    assert all(c.startswith("#") for c in (C.COLOR_OHNE, C.COLOR_MIT, C.COLOR_ZIEL, C.COLOR_SCHAETZ, C.COLOR_ZWEI))
