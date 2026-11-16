"""SplitMix64 (`off_rng.py`, Kopie der geprüften Vorlage): bekannte Ausgabe, Wertebereiche, plattformfeste Normalzahlen."""

import math

import numpy as np

import off_rng as R


def test_splitmix64_bekannter_erster_wert():
    """Seed 0 gibt als erste Ausgabe 0xE220A8397B1DCDAF (Referenzwert des SplitMix64); die Folge ist ganzzahlig und wiederholbar."""
    assert R.SplitMix64(0).next_u64() == 0xE220A8397B1DCDAF
    a, b = R.SplitMix64(123), R.SplitMix64(123)
    assert [a.next_u64() for _ in range(50)] == [b.next_u64() for _ in range(50)]


def test_gleichverteilung_und_ganzzahlen():
    g = R.SplitMix64(7)
    u = [g.random() for _ in range(2000)]
    assert 0.0 <= min(u) and max(u) < 1.0 and abs(np.mean(u) - 0.5) < 0.03
    h = R.SplitMix64(9)
    assert {h.below(5) for _ in range(500)} == set(range(5))


def test_normalzahlen_ohne_libm_sind_standardnormal_und_wiederholbar():
    """`normal()` rechnet Logarithmus und Kosinus mit `decimal` (kein `math.log`/`math.cos`): 4 000 Werte haben Mittel nahe 0 und Standardabweichung nahe 1, und zwei Läufe sind bitgleich."""
    erster = R.SplitMix64(5).normal()
    g = R.SplitMix64(5)
    y = [g.normal() for _ in range(4000)]
    assert abs(np.mean(y)) < 0.06 and abs(np.std(y) - 1.0) < 0.05 and y[0] == erster


def test_exakte_funktionen_gegen_bekannte_werte():
    assert abs(R.exact_log(math.e) - 1.0) < 1e-15
    assert abs(R.exact_log(10.0) - 2.302585092994046) < 1e-15
    assert abs(R.exact_cos(0.0) - 1.0) < 1e-16 and abs(R.exact_cos(math.pi) + 1.0) < 1e-15
