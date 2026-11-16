"""Erzeugt `data/off_messreihen.json`: die Messreihen, die die App nicht live rechnet (vorgerechnet).

Aufruf (Wurzelordner der Demo): <python> tools/off_make_messreihen.py     (Laufzeit wenige Sekunden)
Inhalt (alles mit SplitMix64, Zufalls-Seed in der Datei):
  R                  2 000 Werte R log-gleichverteilt in [1e-8, 1e4]: kleinster und größter Wert von 1/K1, Zahl der Rückgänge nach R sortiert, Zahl der Werte unter dem goldenen Schnitt
  brueche            300 zufällige stabile rationale Gesetze K = (k1, k2): Fixpunkt in Brüchen gleich (d/k1, 0)
  g                  je Beobachter (3 x 0,2 / 0,5 / 0,8) 1 000 zufällige g im stabilen Gebiet, B_dt = g B und B_dt = B: Zahl stabiler Fälle, größter Betrag des Fixpunktfehlers
  zufallsstrecken    fünf Fehlergrößen delta, je 3 000 Zufallsstrecken: Zahl stabiler Strecken, größter Betrag des Ortsfehlers
Die Tests rechnen die Reihen neu (tests/test_claims.py, als Band: Zählungen ± 2, Fehler unter 1e-9)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import off_constants as C  # noqa: E402
import off_evaluation as E  # noqa: E402


def main() -> None:
    data = E.messreihen()
    C.DATA_DIR.mkdir(exist_ok=True)
    C.MESSREIHEN_PATH.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("geschrieben:", C.MESSREIHEN_PATH)
    print(json.dumps(data["zufallsstrecken_summe"]))


if __name__ == "__main__":
    main()
