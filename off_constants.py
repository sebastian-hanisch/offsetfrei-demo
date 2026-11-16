"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
MESSREIHEN_PATH = DATA_DIR / "off_messreihen.json"

# Seitenleiste ------------------------------------------------------------------------------------------------------------------------
STRECKEN = ("wagen", "skalar1", "skalar_halb")
DEFAULT_STRECKE = "wagen"
R_OPTIONS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 20.0)   # Stellgewicht R (Q = I); frühere Rechnungen benutzten 1, 0,1, 0,01, 0,001 und am Skalar 2
DEFAULT_R = 1.0
D_MIN, D_MAX, D_STEP, DEFAULT_D = -2.0, 2.0, 0.1, 1.0                         # konstante Störkraft am Eingang
SOLL_MIN, SOLL_MAX, SOLL_STEP, DEFAULT_SOLL = -5.0, 5.0, 0.5, 0.0             # Sollwert des Ortes
RHO_MIN, RHO_MAX, RHO_STEP, DEFAULT_RHO = 0.05, 0.95, 0.05, 0.5               # Pol des Beobachters (Wagen: dreifach)
B_MIN, B_MAX, B_STEP, DEFAULT_B = -2.0, 2.0, 0.1, 0.0                         # Messbias
UMAX_AUS = "aus"
UMAX_OPTIONS = (UMAX_AUS, 0.5, 1.0, 1.5, 2.0, 3.0)                            # Stellgrenze |u| <= u_max
DEFAULT_UMAX = UMAX_AUS
SEED_MAX, DEFAULT_SEED = 999999, 7

# Experimente im Hauptbereich -------------------------------------------------------------------------------------------------------
G_MIN, G_MAX, G_STEP, DEFAULT_G = 0.2, 2.0, 0.05, 0.8                         # wahre Streckenverstärkung (Entwurf für 1)
DELTA_OPTIONS = (0.02, 0.05, 0.1, 0.2, 0.4)                                   # Zufallsstrecken: Fehler je Eintrag von A und B
DEFAULT_DELTA = 0.1
DCAL_MIN, DCAL_MAX, DCAL_STEP, DEFAULT_DCAL = 0.0, 2.0, 0.1, 0.9             # einmal gesetzte Konstante der Vorsteuerung
QI_OPTIONS = (1e-4, 1e-3, 1e-2, 0.1, 1.0)                                     # Gewicht des Integralzustands
DEFAULT_QI = 0.1
RAMPE_MIN, RAMPE_MAX, RAMPE_STEP, DEFAULT_RAMPE = 0.0, 0.05, 0.005, 0.01      # Steigung s der Rampe d_k = s k
OMEGA_OPTIONS = (0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 1.0, 2.0)                    # Kreisfrequenz der Sinusstörung (Bogenmaß je Periode)
DEFAULT_OMEGA = 0.4

# feste Annahmen -------------------------------------------------------------------------------------------------------------------
RHO_TABELLE = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)                 # Raster der Beobachterpole in Experiment B (Wagen)
QD_KALMAN = (1e-6, 1e-4, 1e-2, 1.0)                                           # Prozessrauschen der Störung für die Kalman-Zeile
R_TABELLE = (1.0, 0.1, 0.01, 0.001)                                           # Raster von R in Experiment A (Wagen)
R_SKALAR = (0.5, 1.0, 2.0, 5.0, 20.0)                                         # Raster von R am Skalar a = 1
STELL_D = (0.3, 0.9, 1.5)                                                     # Störungen der Stellgrenzen-Tabelle (u_max = 1)
STELL_UMAX = 1.0
STELL_START = (4.0, 0.0)                                                      # Start der Stellgrenzen-Tabelle: Ort 4, Geschwindigkeit 0
DCAL_TABELLE = (0.0, 0.5, 0.9, 1.0, 1.1, 1.5)
T_LANG = 6000                                                                 # Länge des Laufs für die Kreuzungsschritte (Summe aller |p_k|)
N_ZUFALL_LIVE = 500                                                           # Zufallsstrecken im Live-Lauf von Experiment C
T_RAUSCH = 240                                                                # Länge des Rauschlaufs (nur zur Anschauung)
T_GRENZE = 300                                                                # Länge der Läufe mit Stellgrenze
KALMAN_V = 0.01                                                               # Prozessrauschen der Geschwindigkeit in der Kalman-Zeile

COLOR_OHNE = "#c0504d"
COLOR_MIT = "#2F6B65"
COLOR_ZIEL = "#7f7f7f"
COLOR_SCHAETZ = "#4c78a8"
COLOR_ZWEI = "#e8a33d"

PRESETS = {
    "Mit Störungsbeobachter (ρ = 0,5)": {"strecke": "wagen", "R": 1.0, "d": 1.0, "r": 0.0, "rho": 0.5, "b": 0.0, "umax": UMAX_AUS},
    "Von Hand (Skalar, d = 1)": {"strecke": "skalar1", "R": 2.0, "d": 1.0, "r": 0.0, "rho": 0.5, "b": 0.0, "umax": UMAX_AUS},
    "Kleines R (R = 0,01)": {"strecke": "wagen", "R": 0.01, "d": 1.0, "r": 0.0, "rho": 0.5, "b": 0.0, "umax": UMAX_AUS},
    "Schneller Beobachter (ρ = 0,1)": {"strecke": "wagen", "R": 1.0, "d": 1.0, "r": 0.0, "rho": 0.1, "b": 0.0, "umax": UMAX_AUS},
    "Langsamer Beobachter (ρ = 0,9)": {"strecke": "wagen", "R": 1.0, "d": 1.0, "r": 0.0, "rho": 0.9, "b": 0.0, "umax": UMAX_AUS},
    "Messbias (b = −0,4)": {"strecke": "wagen", "R": 1.0, "d": 0.7, "r": 0.0, "rho": 0.5, "b": -0.4, "umax": UMAX_AUS},
    "Stellgrenze (d = 1,5)": {"strecke": "wagen", "R": 1.0, "d": 1.5, "r": 0.0, "rho": 0.5, "b": 0.0, "umax": 1.0},
}
PRESET_HELP = {
    "Mit Störungsbeobachter (ρ = 0,5)": "Voreinstellung: Wagen, R = 1, konstante Störkraft d = 1, Beobachter mit dreifachem Pol 0,5. Der Ort bleibt ohne Störungsmodell bei d/K₁ stehen und geht mit Störungsmodell auf 0; die Spitze kostet das.",
    "Von Hand (Skalar, d = 1)": "Das Handbeispiel: Skalar x⁺ = x + u + d, Kosten x² + 2 u², Pole des Beobachters ½ und ¼. Der Dauerfehler ist 2 = d/k, mit Beobachter 0. Die Rechnung in Brüchen steht im Expander unter dem Ergebnis.",
    "Kleines R (R = 0,01)": "Ein stärkerer Eingriff drückt den Dauerfehler des Wagens von 2,369 auf 1,630, aber nicht darunter (nie unter 1,618 in 2 000 gemessenen R).",
    "Schneller Beobachter (ρ = 0,1)": "Pole bei 0,1: die Spitze bleibt klein (6,0), aber das Messrauschen wird stark an die Stellgröße weitergegeben (Experiment B).",
    "Langsamer Beobachter (ρ = 0,9)": "Pole bei 0,9: wenig Verstärkung des Messrauschens (0,20), aber eine Spitze von 55 und eine Abklingzeit von 94 Perioden.",
    "Messbias (b = −0,4)": "Der Ortssensor zeigt 0,4 zu wenig und die Störung ist 0,7: Die gemessene Größe geht auf 0, der wahre Ort bleibt bei 0,4. Offsetfrei heißt: die gemessene Größe erreicht das Ziel.",
    "Stellgrenze (d = 1,5)": "Die Störkraft 1,5 ist größer als die Stellgrenze 1: Der Wagen läuft weg, mit und ohne Störungsmodell.",
}
