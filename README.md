# Offsetfrei regeln: die Störung mitschätzen und was es kostet – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-offsetfrei-demo.streamlit.app/)** -->

Stück 8 der Linie **Regelung (von LQR bis MPC)** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Linie fragt, wann eine Verstärkung reicht, was ein Plan oder eine Optimierung in jedem Schritt zusätzlich leistet und was dabei schiefgehen kann; Vehikel ist durchgehend **ein Wagen** (Doppelintegrator, Takt 1, Ort gemessen) mit einem **Skalar** als Handbeispiel, nicht das Lager.
Die Kante zu den Vorgängern ist `fix`: [`plan-stoerung-demo`](https://github.com/sebastian-hanisch/plan-stoerung-demo) zeigt, dass die Rückführung u = −K x eine konstante Störung hält, aber einen **Dauerfehler** d/K₁ (am Wagen bei R = 1 etwa 2,37·d) stehen lässt, und [`lqr-riccati-demo`](https://github.com/sebastian-hanisch/lqr-riccati-demo) liefert den Gewinn K. Dieses Stück beseitigt den Fehler (Störungsmodell mit Beobachter, als zweiter Weg der Integralzustand) und fragt, **was die Beseitigung kostet** und wo sie nicht hilft. Konventionen sind die der Vorgänger: u = −K x, Kosten Σ (x'Q x + R u²) ohne Faktor ½, Q = I (am Skalar Q = 1).
Das Wort „Dauerfehler“ hat in der Reihe zwei Ursachen: hier die konstante Störkraft am Wagen ohne Integralwirkung; im Lager ist es der Fehler einer falsch geschätzten Lieferzeit, siehe [`leitung-mitrechnen-demo`](https://github.com/sebastian-hanisch/leitung-mitrechnen-demo), [`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo) und [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) (dort ohne Beobachter und ohne Integral). Weitere Nachbarn: [`windup-pi-demo`](https://github.com/sebastian-hanisch/windup-pi-demo) (Integrator mit Stellgrenze), [`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo), [`lqr-reserven-demo`](https://github.com/sebastian-hanisch/lqr-reserven-demo) (Reserven und LQG), [`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo) (Rechenzeit als Verzug) und [`kalman-filter-demo`](https://github.com/sebastian-hanisch/kalman-filter-demo) (Filter, NEES, Einstellung von Q und R; hier nur eine Zeile). **Alle Formeln und Befunde sind eigene Nachrechnung in diesem Modell, bei den genannten Parametern**; eine Neuheit wird nicht behauptet.

## Warum dieses Problem

Die Rückführung u = −K x hält eine konstante Störkraft d am Eingang des Wagens, aber nicht am Ziel: Im Beharrungszustand verlangt der Wagen v = 0 und K₁·p = d, also **p = d/K₁** (R = 1: 2,369·d). Ein kleineres R drückt den Fehler (1,630·d bei R = 0,01), aber in 2 000 gemessenen R nie unter 1,618·d. Ein **Störungsmodell** (die Störung als Zustand mit d⁺ = d) und ein Beobachter, der sie aus dem gemessenen Ort schätzt, beseitigen den Fehler, und zwar in diesem Modell für jede stabile Auslegung und jede wahre Strecke, die den Kreis stabil lässt (Dauerfehler 0 in 6 000 von 6 000 stabilen Fällen mit falscher Streckenverstärkung und in 10 171 von 10 171 stabilen Zufallsstrecken; ein Messbias bleibt als Ortsfehler).
**Bei getrenntem Entwurf (K aus dem LQR ohne Störung, L aus Polvorgabe) hat die Beseitigung einen Preis:** Bei R = 1 und Beobachterpolen 3 × 0,5 steigt die Spitze des Ortes nach einem Störsprung von 2,38 (reines LQR, das bei 2,37 stehen bleibt) auf 9,02, der Frequenzgang |p/d| erreicht 13,1 statt 2,37, und das stabile Gebiet der Streckenverstärkung g schrumpft von (0 bis 1,94) auf (0,41 bis 1,29). Ein schneller Beobachter (ρ = 0,1) hält die Spitze bei 6,0, verstärkt aber das Messrauschen (std(u) 15,9 je std(n)); ein langsamer (ρ = 0,9) rauscht weniger (0,20), hat aber Spitze 55 und Abklingzeit 94.
Die Demo fragt: **Bleibt ein Fehler stehen, wie kommt er weg und was kostet das, und wann hilft es nicht?** Die Fachliteratur (Störungsmodell in der prädiktiven Regelung, Integralzustand) ist nur nach dem Quellenbericht der Planung bekannt, nicht beim Bau erneut geöffnet; „bekannt“ oder „neu“ wird deshalb nicht gesagt.

## Modell

Strecke x⁺ = A x + B (u + d), gemessen y = C x + b (Messbias b), Ziel y = r. **Wagen:** x = (Ort p, Geschwindigkeit v), A = [[1, 1], [0, 1]], B = (0, 1)ᵀ, C = (1, 0), d eine konstante Störkraft am Eingang; **Skalar:** x⁺ = a x + u + d mit a = 1 oder ½ (Handbeispiel, Q = 1, R = 2).

- **Gesetz:** K aus der diskreten Riccati-Gleichung (SciPy), u = −K x. Der Wagen bei R = 1: K = (0,42208; 1,24393).
- **Störungsmodell:** z = (x, d), d⁺ = d, A_e = [[A, B], [0, 1]], B_e = (B; 0), C_e = (C, 0); Beobachter in Vorhersageform ẑ⁺ = A_e ẑ + B_e u + L (y − C_e ẑ) mit der Ackermann-Verstärkung (Wagen: dreifacher Pol ρ; Skalar: Pole ρ und ρ/2 wie im Handbeispiel); Gesetz u = u_s − K (x̂ − x_s) mit Zielwerten aus [[I − A, −B], [C, 0]] (x_s, u_s) = (B d̂, r) (Wagen: x_s = (r, 0), u_s = −d̂).
- **Spitze** = größter Betrag |p_k − r| nach einem Störsprung d bei k = 0 aus der Ruhe im Sollwert, **Abklingzeit** = letzte Periode, in der |p_k − p_Beharrung| über 1 % der Spitze liegt (Definition der Reihe „Rückkopplung und Verzögerung“), **Dauerfehler** = Ort im Beharrungszustand (Fixpunkt des Regelkreises). Zähleinheit: Periode.
- **Beweisskizze der Offsetfreiheit in diesem Modell:** Im Fixpunkt verlangt die d̂-Zeile des Beobachters L_d·(y − ŷ) = 0, also y = ŷ; die v̂-Zeile gibt dann v̂ = 0, die Gesetzeszeile K₁ p̂ = 0, also p = p̂ = 0. Die wahre Strecke geht nicht ein, nur ihre Stabilität (und L_d ≠ 0, K₁ ≠ 0). Gemessen wird das mit Fixpunkten in Gleitkomma und in Brüchen und mit Zufallsstrecken (unten).
- **Trennung:** das charakteristische Polynom des Kreises ist das Produkt aus dem Polynom von A − B K und dem des Beobachters; verglichen wird über die Polynomkoeffizienten, nicht über Eigenwerte (dreifache Pole sind als Eigenwerte schlecht konditioniert: Abstand von der Größe 10⁻⁶).
- **Rangbedingung:** rank [[I − A, −B_d], [C, C_d]] = n + n_d (exakt in Brüchen gerechnet).
- **Integralzustand (zweiter Weg):** ξ⁺ = ξ + p, LQR mit Q = diag(1, 1, q_i), Sollwert 0, Zustand gemessen.
- Die Formeln stehen im Expander 📐 der App.

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Fixpunkte** (`off_dynamics.fixpunkt`, `off_exakt`) | lineares System (I − A_cl) z = B_w w; dieselbe Rechnung mit Brüchen für die Handwerte | setzt einen stabilen Kreis voraus |
| **Spitze und Abklingzeit** (`kennzahlen`, `sprung`) | Simulation des linearen Kreises, Länge aus dem Spektralradius (bis 6 000 Perioden) | die Abklingzeit misst gegen den Beharrungswert, nicht gegen das Ziel |
| **Frequenzgang** (`max_gain`) | (e^{jω} I − A_cl)⁻¹ b_d über 6 000 Frequenzen in (0, π], danach feinere Suche um das Maximum | Gegenprobe: Sinus-Simulation (Band 2 %) |
| **Rauschen** (`rausch_std`) | Lyapunov-Gleichung für weißes Messrauschen mit Varianz 1 auf dem Ort; Anschauung: ein Lauf mit SplitMix64-Normalzahlen (`off_rng.py`) | Gegenprobe: 40 000 Schritte, Band 5 % |
| **Gebiet der Streckenverstärkung** (`g_gebiet`) | Gitter mit 4 000 Punkten (g bis 8), Kanten per Bisektion; wahre Strecke B_t = g B, B_dt = g B | Gegenprobe: Jury-Kriterium an den Kanten |
| **Reserve** (`reserve`) | kleinster Betrag von 1 + L(e^{jω}) am Streckeneingang, Regler C_c(z) = K_z (zI − A_c)⁻¹ L | eine Reserve im Frequenzbereich, keine Aussage über Zeitbereichsgrenzen |
| **Stellgrenze** (`lauf_grenze`) | Abschneiden von u; Strecke **und** Beobachter bekommen die angewendete Stellgröße | kein QP (siehe „Bewusst nicht umgesetzt“) |

Zufall nur über den ganzzahligen SplitMix64 (`off_rng.py`, Kopie der geprüften Vorlage aus dem Website-Repo, `normal()` ohne `math.log` und `math.cos`); die Zufalls-Seeds stehen in `data/off_messreihen.json` und im Text. Die Live-Rechnung der App braucht je Eingabe etwa 0,25 Sekunden (Regler d, g, ρ), 0,9 Sekunden (Änderung von R) und beim Erstaufruf etwa 2,4 Sekunden (Messung mit AppTest auf dem Entwicklungsrechner, nicht festgeschrieben). Die Reihen, die die App nicht live rechnet, sind **vorgerechnet** (`data/off_messreihen.json`, erzeugt von `tools/off_make_messreihen.py`): 2 000 R, 300 rationale Gesetze, 6 000 Fälle mit falscher Verstärkung und 15 000 Zufallsstrecken.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: **Wagen**, konstante Störkraft am Eingang, Ort gemessen, K aus der DARE (Q = I), Beobachter 3 × ρ (Ackermann), Störsprung aus der Ruhe, Messrauschen weiß mit Varianz 1; „vorgerechnet“ steht in `data/off_messreihen.json` und wird im Test neu gerechnet (Zählungen ± 2, Fehler unter 1e-9). Die Spalte „Test“ nennt `tests/test_claims.py`, wenn nichts anderes steht; die Vormessungen (`out_rg_v17_*.txt` der Planung, nicht im Repo) sind die Erwartung, im Bau neu gemessen.

| Frage | Befund | Test |
|---|---|---|
| **M1** Dauerfehler des Gesetzes | p_ss/d = 1/K₁ = **2,369 / 1,727 / 1,630 / 1,619** für R = 1 / 0,1 / 0,01 / 0,001 (v_ss = 0; Spektralradius 0,422 / 0,362 / 0,380 / 0,382); Handwert K = (2/5, 6/5): 5/2. Vorgerechnet: Fixpunkt in Brüchen gleich (d/K₁, 0) in **300 von 300** zufälligen stabilen rationalen Gesetzen (Zufalls-Seed 17); unter **2 000** zufälligen R (10⁻⁸ bis 10⁴, Zufalls-Seed 174) liegt 1/K₁ zwischen 1,618034 und 105,7, wächst mit R (0 Rückgänge) und liegt nie unter 1,618033 (numerisch gleich dem goldenen Schnitt, **nicht hergeleitet**). Skalar a = 1: x_ss/d = 1,366 / 1,618 / 2,000 / 2,791 / 5,000 für R = 0,5 / 1 / 2 / 5 / 20 | `test_m1_*` |
| **M2** Sollwert | Wagen, u = −K (x − (r, 0)), ohne Störung: Fixpunkt p = r = 2,5 (der Wagen integriert selbst, u_s = 0). Skalar a = ½, k = ¼, r = 1: ohne Zielwert x_ss = 1/3 (Fehler 2/3 = r (1 − a)/(1 − a + b k)), mit u_s = ½ Fehler 0 (Brüche) | `test_m2_*` |
| **M3** Störungsmodell mit Beobachter | Fixpunkt (p, v, d̂) = (0, 0, d) in Gleitkomma (R = 1 mit ρ = 0,5 und 0,8; R = 0,1 mit ρ = 0,3) und **exakt in Brüchen** (K = (2/5, 6/5), L = (3/2, 3/4, 1/8) für 3 × ½: Fixpunkt (0, 0, 0, 0, 1)); mit Sollwert r = 1 und r = −3/2 bei d = 1: p = r. Trennung: Koeffizientenabstand unter 1e-10 (9 Beobachter; auf dem Entwicklungsrechner um 1e-14); als Eigenwerte weicht der dreifache Pol 0,5 um etwa 7·10⁻⁶ ab | `test_m3_*` |
| **M4** Rangbedingung | Wagen, Störung am Eingang: Rang 3 von 3 (erfüllt); **Wagen, Störung am Ausgang: Rang 2 von 3 (verletzt)**; Wagen mit zwei Störungen bei einer Messung: 3 von 4; Skalar a = 1: Eingang 2 von 2, Ausgang 1 von 2; Skalar a = ½, Störung am Ausgang: 2 von 2 (erfüllt) | `test_m4_*`, `test_oracle_*::test_h_*` |
| **M5** Rampe und Sinus | Rampe d_k = s·k, s = 0,01 (R = 1, 3 × 0,5): der Ort wird **konstant** 0,5758 (= 57,58·s), nicht 0; in Brüchen mit K = (2/5, 6/5): 59·s, d̂ liegt um 6·s hinter der Rampe; ohne Beobachter läuft der Ort mit der Steigung s/K₁ = 0,02369 je Schritt weg. Sinus d_k = sin(ω k), \|p/d\| mit gegen ohne Beobachter: ω = 0,01: 0,576 gegen 2,369; 0,05: 2,851 gegen 2,366; 0,1: 5,539 gegen 2,357; 0,2: 9,892 gegen 2,322; 0,5: 12,467 gegen 2,074; 1: 4,568 gegen 1,425; 2: 0,183 gegen 0,688. Spitze des Beobachterkreises 13,120 bei ω = 0,3952 | `test_m5_*` |
| **M6** Stellgrenze | Abschneiden \|u\| ≤ 1, Start (Ort 4, Geschwindigkeit 0), R = 1, ρ = 0,5, Ort nach 300 Schritten: d = 0,3: ohne Modell 0,71076 (= d/K₁), mit Modell 0; d = 0,9: ohne 2,13228, mit 0; **d = 1,5 > u_max: der Wagen läuft weg**, mit Modell 665,5 / 2 578,0 / 22 728,0 nach 50 / 100 / 300 Schritten, ohne 616,5 / 2 479,0 / 22 429,0. **Gleichstand d = u_max = 1:** ohne Modell bleibt der Ort bei 4,0, mit Modell läuft er mit der Geschwindigkeit 1 weg (303,0 nach 300 Schritten). Aus der Ruhe, d = 0,9: Spitze 40,5 mit Modell | `test_m6_*` |
| **M7** Modellfehler | Wahre Verstärkung g (B_t = g B, B_dt = g B, Entwurf für g = 1), stabil für g von 0,0005 (Gitterrand) bis 1,936 beim reinen LQR; mit Beobachter 3 × 0,2 / 0,5 / 0,8: 0,786 bis 1,170 / 0,406 bis 1,293 / 0,274 bis 1,785. **Dauerfehler 0 in 2 000 von 2 000 stabilen Fällen je Beobachter** (vorgerechnet, größter Rest 1,1e-14). Falsche Schrittweite h (A_t = [[1, h], [0, 1]]) gibt dasselbe Gebiet 0,406 bis 1,293 und Dauerfehler 0 in 500 Werten. **Zufallsstrecken** (je Eintrag ± δ, B_dt zufällig), vorgerechnet, je 3 000: stabil in 3 000 / 2 999 / 2 641 / 1 155 / 376 für δ = 0,02 / 0,05 / 0,1 / 0,2 / 0,4; **Dauerfehler 0 in 10 171 von 10 171 stabilen** (größter Rest 1,3e-14) | `test_m7_*`, `test_oracle_*::test_n_*` |
| **M8** Falsche Konstante, Messbias, L_d = 0 | Statische Vorsteuerung u = −K x − d_cal ohne Beobachter, d = 1: p_ss = (d − d_cal)/K₁ = 2,369 / 1,185 / 0,237 / 0 / −0,237 / −1,185 für d_cal = 0 / 0,5 / 0,9 / 1,0 / 1,1 / 1,5. Messbias b: wahrer Ort −b (3 von 3 Beobachtern); d = 0,7 und b = −0,4: p_ss = 0,4, d̂ = 0,7. Ohne Störungsmodell im Beobachter (L_d = 0): Eigenwert 1, der Ort bleibt bei 16,509·d (reines LQR: 2,369·d); **Ursache nicht gemessen** | `test_m8_*` |
| **M9** Beobachterpole | Tabelle unten; Kalman-Verstärkung statt Polvorgabe: q_d = 10⁻⁶ / 10⁻⁴ / 10⁻² / 1: Spitze 30,35 / 22,29 / 11,92 / 7,18, Abklingzeit 471 / 57 / 25 / 11. Zustand gemessen (Pole ρ, 0,9 ρ, 0,8 ρ): ρ = 0,1 / 0,3 / 0,5 / 0,8: Spitze 3,000 / 3,230 / 3,910 / 6,172, Abklingzeit 8 / 9 / 13 / 29 | `test_m9_*` |
| **M10** Wann zahlt sich die Offsetfreiheit aus? | \|p_k\| des Beobachterkreises liegt **dauerhaft** unter dem des reinen LQR ab Schritt 7 / 8 / 12 / 20 / 77, die kumulierte Summe ab Schritt 11 / 16 / 27 / 70 / 828 für ρ = 0,1 / 0,3 / 0,5 / 0,7 / 0,9 (Summe aller \|p_k\| des Beobachterkreises 20,3 / 31,1 / 57,6 / 159,0 / 1 955,2) | `test_m10_*` |
| **M11** Zwei Wege (Zustand gemessen, ein Entwurf je Familie) | Integralzustand, q_i = 10⁻⁴ / 10⁻³ / 10⁻² / 0,1 / 1: Spitze 2,294 / 2,164 / 1,899 / 1,494 / 1,132, Abklingzeit 466 / 151 / 51 / 18 / 9, max \|p/d\| 2,357 / 2,333 / 2,259 / 2,076 / 1,772, Spektralradius 0,990 / 0,969 / 0,904 / 0,718 / 0,481. Störungsbeobachter mit Zustand gemessen: Spitze 3,0 bis 6,2, max \|p/d\| 4,2 bis 8,4 | `test_m11_*` |
| **M12** Handbeispiel | Skalar R = 2: P₀, P₁, P₂ = 1, 5/3, 21/11 → P = 2, k = ½, Dauerfehler 2; Beobachter für Pole ½ und ¼: L = (5/4, 3/8); x_k = 0, 1, 2, 2, 25/16, 69/64, …; Polynom des Kreises (z − ½)² (z − ¼); Integralzustand: x_k = k/2^(k−1), Spitze 1 | `test_dynamics.py` |

**Tabelle M9 (Beobachterpole 3 × ρ, R = 1, Störsprung d = 1 aus der Ruhe):** Spalten: Spitze (Schritt), Abklingzeit, max |p/d| (bei ω), std(p) und std(u) je std(n) für weißes Messrauschen mit Varianz 1 auf dem Ort, stabiles Gebiet der Streckenverstärkung g, kleinster Betrag von 1 + L am Streckeneingang (Reserve).

| ρ | Spitze (Schritt) | Abklingzeit | max \|p/d\| (ω) | std(p) | std(u) | g-Gebiet | Reserve |
|---|---|---|---|---|---|---|---|
| 0,05 | 6,000 (4) | 9 | 8,209 (0,740) | 6,749 | 19,386 | 0,876 bis 1,127 | 0,1125 |
| 0,1 | 6,000 (4) | 9 | 8,516 (0,701) | 5,850 | 15,940 | 0,851 bis 1,146 | 0,1264 |
| 0,2 | 6,087 (5) | 10 | 9,239 (0,624) | 4,473 | 10,792 | 0,786 bis 1,170 | 0,1454 |
| 0,3 | 6,942 (5) | 12 | 10,159 (0,549) | 3,483 | 7,271 | 0,691 bis 1,200 | 0,1660 |
| 0,5 | 9,022 (6) | 17 | 13,120 (0,395) | 2,154 | 3,125 | 0,406 bis 1,293 | 0,2195 |
| 0,7 | 14,469 (8) | 30 | 20,727 (0,234) | 1,262 | 1,112 | 0,329 bis 1,510 | 0,3131 |
| 0,8 | 22,523 (11) | 46 | 32,076 (0,153) | 0,894 | 0,558 | 0,274 bis 1,785 | 0,3988 |
| 0,9 | 55,491 (21) | 94 | 78,935 (0,074) | 0,541 | 0,204 | 0,203 bis 2,621 | 0,5563 |
| 0,95 | 162,011 (40) | 191 | 230,385 (0,036) | 0,352 | 0,086 | 0,160 bis 4,309 | 0,7036 |
| ohne Beobachter | 2,376 (Dauerfehler 2,369) | – | 2,369 | – | – | 0,0005 bis 1,936 | 0,4836 |

Zeile für Zeile gilt: keine Einstellung senkt Spitze, Abklingzeit, Rauschen und Frequenzgang zugleich (in diesen Zeilen: Spitze und std(u) laufen gegeneinander). Gegenproben (Test `test_m9_*`, `test_oracle_*::test_e_*`): Rauschvarianz aus der Lyapunov-Gleichung gegen 40 000 Schritte (ρ = 0,3: std(p) 3,483, std(u) 7,271; Band 5 %), Frequenzspitze gegen die Sinus-Simulation (Band 2 %), Spitzenfolge in Brüchen p₁ bis p₈ = 0, 1, 3, 6, 67/8, 147/16, 171/20, 5657/800 (K = (2/5, 6/5), 3 × ½).

### Befunde und Korrekturen gegenüber dem Plan

- **Zählung der stabilen Zufallsstrecken:** Die Vormessung (numpy) fand 3 000 / 3 000 / 2 586 / 1 210 / 367 (Summe 10 163); mit dem Zufallsstrom der Demo (SplitMix64) sind es 3 000 / 2 999 / 2 641 / 1 155 / 376 (Summe 10 171). Die Zählung hängt an der Schwelle „Spektralradius unter 1 − 10⁻⁶“ und an der Zufallsquelle; der Befund „Dauerfehler 0 in allen stabilen“ ändert sich nicht. Die Tests prüfen die Zählung als Band (± 2) und den Fehler mit einer **anderen** Zufallsquelle (numpy) mit gelockerter Annahme (B_dt beliebig in [−1, 1]², auch ohne Bezug zum Eingang).
- **Kanten der g-Gebiete:** Die Vormessung nahm ein Gitter mit Schrittweite 0,001, hier stehen Bisektionskanten: 0,876 / 0,851 / 0,786 / 0,691 / 0,406 / 0,329 / 0,274 / 0,203 / 0,160 statt 0,877 / 0,852 / 0,787 / 0,692 / 0,407 / 0,330 / 0,275 / 0,204 / 0,161 (Abweichung höchstens 0,001, nur im Wert, nicht im Befund). Ebenso die Frequenzspitze: ω = 0,3952 gegen 0,3954 im Gitter, Spitze 13,120 gleich.
- **Stellgrenze nur als Abschneiden:** Der Plan hatte die Zahlen für d = 1,5 aus einem MPC (N = 10, Box-QP); hier steht das Abschneiden von u = −K x̂ − d̂. Die Zahlen für d = 1,5 sind **dieselben** (665,5 / 2 578,0 / 22 728,0), weil die Stellgröße in jedem Schritt an der Grenze liegt und die Wahl des Reglers dort nichts mehr ändert (aus v⁺ ≥ v + d − u_max, für jede Regel mit |u| ≤ u_max). Der Plan nannte an zwei Stellen 22 729; die Ausgabe der Vormessung ist 22 728,0. Das MPC-Orakel (MPC ohne aktive Grenze gleich LQR) ist nicht umgesetzt (siehe unten).
- **Experimentnummern:** Das Experiment 2 des Plans („Die Störung mitschätzen“) ist die **Primäransicht** der App; die Experimente 1 und 3 bis 6 des Plans heißen hier A bis E (A Dauerfehler und Sollwert, B Beobachterpole, C Modellfehler, D zwei Wege, E Grenzen).
- **Presets:** Das Preset „Wagen ohne Störungsmodell (R = 1)“ des Plans wäre mit der Voreinstellung identisch gewesen, weil die Primäransicht beide Kurven immer zeigt; stattdessen „Kleines R (R = 0,01)“. Ein Klick auf ein Preset setzt auch die Regler der Experimente und den Zufalls-Seed auf ihren Standardwert.
- **Gleichstand an der Schwelle |d| = u_max:** wegen v⁺ ≥ v + |d| − u_max kann eine einmal entstandene Geschwindigkeit nicht abgebaut werden; ob der Wagen ankommt, hängt vom Anlauf ab (Start (4; 0): ohne Modell bleibt er bei 4,0, mit Modell läuft er weg). Die App meldet |d| = u_max deshalb getrennt von |d| > u_max.
- **Schwellen der Verdikte:** Das Verdikt der Kernfrage 2 meldet eine Warnung ab dem Faktor 1,5 (Spitze mit gegen Spitze ohne Störungsmodell, Schwelle inklusive) und sagt bei einem Faktor von höchstens 1 (Handbeispiel: 2,00 gegen 2,00), dass die Beseitigung dort keine zusätzliche Spitze kostet; „Spitze 2 gleich Dauerfehler 2“ ist ein Zufall dieser Zahlen, keine Regel.
- **Messbias und Abklingzeit:** Die Abklingzeit misst gegen den Beharrungswert (r − b), nicht gegen das Ziel; sonst stünde bei b ≠ 0 die Länge des Laufs dort.
- **Kante `fix` gegen PLAN-01 und PLAN-02 geprüft:** gleiche Konventionen (u = −K x, Kosten ohne Faktor ½), gleicher Wert 2,369·d; `plan-stoerung-demo` zeigt den Fehler als Zahl bei d = 0,1, hier steht die Formel mit der Brüche-Probe und die Beseitigung.

## Quellen

| Quelle | Stufe | Was hier verwendet wird |
|---|---|---|
| Rawlings, Mayne, Diehl, *Model Predictive Control*, Kap. 1.5.2 (S. 49–53) | im Wortlaut gelesen laut Quellenbericht der Planung; beim Bau nicht erneut geöffnet; das PDF enthält den Vermerk „Weitergabe nur mit Erlaubnis des Verlags“, **nichts ist wörtlich übernommen** | Gleichungen und Befund in eigenen Worten: Störungsmodell d⁺ = d, Rangbedingung, Lemma zur Offsetfreiheit (stabiler Kreis, Beschränkungen im Beharrungszustand inaktiv, so viele Störungen wie Messungen) |
| Borrelli, Bemporad, Morari, *Predictive Control*, Kap. 12.7 (S. 267–272) | im Wortlaut gelesen laut Quellenbericht | die Sätze zur Beobachtbarkeit des erweiterten Modells und zur Offsetfreiheit sowie die Bemerkung zu Integratoren: hat die Strecke Integratoren, ist B_d = 0 unzulässig (hier gemessen: Wagen, Störung am Ausgang, Rang 2 von 3) |
| Åström, Murray, *Feedback Systems*, Kap. 7.4 | im Wortlaut gelesen laut Quellenbericht | Integralzustand: ist der Kreis stabil, gilt im Beharrungszustand y = r, unabhängig vom genauen Modell |
| Pannocchia/Rawlings 2003 (Beweis des Lemmas), Muske/Badgwell, Maeder/Borrelli/Morari (offsetfreies MPC) | **nicht gelesen** | nichts; deshalb keine Aussage „bekannt“ oder „neu“ |

## Ehrliche Grenzen

- **Ein Modell:** linear, eine Störung am Eingang, eine Messung (n_d = 1 = Zahl der Messungen), Ort gemessen, konstanter Entwurf K für R aus der Seitenleiste mit Q = I. Der Fall weniger Störungen als Messungen (n_d < Zahl der Messungen) ist nicht gemessen.
- **Der Preis gilt für diesen Entwurf:** K aus dem LQR ohne Störung, L aus Polvorgabe (dreifacher Pol am Wagen). Andere Auslegungen (z. B. ein gemeinsamer LQR auf (x, d̂), gestaffelte Pole) sind nicht gemessen; der Satz „Integralzustand ist besser“ wäre zu stark: Er gilt in den fünf gemessenen Entwürfen (Zustand gemessen, ein Entwurf je Familie), und der Integralzustand zahlt mit der Abklingzeit (q_i = 10⁻⁴: 466 Perioden gegen 9 bei q_i = 1).
- **Die Offsetfreiheit gilt für konstante Störungen, einen stabilen Kreis, eine Stellgrenze, die die Gegenkraft zulässt (|d| < u_max), und die gemessene Größe.** Bei einer Rampe bleibt ein konstanter Ortsfehler (57,6·s), bei einem Sinus verstärkt der Beobachterkreis mittlere Frequenzen (13,1 gegen 2,37), bei |d| > u_max läuft der Wagen weg, bei einem Messbias bleibt der wahre Ort um −b daneben.
- **Nicht gemessen:** Prozessrauschen (zufällige statt konstanter Störung) mit Beobachter; die befohlene gegen die angewendete Stellgröße im Beobachter außer im Lauf mit Abschneiden (dort die angewendete); Verzug (Rechenzeit, Totzeit) zusammen mit dem Beobachter; ein Störungsmodell an einem anderen Ort als dem Eingang als Regelkreis (nur über die Rangbedingung); andere Q als Q = I; die Ursache des Ortsfehlers bei L_d = 0 (16,5·d, schlechter als das reine LQR).
- **Zahlen aus Gittern und Zufall:** g-Gebiete und Frequenzspitzen kommen aus Gittern mit Bisektion bzw. feiner Suche (Gegenprobe mit dem Jury-Kriterium); Zählungen von Zufallsstrecken sind Messungen an einem Zufallsstrom, kein Beweis. Spitzen, Abklingzeiten und Gebiete werden im Test als Band geprüft, Zählungen ± 2, nur Brüche exakt.
- **Dreifache Pole:** der Beobachter hat am Wagen einen dreifachen Pol; als Eigenwerte sind solche Pole schlecht konditioniert, die Aussagen laufen über Polynomkoeffizienten und Kreise, nicht über `eigvals` der Pole.
- **Nicht mit echtem Mausklick getestet:** die App ist mit AppTest und im Browser über Presets und Permalinks geprüft; Plotly-Interaktionen sind gesperrt.

## Verifikation

Die Suite hat **223 Tests** (`pytest --collect-only -q tests`), alle ohne Netz; auf dem Entwicklungsrechner braucht sie mit `-n 2` etwa 45 Sekunden (Messung, nicht festgeschrieben).

- **Kern und Handbeispiele (`tests/test_dynamics.py`):** Riccati von Hand (P₁ = 5/3, P₂ = 21/11), Dauerfehler 2·d, Beobachterverstärkung (5/4, 3/8), Folge x_k, Polynom des Kreises (z − ½)² (z − ¼), Integralzustand, Sollwert 2/3, Wagen mit K = (2/5, 6/5) (5/2, Polynom, p₁ bis p₈, Rampe 59·s), Rang; Einheiten (Kennzahlen, Kreuzungsschritte, Stellgrenze).
- **Orakel (`tests/test_oracle_offsetfrei.py`, Hilfen in `tests/oracle_helpers.py`):** (a) Dauerfehler: Fixpunkt gegen Simulation gegen Brüche; (b) Riccati: Rekursion gegen `solve_discrete_are`; (c) Trennung über Polynomkoeffizienten; (d) Ackermann gegen `place_poles` (verschiedene Pole) und in Brüchen; (e) Rauschen: Lyapunov gegen 40 000 Schritte SplitMix64 und gegen die Summe der Stoßantwort; (f) Frequenzgang gegen Sinus-Simulation, der 5er-Kreis dabei Zeile für Zeile unabhängig aufgeschrieben; (g) Rampe: Ansatz gegen Simulation gegen s mal Summe der Sprungantwort; (h) Rang gegen Determinante und `matrix_rank`; (i) g-Gebiet: Gitter gegen Bisektion gegen Jury-Kriterium; (j) Stellgrenze gegen eine zweite Schleife in einfachen Zahlen; (k) Kalman gegen Riccati-Iteration; (l) Integralzustand gegen Riccati-Iteration, Fixpunkt; (m) Messbias; (n) Zufallsstrecken mit anderer Zufallsquelle. Das MPC-Orakel des Plans (MPC ohne aktive Grenze gleich LQR) ist nicht umgesetzt.
- **Eigenschaften (`tests/test_eigenschaften_offsetfrei.py`):** Linearität, Spiegelung, Überlagerung, Zeitverschiebung, Offsetfreiheit unabhängig von K und L (300 rationale Paare in Brüchen), 1/K₁ wächst mit R und liegt nie unter 1,618033 (2 000 R), L_d = 0 gibt den Eigenwert 1, Sollwert und Störung wirken addierend, Messbias verschiebt um −b, Rampe skaliert linear, g-Gebiete, gleiche Zufalls-Seeds gleiche Ausgabe.
- **Claims (`tests/test_claims.py`):** jede Zahl der Befundtabelle und der Tabelle M9 als Band oder exakt, die vorgerechneten Reihen gegen die Neu-Rechnung, je Preset die erwartete Verdikt-Zeile, die Zahlen der App-Texte; **Verdikte (`tests/test_verdicts.py`):** jede Schwelle (Faktor 1,5, Gleichstand an der Schwelle) und jede Zahl im Text an handgesetzten Werten; **Randfälle (`tests/test_edge_cases.py`):** ρ bis 0 und 0,95, R = 10⁻⁸ und 10⁴, d = 0, Messbias, Gleichstand |d| = u_max, Rang 2 von 3.
- **Presets und Oberfläche:** Presets vollständig, kein Preset ohne Effekt; AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, alle Minima und alle Maxima zusammen, jede Auswahl am ersten und letzten Eintrag, Permalink mit Begrenzung und mit ungültigen Werten), Layout (Schnellstart in der Seitenleiste, aktives Preset, jeder Regler in `SETTING_SPECS`), Footer, Laufzeit-Importe.
- **Nicht im Test:** die Formeln im Expander und die Texte der Erklärung sind Behauptungen mit eigenem Prüfbedarf und stehen in der Aussagenprüfung der Einbindung; Messungen der Vormessung, die der Bau nicht wiederholt, stehen nicht in den Zahlen oben.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Störsprung mit und ohne Störungsmodell, Handbeispiel in Brüchen, Kernfragen, fünf Experimente (Dauerfehler und Sollwert, Beobachterpole, Modellfehler, zwei Wege, Grenzen) |
| `off_dynamics.py` | Kern: LQR, Störungsmodell und Beobachter (Stapel von Regelkreisen), Integralzustand, Frequenzgang, Rauschen, g-Gebiete, Reserve, Rampe, Stellgrenze, Rang |
| `off_exakt.py` | Bruchrechnung (Gleichungssysteme, Determinante, Rang, charakteristisches Polynom, Ackermann) |
| `off_evaluation.py` | Auswertungen der Experimente, vorgerechnete Reihen, Handbeispiel, Verdikte (reine Funktionen) |
| `off_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `off_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `off_layout.py` | Fester Kopf (Stylesheet) |
| `off_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `off_rng.py` | SplitMix64 (Kopie der geprüften Vorlage `rng_splitmix64_exact.py` des Website-Repos) |
| `data/off_messreihen.json` | Vorgerechnete Reihen (2 000 R, 300 Gesetze, 6 000 g-Fälle, 15 000 Zufallsstrecken) |
| `tools/off_make_messreihen.py` | Erzeugt `data/off_messreihen.json` (wenige Sekunden) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, plotly |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Kern und Handbeispiele, Orakel mit Hilfen, Eigenschaften, Claims, Verdikte, Randfälle, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **MPC mit Box-QP** (Stück 5 der Linie baut den QP-Löser): die Stellgrenze wird nur abgeschnitten; für d > u_max ist das Ergebnis dasselbe (siehe oben). Der Preis eines MPC mit Stellgrenze und Störungsmodell steht in [`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo).
- **Integrator mit Stellgrenze und Rückrechnung:** [`windup-pi-demo`](https://github.com/sebastian-hanisch/windup-pi-demo); hier hat keiner der beiden Wege eine Stellgrenze im Kreis des Vergleichs.
- **Filter, NEES, Einstellung von Q und R, LQG nach Doyle:** [`kalman-filter-demo`](https://github.com/sebastian-hanisch/kalman-filter-demo), [`lqr-reserven-demo`](https://github.com/sebastian-hanisch/lqr-reserven-demo); hier eine Kalman-Zeile und die Reserve als Zahl.
- **Prozessrauschen, mehrere Störungen oder Messungen, Verzug:** nicht gemessen (siehe Ehrliche Grenzen); der Verzug gehört zu [`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo).
- **PDF-Export:** nicht eingebaut.
- **Literaturabgleich** (Pannocchia/Rawlings, Maeder/Borrelli/Morari, Muske/Badgwell): erst nach dem Lesen der Primärarbeiten; bis dahin keine Neuheitsaussage.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
