# Getrimmt, Hodges–Lehmann, Huber: wie viel Median braucht man? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach). Der Ordnername enthält „hub“ (huber): Streamlit Cloud lehnt solche App-Namen ab, also einen Alias wählen:
**[→ Demo live ausprobieren](https://sebastianhanisch-<alias>.streamlit.app/)** -->

Stück 4 der Reihe **Mittelwerte und Lagemaße** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, mit einem Abschnitt **Kappen und privates Mittel**. Die Kante im Abhängigkeitsgraphen: Stück 2 (Median einer Zahlenreihe) → 4 als Fix (der Median verliert auf sauberen Daten gut ein Drittel der Genauigkeit). Die Demo setzt Mittel und Median als bekannt voraus und wiederholt sie nur in einem Satz.

Vehikel: **die Lieferzeit-Abweichung einer Sendung** (Stunden gegenüber Plan, der Kern ist auf Streuung 1 skaliert). Zwischen den normalen Sendungen stecken **Lieferausfälle** (Ausreißer): genau m = round(ε · n) Sendungen werden durch d + N(0, 1) ersetzt, einseitig (alle verspätet, der ungünstigste Fall für die Lage) oder zweiseitig. Die Demo zeigt (a) den Preis der Robustheit auf sauberen Daten gegen die Verschiebung bei Ausfällen, (b) die Verschiebung über dem Ausfallanteil als Kurve (Klippe oder Anstieg), (c) das Verhalten bei wachsendem Abstand, (d) worauf ein Schätzer bei schiefen Daten zielt, (e) die Fallen (Treppe, Skala 0, Gleitkomma-Kante), (f) Catoni nach der Definition, (g) im Abschnitt **Kappen und privates Mittel**: die U-Kurve Kappen gegen Laplace-Rauschen, die Skalierung der besten Kappungsgrenze, ein Gegenbeispiel (C aus den Daten gewählt ist nicht ε_DP-privat) und die Verbindung zum Winsorisieren.

**Buchstaben.** ε ist der Anteil der Ausfälle, d ihr Abstand, α der Trimm- bzw. Winsor-Anteil je Seite, c die Huber-Schranke, λ das Gewicht des Catoni-Schätzers (ein Vielfaches des Wertes aus der Definition). Im Abschnitt Kappen heißt das Datenschutz-Budget **ε_DP** und die Kappungsgrenze **C**.

Abgrenzung (Stichwortsuche in 455 lokalen und den per `gh api` gelesenen READMEs, Stand 2026-10-24): Der Lageschätzer in einer Dimension ist in keiner anderen Demo der Gegenstand. `permutation-test-demo` winsorisiert nur als Vorbehandlung eines Tests, `forecast-combination-demo` nutzt das getrimmte Mittel als eine Kombinationsregel, `kmeans-demo` vergleicht Mittelwert und Medoid, `robust-filter-demo` und `imm-demo` gewichten Messvarianzen nach Huber im Kalman-Filter, `gradient-boosting-demo` nutzt den Huber-Verlust, `elliptic-envelope-demo` schätzt robust in mehreren Dimensionen (MCD). Zum Datenschutz-Abschnitt gibt es in keiner Demo einen Treffer zu differentieller Privatheit. Weitere Stücke der Reihe, die hier nur beim Namen vorkommen: Mittelwert und Median, Median-of-Means, Lp-Mitte, Weiszfeld.

## Warum dieses Problem

„Wie viel Median braucht man?“ ist keine Ja-Nein-Frage, sondern eine Kurve. Der Hook sind gemessene Befunde statt eines Lehrbuchbilds: (1) die drei Hauptkandidaten **Huber, Hodges–Lehmann und Trimmen 10 %** liegen auf fast derselben Effizienz-gegen-Verzerrung-Kurve; (2) sie unterscheiden sich dort, wo man es nicht erwartet: **Trimmen und Winsorisieren haben eine Klippe genau bei ε > α, Hodges–Lehmann eine zwischen 29 und 30 %**, danach ist er schlechter als das Mittel, während Huber und Median stetig steigen; (3) bei **schiefen Daten** zielt jeder robuste Schätzer irgendwo zwischen Median und Mittelwert, und dieser Abstand **schrumpft nicht mit n**; (4) **Kappen auf eine feste Grenze** tauscht Verzerrung gegen Rauschen, und das Minimum der U-Kurve ist im Beispiel fast reiner Stichprobenfehler.

Kernfrage: *Wie viel Median braucht man, wie viel Mittel darf bleiben, und worauf zielt das Ergebnis dann?*

## Befunde und Korrekturen gegenüber dem Plan

Die Hypothesen des Einzelplans (`mittelwerte-planung/STUECK-4-trimmen-huber.md`, Abschnitt Kappen aus `STUECK-9-privates-mittel.md`) wurden im Bau mit dem **eigenen SplitMix64-Strom** neu gemessen; die Vormessung des Plans (numpy-Zufall, Wegwerfskripte) liefert keine README-Zahl. Alle Zahlen: vier feste Zufalls-Seeds je Zelle, n = 100 und Normal als Kern, wo nichts anderes steht.

| | Hypothese | Ergebnis | Zahl (diese Demo) |
|---|---|---|---|
| H1 | Hodges–Lehmann und Huber (c = 1,345) liegen bei Effizienz 0,95 und Verschiebung 0,19 bis 0,21; die drei Familien sind austauschbar | **Ja** | Effizienz / Verschiebung bei ε = 10 %, d = 5: Huber 0,952 / 0,196, Hodges–Lehmann 0,959 / 0,210, Trimmen 10 % 0,948 / 0,210; Mittel 1 / 0,501, Median 0,659 / 0,139 |
| H2 | Trimmen unter dem Ausfallanteil hilft kaum; Winsorisieren ist dominiert | **Teilweise** | Trimmen 5 % bei 10 % Ausfall: Verschiebung 0,347 (Mittel 0,501). Winsorisieren 10 % (0,969 / 0,294) wird von Huber c = 2 (0,991 / 0,260) dominiert, aber nur 5 von 8 Winsor-Anteilen haben einen dominierenden Huber-Wert im Raster der Messung (nicht bei 20, 40 und 45 %) |
| H3 | Die Verschiebung hat zwei Kurventypen: Klippe (Trimmen bei ε = α, Hodges–Lehmann bei 25 bis 30 %) und stetiger Anstieg (Huber, Median) | **Ja** | d = 1000: Trimmen 10 % 0,21 bei 10 % und 62,64 bei 15 %; Trimmen 25 % 0,54 bei 25 % und 20,49 bei 26 %; Hodges–Lehmann 1,58 bei 29 % und 498,56 bei 30 %; Huber (c = 1,345) 4,42 und Median 1,32 bei 45 % |
| H4 | Jenseits der Klippe ist Hodges–Lehmann schlechter als das Mittel | **Ja** | ε = 35 %, d = 1000: 499,32 gegen 349,99 beim Mittel |
| H5 | Beschränkte Schätzer sind vom Abstand unabhängig, Catoni wächst logarithmisch | **Ja** | ε = 10 %: Huber 0,185 bis 0,191 von d = 3 bis 10⁶, Median 0,137; Catoni (λ × 1) 0,288 bei d = 3, 6,006 bei d = 1000, 20,158 bei d = 10⁶; Mittel 0,296 bis 99999,996 |
| H6 | Kein Schätzer gewinnt auf allen Rändern; Hodges–Lehmann und Huber verlieren auf Normal höchstens 4 % | **Ja** | RMSE-Verhältnis zum Mittel bei n = 10 / 100: Hodges–Lehmann 1,039 / 1,025, Huber 1,040 / 1,028, Median 1,186 / 1,247; Laplace: Median 0,753, Hodges–Lehmann 0,820; t(5): Median 1,019 (schlechter als das Mittel) |
| H7 | Bei Schiefe schrumpft der Bias nicht mit n; das Mittel gewinnt ab n ≈ 30 bis 100, wenn das Ziel der Mittelwert ist | **Ja, mit einer Ausnahme (Catoni)** | Lognormal (σ = 1), Bias von Trimmen 10 % gegen den Mittelwert −0,327 / −0,379 / −0,402 / −0,410 bei n = 10 / 30 / 100 / 300; das Mittel hat ab n = 30 den kleineren RMSE als Median, Trimmen 10 und 25 %, Huber (c = 0,5 und 1,345) und Hodges–Lehmann, gegen Winsorisieren 10 % erst ab n = 100. Catoni (λ × 1) liegt bei n = 10 / 30 / 100 unter dem Mittel (0,567 / 0,338 / 0,200 gegen 0,687 / 0,393 / 0,216) |
| H8 | Catoni nach Definition verhält sich bei Kontamination wie das Mittel, Huber dominiert ihn | **Teilweise** | Bei kleinem Abstand ja (Verschiebung 0,470 gegen 0,501 bei ε = 10 %, d = 5), bei d = 1000 nicht: 6,013 gegen 100,001 beim Mittel und 0,196 bei Huber. Mit λ × 32: Effizienz 0,937, Verschiebung 0,245 / 0,687; Huber (0,952 / 0,196 / 0,196) ist besser |
| H9 | Fallen: Treppe bei kleinem n, Skala 0 bei diskreten Daten, Gleitkomma-Kante | **Ja** | n = 10: α = 5 / 10 / 15 / 20 % schneiden 0 / 1 / 1 / 2 Werte je Seite ab; `int(α·n)` weicht in 13 von 44820 Paaren ab (α = 0,29, n = 100: 28 statt 29); Poisson(0,5): MAD = 0 in 84,9 % (n = 30) und 97,2 % (n = 100) der Stichproben |
| H10 | Lage der Hodges–Lehmann-Klippe (theoretisch 0,293?) und Verhalten bei ε = α exakt | **Messung ja, Theorie nicht geprüft** | Raster 1 Prozentpunkt: Sprung zwischen 29 % (1,58) und 30 % (498,56) bei n = 100 und 300; Trimmen 10 % hält bei ε = 10 % (0,21) und bricht bei 11 % (12,69). Der Wert 0,293 wurde nicht geprüft, 29 bis 30 % ist damit verträglich |
| H11 | Catoni-Garantie (Fehlerquantil bei Niveau δ) | **Offen** | gehört zu Median-of-Means (Stück 6 der Reihe), hier nicht gemessen |
| K1 | U-Kurve: lognormal(0; 1), n = 1000, ε_DP = 1: Minimum bei C ≈ 14 bis 25, Fehler überwiegend Stichprobenfehler | **Ja** | Minimum bei C* = 14,89 (Formel; Raster der Simulation 14,46), Anteil des Rauschens an der MSE dort 9,5 %; RMSE der Formel dort 0,0681, RMSE des nicht privaten Stichprobenmittels (Simulation) 0,0680 |
| K2 | MSE = Verzerrung² + Varianz/n + 2b² trägt die Simulation | **Ja** | RMSE²/MSE-Formel über das Raster: Median 1,000 (kleinster 0,970, größter 1,028) |
| K3 | C* wächst mit n; Pareto folgt (n·ε_DP)^(1/α), Lognormal nicht | **Ja** | Steigung d ln C*/d ln n: Pareto (α = 1,5) 0,66 bis 0,67, Pareto (α = 3) 0,32 bis 0,33 (Herleitung 0,667 und 0,333); Lognormal (σ = 1) fällt von 0,38 auf 0,22 |
| K4 | Ein Faktor 4 zu kleines C kostet mehr als ein Faktor 4 zu großes | **Ja (in 4 von 6 Zellen), nicht bei Pareto (α = 1,5)** | Lognormal (σ = 1), n = 1000, ε_DP = 1: RMSE-Verhältnis 3,99 (C*/4), 1,59 (C* · 4), 5,04 (C* · 16); bei Pareto (α = 1,5) ist C* · 4 teurer als C*/4 (1,87 gegen 1,50) |
| K5 | C aus den Daten gewählt, ohne Budget dafür, ist nicht ε_DP-privat | **Ja** | größtes log-Verhältnis bei ε_DP = 1 an der Stelle y = 3 / 10 / 100 / 1000: 6,39 / 29,87 / 329,87 / 3329,87 (wächst linear, keine Schranke) |
| K6 | Die Empfindlichkeit (hi − lo)/n ist scharf, und bei Schwellen aus den Daten ist die Änderung größer | **Ja** | Aufzählung: größtes log-Verhältnis = ε_DP (0,5000 bei 0,5; 1,0000 bei 1); Fehler mit n + 1: 0,6667 / 1,3333 (n = 3); Schwellen aus den Daten (n = 100, 1 Wert je Seite): im Mittel das 2,3-fache von (hi − lo)/n, größter Datensatz 5,7-fach |

**Korrekturen gegenüber dem Plan.**

- *Catoni (PLAN E1 „verhält sich wie das Mittel“):* mit korrekter Wurzelsuche und dem Wert aus der Definition ergibt sich ein **teilweise** anderes Bild als im Plan: bei kleinem Abstand wie das Mittel, bei großem Abstand wächst die Verschiebung nur logarithmisch (6,013 bei d = 1000 statt 100,001). Der Befund „Catoni ist kein robuster Lageschätzer unter Kontamination“ bleibt, „wie das Mittel“ gilt nur für kleine d. Die Definition steht **aus dem Gedächtnis**; die Originalarbeit wurde nicht eingesehen.
- *Catoni bei Schiefe:* Der Plan maß nur, dass er auf den Mittelwert zielt. Hier zusätzlich: bei lognormal (σ = 2) ist sein RMSE gegen den Mittelwert bei n = 100 kleiner als der des Mittels (3,583 gegen 4,905), bei n = 300 nicht (3,074 gegen 2,865).
- *Winsorisieren dominiert (Plan):* gilt nur für 5 von 8 Winsor-Anteilen (siehe H2).
- *Verschiebungsschwelle:* durch die Kurve ersetzt; die Klippe von Trimmen 10 % liegt bei 11 % im 1-Prozent-Raster, nicht beim Rasterpunkt 15 %.
- *Neu gegenüber dem Plan:* Zweiseitige Ausfälle (Winsorisieren 10 % ist bei ε = 20 %, d = 10 mit RMSE 0,867 schlechter als das Mittel mit 0,460), Skala bekannt gegen MAD, Skala 0, die Zerlegung des Rauschens im Abschnitt Kappen.

## Modell

Eine Stichprobe von n Lieferzeit-Abweichungen (Kern: Normal, Laplace, t(5), t(3), Lognormal mit σ = 0,5 / 1 / 2, Exponential oder diskrete Tage mit Poisson 0,5), darunter m = round(ε · n) Ausfälle d + N(0, 1). Zielgröße der Lage: der Median des Kerns (bei symmetrischen Kernen gleich dem Mittelwert, 0); bei schiefen Kernen zeigt Experiment E4 beide. **Verschiebung** = Mittelwert des Schätzers über die Läufe (Ziel 0), **Effizienz** = MSE des Mittels geteilt durch MSE des Schätzers an denselben sauberen Normaldaten (gepaart). Formeln: Expander 📐 Mathematische Formulierung in der App.

Abschnitt Kappen: n öffentlich, Nachbarn = ein ersetzter Wert, ε_DP-Privatheit als Schranke für das log-Verhältnis der Dichten, Kappung auf eine öffentliche Grenze [0, C], Laplace-Rauschen der Skala C/(n · ε_DP). Die Demo ist hier Lehrmaterial über Definition, Fehlerzerlegung und ein Gegenbeispiel, **kein Sicherheitsnachweis**.

## Methodik

- **Schätzer** (`thu_robust.py`): Mittel, Median, getrimmtes und winsorisiertes Mittel (k = ⌊α·n + 10⁻⁹⌋ je Seite), Huber (gewichtete Iteration ab dem Median, Skala MAD/0,6745 oder bekannt 1), Hodges–Lehmann (Median aller Paarmittel, exakt), Catoni (Bisektion der Nullstelle, α aus Stichprobenumfang und MAD-Varianz).
- **Zufall:** SplitMix64 (Kopie von `reduktionskette-demo`), Normalwerte nach Box-Muller; alle Anteile und Abstände einer Zelle teilen sich dieselben Zufallszahlen (glatte Kurven, gepaarte Vergleiche).
- **Messreihen:** `tools/make_results.py` erzeugt `data/thu_results.json` mit der Prüfsumme der Quellen; die App zeigt die vorgerechneten Zellen und rechnet auf Knopfdruck mit den Einstellungen der Seitenleiste neu (Verschiebungskurve, U-Kurve, Aufzählung).
- **Kappen:** geschlossene Zerlegung des Fehlers (gekappte Momente von Lognormal und Pareto), Simulation mit Laplace-Rauschen per Inversion, Aufzählung aller benachbarten Datensätze auf kleinen Gittern.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und werden dort aus `data/thu_results.json` gebildet.

| Frage | Befund |
|---|---|
| Hängt die Effizienz an n? | Kaum. Hodges–Lehmann 0,933 / 0,948 / 0,959 / 0,956 und Huber (c = 1,345) 0,933 / 0,945 / 0,952 / 0,952 bei n = 10 / 30 / 100 / 300; der Median fällt von 0,721 auf 0,636 |
| Wie breit ist der Streifen der drei Hauptkandidaten? | Verschiebung bei ε = 10 %, d = 5: 0,196 (Huber) bis 0,210 (Hodges–Lehmann und Trimmen 10 %), Unterschied 0,015 |
| Wann springt die Verschiebung? | Klippe bei Trimmen 10 % zwischen ε = 10 % (0,21) und 11 % (12,69), bei Trimmen 25 % zwischen 25 % (0,54) und 26 % (20,49), bei Winsorisieren 25 % zwischen 25 % (0,76) und 26 % (259,65), bei Hodges–Lehmann zwischen 29 % (1,58) und 30 % (498,56); bei n = 300 ebenso (1,66 und 498,58) |
| Wächst die Verschiebung mit dem Abstand? | Bei ε = 20 %: Huber 0,450 bei d = 3 und 0,496 ab d = 5, Median 0,314 bis 0,317, Hodges–Lehmann 0,460 bis 0,540; Mittel 0,596 bis 199999,996; Trimmen 10 % wächst bis 125000,103 (Anteil über α) |
| Und bei zweiseitigen Ausfällen? | d = 10, ε = 10 %: RMSE um 0 beim Mittel 0,330 (nach der Formel √((1 + ε·d²)/n) 0,332), Median 0,138, Huber 0,123, Trimmen 10 % 0,124; bei ε = 20 % Winsorisieren 10 % 0,867 gegen 0,460 beim Mittel |
| Schwere Ränder | t(3): RMSE-Verhältnis zum Mittel bei n = 100: Hodges–Lehmann 0,736, Huber 0,736, Trimmen 25 % 0,721, Median 0,792; Laplace bei n = 300: Median 0,737 |
| Worauf zielt der Schätzer bei Lognormal (σ = 1)? | Mittelwert 1,649, Median 1,000; an der Population Trimmen 10 % 1,236, Hodges–Lehmann 1,223, Huber (c = 1,345) 1,199, Catoni (λ × 1) 1,650 |
| Bei Lognormal (σ = 2) | Mittelwert 7,389; Trimmen 10 % 2,175, Hodges–Lehmann 1,835, Huber (c = 1,345) 1,502; das Mittel gewinnt gegen Median, Trimmen 10 %, Hodges–Lehmann und Huber (c = 1,345) ab n = 100, gegen Winsorisieren 10 % ab n = 300; seine eigene Streuung ist dort groß (SD 4,905 bei n = 100, Median 0,261) |
| Bei Exponential | das Mittel hat bei n = 10 bis 300 einen kleineren RMSE gegen den Mittelwert als alle hier gemessenen Robusten außer Catoni; bei n = 10: 0,316 gegen 0,401 beim Median und 0,317 bei Winsorisieren 10 % |
| Skala 0 | Poisson(0,5), n = 100: MAD = 0 in 97,2 % der Stichproben; Lage Median 0,023, Huber 0,018, Hodges–Lehmann 0,490, Mittel 0,500. 60 % Nullen + Poisson(5): 84,5 % (n = 30) und 97,8 % (n = 100) |
| Skala bekannt gegen MAD | Huber (c = 1,345) mit Skala 1: Effizienz 0,954 gegen 0,952; Verschiebung bei ε = 20 %, d = 5: 0,416 gegen 0,501; bei ε = 30 %, d = 1000: 0,734 gegen 1,096 |
| Iterationen | Größte Schrittzahl der Huber-Iteration im Stapel von 2000 Stichproben (Mittel über Seeds): 38,0 bei c = 0,5, 16,8 bei c = 1,345, 8,0 bei c = 3 |
| Catoni | λ × 1 / 8 / 32: Effizienz 1,000 / 0,985 / 0,937, Verschiebung (ε = 10 %, d = 5) 0,470 / 0,322 / 0,245 und (ε = 10 %, d = 1000) 6,013 / 1,348 / 0,687; wächst mit n: bei d = 1000 3,54 / 6,01 / 8,94 für n = 20 / 100 / 300; t(3): RMSE-Verhältnis 0,910 (λ × 1), 0,783 (λ × 8), Huber 0,736 |
| Kappen: Skalierung | C* bei n = 1000, ε_DP = 1: Lognormal (σ = 1) 14,89, Pareto (α = 1,5) 85,23, Pareto (α = 3) 6,99; Anteil des Rauschens an der MSE im Minimum 9,5 % / 16,6 % / 14,0 % |
| Kappen: Aufzählung | Empfindlichkeit durch Aufzählung (Werte 0 bis 6, Kappung [0, 3], 57624 Paare bei n = 4): größtes log-Verhältnis genau ε_DP; Fehler mit n + 1: 0,6250 / 1,2500 (n = 4, ε_DP = 0,5 / 1); ohne Kappung 1,0000 / 2,0000 |

## Ehrliche Grenzen

- **Ein Modell der Ausfälle.** Genau m = ε · n Ausfälle, unabhängig vom Kern, mit festem Abstand d (± Rauschen). Echte Ausfälle häufen sich und haben verschiedene Höhen; die Lage der Klippen gilt für dieses Modell.
- **Eine Dimension, keine Konfidenzintervalle, keine Tests.** Yuen-Test und Zwei-Stichproben-Hodges–Lehmann sind nicht gebaut; mehrdimensionale Schätzer gehören zu `elliptic-envelope-demo`.
- **Durchschlagspunkte nur gemessen.** Die Lage der Klippen (bei ε > α, bei Hodges–Lehmann zwischen 29 und 30 %) ist eine Messung an vier Seeds. Die theoretischen Werte (zum Beispiel 0,293 für Hodges–Lehmann) wurden **nicht geprüft**.
- **Catoni nach der Definition aus dem Gedächtnis.** Die Originalarbeit (Catoni 2012) wurde nicht eingesehen, ebenso nicht Hodges–Lehmann (1963) und Huber (1964): Alle Aussagen beruhen auf eigener Rechnung und Messungen gegen unabhängige Rechenwege (siehe Verifikation).
- **Catoni-Zweck nicht gemessen.** Das Fehlerquantil bei Niveau δ gehört zu Median-of-Means.
- **Vier Seeds je Zelle.** Die Streuung ist die zwischen vier festen Seeds, kein Konfidenzintervall. Bei Lognormal (σ = 2) ist der Mittelwert-Schätzer selbst stark verrauscht; die Zahlen dort nur als Größenordnung lesen.
- **Skala aus dem MAD** (bei diskreten Daten oft 0) gegen **bekannte Skala 1**: beide stehen zur Wahl, die Messung mit bekannter Skala läuft mit der wahren Streuung des Kerns (Normal).
- **Kappen und privates Mittel: kein Sicherheitsnachweis.** n ist öffentlich, die Nachbarschaft ist „ein Wert ersetzt“, das Kappungsintervall ist öffentlich; Gleitkommazahlen und Zufallszahlen sind **nicht gehärtet** (der Laplace-Sampler ist eine Inversion aus gleichverteilten Doubles); mehrere Veröffentlichungen verbrauchen Budget (nicht gerechnet); die Daten sind simuliert. Die Aufzählung prüft die Mathematik auf kleinen Gittern, nicht die Umsetzung. Arbeiten zur differentiellen Privatheit wurden **nicht eingesehen**.
- **Mit echtem Mausklick nicht geprüft:** Alle Eingaben laufen über Regler, Auswahlfelder und Knöpfe; die Diagramme sind nicht anklickbar.

## Verifikation

- **Kern gegen Handrechnung** (`tests/conftest.py`, `tests/test_robust.py`, `tests/test_private.py`): x = (0, 1, 2, 5, 12) (Mittel 4, Median 2, Trimmen 20 % 8/3, Winsorisieren 20 % 2,8, Hodges–Lehmann 3 mit den 15 sortierten Walsh-Mitteln), Huber c = 1, Skala 1 auf (0, 1, 10), Catoni α = 1 auf (0, 1) mit ψ(1) = log 2,5, MAD = 0 auf (0, 0, 0, 1, 7), Kappen von (0, 2, 10) auf [0, 4] mit Empfindlichkeit 4/3.
- **Orakel** (`tests/test_oracle_lagemasse.py`, `tests/test_oracle_privatheit.py`): Trimmen und Winsorisieren gegen `scipy.stats.trim_mean` und `mstats.winsorize` (die 13 Kanten-Paare einzeln), Hodges–Lehmann gegen die Aufzählung aller Paarmittel (n bis 9, auch mit Gleichständen), Huber gegen Bisektion und `minimize_scalar`, Catoni gegen `brentq` und eine Fixpunktiteration, gekappte Momente gegen `scipy.integrate.quad`, die Zerlegung des Fehlers gegen eine Simulation, die Empfindlichkeit und das Gegenbeispiel durch Aufzählung aller benachbarten Datensätze (mit absichtlichen Fehlern, die gefangen werden), der Laplace-Sampler gegen die Verteilungsfunktion.
- **Eigenschaftstests** (`tests/test_eigenschaften_lagemasse.py`): Verschiebungs- und Skalenäquivarianz, Spiegelung, Permutationsinvarianz, Monotonie (wo belegt), Schranken, Konstante.
- **Zweig-Tests** (`tests/test_zweige.py`, `tests/test_evaluation.py`): jedes Verfahren weicht auf mindestens einer Instanz von jedem anderen ab, Trimmen 5 % ist bei n < 20 das Mittel, jeder Regler und jede Verteilung ändert mindestens eine Kennzahl, die Klippen liegen als Bänder bei ε > α.
- **Schleifenabbruch und Rechenbudget** (`tests/test_termination.py`, `tests/test_worstcase.py`): Huber, Catoni-Bisektion und Hodges–Lehmann enden im eigenen Prozess unter einer Zeitgrenze, auch mit Skala 0 und konstanten Daten; die größte Instanz (n = 3000) läuft durch; zweimal gerechnet mit zwei Hash-Seeds gleiche Zahlen.
- **Messreihen** (`tests/test_results.py`): die Datei ist aktuell gegen die Quelldateien (Prüfsumme), vorgerechnete Zellen werden von den Live-Funktionen mit denselben Seeds reproduziert.
- **Mutationsprüfung** (`oracle_mutation.py` aus dem Website-Repo): `oracle_mutation.py` über `mutate_demo.py` aus dem Website-Repo, inkrementell, Zustand in `tests/mutation_state.json`, ohne `tests/test_results.py` (die Prüfsumme der Quellen würde jeden Mutanten fangen) und ohne Szenario-Erzeugung, Auswertung, Figuren, PDF und Presets: gezogen wurden alle 156 Stellen von `thu_robust.py`, `thu_private.py` und `thu_neighbors.py` (die Lageschätzer, das Kappen und die Aufzählung). Stand beim Abschluss: **140 gefangen, 16 als äquivalent festgehalten, 0 offen**. Im ersten Lauf (70 Stellen) überlebten 13 Mutanten die ganze Suite (Rate 81 %); geschlossen sind unter anderem: die Aufzählung des Gegenbeispiels gegen eine unabhängige Fassung mit anderem Rang und anderem Gitter, die Handrechnung der Empfindlichkeit bei Schwellen aus den Daten (auch an der letzten Position), die Stichprobenvarianz im Catoni-Schätzer (auch unter 1 und bei n = 1 und 2), die Grenze n = 2·log(1/δ), die Zahl der Huber-Iterationen und die Verfeinerung von C* samt Gitterrändern. Äquivalent festgehalten (jeweils begründet): sieben Vergleiche und Konstanten um die parabolische Verfeinerung von C* (am Gitterrand entartet die Parabel und fällt auf den Randwert zurück; die Verfeinerung nur bei den Gitterpunkten 1 und 698 auszulassen ändert kein getestetes Ergebnis), `c < 1` gegen `c <= 1` in den gekappten Momenten von Pareto (bei c = 1 stetig), `n > 1` gegen `n > 2` bei Catoni (bei zwei Werten liegt die Wurzel immer in der Mitte), 99 gegen 100 Bisektionsschritte und `f > 0` gegen `f >= 0` bei Catoni, die Schleifengrenze der Huber-Iteration um einen Schritt und `step < tol` gegen `step <= tol`.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede Zahl im Text an handgesetzten Werten. **Aussagen** (`tests/test_claims.py`): jede Zahl dieser README.
- **Oberfläche:** AppTest (jeder Knopf, jeder Regler am Minimum und Maximum, jede Option der Auswahlregler, Eckkombinationen), Layout (aktives Preset, Hervorhebung), Footer, PDF, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App |
| `thu_constants.py` | Regler-Grenzen, Raster, Farben, Presets mit Hilfetexten |
| `thu_robust.py` | Kern: die Lageschätzer |
| `thu_private.py`, `thu_neighbors.py` | Kappen, Laplace-Mechanismus, geschlossene Fehlerzerlegung; Aufzählung benachbarter Datensätze |
| `thu_scenario.py`, `thu_rng.py` | Stichproben und Ausfälle; SplitMix64 (Kopie) |
| `thu_curves.py`, `thu_kappen.py`, `thu_experiments.py` | Messreihen-Funktionen, Zusammenstellung der Teile, Prüfsumme der Quellen |
| `thu_results.py` | Laden der vorgerechneten Messreihen |
| `thu_evaluation.py` | Einstellungen, Analyse einer Stichprobe, Zusammenfassungen, Verdikte |
| `thu_visualization.py`, `thu_pdf_export.py`, `thu_presets.py`, `thu_layout.py` | Figuren, PDF, Permalink und Presets, fester Kopf |
| `tools/make_results.py`, `tools/run_parts.sh` | erzeugt `data/thu_results.json` (Teile einzeln oder alle) |
| `data/thu_results.json` | die vorgerechneten Messreihen mit Prüfsumme |
| `tests/` | Tests (siehe oben) |
| `requirements.txt`, `requirements-dev.txt`, `.github/workflows/tests.yml` | Laufzeit, Test und Gegenproben (scipy nur dort), CI |

## Bewusst nicht umgesetzt

- **Yuen-Test, Zwei-Stichproben-Hodges–Lehmann, Konfidenzintervalle:** hier steht nur die Lage.
- **Mehrdimensionale Schätzer:** `elliptic-envelope-demo` (MCD).
- **Garantien für Catoni und Median-of-Means:** Stück 6 der Reihe. **Lp-Mitte** und **Weiszfeld:** Stücke 3 und 5 der Reihe.
- **Exponential-Mechanismus, Gauß-Mechanismus, gestufte Verfahren, glatte Empfindlichkeit, Mehrfachabfragen:** im Abschnitt Kappen nicht umgesetzt; ebenso kein Sicherheitsversprechen.
- **Durchschlagspunkte und Effizienzen als Sätze aus der Literatur:** nur als Messung.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Der Code verwendet nur Sprach- und Bibliotheksmerkmale, die unter Python 3.12 (CI) laufen; die Python-Version der Streamlit Cloud ist eine App-Einstellung.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Messreihen werden mit `sh tools/run_parts.sh` (alle Teile mit höchstens vier Prozessen gleichzeitig, danach Zusammenfügen) oder einzeln mit `python tools/make_results.py --part <Name>` (Namen: `--list`) und `--merge` neu erzeugt; `tests/test_results.py` meldet, wenn die Datei hinter den Quelldateien zurückliegt.

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
