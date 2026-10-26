"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Die Erwartungswerte stehen im Test als Rechnung.

Handrechnungen:
* x = (0, 1, 2, 5, 12): Mittel 20/5 = 4, Median 2. Trimmen 20 % (n_cut = ⌊0,2·5⌋ = 1): (1 + 2 + 5)/3 = 8/3. Winsorisieren 20 %: (1, 1, 2, 5, 5) → 14/5 = 2,8.
  Walsh-Mittel (15 Stück, sortiert): 0; 0,5; 1; 1; 1,5; 2; 2,5; 3; 3,5; 5; 6; 6,5; 7; 8,5; 12 — der 8. Wert, der Median, ist 3.
* Huber c = 1, Skala 1, x = (0, 1, 10): bei μ = 1 sind die Reste (−1, 0, 9), ψ = (−1, 0, 1), Summe 0, also μ = 1.
* Catoni, α = 1, x = (0, 1): ψ ist ungerade, also Nullstelle bei 1/2. ψ(1) = log(1 + 1 + 1/2) = log 2,5 = 0,91629.
* x = (0, 0, 0, 1, 7): Median 0, Abweichungen (0, 0, 0, 1, 7), MAD = 0: Skala 0, Huber fällt auf den Median 0 zurück.
* n = 10, α = 0,05: ⌊0,5⌋ = 0 Werte je Seite, das getrimmte Mittel ist das Mittel. n = 20: ⌊1⌋ = 1.
* Kappen: x = (0, 2, 10), C = 4, n = 3: gekappt (0, 2, 4), Mittel 2, Empfindlichkeit 4/3, bei ε_DP = 1 Skala 4/3. Nachbar (0, 2, 0): Mittel 2/3, Abstand 4/3 = b · ε_DP (scharf).
* Winsorisieren: (1, 2, 3, 4, 100), k = 1 → (2, 2, 3, 4, 4), Mittel 3, gleich dem Kappen auf [2, 4].
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand_x():
    return np.array([0.0, 1.0, 2.0, 5.0, 12.0])


@pytest.fixture
def zero_scale_x():
    return np.array([0.0, 0.0, 0.0, 1.0, 7.0])


LOG_2_5 = math.log(2.5)
