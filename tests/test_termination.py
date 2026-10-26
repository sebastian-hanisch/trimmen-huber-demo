"""Schleifen mit Abbruchbedingung (Huber-Iteration, Catoni-Bisektion, Hodges–Lehmann in Stücken) enden im eigenen Prozess innerhalb einer Zeitgrenze, auch auf Randfällen.

Ein Mutant, der eine solche Schleife endlos laufen lässt, wird sonst erst vom Zeitlimit des Mutationswerkzeugs gefangen (und nicht in tests/mutation_state.json vermerkt).
Zweiter Teil: dasselbe Szenario mit zwei verschiedenen PYTHONHASHSEED rechnet dieselben Zahlen (kein Zufall aus der Hash-Reihenfolge)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = r"""
import numpy as np
import thu_robust as R
from thu_scenario import Stream, Contamination

x = Contamination("Normal", 200, 100, Stream(3)).at(30, 1e9)
for c in (1e-6, 0.25, 1.345, 5.0, 1e6):
    mu, its = R.huber_location(x, c, return_iterations=True)
    assert np.isfinite(mu).all() and its <= R.HUBER_MAXIT
assert np.isfinite(R.huber_location(np.full((3, 7), 2.0))).all()                 # alle gleich: Skala 0
assert np.isfinite(R.huber_location(np.array([[0.0, 0.0, 0.0, 1.0, 7.0]]))).all()
for mult in (1.0, 8.0, 32.0):
    assert np.isfinite(R.catoni_location(x, 0.05, mult)).all()
assert np.isfinite(R.catoni_location(np.full((2, 20), 5.0))).all()              # Varianz 0
big = Stream(5).normal(1200).reshape(1, 1200)
assert np.isfinite(R.hodges_lehmann(big)).all()
assert np.isfinite(R.huber_location(big)).all() and np.isfinite(R.catoni_location(big)).all()
print("fertig")
"""

REPRO = r"""
import thu_curves as K
cell = K.shift_cell(("mean", "median", "trim:0.1", "huber:1.345", "hl", "catoni:1"), 30, (0.0, 0.1), (5.0,), 20, (1, 2))
print(repr(cell["rows"]))
"""


def test_all_iterations_terminate_within_the_time_limit():
    done = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=120, encoding="utf-8")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("fertig")


def test_results_do_not_depend_on_the_hash_seed():
    outputs = []
    for hash_seed in ("1", "2"):
        done = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, '.'); " + REPRO.strip().replace("\n", "; ")],
                              cwd=ROOT, capture_output=True, text=True, timeout=120, encoding="utf-8", env={**os.environ, "PYTHONHASHSEED": hash_seed})
        assert done.returncode == 0, done.stderr[-2000:]
        outputs.append(done.stdout)
    assert outputs[0] == outputs[1] and outputs[0].startswith("[{")
