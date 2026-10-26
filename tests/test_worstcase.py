"""Worst-Case-Test am Slider-Maximum (DEMO-PLAYBOOK §1c): keine Zeit-Assertion, nur ein Timeout im eigenen Prozess als Netz gegen Hänger und Aufblähen.

Komplexität am Maximum n = 3000: Mittel, Median, Trimmen, Winsorisieren O(n log n); Huber O(n · Iterationen); Catoni O(n · 100); Hodges–Lehmann O(n²) mit 4,5 Mio. Paarmitteln."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = r"""
import numpy as np
import thu_constants as C
import thu_evaluation as V

for dist in ("Normal", "Lognormal (σ = 2)", "Diskrete Tage (Poisson 0,5)"):
    a = V.analyse(V.Settings(dist=dist, n=C.N_MAX, eps=C.EPS_MAX, d=1000000.0, alpha=C.ALPHA_MAX, delta=0.01, mult=32.0))
    assert a.x.shape == (C.N_MAX,) and np.isfinite(a.values).all(), dist
print("fertig")
"""


def test_largest_instance_finishes():
    done = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=300, encoding="utf-8")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("fertig")
