"""Erzeugt data/thu_results.json (die vorgerechneten Messreihen aller Experimente) aus thu_experiments.py.

    python tools/make_results.py --part m2_n100     # eine Messreihe nach data/_parts/m2_n100.json (läuft allein, mehrere Teile können parallel laufen)
    python tools/make_results.py --all-parts        # alle Teile nacheinander (dauert; für Parallelläufe die Teile einzeln starten)
    python tools/make_results.py --merge            # alle Teile zu data/thu_results.json zusammenfügen, mit Prüfsumme der Quelldateien
    python tools/make_results.py --list             # Namen der Teile

Teile: m1_n* (Effizienz und Verzerrung je n), m2_n* (Verschiebung über dem Ausfallanteil), m2fine (feines Raster), m2b (Abstand), m2c (zweiseitig), m3 (Ränder),
m4_* (Zielgröße je schiefer Verteilung), m5 (Catoni), m6 (Fallen), m7_* (U-Kurve Kappen), m8 (Skalierung und falsches C), m9 (Aufzählung und Winsorisieren).
Die Prüfsumme (`erzeuger_hash`) deckt thu_rng, thu_scenario, thu_robust, thu_private, thu_neighbors, thu_curves, thu_kappen, thu_experiments und dieses Skript ab;
tests/test_results.py schlägt fehl, wenn sich eine dieser Dateien nach dem Erzeugen geändert hat."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import thu_constants as C  # noqa: E402
import thu_experiments as X  # noqa: E402

PARTS_DIR = ROOT / "data" / "_parts"
STAND = "2026-10-26"


def run_part(name: str) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = X.PARTS[name]()
    out = PARTS_DIR / f"{name}.json"
    out.write_text(json.dumps({"hash": X.source_hash(), "data": result}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"{name}: fertig in {time.perf_counter() - t0:.0f} s -> {out}", flush=True)


def merge() -> None:
    parts, hashes = {}, set()
    for name in X.PARTS:
        raw = json.loads((PARTS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        parts[name] = raw["data"]
        hashes.add(raw["hash"])
    if hashes != {X.source_hash()}:
        sys.exit(f"Teile sind mit anderem Quellstand erzeugt ({sorted(hashes)} gegen {X.source_hash()}); Teile neu erzeugen")
    meta = {"stand": STAND, "erzeuger_hash": X.source_hash(), "seeds": list(C.SEEDS), "quellen": list(X.SOURCES)}
    (ROOT / "data" / C.RESULTS_FILE).write_text(json.dumps({"meta": meta, **parts}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print("data/thu_results.json geschrieben", X.source_hash())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(X.PARTS))
    ap.add_argument("--all-parts", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print(" ".join(X.PARTS))
    elif args.part:
        run_part(args.part)
    elif args.all_parts:
        for part in X.PARTS:
            run_part(part)
    elif args.merge:
        merge()
    else:
        ap.error("--part, --all-parts, --merge oder --list angeben")
