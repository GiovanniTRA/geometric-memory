"""
Aggregates every results/gis_edit_*.json + results/gis_acc_edit_*.json into
the paper's headline numbers: edit success rate, mean nGIS_theta, mean
delta_GIS_acc -- overall and broken down by domain.

Usage: python stats.py
"""
import glob
import json
import os

import numpy as np

from config import EDIT_TABLE

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_results():
    rows = []
    for path in sorted(glob.glob(os.path.join(_HERE, "results", "gis_edit_*.json"))):
        gis = json.load(open(path))
        acc_path = os.path.join(_HERE, "results", f"gis_acc_edit_{gis['edit_idx']}.json")
        gis_acc = json.load(open(acc_path)) if os.path.exists(acc_path) else {}
        rows.append({**gis, "delta_gis_acc": gis_acc.get("delta_gis_acc")})
    return rows


def summarize(rows, label):
    if not rows:
        print(f"{label:20s}  (no edits)")
        return
    n = len(rows)
    success = np.mean([r["edit_success"] for r in rows])
    nGIS = np.mean([r["nGIS"] for r in rows])
    deltas = [r["delta_gis_acc"] for r in rows if r["delta_gis_acc"] is not None]
    delta_str = f"{np.mean(deltas):+.4f}" if deltas else "n/a"
    print(f"{label:20s}  n={n:3d}  edit_success={success*100:5.1f}%  "
          f"nGIS_theta={nGIS:+.4f}  delta_GIS_acc={delta_str}")


def main():
    rows = load_results()
    if not rows:
        print("No results yet. Run extract.py, then run_edit.py / run_all.py.")
        return

    print(f"Loaded {len(rows)}/{len(EDIT_TABLE)} edits.\n")
    summarize(rows, "ALL")
    print()
    domains = sorted(set(r["domain"] for r in rows))
    for d in domains:
        summarize([r for r in rows if r["domain"] == d], d)


if __name__ == "__main__":
    main()
