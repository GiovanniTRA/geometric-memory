"""
Runs run_edit.py once per edit in a fresh subprocess (ROME's own context-
template sampling is unseeded and module-global -- a fresh process per edit
keeps every edit independent; see README.md).

Usage:
  python run_all.py                 # all 100 edits
  python run_all.py --limit 5       # smoke test
"""
import argparse
import os
import subprocess
import sys

from config import EDIT_TABLE

_HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RUNNER = os.path.join(_HERE, "run_edit.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(EDIT_TABLE))
    args = ap.parse_args()

    n = min(args.limit, len(EDIT_TABLE))
    failed = []
    for edit_idx in range(n):
        print(f"\n[{edit_idx + 1}/{n}] edit_idx={edit_idx}")
        result = subprocess.run([PY, RUNNER, str(edit_idx)], cwd=_HERE)
        if result.returncode != 0:
            print(f"  !! edit_idx={edit_idx} FAILED (returncode={result.returncode})")
            failed.append(edit_idx)

    print(f"\nAll done. ({n} attempted, {len(failed)} failed)")
    if failed:
        print("Failed:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
