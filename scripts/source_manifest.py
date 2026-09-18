#!/usr/bin/env python3
"""Generate source inventory; optional check detects accidental snapshot changes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    files = {}
    for directory in ("src", "training", "workspaces/scripts", "workspaces/config"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or any(p in {"__pycache__", ".pytest_cache"} for p in path.parts):
                continue
            files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"snapshot_date": "2026-09-18", "training_upstream": "https://github.com/pairlab-usyd/Stereo-patch",
            "training_commit": "26290bcd247f1fb373a517a7057f66fd40ef1198", "sha256": files}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    target = ROOT / "SOURCE_MANIFEST.json"
    current = manifest()
    if args.check:
        if json.loads(target.read_text()) != current:
            raise SystemExit("Source manifest mismatch")
        print("Source hashes match")
    else:
        target.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {len(current['sha256'])} source hashes")
