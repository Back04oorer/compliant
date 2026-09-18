#!/usr/bin/env python3
"""Static checks only: no imports of hardware modules, no ROS or GPU operations."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "__pycache__", ".pytest_cache", ".venv", "ws_robot"}


def main():
    errors = []
    counts = {"python": 0, "shell": 0, "xml": 0, "files": 0}
    # Construct in parts so the scanner does not match its own test patterns.
    patterns = [re.compile(x) for x in (
        r"hf_" + r"[A-Za-z0-9]{25,}", r"gh[pousr]_" + r"[A-Za-z0-9]{25,}",
        r"github_pat_" + r"[A-Za-z0-9_]{30,}", r"-----BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    )]
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if any(p in SKIP or p.endswith(".egg-info") for p in rel.parts) or not path.is_file():
            continue
        counts["files"] += 1
        if path.is_symlink():
            errors.append(f"Unexpected source symlink: {rel}")
            continue
        if path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f"File too large: {rel}")
        if path.suffix.lower() in {".bag", ".mp4", ".safetensors", ".pth", ".pt", ".token", ".pem", ".key"}:
            errors.append(f"Forbidden artifact: {rel}")
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(), filename=str(rel))
                counts["python"] += 1
            elif path.suffix == ".sh":
                result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                if result.returncode:
                    errors.append(f"Shell syntax: {rel}: {result.stderr.strip()}")
                counts["shell"] += 1
            elif path.suffix in {".xml", ".launch", ".urdf", ".xacro"}:
                ET.parse(path)
                counts["xml"] += 1
            if path.stat().st_size < 3 * 1024 * 1024:
                try:
                    data = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                if any(p.search(data) for p in patterns):
                    errors.append(f"Potential credential (value not printed): {rel}")
        except (SyntaxError, ET.ParseError, UnicodeDecodeError) as exc:
            errors.append(f"Parse error: {rel}: {exc}")
    print(json.dumps({"checked": counts, "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
