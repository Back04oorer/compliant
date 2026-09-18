#!/usr/bin/env python3
"""Download a private/public Stereo-Patch checkpoint from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repository")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.output_dir.expanduser(),
        token=True,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
