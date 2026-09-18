"""Console entry points installed by the package."""

from __future__ import annotations


def train_main() -> int:
    from scripts.train_policy import main

    return main()


def infer_main() -> int:
    from scripts.infer_checkpoint import main

    return main()


def download_main() -> int:
    from scripts.download_checkpoint import main

    return main()
