#!/usr/bin/env python3
"""Import checks, including FFmpeg/TorchCodec. No datasets, weights or robot I/O."""
import argparse
import importlib
import importlib.metadata
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Also run one tiny CUDA operation; not a model smoke test")
    args = parser.parse_args()
    versions = {}
    for package, module in [("torch", "torch"), ("torchvision", "torchvision"), ("torchcodec", "torchcodec.decoders"),
                            ("lerobot", "lerobot"), ("numpy", "numpy"), ("opencv-python-headless", "cv2"),
                            ("transformers", "transformers"), ("rosbags", "rosbags"), ("pyarrow", "pyarrow")]:
        importlib.import_module(module)
        versions[package] = importlib.metadata.version(package)
    if args.cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check the host NVIDIA driver and chosen wheel")
        value = torch.ones(1, device="cuda").sum().item()
        assert value == 1
        versions["gpu"] = torch.cuda.get_device_name(0)
        versions["cuda_runtime"] = torch.version.cuda
    print(json.dumps(versions, indent=2))
    print("PASS: imports" + (" and CUDA probe" if args.cuda else " (GPU not tested)"))


if __name__ == "__main__":
    main()
