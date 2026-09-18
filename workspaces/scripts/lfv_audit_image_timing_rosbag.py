#!/usr/bin/env python3
"""Audit image topic timing and compressed sizes in a ROS bag."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import rosbag


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return float(ordered[max(0, min(len(ordered) - 1, idx))])


def summarize_times(times: list[float]) -> dict[str, Any]:
    if not times:
        return {
            "count": 0,
            "duration_sec": 0.0,
            "hz": None,
            "dt_sec": {},
        }
    duration = max(0.0, times[-1] - times[0])
    hz = (len(times) - 1) / duration if duration > 0 and len(times) > 1 else None
    dts = [b - a for a, b in zip(times, times[1:]) if b >= a]
    return {
        "count": len(times),
        "first_sec": times[0],
        "last_sec": times[-1],
        "duration_sec": duration,
        "hz": hz,
        "dt_sec": {
            "min": min(dts) if dts else None,
            "max": max(dts) if dts else None,
            "mean": statistics.fmean(dts) if dts else None,
            "p50": percentile(dts, 0.50),
            "p95": percentile(dts, 0.95),
            "p99": percentile(dts, 0.99),
        },
    }


def summarize_sizes(sizes: list[int]) -> dict[str, Any]:
    if not sizes:
        return {}
    return {
        "min_bytes": min(sizes),
        "max_bytes": max(sizes),
        "mean_bytes": statistics.fmean(sizes),
        "p50_bytes": percentile([float(x) for x in sizes], 0.50),
        "p95_bytes": percentile([float(x) for x in sizes], 0.95),
    }


def audit_bag(bag_path: Path, topics: list[str]) -> dict[str, Any]:
    by_topic: dict[str, dict[str, list[float] | list[int]]] = {
        topic: {"bag_times": [], "header_times": [], "sizes": []} for topic in topics
    }
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, bag_time in bag.read_messages(topics=topics):
            entry = by_topic.setdefault(topic, {"bag_times": [], "header_times": [], "sizes": []})
            entry["bag_times"].append(float(bag_time.to_sec()))
            header = getattr(msg, "header", None)
            stamp = getattr(header, "stamp", None)
            if stamp is not None:
                stamp_sec = float(stamp.to_sec())
                if stamp_sec > 0:
                    entry["header_times"].append(stamp_sec)
            data = getattr(msg, "data", None)
            if data is not None:
                entry["sizes"].append(len(data))

    summary: dict[str, Any] = {"bag": str(bag_path), "topics": {}}
    for topic, entry in by_topic.items():
        bag_times = list(entry["bag_times"])
        header_times = list(entry["header_times"])
        sizes = list(entry["sizes"])
        summary["topics"][topic] = {
            "bag_time": summarize_times(bag_times),
            "header_time": summarize_times(header_times),
            "compressed_size": summarize_sizes([int(x) for x in sizes]),
            "uses_header_time": bool(header_times),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("--topic", action="append", default=[])
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    topics = args.topic or [
        "/top_stereo/color/image_raw/compressed",
        "/top_stereo/left/image_raw/compressed",
        "/top_stereo/right/image_raw/compressed",
    ]
    bag_path = Path(args.bag)
    summary = audit_bag(bag_path, topics)

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
