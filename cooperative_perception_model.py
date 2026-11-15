"""Local-first cooperative detection-fusion prototype.

Input CSV columns:
    source,timestamp_s,x_m,y_m,class_name,confidence

Coordinates must already be expressed in the same reference frame. This
prototype fuses detector outputs; it does not process camera or LiDAR data,
train a detector, or produce vehicle-control commands.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


REQUIRED_COLUMNS = {
    "source",
    "timestamp_s",
    "x_m",
    "y_m",
    "class_name",
    "confidence",
}


@dataclass(frozen=True)
class Detection:
    source: str
    timestamp_s: float
    x_m: float
    y_m: float
    class_name: str
    confidence: float


@dataclass(frozen=True)
class FusedDetection:
    timestamp_s: float
    x_m: float
    y_m: float
    class_name: str
    confidence: float
    sources: str
    local_present: bool


@dataclass(frozen=True)
class FusionSummary:
    input_count: int
    accepted_remote_count: int
    dropped_remote_count: int
    output_count: int


def _finite_float(value: str, column: str, row_number: int) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {column} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: {column} must be finite")
    return result


def read_detections(path: Path) -> list[Detection]:
    """Load and validate detector messages from a CSV file."""
    detections: list[Detection] = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("input CSV is empty or has no header")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "input CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            source = (row["source"] or "").strip()
            class_name = (row["class_name"] or "").strip()
            if not source or not class_name:
                raise ValueError(
                    f"row {row_number}: source and class_name cannot be empty"
                )

            confidence = _finite_float(row["confidence"] or "", "confidence", row_number)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"row {row_number}: confidence must be between 0 and 1"
                )
            detections.append(
                Detection(
                    source=source,
                    timestamp_s=_finite_float(
                        row["timestamp_s"] or "", "timestamp_s", row_number
                    ),
                    x_m=_finite_float(row["x_m"] or "", "x_m", row_number),
                    y_m=_finite_float(row["y_m"] or "", "y_m", row_number),
                    class_name=class_name,
                    confidence=confidence,
                )
            )

    if not detections:
        raise ValueError("input CSV contains no detections")
    return detections


def fuse_detections(
    detections: list[Detection],
    *,
    now_s: float,
    ego_source: str = "ego",
    max_age_s: float = 0.5,
    match_distance_m: float = 2.0,
) -> tuple[list[FusedDetection], FusionSummary]:
    """Fuse nearby same-class detections, retaining all local detections.

    Remote detections are used only when their timestamps are not in the
    future and are no older than ``max_age_s``. Position estimates use
    confidence-weighted averaging; the fused confidence is the maximum input
    confidence, avoiding an independence assumption about detector outputs.
    """
    if not math.isfinite(now_s):
        raise ValueError("now_s must be finite")
    if not math.isfinite(max_age_s) or max_age_s < 0:
        raise ValueError("max_age_s must be a finite non-negative number")
    if not math.isfinite(match_distance_m) or match_distance_m < 0:
        raise ValueError("match_distance_m must be a finite non-negative number")
    if not ego_source:
        raise ValueError("ego_source cannot be empty")

    local = [d for d in detections if d.source == ego_source]
    remote = [d for d in detections if d.source != ego_source]
    fresh_remote = [
        d for d in remote if 0.0 <= now_s - d.timestamp_s <= max_age_s
    ]
    dropped_remote_count = len(remote) - len(fresh_remote)

    clusters: list[list[Detection]] = [[d] for d in local]
    for detection in sorted(
        fresh_remote, key=lambda item: (item.confidence, item.timestamp_s), reverse=True
    ):
        candidates: list[tuple[float, int]] = []
        for index, cluster in enumerate(clusters):
            if detection.class_name != cluster[0].class_name:
                continue
            if any(member.source == detection.source for member in cluster):
                continue
            center_x, center_y = _weighted_center(cluster)
            distance = math.hypot(detection.x_m - center_x, detection.y_m - center_y)
            if distance <= match_distance_m:
                candidates.append((distance, index))

        if candidates:
            _, best_index = min(candidates)
            clusters[best_index].append(detection)
        else:
            clusters.append([detection])

    fused = [
        _fuse_cluster(cluster, ego_source=ego_source)
        for cluster in clusters
    ]
    fused.sort(key=lambda item: (item.class_name, item.x_m, item.y_m))
    return fused, FusionSummary(
        input_count=len(detections),
        accepted_remote_count=len(fresh_remote),
        dropped_remote_count=dropped_remote_count,
        output_count=len(fused),
    )


def _weighted_center(cluster: list[Detection]) -> tuple[float, float]:
    total_weight = sum(d.confidence for d in cluster)
    if total_weight == 0.0:
        return (
            sum(d.x_m for d in cluster) / len(cluster),
            sum(d.y_m for d in cluster) / len(cluster),
        )
    return (
        sum(d.x_m * d.confidence for d in cluster) / total_weight,
        sum(d.y_m * d.confidence for d in cluster) / total_weight,
    )


def _fuse_cluster(cluster: list[Detection], *, ego_source: str) -> FusedDetection:
    x_m, y_m = _weighted_center(cluster)
    return FusedDetection(
        timestamp_s=max(d.timestamp_s for d in cluster),
        x_m=x_m,
        y_m=y_m,
        class_name=cluster[0].class_name,
        confidence=max(d.confidence for d in cluster),
        sources=";".join(sorted({d.source for d in cluster})),
        local_present=any(d.source == ego_source for d in cluster),
    )


def write_detections(detections: list[FusedDetection], file: TextIO) -> None:
    """Write fused detections as CSV."""
    fields = [
        "timestamp_s",
        "x_m",
        "y_m",
        "class_name",
        "confidence",
        "sources",
        "local_present",
    ]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    for detection in detections:
        writer.writerow(
            {
                "timestamp_s": f"{detection.timestamp_s:.6f}",
                "x_m": f"{detection.x_m:.6f}",
                "y_m": f"{detection.y_m:.6f}",
                "class_name": detection.class_name,
                "confidence": f"{detection.confidence:.6f}",
                "sources": detection.sources,
                "local_present": str(detection.local_present).lower(),
            }
        )


def _demo() -> tuple[list[Detection], float]:
    """Return a small example with a fusion, a remote-only object, and stale data."""
    return (
        [
            Detection("ego", 10.0, 10.0, 0.0, "car", 0.90),
            Detection("vehicle_2", 9.9, 10.4, 0.2, "car", 0.80),
            Detection("vehicle_2", 9.9, 18.0, 1.0, "pedestrian", 0.75),
            Detection("vehicle_3", 8.0, 24.0, 0.0, "truck", 0.95),
        ],
        10.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuse fresh cooperative detections with local ego detections."
    )
    parser.add_argument("--input", type=Path, help="input detection CSV")
    parser.add_argument("--output", type=Path, help="output fused-detection CSV")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run a built-in synthetic example instead of reading a CSV",
    )
    parser.add_argument(
        "--now-s",
        type=float,
        help="replay time in seconds (defaults to the latest input timestamp)",
    )
    parser.add_argument("--ego-source", default="ego", help="local vehicle source label")
    parser.add_argument("--max-age-s", type=float, default=0.5)
    parser.add_argument("--match-distance-m", type=float, default=2.0)
    args = parser.parse_args()

    if args.demo == (args.input is not None):
        parser.error("provide exactly one of --demo or --input")

    try:
        if args.demo:
            detections, default_now_s = _demo()
        else:
            detections = read_detections(args.input)
            default_now_s = max(d.timestamp_s for d in detections)

        fused, summary = fuse_detections(
            detections,
            now_s=default_now_s if args.now_s is None else args.now_s,
            ego_source=args.ego_source,
            max_age_s=args.max_age_s,
            match_distance_m=args.match_distance_m,
        )

        if args.output is None:
            write_detections(fused, sys.stdout)
        else:
            with args.output.open("w", newline="", encoding="utf-8") as file:
                write_detections(fused, file)

        print(
            "detections={input_count} accepted_remote={accepted_remote_count} "
            "dropped_stale_or_future_remote={dropped_remote_count} "
            "fused_outputs={output_count}".format(**summary.__dict__),
            file=sys.stderr,
        )
    except (OSError, csv.Error, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
