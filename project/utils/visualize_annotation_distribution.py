#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Visualize driver-action annotation distributions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_ROOT = Path("/home/data/xchen/drive/multi_view_driver_action")
DEFAULT_OUTPUT_DIR = Path("outputs/annotation_visualizations")

LABEL_8_TO_4 = {
    "left": "left",
    "left_up": "left",
    "left_down": "left",
    "right": "right",
    "right_up": "right",
    "right_down": "right",
    "up": "up",
    "down": "down",
}

ENV_ORDER = ["day_high", "day_low", "night_high", "night_low"]
COARSE_LABEL_ORDER = ["left", "right", "down", "up"]


def parse_label_filename(path: Path) -> tuple[str, str]:
    """Return (person_id, env_key) from person_01_day_high_h265.json."""
    match = re.match(
        r"person_(?P<person>\d+)_(?P<daynight>day|night)_(?P<level>high|low)_h265\.json$",
        path.name,
    )
    if not match:
        raise ValueError(f"Unexpected label filename: {path.name}")
    return match.group("person"), f"{match.group('daynight')}_{match.group('level')}"


def parse_video_reference(video: str) -> tuple[str, str] | None:
    """Return (person_id, env_key) from a Label Studio video reference."""
    filename = Path(video.split("?d=")[-1]).name
    match = re.match(
        r"person_(?P<person>\d+)_(?P<daynight>day|night)_(?P<level>high|low)_h265\.mp4$",
        filename,
    )
    if not match:
        return None
    return match.group("person"), f"{match.group('daynight')}_{match.group('level')}"


def normalize_label(label: str) -> str:
    return LABEL_8_TO_4.get(str(label), str(label))


def collect_segments(label_dir: Path) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for label_file in sorted(label_dir.glob("person_*_*.json")):
        person_id, env_key = parse_label_filename(label_file)
        with label_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        for annotation_index, annotation in enumerate(payload.get("annotations", [])):
            annotator = annotation.get("annotator", "unknown")
            for label_obj in annotation.get("videoLabels", []):
                labels = label_obj.get("timelinelabels", [])
                ranges = label_obj.get("ranges", [])
                for label in labels:
                    for range_obj in ranges:
                        start = range_obj.get("start")
                        end = range_obj.get("end")
                        if start is None or end is None:
                            continue
                        segments.append(
                            {
                                "file": label_file.name,
                                "person_id": person_id,
                                "env_key": env_key,
                                "annotator": annotator,
                                "annotation_index": annotation_index,
                                "raw_label": str(label),
                                "coarse_label": normalize_label(str(label)),
                                "start": float(start),
                                "end": float(end),
                                "duration": float(end) - float(start),
                            }
                        )
    return segments


def collect_split_markers(split_mid_end_file: Path) -> list[dict[str, Any]]:
    if not split_mid_end_file.exists():
        return []

    with split_mid_end_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    markers: list[dict[str, Any]] = []
    for item in payload:
        parsed = parse_video_reference(item.get("video", ""))
        if parsed is None:
            continue
        person_id, env_key = parsed
        frames: dict[str, float] = {}
        for label_obj in item.get("videoLabels", []):
            labels = label_obj.get("timelinelabels", [])
            ranges = label_obj.get("ranges", [])
            if not labels or not ranges:
                continue
            label = labels[0]
            if label in {"start", "mid", "end"}:
                frames[label] = float(ranges[0].get("start"))
        if {"start", "mid", "end"}.issubset(frames):
            markers.append(
                {
                    "person_id": person_id,
                    "env_key": env_key,
                    "start": frames["start"],
                    "mid": frames["mid"],
                    "end": frames["end"],
                    "start_to_mid": frames["mid"] - frames["start"],
                    "mid_to_end": frames["end"] - frames["mid"],
                    "start_to_end": frames["end"] - frames["start"],
                }
            )
    return markers


def _basic_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def build_summary(label_dir: Path | str, split_mid_end_file: Path | str) -> dict[str, Any]:
    label_dir = Path(label_dir)
    split_mid_end_file = Path(split_mid_end_file)
    segments = collect_segments(label_dir)
    markers = collect_split_markers(split_mid_end_file)

    samples = {(item["person_id"], item["env_key"]) for item in segments}
    annotations = {
        (item["file"], item["annotation_index"], item["annotator"]) for item in segments
    }

    return {
        "sample_count": len(samples),
        "annotation_count": len(annotations),
        "segment_count": len(segments),
        "raw_label_counts": dict(sorted(Counter(s["raw_label"] for s in segments).items())),
        "coarse_label_counts": dict(
            sorted(Counter(s["coarse_label"] for s in segments).items())
        ),
        "annotator_counts": dict(sorted(Counter(s["annotator"] for s in segments).items())),
        "split_marker_count": len(markers),
        "duration_stats": _basic_stats([s["duration"] for s in segments]),
        "split_duration_stats": _basic_stats([m["start_to_end"] for m in markers]),
    }


def _save_bar(counts: dict[Any, int], title: str, xlabel: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [str(k) for k in counts.keys()]
    values = list(counts.values())
    sns.barplot(x=names, y=values, ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    for i, value in enumerate(values):
        ax.text(i, value, str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_visualizations(
    label_dir: Path,
    split_mid_end_file: Path,
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    segments = collect_segments(label_dir)
    markers = collect_split_markers(split_mid_end_file)
    summary = build_summary(label_dir, split_mid_end_file)

    seg_df = pd.DataFrame(segments)
    marker_df = pd.DataFrame(markers)
    saved: list[Path] = []

    raw_path = output_dir / "raw_label_distribution.png"
    _save_bar(
        summary["raw_label_counts"],
        "Raw 8-Class Label Distribution",
        "Raw label",
        "Segment count",
        raw_path,
    )
    saved.append(raw_path)

    coarse_counts = {
        label: summary["coarse_label_counts"].get(label, 0)
        for label in COARSE_LABEL_ORDER
    }
    coarse_path = output_dir / "coarse_4class_distribution.png"
    _save_bar(
        coarse_counts,
        "Merged 4-Class Label Distribution",
        "Merged label",
        "Segment count",
        coarse_path,
    )
    saved.append(coarse_path)

    annotator_label_path = output_dir / "annotator_label_distribution.png"
    annotator_counts = (
        seg_df.pivot_table(
            index="annotator",
            columns="coarse_label",
            values="duration",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=COARSE_LABEL_ORDER, fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    annotator_counts.plot(kind="bar", stacked=False, ax=ax, width=0.8)
    ax.set_title("Merged Label Counts by Annotator")
    ax.set_xlabel("Annotator")
    ax.set_ylabel("Segment count")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(annotator_label_path, dpi=180)
    plt.close(fig)
    saved.append(annotator_label_path)

    heatmap_path = output_dir / "segments_per_person_env_heatmap.png"
    person_order = sorted(seg_df["person_id"].unique())
    heatmap_df = (
        seg_df.pivot_table(
            index="person_id",
            columns="env_key",
            values="duration",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=person_order, columns=ENV_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 9))
    sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title("Segment Count per Person and Environment")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Person")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)
    saved.append(heatmap_path)

    duration_path = output_dir / "segment_duration_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(seg_df["duration"], bins=60, ax=ax, color="#59A14F")
    ax.set_title("Action Segment Duration Distribution")
    ax.set_xlabel("Duration (frames)")
    ax.set_ylabel("Segment count")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(duration_path, dpi=180)
    plt.close(fig)
    saved.append(duration_path)

    if not marker_df.empty:
        marker_path = output_dir / "split_marker_distribution.png"
        marker_long = marker_df.melt(
            id_vars=["person_id", "env_key"],
            value_vars=["start", "mid", "end"],
            var_name="marker",
            value_name="frame",
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(data=marker_long, x="marker", y="frame", ax=axes[0], color="#F28E2B")
        axes[0].set_title("Start/Mid/End Frame Distribution")
        axes[0].set_xlabel("Marker")
        axes[0].set_ylabel("Frame")
        sns.boxplot(
            data=marker_df.melt(
                id_vars=["person_id", "env_key"],
                value_vars=["start_to_mid", "mid_to_end", "start_to_end"],
                var_name="interval",
                value_name="frames",
            ),
            x="interval",
            y="frames",
            ax=axes[1],
            color="#E15759",
        )
        axes[1].set_title("Split Interval Lengths")
        axes[1].set_xlabel("Interval")
        axes[1].set_ylabel("Frames")
        axes[1].tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(marker_path, dpi=180)
        plt.close(fig)
        saved.append(marker_path)

    summary_path = output_dir / "summary.md"
    summary_path.write_text(_render_summary(summary, output_dir, saved), encoding="utf-8")
    saved.append(summary_path)

    return saved


def _fmt_stats(stats: dict[str, float | None]) -> str:
    if stats["min"] is None:
        return "n/a"
    return (
        f"min={stats['min']:.0f}, "
        f"median={stats['median']:.0f}, "
        f"max={stats['max']:.0f}"
    )


def _render_summary(summary: dict[str, Any], output_dir: Path, images: list[Path]) -> str:
    image_lines = "\n".join(f"- `{path.name}`" for path in images)
    raw_lines = "\n".join(
        f"- `{label}`: {count}" for label, count in summary["raw_label_counts"].items()
    )
    coarse_lines = "\n".join(
        f"- `{label}`: {count}" for label, count in summary["coarse_label_counts"].items()
    )
    annotator_lines = "\n".join(
        f"- `{annotator}`: {count}"
        for annotator, count in summary["annotator_counts"].items()
    )
    return f"""# Annotation Distribution Summary

Output directory: `{output_dir}`

## Dataset

- Samples: {summary["sample_count"]}
- Annotation instances: {summary["annotation_count"]}
- Action segments: {summary["segment_count"]}
- Start/mid/end marker samples: {summary["split_marker_count"]}
- Action duration stats: {_fmt_stats(summary["duration_stats"])} frames
- Start-to-end stats: {_fmt_stats(summary["split_duration_stats"])} frames

## Raw Label Counts

{raw_lines}

## Merged 4-Class Counts

{coarse_lines}

## Annotator Segment Counts

{annotator_lines}

## Generated Files

{image_lines}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create annotation distribution visualizations."
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_ROOT / "label",
        help="Directory containing person_*_*.json label files.",
    )
    parser.add_argument(
        "--split-mid-end",
        type=Path,
        default=DEFAULT_ROOT / "split_mid_end" / "mini.json",
        help="Path to split_mid_end/mini.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figures and summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = create_visualizations(args.label_dir, args.split_mid_end, args.output_dir)
    print(f"Saved {len(saved)} files to {args.output_dir}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
