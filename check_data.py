"""Check custom BEVFusion clip annotation completeness.

The expected custom clip layout is:

    data/clip_xxx/annotations/sample_annotation/*.json

By default this script checks that each clip has exactly 40 annotation JSON
files and that those JSON files contain the 3D annotation keys required by
custom_dataset_converter.py.
"""

import argparse
import json
import os
from os import path as osp


REQUIRED_TOP_LEVEL_KEYS = (
    "timestamp",
    "ego2global_transformation_matrix",
    "sensors",
    "objects",
)

STATUS_OK = "ok"
STATUS_MISSING_ANNOTATIONS = "missing_annotations"
STATUS_MISSING_SAMPLE_ANNOTATION = "missing_sample_annotation"
STATUS_BAD_JSON_COUNT = "bad_json_count"
STATUS_INVALID_SCHEMA = "invalid_schema"

STATUS_LABELS = {
    STATUS_OK: "complete",
    STATUS_MISSING_ANNOTATIONS: "missing annotations directory",
    STATUS_MISSING_SAMPLE_ANNOTATION: "missing sample_annotation directory",
    STATUS_BAD_JSON_COUNT: "wrong json count",
    STATUS_INVALID_SCHEMA: "invalid annotation schema",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="data",
        help="Directory containing clip folders. Default: data",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=40,
        help="Expected number of sample_annotation JSON files per clip.",
    )
    parser.add_argument(
        "--no-schema",
        action="store_true",
        help="Only check JSON file count; do not parse JSON or validate 3D keys.",
    )
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="Print clips that pass all checks as well as failed clips.",
    )
    return parser.parse_args()


def iter_clip_dirs(root):
    for name in sorted(os.listdir(root)):
        clip_dir = osp.join(root, name)
        if osp.isdir(clip_dir):
            yield name, clip_dir


def validate_annotation_schema(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
    except json.JSONDecodeError as exc:
        return f"invalid json: {exc}"

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in ann]
    if missing:
        return "missing keys: " + ", ".join(missing)

    sensors = ann.get("sensors")
    if not isinstance(sensors, dict):
        return "sensors is not a dict"
    if "lidar" not in sensors:
        return "missing sensors.lidar"
    if "cams" not in sensors:
        return "missing sensors.cams"
    return None


def check_clip(clip_dir, expected_count, check_schema):
    annotations_dir = osp.join(clip_dir, "annotations")
    if not osp.isdir(annotations_dir):
        return STATUS_MISSING_ANNOTATIONS, [f"missing directory: {annotations_dir}"], 0

    ann_dir = osp.join(annotations_dir, "sample_annotation")
    if not osp.isdir(ann_dir):
        return STATUS_MISSING_SAMPLE_ANNOTATION, [f"missing directory: {ann_dir}"], 0

    json_files = sorted(name for name in os.listdir(ann_dir) if name.endswith(".json"))
    if len(json_files) != expected_count:
        return (
            STATUS_BAD_JSON_COUNT,
            [f"json count {len(json_files)} != {expected_count}"],
            len(json_files),
        )

    if check_schema:
        for name in json_files:
            json_path = osp.join(ann_dir, name)
            error = validate_annotation_schema(json_path)
            if error:
                return STATUS_INVALID_SCHEMA, [f"{name}: {error}"], len(json_files)

    return STATUS_OK, [], len(json_files)


def main():
    args = parse_args()
    root = osp.abspath(args.root)
    check_schema = not args.no_schema

    if not osp.isdir(root):
        raise FileNotFoundError(root)

    total = 0
    counts = {status: 0 for status in STATUS_LABELS}
    bad_clips = []

    for clip_name, clip_dir in iter_clip_dirs(root):
        total += 1
        status, errors, json_count = check_clip(
            clip_dir,
            expected_count=args.expected_count,
            check_schema=check_schema,
        )
        counts[status] += 1
        if status == STATUS_OK:
            if args.show_ok:
                print(f"[OK] {clip_name}: {json_count} json files")
        else:
            bad_clips.append((clip_name, status, json_count, errors))
            print(f"[BAD:{status}] {clip_name}: {json_count} json files")
            for error in errors:
                print(f"  - {error}")

    bad_count = total - counts[STATUS_OK]
    print("=" * 72)
    print(f"Root: {root}")
    print(f"Total clips: {total}")
    print(f"Complete clips: {counts[STATUS_OK]}")
    print(f"Bad clips: {bad_count}")
    print(f"Missing annotations directory clips: {counts[STATUS_MISSING_ANNOTATIONS]}")
    print(
        "Missing sample_annotation directory clips: "
        f"{counts[STATUS_MISSING_SAMPLE_ANNOTATION]}"
    )
    print(f"Wrong json count clips: {counts[STATUS_BAD_JSON_COUNT]}")
    print(f"Invalid annotation schema clips: {counts[STATUS_INVALID_SCHEMA]}")
    print(
        "Clips with annotations but changed/invalid json format: "
        f"{counts[STATUS_INVALID_SCHEMA]}"
    )
    print(f"Expected annotation json count per clip: {args.expected_count}")
    print(f"Schema check: {'on' if check_schema else 'off'}")

    if bad_clips:
        raise SystemExit(1)


if __name__ == "__main__":
    main()