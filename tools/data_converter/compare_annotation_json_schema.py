#!/usr/bin/env python3
"""Compare new sample_annotation JSON files with the demo annotation schema.

The script uses only Python's standard library so it can be copied to and run
on a cloud server without installing the BEVFusion environment.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


CAMERAS = (
    "center_camera_fov120",
    "left_front_camera",
    "right_front_camera",
    "rear_camera",
    "left_rear_camera",
    "right_rear_camera",
)


def converter_requirements():
    required = {
        "timestamp": "number",
        "ego2global_transformation_matrix": "array",
        "sensors": "object",
        "sensors.lidar": "object",
        "sensors.lidar.perception": "object",
        "sensors.lidar.perception.extrinsic": "array",
        "sensors.cams": "object",
        "objects": "array",
    }
    for camera in CAMERAS:
        base = f"sensors.cams.{camera}"
        required[base] = "object"
        required[f"{base}.timestamp"] = "number"
        required[f"{base}.extrinsic"] = "array"
        required[f"{base}.cam_intrinsic"] = "array"
    return required


OBJECT_REQUIREMENTS = {
    "objects[].type": "string",
    "objects[].size": "array",
    "objects[].location": "array",
}


def type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def collect_paths(value, path, found):
    """Collect recursive JSON paths, normalizing every list index to []."""
    found[path].add(type_name(value))
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            collect_paths(child, child_path, found)
    elif isinstance(value, list):
        item_path = f"{path}[]"
        if not value:
            found[item_path].add("<empty>")
        for child in value:
            collect_paths(child, item_path, found)


def find_json_files(source, limit):
    path = Path(source)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.json"))
    else:
        raise FileNotFoundError(source)
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No JSON files found under: {source}")
    return files


def inspect(source, limit=0):
    files = find_json_files(source, limit)
    schema = defaultdict(set)
    path_file_count = Counter()
    parse_errors = []
    top_level_shapes = Counter()
    class_counts = Counter()
    converter_issues = []

    requirements = converter_requirements()
    all_requirements = dict(requirements)
    all_requirements.update(OBJECT_REQUIREMENTS)

    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8-sig") as stream:
                data = json.load(stream)
        except Exception as exc:
            parse_errors.append({"file": str(file_path), "error": repr(exc)})
            continue

        one_file = defaultdict(set)
        collect_paths(data, "", one_file)
        top_level_shapes[tuple(sorted(data)) if isinstance(data, dict) else (type_name(data),)] += 1
        for json_path, types in one_file.items():
            schema[json_path].update(types)
            path_file_count[json_path] += 1

        objects = data.get("objects", []) if isinstance(data, dict) else []
        if isinstance(objects, list):
            for obj in objects:
                if isinstance(obj, dict):
                    class_counts[str(obj.get("type", "<missing>"))] += 1

        issues = []
        for json_path, expected in all_requirements.items():
            # Object fields are only mandatory when at least one object exists.
            if json_path.startswith("objects[]") and not objects:
                continue
            actual = one_file.get(json_path)
            if not actual:
                issues.append(f"missing {json_path}")
            elif expected not in actual:
                issues.append(
                    f"type {json_path}: expected {expected}, got {sorted(actual)}"
                )
        if issues:
            converter_issues.append({"file": str(file_path), "issues": issues})

    valid_count = len(files) - len(parse_errors)
    paths = {}
    for json_path in sorted(schema):
        paths[json_path or "<root>"] = {
            "types": sorted(schema[json_path]),
            "files": path_file_count[json_path],
            "presence_percent": round(100.0 * path_file_count[json_path] / valid_count, 2)
            if valid_count
            else 0.0,
        }
    return {
        "source": str(Path(source).resolve()),
        "files_selected": len(files),
        "files_valid": valid_count,
        "parse_errors": parse_errors,
        "top_level_shapes": [
            {"keys": list(keys), "files": count}
            for keys, count in top_level_shapes.most_common()
        ],
        "paths": paths,
        "object_class_counts": dict(class_counts.most_common()),
        "converter_compatibility": {
            "files_with_issues": len(converter_issues),
            "issues_by_file": converter_issues,
        },
    }


def compare(old, new):
    old_paths, new_paths = old["paths"], new["paths"]
    old_set, new_set = set(old_paths), set(new_paths)
    changed = []
    for path in sorted(old_set & new_set):
        if old_paths[path]["types"] != new_paths[path]["types"]:
            changed.append(
                {
                    "path": path,
                    "old_types": old_paths[path]["types"],
                    "new_types": new_paths[path]["types"],
                }
            )
    return {
        "removed_from_new": sorted(old_set - new_set),
        "added_in_new": sorted(new_set - old_set),
        "type_changes": changed,
    }


def render_text(result):
    old, new, diff = result["old"], result["new"], result["difference"]
    lines = [
        "sample_annotation JSON schema comparison",
        "=" * 48,
        f"OLD: {old['source']}",
        f"  selected={old['files_selected']}, valid={old['files_valid']}, "
        f"parse_errors={len(old['parse_errors'])}",
        f"NEW: {new['source']}",
        f"  selected={new['files_selected']}, valid={new['files_valid']}, "
        f"parse_errors={len(new['parse_errors'])}",
        "",
        f"Removed fields ({len(diff['removed_from_new'])})",
        "-" * 48,
        *[f"- {p}" for p in diff["removed_from_new"]],
        "",
        f"Added fields ({len(diff['added_in_new'])})",
        "-" * 48,
        *[f"+ {p}" for p in diff["added_in_new"]],
        "",
        f"Type changes ({len(diff['type_changes'])})",
        "-" * 48,
        *[
            f"~ {x['path']}: {x['old_types']} -> {x['new_types']}"
            for x in diff["type_changes"]
        ],
        "",
        "Converter compatibility of NEW files",
        "-" * 48,
        f"files_with_issues={new['converter_compatibility']['files_with_issues']}",
    ]
    for item in new["converter_compatibility"]["issues_by_file"]:
        lines.append(f"FILE: {item['file']}")
        lines.extend(f"  ! {issue}" for issue in item["issues"])
    lines.extend(["", "All NEW recursive paths (type; presence)", "-" * 48])
    for path, meta in new["paths"].items():
        lines.append(
            f"{path}: {','.join(meta['types'])}; "
            f"{meta['files']}/{new['files_valid']} ({meta['presence_percent']}%)"
        )
    if new["parse_errors"]:
        lines.extend(["", "NEW parse errors", "-" * 48])
        for error in new["parse_errors"]:
            lines.append(f"{error['file']}: {error['error']}")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="Demo JSON file or directory")
    parser.add_argument("--new", required=True, help="New JSON file or directory")
    parser.add_argument("--out", default="annotation_schema_report", help="Output prefix")
    parser.add_argument(
        "--limit", type=int, default=0, help="Max files per side; 0 means all files"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        old = inspect(args.old, args.limit)
        new = inspect(args.new, args.limit)
    except (FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    result = {"old": old, "new": new, "difference": compare(old, new)}
    output_prefix = Path(args.out)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    text_path = output_prefix.with_suffix(".txt")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    with text_path.open("w", encoding="utf-8") as stream:
        stream.write(render_text(result))
    print(f"Wrote machine-readable report: {json_path.resolve()}")
    print(f"Wrote readable report:        {text_path.resolve()}")
    print(
        f"Differences: removed={len(result['difference']['removed_from_new'])}, "
        f"added={len(result['difference']['added_in_new'])}, "
        f"type_changes={len(result['difference']['type_changes'])}"
    )
    print(
        "New files incompatible with converter: "
        f"{new['converter_compatibility']['files_with_issues']}"
    )
    return 1 if new["parse_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
