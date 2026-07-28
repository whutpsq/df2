"""Audit custom clip data before BEVFusion conversion or training.

The audit is read-only.  It checks sample/asset completeness, annotation values,
class mapping, six-camera calibration matrices, image/calibration dimensions,
and the consistency between projected 3D boxes and annotation-provided 2D boxes.
It writes a machine-readable JSON report outside the source data tree.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from custom_dataset_converter_V2 import CAMERA_MAP, CLASS_MAP, resolve_annotation_path
from test_custom_gt_projection import (
    ANNOTATION_CAMERA_MAP,
    box_corners_lidar,
    lidar_to_camera,
    project_camera_points,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-path",
        default="data2",
        help="One clip root or a directory containing clip roots.",
    )
    parser.add_argument(
        "--out-dir",
        default="data_write/custom_validation",
        help="Report directory; must be separate from the source data tree.",
    )
    parser.add_argument(
        "--xy-limit",
        type=float,
        default=51.2,
        help="Training-range half-width used for center statistics.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=200,
        help="Maximum detailed structural issues retained in the JSON report.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def discover_clips(root):
    root = root.resolve()
    if (root / "annotations" / "sample.json").is_file():
        return [root]
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "annotations" / "sample.json").is_file()
    )


def percentile(values, probability):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), probability))


def distribution(values):
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def rotation_metrics(matrix):
    rotation = matrix[:3, :3]
    error = rotation.T @ rotation - np.eye(3)
    return {
        "determinant": float(np.linalg.det(rotation)),
        "max_orthogonality_error": float(np.max(np.abs(error))),
    }


def clipped_rect(rect, width, height):
    x0, y0, x1, y1 = rect
    return np.array(
        [
            np.clip(x0, 0.0, width - 1.0),
            np.clip(y0, 0.0, height - 1.0),
            np.clip(x1, 0.0, width - 1.0),
            np.clip(y1, 0.0, height - 1.0),
        ],
        dtype=np.float64,
    )


def rect_iou(first, second):
    ix0, iy0 = np.maximum(first[:2], second[:2])
    ix1, iy1 = np.minimum(first[2:], second[2:])
    intersection = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    first_area = max(first[2] - first[0], 0.0) * max(first[3] - first[1], 0.0)
    second_area = max(second[2] - second[0], 0.0) * max(second[3] - second[1], 0.0)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def projected_box_iou(obj, calibration, bbox):
    transform = np.asarray(calibration["T"], dtype=np.float64)
    intrinsic = np.asarray(calibration["K"], dtype=np.float64)
    distortion = np.asarray(calibration["D"], dtype=np.float64)
    corners_camera = lidar_to_camera(box_corners_lidar(obj), transform)
    if np.min(corners_camera[:, 2]) <= 0.1:
        return None
    pixels = project_camera_points(
        corners_camera, intrinsic, distortion, use_distortion=True
    )
    if not np.isfinite(pixels).all():
        return None
    predicted = np.array(
        [
            np.min(pixels[:, 0]),
            np.min(pixels[:, 1]),
            np.max(pixels[:, 0]),
            np.max(pixels[:, 1]),
        ]
    )
    center_x, center_y, box_width, box_height = [float(value) for value in bbox]
    annotated = np.array(
        [
            center_x - box_width / 2.0,
            center_y - box_height / 2.0,
            center_x + box_width / 2.0,
            center_y + box_height / 2.0,
        ]
    )
    width, height = int(calibration["width"]), int(calibration["height"])
    return rect_iou(
        clipped_rect(predicted, width, height),
        clipped_rect(annotated, width, height),
    )


def pcd_point_count(path):
    with open(path, "rb") as file:
        for _ in range(200):
            line = file.readline()
            if not line:
                break
            decoded = line.decode("ascii", errors="replace").strip()
            if decoded.upper().startswith("POINTS "):
                return int(decoded.split()[1])
            if decoded.upper().startswith("DATA "):
                break
    return None


def add_issue(issues, limit, clip, category, detail):
    if len(issues) < limit:
        issues.append({"clip": clip, "category": category, "detail": detail})


def audit_clip(clip, xy_limit, issues, issue_limit):
    samples = load_json(clip / "annotations" / "sample.json")
    camera_keys = [sample_key for _, sample_key in CAMERA_MAP.values()]
    calibrations = {}
    calibration_report = {}

    lidar_path = clip / "calib" / "lidar" / "top-center-lidar.json"
    lidar_matrix = np.asarray(load_json(lidar_path)["extrinsic_matrix"], dtype=float)
    calibration_report["top-center-lidar"] = {
        **rotation_metrics(lidar_matrix),
        "translation_m": lidar_matrix[:3, 3].tolist(),
        "bottom_row": lidar_matrix[3].tolist(),
    }

    for camera_key in camera_keys:
        path = clip / "calib" / "camera" / f"{camera_key}.json"
        item = load_json(path)
        calibrations[camera_key] = item
        matrix = np.asarray(item["T"], dtype=float)
        center_lidar = -matrix[:3, :3].T @ matrix[:3, 3]
        metrics = rotation_metrics(matrix)
        metrics.update(
            {
                "translation_in_lidar_to_camera_matrix": matrix[:3, 3].tolist(),
                "camera_center_in_lidar_m": center_lidar.tolist(),
                "bottom_row": matrix[3].tolist(),
                "intrinsic_shape": list(np.asarray(item["K"]).shape),
                "distortion_count": len(item["D"]),
                "calibration_size": [int(item["width"]), int(item["height"])],
            }
        )
        calibration_report[camera_key] = metrics

    raw_classes = Counter()
    mapped_classes = Counter()
    unknown_classes = Counter()
    total_objects = 0
    valid_objects = 0
    objects_in_range = 0
    invalid_dimensions = 0
    raw_speeds = []
    converter_speeds = []
    point_counts = []
    lidar_frame_points = []
    projection_ious = defaultdict(list)
    matched_2d = Counter()
    structural_missing = 0

    for sample in samples:
        token = str(sample.get("sample_annotation"))
        required_assets = [
            clip / "concated_pcl" / f"{sample.get('concated_pcl')}.pcd",
            *[
                clip / "camera" / key / f"{sample.get(key)}.jpg"
                for key in camera_keys
            ],
        ]
        for path in required_assets:
            if not path.is_file():
                structural_missing += 1
                add_issue(issues, issue_limit, clip.name, "missing_asset", str(path))

        pcd_path = required_assets[0]
        if pcd_path.is_file():
            count = pcd_point_count(pcd_path)
            if count is not None:
                lidar_frame_points.append(count)

        for camera_key, image_path in zip(camera_keys, required_assets[1:]):
            if not image_path.is_file():
                continue
            with Image.open(image_path) as image:
                expected = (
                    int(calibrations[camera_key]["width"]),
                    int(calibrations[camera_key]["height"]),
                )
                if image.size != expected:
                    add_issue(
                        issues,
                        issue_limit,
                        clip.name,
                        "image_size_mismatch",
                        f"{image_path}: image={image.size}, calib={expected}",
                    )

        try:
            annotation_path = Path(resolve_annotation_path(str(clip), token))
            annotation = load_json(annotation_path)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as error:
            structural_missing += 1
            add_issue(
                issues, issue_limit, clip.name, "annotation_error", f"{token}: {error}"
            )
            continue

        for obj in annotation.get("objects", []):
            total_objects += 1
            raw_name = obj.get("type")
            raw_classes[str(raw_name)] += 1
            mapped_name = CLASS_MAP.get(raw_name)
            if mapped_name is None:
                unknown_classes[str(raw_name)] += 1
            else:
                mapped_classes[mapped_name] += 1

            try:
                location = [float(value) for value in obj["location"]]
                size = [float(value) for value in obj["size"]]
                rotation = [float(value) for value in obj["rotation"]]
                finite = np.isfinite(location + size + rotation).all()
            except (KeyError, TypeError, ValueError):
                finite = False
                location, size = [math.nan] * 3, [math.nan] * 3
            if not finite or any(value <= 0 for value in size):
                invalid_dimensions += 1
            if finite and abs(location[0]) <= xy_limit and abs(location[1]) <= xy_limit:
                objects_in_range += 1

            points = obj.get("num_points", obj.get("clip_points", 0))
            if isinstance(points, dict):
                points = sum(points.values())
            points = int(points or 0)
            point_counts.append(points)
            if points > 0:
                valid_objects += 1

            velocity = obj.get("velocity") or [0.0, 0.0, 0.0]
            if len(velocity) >= 2:
                vx, vy = float(velocity[0]), float(velocity[1])
                raw_speeds.append(math.hypot(vx, vy))
                if max(abs(vx), abs(vy)) > 200.0:
                    vx, vy = vx / 1000.0, vy / 1000.0
                converter_speeds.append(math.hypot(vx, vy))

            for item in obj.get("info2d", []):
                camera_key = ANNOTATION_CAMERA_MAP.get(item.get("camera"))
                if camera_key not in calibrations or not item.get("bbox"):
                    continue
                matched_2d[camera_key] += 1
                iou = projected_box_iou(obj, calibrations[camera_key], item["bbox"])
                if iou is not None:
                    projection_ious[camera_key].append(iou)

    projection_report = {}
    for camera_key in camera_keys:
        values = projection_ious[camera_key]
        projection_report[camera_key] = {
            "annotation_2d_matches": int(matched_2d[camera_key]),
            "projectable_pairs": len(values),
            "iou": distribution(values),
            "fraction_iou_ge_0_25": (
                float(np.mean(np.asarray(values) >= 0.25)) if values else None
            ),
            "fraction_iou_ge_0_50": (
                float(np.mean(np.asarray(values) >= 0.50)) if values else None
            ),
        }

    return {
        "clip": clip.name,
        "sample_count": len(samples),
        "missing_or_broken_assets": structural_missing,
        "lidar_points_per_frame": distribution(lidar_frame_points),
        "annotations": {
            "total_objects": total_objects,
            "objects_with_lidar_points": valid_objects,
            "objects_with_center_in_training_xy_range": objects_in_range,
            "invalid_or_nonpositive_box_dimensions": invalid_dimensions,
            "lidar_points_per_object": distribution(point_counts),
            "raw_speed_magnitude": distribution(raw_speeds),
            "speed_after_current_converter_rule": distribution(converter_speeds),
            "raw_class_counts": dict(raw_classes.most_common()),
            "mapped_class_counts": dict(mapped_classes.most_common()),
            "unmapped_class_counts": dict(unknown_classes.most_common()),
        },
        "calibration": calibration_report,
        "projection_3d_box_vs_annotation_2d": projection_report,
    }


def main():
    args = parse_args()
    clips = discover_clips(Path(args.root_path))
    if not clips:
        raise FileNotFoundError(f"No clip roots found under {args.root_path}")

    issues = []
    clip_reports = [
        audit_clip(clip, args.xy_limit, issues, args.max_issues) for clip in clips
    ]
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "root_path": str(Path(args.root_path).resolve()),
        "clip_count": len(clips),
        "assumptions": {
            "camera_T": "lidar_to_camera",
            "box_location": "geometric_center_in_lidar",
            "box_size": "length_width_height",
            "box_yaw": "counter_clockwise_about_lidar_positive_z",
            "info2d_bbox": "center_x_center_y_width_height",
            "camera_projection": "OpenCV_rational_model_K_plus_8D",
        },
        "clips": clip_reports,
        "issues": issues,
    }

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"custom_dataset_validation_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"Clips: {len(clips)}")
    for item in clip_reports:
        annotation = item["annotations"]
        print(
            f"{item['clip']}: samples={item['sample_count']}, "
            f"objects={annotation['total_objects']}, "
            f"valid={annotation['objects_with_lidar_points']}, "
            f"in_range={annotation['objects_with_center_in_training_xy_range']}, "
            f"unmapped={sum(annotation['unmapped_class_counts'].values())}, "
            f"missing={item['missing_or_broken_assets']}"
        )
        for camera, projection in item[
            "projection_3d_box_vs_annotation_2d"
        ].items():
            print(
                f"  {camera}: pairs={projection['projectable_pairs']}, "
                f"median_iou={projection['iou']['p50']}, "
                f"iou>=0.25={projection['fraction_iou_ge_0_25']}"
            )
    print(f"Detailed issues retained: {len(issues)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
