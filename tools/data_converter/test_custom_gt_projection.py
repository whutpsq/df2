"""Project one random custom-dataset frame onto its six camera images.

The script visualizes both LiDAR points and annotated 3D boxes.  It uses the
data2 conventions verified against the annotation-provided 2D boxes:

* calib/camera/<camera>.json:T maps LiDAR coordinates to camera coordinates;
* object location is the geometric box center;
* object size is [length, width, height];
* yaw is counter-clockwise around LiDAR +z;
* camera projection uses both K and D by default.

Only NumPy and Pillow are required.  PCD parsing is reused from
custom_dataset_converter.py, so Open3D and OpenCV are not needed.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from custom_dataset_converter import load_pcd_xyzi, resolve_annotation_path


CAMERAS = [
    ("front_left", "left-front-camera"),
    ("front", "front-camera-fov120"),
    ("front_right", "right-front-camera"),
    ("rear_left", "left-rear-camera"),
    ("rear", "rear-camera"),
    ("rear_right", "right-rear-camera"),
]

ANNOTATION_CAMERA_MAP = {
    "front_camera_fov120": "front-camera-fov120",
    "center_camera_fov120": "front-camera-fov120",
    "left_front_camera": "left-front-camera",
    "right_front_camera": "right-front-camera",
    "rear_camera": "rear-camera",
    "left_rear_camera": "left-rear-camera",
    "right_rear_camera": "right-rear-camera",
}

BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
FRONT_FACE_EDGES = {(0, 1), (4, 5), (0, 4), (1, 5)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-path",
        default="data2",
        help="One clip root or a directory containing multiple clip roots.",
    )
    parser.add_argument(
        "--out-dir",
        default="data_write/gt_projection_test",
        help="Output root for six images, montage.jpg, and summary.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clip-name",
        help="Optional clip directory name. By default all discovered clips are used.",
    )
    parser.add_argument(
        "--sample-token",
        help="Optional sample_annotation token. Otherwise one frame is random.",
    )
    parser.add_argument(
        "--xy-limit",
        type=float,
        default=50.0,
        help="Project boxes only when abs(x) and abs(y) are both within this limit.",
    )
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-lidar-points", type=int, default=80000)
    parser.add_argument(
        "--point-radius",
        type=int,
        default=2,
        help="Projected LiDAR point radius in pixels. Default 2 draws 5x5 dots.",
    )
    parser.add_argument(
        "--no-lidar",
        action="store_true",
        help="Draw only 3D GT boxes, without projected LiDAR points.",
    )
    parser.add_argument(
        "--no-distortion",
        dest="use_distortion",
        action="store_false",
        help="Use K only. By default projection uses both K and D.",
    )
    parser.set_defaults(use_distortion=True)
    parser.add_argument(
        "--draw-labels", action="store_true", help="Draw object type next to boxes."
    )
    parser.add_argument(
        "--draw-2d-boxes",
        action="store_true",
        help="Also draw annotation info2d boxes in cyan for comparison.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include 3D objects whose num_points sum is zero.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def is_clip_root(path):
    return (path / "annotations" / "sample.json").is_file()


def discover_clip_roots(root_path, clip_name=None):
    root_path = root_path.resolve()
    if is_clip_root(root_path):
        clips = [root_path]
    else:
        clips = sorted(
            path for path in root_path.iterdir() if path.is_dir() and is_clip_root(path)
        )
    if clip_name:
        clips = [path for path in clips if path.name == clip_name]
    if not clips:
        raise FileNotFoundError(f"No matching clip root found under {root_path}")
    return clips


def choose_frame(root_path, seed, clip_name=None, sample_token=None):
    records = []
    for clip_root in discover_clip_roots(root_path, clip_name):
        samples = load_json(clip_root / "annotations" / "sample.json")
        records.extend((clip_root, sample) for sample in samples)

    if sample_token is not None:
        records = [
            item
            for item in records
            if str(item[1]["sample_annotation"]) == str(sample_token)
        ]
        if not records:
            raise KeyError(f"sample token not found: {sample_token}")

    rng = random.Random(seed)
    return records[rng.randrange(len(records))]


def make_unique_output_dir(out_root, clip_name, token):
    base = out_root.resolve() / f"{clip_name}_{token}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def object_num_points(obj):
    value = obj.get("num_points", 0)
    if isinstance(value, dict):
        return int(sum(value.values()))
    return int(value or obj.get("clip_points", 0) or 0)


def box_corners_lidar(obj):
    """Return 8 corners using the raw annotation box convention."""
    length, width, height = [float(value) for value in obj["size"]]
    local = np.array(
        [
            [length / 2, width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [length / 2, width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ],
        dtype=np.float64,
    )
    yaw = float(obj.get("rotation", [0.0, 0.0, 0.0])[2])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    local[:, :2] = local[:, :2] @ rotation.T
    return local + np.asarray(obj["location"], dtype=np.float64)


def lidar_to_camera(points, transform):
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_camera_points(points_camera, intrinsic, distortion, use_distortion):
    points_camera = np.asarray(points_camera, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = points_camera[:, 0] / points_camera[:, 2]
        y = points_camera[:, 1] / points_camera[:, 2]
        if use_distortion:
            k1, k2, p1, p2, k3, k4, k5, k6 = distortion[:8]
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
            radial = numerator / denominator
            distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            distorted_y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x, y = distorted_x, distorted_y
        normalized = np.stack([x, y, np.ones_like(x)], axis=1)
        pixels_h = normalized @ intrinsic.T
        return pixels_h[:, :2] / pixels_h[:, 2:3]


def clip_segment_to_near_plane(point0, point1, min_depth):
    point0 = point0.copy()
    point1 = point1.copy()
    if point0[2] < min_depth and point1[2] < min_depth:
        return None
    if point0[2] < min_depth:
        ratio = (min_depth - point0[2]) / (point1[2] - point0[2])
        point0 += ratio * (point1 - point0)
    elif point1[2] < min_depth:
        ratio = (min_depth - point1[2]) / (point0[2] - point1[2])
        point1 += ratio * (point0 - point1)
    return point0, point1


def clip_line_to_image(point0, point1, width, height):
    """Liang-Barsky clipping for one 2D segment."""
    x0, y0 = [float(value) for value in point0]
    x1, y1 = [float(value) for value in point1]
    dx, dy = x1 - x0, y1 - y0
    lower, upper = 0.0, 1.0
    for p_value, q_value in (
        (-dx, x0),
        (dx, width - 1 - x0),
        (-dy, y0),
        (dy, height - 1 - y0),
    ):
        if abs(p_value) < 1e-12:
            if q_value < 0:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return (
        (x0 + lower * dx, y0 + lower * dy),
        (x0 + upper * dx, y0 + upper * dy),
    )


def depth_colors(depth):
    if len(depth) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    near, far = np.percentile(depth, [5, 95])
    if far <= near:
        far = near + 1.0
    value = np.clip((depth - near) / (far - near), 0.0, 1.0)
    red = 255.0 * (1.0 - value)
    green = 255.0 * (1.0 - np.abs(2.0 * value - 1.0))
    blue = 255.0 * value
    return np.stack([red, green, blue], axis=1).astype(np.uint8)


def overlay_lidar_points(
    image,
    points_lidar,
    transform,
    intrinsic,
    distortion,
    min_depth,
    use_distortion,
    point_radius,
):
    points_camera = lidar_to_camera(points_lidar, transform)
    front = points_camera[:, 2] >= min_depth
    points_camera = points_camera[front]
    pixels = project_camera_points(
        points_camera, intrinsic, distortion, use_distortion
    )
    finite = np.isfinite(pixels).all(axis=1)
    pixels = pixels[finite]
    depth = points_camera[finite, 2]
    rounded = np.rint(pixels).astype(np.int64)
    width, height = image.size
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    rounded = rounded[inside]
    depth = depth[inside]
    colors = depth_colors(depth)

    order = np.argsort(depth)[::-1]
    rounded = rounded[order]
    colors = colors[order]
    array = np.asarray(image).copy()
    offsets = [
        (offset_x, offset_y)
        for offset_y in range(-point_radius, point_radius + 1)
        for offset_x in range(-point_radius, point_radius + 1)
        if offset_x * offset_x + offset_y * offset_y <= point_radius * point_radius
    ]
    for offset_x, offset_y in offsets:
        x = rounded[:, 0] + offset_x
        y = rounded[:, 1] + offset_y
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        old = array[y[valid], x[valid]].astype(np.float32)
        array[y[valid], x[valid]] = (0.30 * old + 0.70 * colors[valid]).astype(
            np.uint8
        )
    return Image.fromarray(array), int(len(rounded))


def draw_projected_box(
    draw,
    corners_camera,
    intrinsic,
    distortion,
    image_size,
    min_depth,
    use_distortion,
    line_width,
):
    width, height = image_size
    anchor = None
    visible_edges = 0
    sample_count = 12 if use_distortion else 2
    for edge in BOX_EDGES:
        clipped_3d = clip_segment_to_near_plane(
            corners_camera[edge[0]], corners_camera[edge[1]], min_depth
        )
        if clipped_3d is None:
            continue
        point0, point1 = clipped_3d
        ratios = np.linspace(0.0, 1.0, sample_count)
        edge_points = point0[None, :] + ratios[:, None] * (point1 - point0)[None, :]
        edge_pixels = project_camera_points(
            edge_points, intrinsic, distortion, use_distortion
        )
        if not np.isfinite(edge_pixels).all():
            continue
        color = (255, 165, 0) if edge in FRONT_FACE_EDGES else (40, 255, 40)
        for index in range(len(edge_pixels) - 1):
            clipped_2d = clip_line_to_image(
                edge_pixels[index], edge_pixels[index + 1], width, height
            )
            if clipped_2d is None:
                continue
            draw.line(clipped_2d, fill=color, width=line_width)
            visible_edges += 1
            if anchor is None:
                anchor = clipped_2d[0]
    return visible_edges > 0, anchor


def bbox_iou_from_center_xywh(projected_pixels, bbox):
    pred_min = np.min(projected_pixels, axis=0)
    pred_max = np.max(projected_pixels, axis=0)
    center_x, center_y, width, height = [float(value) for value in bbox]
    gt_min = np.array([center_x - width / 2, center_y - height / 2])
    gt_max = np.array([center_x + width / 2, center_y + height / 2])
    intersection_min = np.maximum(pred_min, gt_min)
    intersection_max = np.minimum(pred_max, gt_max)
    intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
    intersection = float(intersection_size[0] * intersection_size[1])
    pred_area = float(np.prod(np.maximum(pred_max - pred_min, 0.0)))
    gt_area = float(width * height)
    union = pred_area + gt_area - intersection
    return intersection / union if union > 0 else 0.0


def load_font(image_width):
    size = max(14, image_width // 120)
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate_camera_image(
    image,
    camera_key,
    calibration,
    objects,
    points_lidar,
    args,
    token,
):
    transform = np.asarray(calibration["T"], dtype=np.float64)
    intrinsic = np.asarray(calibration["K"], dtype=np.float64)
    distortion = np.asarray(calibration["D"], dtype=np.float64)
    expected_size = (int(calibration["width"]), int(calibration["height"]))
    if image.size != expected_size:
        raise ValueError(
            f"{camera_key}: image size {image.size} != calibration {expected_size}"
        )

    visible_lidar_points = 0
    if not args.no_lidar:
        image, visible_lidar_points = overlay_lidar_points(
            image,
            points_lidar,
            transform,
            intrinsic,
            distortion,
            args.min_depth,
            args.use_distortion,
            args.point_radius,
        )

    draw = ImageDraw.Draw(image)
    font = load_font(image.width)
    line_width = max(2, image.width // 900)
    visible_boxes = 0
    ious = []

    for obj in objects:
        if not args.include_empty and object_num_points(obj) <= 0:
            continue
        location = np.asarray(obj["location"], dtype=np.float64)
        if abs(location[0]) > args.xy_limit or abs(location[1]) > args.xy_limit:
            continue
        corners_camera = lidar_to_camera(box_corners_lidar(obj), transform)
        visible, anchor = draw_projected_box(
            draw,
            corners_camera,
            intrinsic,
            distortion,
            image.size,
            args.min_depth,
            args.use_distortion,
            line_width,
        )
        if visible:
            visible_boxes += 1
            if args.draw_labels and anchor is not None:
                label = f"{obj.get('type', 'unknown')}#{obj.get('id', '?')}"
                draw.text(
                    (anchor[0] + 3, anchor[1] + 3),
                    label,
                    fill=(255, 255, 255),
                    font=font,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )

        if np.all(corners_camera[:, 2] >= args.min_depth):
            projected = project_camera_points(
                corners_camera, intrinsic, distortion, args.use_distortion
            )
            for item in obj.get("info2d", []):
                if ANNOTATION_CAMERA_MAP.get(item.get("camera")) != camera_key:
                    continue
                bbox = item.get("bbox")
                if not bbox:
                    continue
                ious.append(bbox_iou_from_center_xywh(projected, bbox))
                if args.draw_2d_boxes:
                    center_x, center_y, box_width, box_height = [
                        float(value) for value in bbox
                    ]
                    draw.rectangle(
                        (
                            center_x - box_width / 2,
                            center_y - box_height / 2,
                            center_x + box_width / 2,
                            center_y + box_height / 2,
                        ),
                        outline=(0, 255, 255),
                        width=line_width,
                    )

    distortion_text = "K+D" if args.use_distortion else "K (rectified)"
    header = (
        f"{camera_key} | token={token} | boxes={visible_boxes} | "
        f"lidar_points={visible_lidar_points} | {distortion_text}"
    )
    text_box = draw.textbbox((0, 0), header, font=font, stroke_width=1)
    draw.rectangle(
        (0, 0, min(image.width, text_box[2] + 20), text_box[3] + 16),
        fill=(0, 0, 0),
    )
    draw.text(
        (10, 8),
        header,
        fill=(255, 255, 255),
        font=font,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    return image, {
        "visible_3d_boxes": visible_boxes,
        "visible_lidar_points": visible_lidar_points,
        "matched_2d_boxes": len(ious),
        "median_3d_to_2d_iou": float(np.median(ious)) if ious else None,
        "mean_3d_to_2d_iou": float(np.mean(ious)) if ious else None,
        "iou_values": ious,
    }


def create_montage(images, output_path):
    cell_width, cell_height = 800, 520
    montage = Image.new("RGB", (cell_width * 3, cell_height * 2), (20, 20, 20))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for index, image in enumerate(images):
        thumbnail = image.copy()
        thumbnail.thumbnail((cell_width, cell_height), resampling)
        left = (index % 3) * cell_width + (cell_width - thumbnail.width) // 2
        top = (index // 3) * cell_height + (cell_height - thumbnail.height) // 2
        montage.paste(thumbnail, (left, top))
    montage.save(output_path, quality=94)


def main():
    args = parse_args()
    if args.xy_limit <= 0:
        raise ValueError("--xy-limit must be positive")
    if args.point_radius < 0:
        raise ValueError("--point-radius must be non-negative")
    root_path = Path(args.root_path)
    clip_root, sample = choose_frame(
        root_path, args.seed, args.clip_name, args.sample_token
    )
    token = str(sample["sample_annotation"])
    annotation_path = Path(resolve_annotation_path(str(clip_root), token))
    annotation = load_json(annotation_path)
    objects = annotation.get("objects", [])

    pcd_path = clip_root / "concated_pcl" / f"{sample['concated_pcl']}.pcd"
    points = load_pcd_xyzi(str(pcd_path))[:, :3]
    rng = np.random.default_rng(args.seed)
    if len(points) > args.max_lidar_points:
        indices = rng.choice(len(points), args.max_lidar_points, replace=False)
        points_to_draw = points[indices]
    else:
        points_to_draw = points

    output_dir = make_unique_output_dir(Path(args.out_dir), clip_root.name, token)
    rendered_images = []
    camera_results = {}

    for index, (display_name, camera_key) in enumerate(CAMERAS, 1):
        image_path = (
            clip_root / "camera" / camera_key / f"{sample[camera_key]}.jpg"
        )
        calibration_path = clip_root / "calib" / "camera" / f"{camera_key}.json"
        image = Image.open(image_path).convert("RGB")
        calibration = load_json(calibration_path)
        rendered, metrics = annotate_camera_image(
            image,
            camera_key,
            calibration,
            objects,
            points_to_draw,
            args,
            token,
        )
        output_path = output_dir / f"{index:02d}_{display_name}.jpg"
        rendered.save(output_path, quality=94)
        rendered_images.append(rendered)
        metrics.pop("iou_values")
        camera_results[camera_key] = {
            "image_path": str(image_path.resolve()),
            "calibration_path": str(calibration_path.resolve()),
            "output_path": str(output_path.resolve()),
            **metrics,
        }

    montage_path = output_dir / "montage.jpg"
    create_montage(rendered_images, montage_path)
    matched_medians = [
        item["median_3d_to_2d_iou"]
        for item in camera_results.values()
        if item["median_3d_to_2d_iou"] is not None
    ]
    summary = {
        "seed": args.seed,
        "clip": clip_root.name,
        "sample_token": token,
        "pointcloud_path": str(pcd_path.resolve()),
        "annotation_path": str(annotation_path.resolve()),
        "pointcloud_points": int(len(points)),
        "sampled_lidar_points": int(len(points_to_draw)),
        "object_count": int(len(objects)),
        "box_xy_limit_m": float(args.xy_limit),
        "point_radius_pixels": int(args.point_radius),
        "coordinate_convention": {
            "lidar_to_camera": "calib T applied as P_cam = R @ P_lidar + t",
            "box_location": "geometric center",
            "box_size": "[length, width, height]",
            "box_yaw": "counter-clockwise around +z",
            "distortion": bool(args.use_distortion),
        },
        "camera_results": camera_results,
        "median_of_camera_median_ious": (
            float(np.median(matched_medians)) if matched_medians else None
        ),
        "montage_path": str(montage_path.resolve()),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Selected clip: {clip_root.name}")
    print(f"Selected token: {token}")
    print(f"Point cloud: {pcd_path}")
    print(f"Objects: {len(objects)}, points: {len(points)}")
    for camera_key, metrics in camera_results.items():
        print(
            f"{camera_key}: boxes={metrics['visible_3d_boxes']}, "
            f"points={metrics['visible_lidar_points']}, "
            f"median IoU={metrics['median_3d_to_2d_iou']}"
        )
    print(f"Montage: {montage_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
