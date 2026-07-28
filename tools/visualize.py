import argparse
import copy
import os
import json
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack import distributed as dist
from torchpack.utils.config import configs
# from torchpack.utils.tqdm import tqdm
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def override_map_line_width(cfg, split, line_width):
    """Override map rasterization width for this visualization run only."""
    if line_width is None:
        return
    if line_width < 1:
        raise ValueError("--map-line-width must be at least 1")

    updated = False
    for transform in cfg.data[split].pipeline:
        if transform.get("type") == "LoadCustomBEVSegmentation":
            transform["line_width"] = line_width
            updated = True
    if not updated:
        raise ValueError(
            "The selected data pipeline has no LoadCustomBEVSegmentation step"
        )


def save_map_channels(out_dir, name, masks, classes):
    """Save each semantic map channel as a separate black-and-white PNG."""
    for class_name, mask in zip(classes, masks):
        class_dir = os.path.join(out_dir, "map-classes", class_name)
        mmcv.mkdir_or_exist(class_dir)
        mmcv.imwrite(
            mask.astype(np.uint8) * 255,
            os.path.join(class_dir, f"{name}.png"),
        )


BOX_EDGES = (
    (0, 1),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 7),
    (5, 6),
    (6, 7),
)


def load_raw_camera_model(image_path):
    """Load K and D for the camera owning one raw image."""
    image_path = Path(image_path)
    camera_name = image_path.parent.name
    clip_root = image_path.parents[2]
    calibration_path = (
        clip_root / "calib" / "camera" / f"{camera_name}.json"
    )
    with calibration_path.open("r", encoding="utf-8-sig") as f:
        calibration = json.load(f)
    intrinsic = np.asarray(calibration["K"], dtype=np.float64)
    distortion = np.asarray(calibration.get("D", []), dtype=np.float64)
    return intrinsic, distortion


def project_camera_points(points_camera, intrinsic, distortion):
    """Project camera-frame points using OpenCV rational K+D equations."""
    points_camera = np.asarray(points_camera, dtype=np.float64)
    depth = points_camera[:, 2]
    x = points_camera[:, 0] / depth
    y = points_camera[:, 1] / depth

    coefficients = np.zeros(8, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    count = min(8, distortion.size)
    coefficients[:count] = distortion[:count]
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial = numerator / denominator
    distorted_x = (
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    )
    distorted_y = (
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    )

    normalized = np.stack(
        [distorted_x, distorted_y, np.ones_like(distorted_x)], axis=1
    )
    pixels_h = normalized @ np.asarray(intrinsic, dtype=np.float64).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]

def visualize_camera_full(
    fpath,
    image_path,
    corners_lidar,
    labels,
    lidar2camera,
    classes,
    max_width,
    max_height,
):
    """Draw boxes on a complete raw camera image without cropping."""
    image = mmcv.imread(str(image_path))
    height, width = image.shape[:2]
    scales = [1.0]
    if max_width > 0:
        scales.append(max_width / float(width))
    if max_height > 0:
        scales.append(max_height / float(height))
    scale = min(scales)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    intrinsic, distortion = load_raw_camera_model(image_path)
    intrinsic = intrinsic.copy()
    intrinsic[0, :] *= scale
    intrinsic[1, :] *= scale

    if corners_lidar is not None and len(corners_lidar) > 0:
        corners_h = np.concatenate(
            [
                corners_lidar.reshape(-1, 3),
                np.ones((corners_lidar.shape[0] * 8, 1), dtype=np.float64),
            ],
            axis=1,
        )
        corners_camera = corners_h @ np.asarray(lidar2camera).T
        corners_camera = corners_camera[:, :3].reshape(-1, 8, 3)

        palette = (
            (0, 165, 255),
            (0, 255, 0),
            (255, 128, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 0),
        )
        for box_index, corners in enumerate(corners_camera):
            depths = corners[:, 2]
            projected = project_camera_points(
                corners, intrinsic, distortion
            )
            label = int(labels[box_index]) if labels is not None else 0
            color = palette[label % len(palette)]
            visible_points = []
            for start, end in BOX_EDGES:
                if depths[start] <= 0.1 or depths[end] <= 0.1:
                    continue
                point0 = tuple(np.rint(projected[start]).astype(np.int32))
                point1 = tuple(np.rint(projected[end]).astype(np.int32))
                cv2.line(image, point0, point1, color, 2, cv2.LINE_AA)
                visible_points.extend((projected[start], projected[end]))

            if visible_points and labels is not None and 0 <= label < len(classes):
                anchor = np.min(np.asarray(visible_points), axis=0)
                cv2.putText(
                    image,
                    classes[label],
                    tuple(np.rint(anchor).astype(np.int32)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(image, fpath)

def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument(
        "--map-line-width",
        type=int,
        default=None,
        help=(
            "Override LoadCustomBEVSegmentation line width for this GT "
            "visualization run only."
        ),
    )
    parser.add_argument(
        "--save-map-channels",
        action="store_true",
        help="Save every map class as an individual black-and-white PNG.",
    )
    parser.add_argument(
        "--camera-max-width",
        type=int,
        default=1920,
        help="Maximum output width for full camera images; 0 disables it.",
    )
    parser.add_argument(
        "--camera-max-height",
        type=int,
        default=1080,
        help="Maximum output height for full camera images; 0 disables it.",
    )
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    if args.map_line_width is not None and args.mode != "gt":
        raise ValueError("--map-line-width is only valid with --mode gt")
    override_map_line_width(cfg, args.split, args.map_line_width)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    # if args.mode == "pred":
    #     model = build_model(cfg.model)
    #     load_checkpoint(model, args.checkpoint, map_location="cpu")

    if args.mode == "pred":
        model = build_model(cfg.model)
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    reported_map_stats = False
    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0]
            if masks.ndim == 4:
                if masks.shape[0] != 1:
                    raise ValueError(
                        f"Expected visualization batch size 1, got {tuple(masks.shape)}"
                    )
                masks = masks[0]
            if masks.ndim != 3:
                raise ValueError(
                    f"Expected map GT shaped (C, H, W), got {tuple(masks.shape)}"
                )
            masks = masks.numpy().astype(np.bool_)
            if not reported_map_stats:
                counts = masks.reshape(masks.shape[0], -1).sum(axis=1)
                overlap_count = masks.sum(axis=0)
                stats = ", ".join(
                    f"{name}={int(count)}"
                    for name, count in zip(cfg.map_classes, counts)
                )
                tqdm.write(
                    f"First-frame map GT shape={masks.shape}, nonzero pixels: "
                    f"{stats}; cross-class overlap pixels="
                    f"{int((overlap_count > 1).sum())}, max classes per pixel="
                    f"{int(overlap_count.max())}"
                )
                reported_map_stats = True
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        if "img" in data:
            lidar2camera_matrices = data["lidar2camera"].data[0][0].numpy()
            lidar_aug_matrix = data["lidar_aug_matrix"].data[0][0].numpy()
            inverse_lidar_aug = np.linalg.inv(lidar_aug_matrix)

            corners_lidar = None
            if bboxes is not None and len(bboxes) > 0:
                corners = bboxes.corners.numpy()
                corners_h = np.concatenate(
                    [
                        corners.reshape(-1, 3),
                        np.ones((corners.shape[0] * 8, 1), dtype=np.float64),
                    ],
                    axis=1,
                )
                corners_lidar = (
                    corners_h @ inverse_lidar_aug.T
                )[:, :3].reshape(-1, 8, 3)

            for k, image_path in enumerate(metas["filename"]):
                visualize_camera_full(
                    os.path.join(
                        args.out_dir, f"camera-{k}", f"{name}.png"
                    ),
                    image_path,
                    corners_lidar,
                    labels,
                    lidar2camera_matrices[k],
                    cfg.object_classes,
                    args.camera_max_width,
                    args.camera_max_height,
                )
        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )
            if args.save_map_channels:
                save_map_channels(
                    args.out_dir,
                    name,
                    masks,
                    cfg.map_classes,
                )


if __name__ == "__main__":
    main()
