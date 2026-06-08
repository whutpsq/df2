from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import RsclAdapterConfig
from .sync import SyncedFrame


def load_calibration(path: str | None, camera_order: Sequence[str]) -> Dict[str, Any]:
    if not path:
        return _identity_calibration(camera_order)
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")
    if path_obj.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    return data


def build_model_input(
    frame: SyncedFrame,
    cfg: RsclAdapterConfig,
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    image_tensors, img_aug_matrix = _build_images(frame, cfg)
    points = _build_points(frame, cfg)
    matrices = _build_matrices(calibration, cfg.camera_order)
    lidar_aug_matrix = torch.eye(4, dtype=torch.float32).unsqueeze(0)

    metas = [
        {
            "timestamp": frame.timestamp_us,
            "token": str(frame.timestamp_us),
            "lidar_path": "rscl://{}".format(cfg.lidar_topic),
            "filename": ["rscl://{}".format(t) for t in cfg.camera_topics],
            "img_shape": [tuple(cfg.image_size)] * len(cfg.camera_topics),
            "ori_shape": [tuple(cfg.image_size)] * len(cfg.camera_topics),
            "pad_shape": [tuple(cfg.image_size)] * len(cfg.camera_topics),
            "lidar2image": matrices["lidar2image"].numpy().tolist(),
        }
    ]

    return {
        "img": image_tensors,
        "points": [points],
        "camera2ego": matrices["camera2ego"].unsqueeze(0),
        "lidar2ego": matrices["lidar2ego"].unsqueeze(0),
        "lidar2camera": matrices["lidar2camera"].unsqueeze(0),
        "lidar2image": matrices["lidar2image"].unsqueeze(0),
        "camera_intrinsics": matrices["camera_intrinsics"].unsqueeze(0),
        "camera2lidar": matrices["camera2lidar"].unsqueeze(0),
        "img_aug_matrix": img_aug_matrix.unsqueeze(0),
        "lidar_aug_matrix": lidar_aug_matrix,
        "metas": metas,
    }


def _build_images(frame: SyncedFrame, cfg: RsclAdapterConfig) -> tuple[torch.Tensor, torch.Tensor]:
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=list(cfg.image_mean), std=list(cfg.image_std)),
        ]
    )
    tensors: List[torch.Tensor] = []
    aug_mats: List[torch.Tensor] = []
    for name in cfg.camera_order:
        image = frame.cameras[name]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Camera payload for {name} must be PIL.Image or numpy array")
        image, aug = _resize_crop_image(image.convert("RGB"), cfg)
        tensors.append(normalize(image))
        aug_mats.append(aug)
    return torch.stack(tensors, dim=0).unsqueeze(0), torch.stack(aug_mats, dim=0)


def _resize_crop_image(image: Image.Image, cfg: RsclAdapterConfig) -> tuple[Image.Image, torch.Tensor]:
    width, height = image.size
    final_h, final_w = int(cfg.image_size[0]), int(cfg.image_size[1])
    resize = float(getattr(cfg, "image_resize", 0.48))
    resize_w = max(int(width * resize), final_w)
    resize_h = max(int(height * resize), final_h)
    crop_w = int(max(0, resize_w - final_w) / 2)
    crop_h = int(max(0, resize_h - final_h))
    image = image.resize((resize_w, resize_h), Image.BILINEAR)
    image = image.crop((crop_w, crop_h, crop_w + final_w, crop_h + final_h))

    aug = torch.eye(4, dtype=torch.float32)
    aug[0, 0] = resize
    aug[1, 1] = resize
    aug[0, 3] = -float(crop_w)
    aug[1, 3] = -float(crop_h)
    return image, aug


def _build_points(frame: SyncedFrame, cfg: RsclAdapterConfig) -> torch.Tensor:
    points = np.asarray(frame.lidar, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, cfg.point_dim)
    elif points.ndim != 2:
        raise ValueError(f"Expected lidar points to be 1D or 2D, got shape {points.shape}")
    pcd_range = np.asarray(cfg.point_cloud_range, dtype=np.float32)
    mask = (
        (points[:, 0] > pcd_range[0])
        & (points[:, 0] < pcd_range[3])
        & (points[:, 1] > pcd_range[1])
        & (points[:, 1] < pcd_range[4])
        & (points[:, 2] > pcd_range[2])
        & (points[:, 2] < pcd_range[5])
    )
    points = points[mask]
    if points.shape[1] < 5:
        pad = np.zeros((points.shape[0], 5 - points.shape[1]), dtype=np.float32)
        points = np.concatenate([points, pad], axis=1)
    return torch.from_numpy(points.astype(np.float32, copy=False))


def _build_matrices(calibration: Mapping[str, Any], camera_order: Sequence[str]) -> Dict[str, torch.Tensor]:
    camera2ego = _stack_camera_matrix(calibration, camera_order, "camera2ego")
    lidar2ego = _matrix(calibration.get("lidar2ego", np.eye(4, dtype=np.float32)))
    camera_intrinsics = _stack_camera_matrix(calibration, camera_order, "camera_intrinsics")

    ego2lidar = torch.linalg.inv(lidar2ego)
    camera2lidar = torch.matmul(ego2lidar.unsqueeze(0), camera2ego)
    lidar2camera = torch.linalg.inv(camera2lidar)
    lidar2image = torch.matmul(camera_intrinsics, lidar2camera)
    return {
        "camera2ego": camera2ego,
        "lidar2ego": lidar2ego,
        "camera_intrinsics": camera_intrinsics,
        "camera2lidar": camera2lidar,
        "lidar2camera": lidar2camera,
        "lidar2image": lidar2image,
    }


def _stack_camera_matrix(
    calibration: Mapping[str, Any], camera_order: Sequence[str], key: str
) -> torch.Tensor:
    cameras = calibration.get("cameras", {})
    values = []
    for name in camera_order:
        cam = cameras.get(name, {})
        if key in cam:
            values.append(_matrix(cam[key]))
        elif key == "camera_intrinsics":
            values.append(torch.eye(4, dtype=torch.float32))
        else:
            values.append(torch.eye(4, dtype=torch.float32))
    return torch.stack(values, dim=0)


def _matrix(value: Any) -> torch.Tensor:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (3, 3):
        padded = np.eye(4, dtype=np.float32)
        padded[:3, :3] = arr
        arr = padded
    if arr.shape != (4, 4):
        raise ValueError(f"Expected matrix shape (4, 4) or (3, 3), got {arr.shape}")
    return torch.from_numpy(arr)


def _identity_calibration(camera_order: Iterable[str]) -> Dict[str, Any]:
    return {
        "lidar2ego": np.eye(4, dtype=np.float32).tolist(),
        "cameras": {
            name: {
                "camera2ego": np.eye(4, dtype=np.float32).tolist(),
                "camera_intrinsics": np.eye(4, dtype=np.float32).tolist(),
            }
            for name in camera_order
        },
    }
