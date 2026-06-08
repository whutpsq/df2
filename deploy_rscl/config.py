from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_CAMERA_TOPICS = [
    "/sensor/camera/center_camera_fov120/encode",
    "/sensor/camera/left_front_camera/encode",
    "/sensor/camera/right_front_camera/encode",
    "/sensor/camera/rear_camera/encode",
    "/sensor/camera/left_rear_camera/encode",
    "/sensor/camera/right_rear_camera/encode",
]


@dataclass
class RsclAdapterConfig:
    config_path: str
    checkpoint_path: str
    output_topic: str = "/perception/bevfusion/objects"
    lidar_topic: str = "/perception/lidar/preproc_points_cloud"
    camera_topics: List[str] = field(default_factory=lambda: list(DEFAULT_CAMERA_TOPICS))
    camera_order: List[str] = field(
        default_factory=lambda: [
            "front",
            "front_left",
            "front_right",
            "rear",
            "rear_left",
            "rear_right",
        ]
    )
    node_name: str = "bevfusion_rscl"
    module_name: str = "bevfusion_rscl"
    device: str = "cuda:0"
    score_threshold: float = 0.2
    sync_tolerance_ms: float = 50.0
    image_size: Sequence[int] = (128, 352)
    image_resize: float = 0.48
    image_mean: Sequence[float] = (0.485, 0.456, 0.406)
    image_std: Sequence[float] = (0.229, 0.224, 0.225)
    point_dim: int = 5
    point_cloud_range: Sequence[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    calibration_file: str | None = None
    input_message_type: str = "RawMessage"
    output_message_type: str = "RawMessage"
    publish_empty_frame: bool = True
    bag_path: str | None = None
    max_frames: int | None = None
    output_file: str | None = None

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "RsclAdapterConfig":
        return cls(
            config_path=str(data["config_path"]),
            checkpoint_path=str(data["checkpoint_path"]),
            output_topic=str(data.get("output_topic", "/perception/bevfusion/objects")),
            lidar_topic=str(data.get("lidar_topic", "/perception/lidar/preproc_points_cloud")),
            camera_topics=list(data.get("camera_topics", DEFAULT_CAMERA_TOPICS)),
            camera_order=list(
                data.get(
                    "camera_order",
                    ["front", "front_left", "front_right", "rear", "rear_left", "rear_right"],
                )
            ),
            node_name=str(data.get("node_name", "bevfusion_rscl")),
            module_name=str(data.get("module_name", "bevfusion_rscl")),
            device=str(data.get("device", "cuda:0")),
            score_threshold=float(data.get("score_threshold", 0.2)),
            sync_tolerance_ms=float(data.get("sync_tolerance_ms", 50.0)),
            image_size=tuple(data.get("image_size", (128, 352))),
            image_resize=float(data.get("image_resize", 0.48)),
            image_mean=tuple(data.get("image_mean", (0.485, 0.456, 0.406))),
            image_std=tuple(data.get("image_std", (0.229, 0.224, 0.225))),
            point_dim=int(data.get("point_dim", 5)),
            point_cloud_range=tuple(
                data.get("point_cloud_range", (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0))
            ),
            calibration_file=data.get("calibration_file"),
            input_message_type=str(data.get("input_message_type", "RawMessage")),
            output_message_type=str(data.get("output_message_type", "RawMessage")),
            publish_empty_frame=bool(data.get("publish_empty_frame", True)),
            bag_path=data.get("bag_path"),
            max_frames=int(data["max_frames"]) if data.get("max_frames") is not None else None,
            output_file=data.get("output_file"),
        )


def load_adapter_config(path: str | Path) -> RsclAdapterConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    cfg = RsclAdapterConfig.from_mapping(data)
    base_dir = path.resolve().parent
    cfg.config_path = _resolve_path(cfg.config_path, base_dir)
    cfg.checkpoint_path = _resolve_path(cfg.checkpoint_path, base_dir)
    if cfg.calibration_file:
        cfg.calibration_file = _resolve_path(cfg.calibration_file, base_dir)
    if cfg.bag_path:
        cfg.bag_path = _resolve_path(cfg.bag_path, base_dir)
    if cfg.output_file:
        cfg.output_file = _resolve_path(cfg.output_file, base_dir)
    return cfg


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())
