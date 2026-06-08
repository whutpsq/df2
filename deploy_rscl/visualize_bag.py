from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw

from .codecs import (
    StatefulCameraDecoder,
    VideoFrameNotReady,
    decode_lidar_message,
)
from .config import RsclAdapterConfig, load_adapter_config
from .sync import FrameSynchronizer, SyncedFrame


class BagVisualizer:
    def __init__(self, cfg: RsclAdapterConfig, output_dir: str) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sync = FrameSynchronizer(
            camera_topics=cfg.camera_topics,
            camera_order=cfg.camera_order,
            lidar_topic=cfg.lidar_topic,
            tolerance_ms=cfg.sync_tolerance_ms,
        )
        self.camera_decoders = {
            topic: StatefulCameraDecoder() for topic in cfg.camera_topics
        }
        self.frames = 0
        self.decode_errors = 0

    def run(self, bag_path: str) -> None:
        import rsclpy

        reader = self._open_reader(rsclpy, bag_path, set([*self.cfg.camera_topics, self.cfg.lidar_topic]))
        if not reader.is_valid():
            raise RuntimeError(f"Invalid rsclbag: {bag_path}")

        header = reader.get_bag_header()
        print(f"bag begin_time={getattr(header, 'begin_time', None)} end_time={getattr(header, 'end_time', None)}")
        while True:
            msg = reader.read_next_message()
            if msg is None:
                break
            frame = self._add_message(msg)
            if frame is None:
                continue
            self._save_frame(frame)
            self.frames += 1
            if self.cfg.max_frames is not None and self.frames >= self.cfg.max_frames:
                break
        print(f"saved_frames={self.frames} decode_errors={self.decode_errors} output_dir={self.output_dir}")

    def _open_reader(self, rsclpy: Any, bag_path: str, channels: set[str]) -> Any:
        if hasattr(rsclpy, "BagReaderAttribute"):
            attr = rsclpy.BagReaderAttribute()
            attr.included_channels = channels
            return rsclpy.BagReader(bag_path, attr)
        return rsclpy.BagReader(bag_path)

    def _add_message(self, msg: Any) -> Optional[SyncedFrame]:
        topic = _channel_name(msg)
        try:
            if topic in self.cfg.camera_topics:
                timestamp_us, image = self.camera_decoders[topic].decode(msg)
                return self.sync.add_camera(topic, timestamp_us, image)
            if topic == self.cfg.lidar_topic:
                timestamp_us, points = decode_lidar_message(msg, point_dim=self.cfg.point_dim)
                return self.sync.add_lidar(timestamp_us, points)
        except VideoFrameNotReady:
            return None
        except Exception:
            self.decode_errors += 1
            print(f"failed to decode topic={topic}")
            import traceback

            traceback.print_exc()
        return None

    def _save_frame(self, frame: SyncedFrame) -> None:
        frame_dir = self.output_dir / f"frame_{self.frames:06d}_{frame.timestamp_us}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for camera_name, image in frame.cameras.items():
            image.save(frame_dir / f"{camera_name}.jpg", quality=90)
        _make_mosaic(frame.cameras, self.cfg.camera_order).save(frame_dir / "cameras_mosaic.jpg", quality=90)
        stats = _lidar_stats(frame.lidar, self.cfg.point_cloud_range)
        (frame_dir / "lidar_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _render_lidar_bev(frame.lidar, self.cfg.point_cloud_range).save(frame_dir / "lidar_bev.png")
        print(
            f"[{self.frames}] saved {frame_dir} "
            f"lidar_points={stats['num_points']} in_range={stats['num_in_config_range']}"
        )


def _make_mosaic(cameras: dict[str, Image.Image], camera_order: Iterable[str]) -> Image.Image:
    thumbs = []
    for name in camera_order:
        image = cameras[name].convert("RGB")
        image.thumbnail((480, 270), Image.BILINEAR)
        canvas = Image.new("RGB", (480, 300), (20, 20, 20))
        canvas.paste(image, ((480 - image.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 276), name, fill=(255, 255, 255))
        thumbs.append(canvas)

    mosaic = Image.new("RGB", (480 * 3, 300 * 2), (0, 0, 0))
    for index, image in enumerate(thumbs):
        mosaic.paste(image, ((index % 3) * 480, (index // 3) * 300))
    return mosaic


def _render_lidar_bev(points: Any, point_cloud_range: Iterable[float], size: int = 900) -> Image.Image:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected lidar points as NxC with C>=3, got {arr.shape}")

    pc_range = np.asarray(list(point_cloud_range), dtype=np.float32)
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range[:6]
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    mask = finite_mask & (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    x = x[mask]
    y = y[mask]
    z = z[mask]

    image = Image.new("RGB", (size, size), (5, 8, 12))
    draw = ImageDraw.Draw(image)
    if len(x) == 0:
        x = arr[finite_mask, 0]
        y = arr[finite_mask, 1]
        z = arr[finite_mask, 2]
        if len(x) == 0:
            return image
        x_min, x_max = _percentile_range(x)
        y_min, y_max = _percentile_range(y)
        z_min, z_max = _percentile_range(z)

    px = ((y - y_min) / max(y_max - y_min, 1e-6) * (size - 1)).astype(np.int32)
    py = ((x_max - x) / max(x_max - x_min, 1e-6) * (size - 1)).astype(np.int32)
    z_norm = np.clip((z - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0)
    for col, row, zn in zip(px.tolist(), py.tolist(), z_norm.tolist()):
        color = (int(40 + 180 * zn), int(220 - 120 * zn), int(255 - 180 * zn))
        draw.point((col, row), fill=color)

    center = size // 2
    draw.line((center, 0, center, size), fill=(80, 80, 80))
    draw.line((0, center, size, center), fill=(80, 80, 80))
    draw.polygon([(center, center - 12), (center - 8, center + 10), (center + 8, center + 10)], outline=(255, 255, 255))
    return image


def _lidar_stats(points: Any, point_cloud_range: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(points, dtype=np.float32)
    stats: dict[str, Any] = {
        "shape": list(arr.shape),
        "num_points": int(arr.shape[0]) if arr.ndim >= 1 else 0,
    }
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return stats

    pc_range = np.asarray(list(point_cloud_range), dtype=np.float32)
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range[:6]
    xyz = arr[:, :3]
    finite_mask = np.isfinite(xyz).all(axis=1)
    finite_xyz = xyz[finite_mask]
    if finite_xyz.shape[0] == 0:
        stats.update(
            {
                "num_finite_xyz": 0,
                "num_in_config_range": 0,
                "config_point_cloud_range": pc_range.astype(float).tolist(),
            }
        )
        return stats
    mask = (
        (finite_xyz[:, 0] >= x_min)
        & (finite_xyz[:, 0] <= x_max)
        & (finite_xyz[:, 1] >= y_min)
        & (finite_xyz[:, 1] <= y_max)
        & (finite_xyz[:, 2] >= z_min)
        & (finite_xyz[:, 2] <= z_max)
    )
    stats.update(
        {
            "num_finite_xyz": int(finite_xyz.shape[0]),
            "xyz_min": finite_xyz.min(axis=0).astype(float).tolist(),
            "xyz_max": finite_xyz.max(axis=0).astype(float).tolist(),
            "xyz_mean": finite_xyz.mean(axis=0).astype(float).tolist(),
            "num_in_config_range": int(mask.sum()),
            "config_point_cloud_range": pc_range.astype(float).tolist(),
        }
    )
    if arr.shape[1] > 3:
        stats["extra_min"] = arr[:, 3:].min(axis=0).astype(float).tolist()
        stats["extra_max"] = arr[:, 3:].max(axis=0).astype(float).tolist()
    return stats


def _percentile_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.0]).astype(float).tolist()
    if abs(hi - lo) < 1e-6:
        lo -= 1.0
        hi += 1.0
    return float(lo), float(hi)


def _channel_name(msg: Any) -> str:
    for attr in ("channel_name", "channelName", "topic", "topic_name"):
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            return str(value() if callable(value) else value)
    raise ValueError(f"RSCL bag message has no channel name field: {type(msg)!r}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export simple camera/lidar visualizations from rsclbag.")
    parser.add_argument("--adapter-config", default="deploy_rscl/configs/bevfusion_rscl.yaml")
    parser.add_argument("--bag", required=True, help="Path to .rsclbag.")
    parser.add_argument("--output-dir", default="runs/rsclbag_visualization")
    parser.add_argument("--max-frames", type=int, default=5)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = load_adapter_config(args.adapter_config)
    cfg.max_frames = args.max_frames
    visualizer = BagVisualizer(cfg, args.output_dir)
    visualizer.run(args.bag)


if __name__ == "__main__":
    main()
