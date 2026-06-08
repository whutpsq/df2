from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

from .codecs import (
    decode_camera_message,
    decode_lidar_message,
    decode_message_timestamp_us,
    encode_detection_message,
    StatefulCameraDecoder,
    VideoFrameNotReady,
)
from .config import RsclAdapterConfig, load_adapter_config
from .sync import FrameSynchronizer, SyncedFrame


class RsclBagFrameRunner:
    def __init__(
        self, cfg: RsclAdapterConfig, decode_only: bool = False, decode_images_only: bool = False
    ) -> None:
        self.cfg = cfg
        self.decode_only = decode_only
        self.decode_images_only = decode_images_only
        self.sync = FrameSynchronizer(
            camera_topics=cfg.camera_topics,
            camera_order=cfg.camera_order,
            lidar_topic=cfg.lidar_topic,
            tolerance_ms=cfg.sync_tolerance_ms,
        )
        self.calibration = None
        self.runner = None
        if not decode_only and not decode_images_only:
            from .preprocess import load_calibration
            from .runner import BevFusionRunner

            self.calibration = load_calibration(cfg.calibration_file, cfg.camera_order)
            self.runner = BevFusionRunner(cfg.config_path, cfg.checkpoint_path, cfg.device)
        self.output_fp = _open_output(cfg.output_file)
        self.frames = 0
        self.decode_errors = 0
        self.camera_decoders = {
            topic: StatefulCameraDecoder() for topic in cfg.camera_topics
        }

    def close(self) -> None:
        if self.output_fp is not None:
            self.output_fp.close()

    def run(self, bag_path: str) -> int:
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
            self._handle_frame(frame)
            if self.cfg.max_frames is not None and self.frames >= self.cfg.max_frames:
                break
        print(f"synced_frames={self.frames} decode_errors={self.decode_errors}")
        return self.frames

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
                if self.decode_only:
                    timestamp_us = decode_message_timestamp_us(msg)
                    image = None
                else:
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
            traceback.print_exc()
        return None

    def _handle_frame(self, frame: SyncedFrame) -> None:
        self.frames += 1
        if self.decode_only:
            record = {
                "timestamp_us": frame.timestamp_us,
                "cameras": sorted(frame.cameras.keys()),
                "lidar_points": int(len(frame.lidar)),
            }
        elif self.decode_images_only:
            record = {
                "timestamp_us": frame.timestamp_us,
                "cameras": {
                    name: list(image.size) if hasattr(image, "size") else None
                    for name, image in frame.cameras.items()
                },
                "lidar_points": int(len(frame.lidar)),
            }
        else:
            assert self.runner is not None
            assert self.calibration is not None
            from .preprocess import build_model_input

            data = build_model_input(frame, self.cfg, self.calibration)
            outputs = self.runner.infer(data)
            record = json.loads(encode_detection_message(outputs[0], self.cfg.score_threshold))
            record["timestamp_us"] = frame.timestamp_us
        _write_record(record, self.output_fp)


def _channel_name(msg: Any) -> str:
    for attr in ("channel_name", "channelName", "topic", "topic_name"):
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            return str(value() if callable(value) else value)
    raise ValueError(f"RSCL bag message has no channel name field: {type(msg)!r}")


def _open_output(path: str | None) -> Optional[TextIO]:
    if not path:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("w", encoding="utf-8")


def _write_record(record: dict[str, Any], output_fp: Optional[TextIO]) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    if output_fp is None:
        print(line)
    else:
        output_fp.write(line + "\n")
        output_fp.flush()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BEVFusion directly on an rsclbag.")
    parser.add_argument(
        "--adapter-config",
        default="deploy_rscl/configs/bevfusion_rscl.yaml",
        help="Path to the RSCL adapter yaml/json config.",
    )
    parser.add_argument("--bag", help="Path to .rsclbag. Overrides bag_path in config.")
    parser.add_argument("--max-frames", type=int, help="Optional frame limit for smoke tests.")
    parser.add_argument("--output-file", help="Optional JSONL output path.")
    parser.add_argument("--decode-only", action="store_true", help="Only decode and sync RSCL messages.")
    parser.add_argument(
        "--decode-images-only",
        action="store_true",
        help="Decode camera images and lidar, but do not run BEVFusion.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = load_adapter_config(args.adapter_config)
    if args.max_frames is not None:
        cfg.max_frames = args.max_frames
    if args.output_file:
        cfg.output_file = str(Path(args.output_file).resolve())

    bag_path = args.bag or cfg.bag_path
    if not bag_path:
        raise SystemExit("Pass --bag or set bag_path in the adapter config.")

    bag_runner = RsclBagFrameRunner(
        cfg,
        decode_only=args.decode_only,
        decode_images_only=args.decode_images_only,
    )
    try:
        bag_runner.run(bag_path)
    finally:
        bag_runner.close()


if __name__ == "__main__":
    main()
