from __future__ import annotations

import argparse
import traceback
from typing import Any, Optional

from .codecs import (
    decode_lidar_message,
    encode_detection_message,
    StatefulCameraDecoder,
    VideoFrameNotReady,
)
from .config import RsclAdapterConfig, load_adapter_config
from .preprocess import build_model_input, load_calibration
from .runner import BevFusionRunner
from .sync import FrameSynchronizer, SyncedFrame


class BevFusionRsclNode:
    def __init__(self, cfg: RsclAdapterConfig) -> None:
        import rsclpy

        self.rsclpy = rsclpy
        self.cfg = cfg
        self.node = rsclpy.Node(cfg.node_name)
        self.sync = FrameSynchronizer(
            camera_topics=cfg.camera_topics,
            camera_order=cfg.camera_order,
            lidar_topic=cfg.lidar_topic,
            tolerance_ms=cfg.sync_tolerance_ms,
        )
        self.calibration = load_calibration(cfg.calibration_file, cfg.camera_order)
        self.runner = BevFusionRunner(cfg.config_path, cfg.checkpoint_path, cfg.device)
        self.publisher = self.node.create_publisher(cfg.output_topic, cfg.output_message_type)
        self.camera_decoders = {
            topic: StatefulCameraDecoder() for topic in cfg.camera_topics
        }
        self._create_subscribers()

    def spin(self) -> None:
        self.node.spin()

    def _create_subscribers(self) -> None:
        for topic in self.cfg.camera_topics:
            self.node.create_subscriber(
                topic,
                self.cfg.input_message_type,
                lambda msg, topic=topic: self._on_camera(topic, msg),
            )
        self.node.create_subscriber(
            self.cfg.lidar_topic,
            self.cfg.input_message_type,
            self._on_lidar,
        )

    def _on_camera(self, topic: str, msg: Any) -> None:
        try:
            timestamp_us, image = self.camera_decoders[topic].decode(msg)
            frame = self.sync.add_camera(topic, timestamp_us, image)
            self._process_if_ready(frame)
        except VideoFrameNotReady:
            return
        except Exception:
            traceback.print_exc()

    def _on_lidar(self, msg: Any) -> None:
        try:
            timestamp_us, points = decode_lidar_message(msg, point_dim=self.cfg.point_dim)
            frame = self.sync.add_lidar(timestamp_us, points)
            self._process_if_ready(frame)
        except Exception:
            traceback.print_exc()

    def _process_if_ready(self, frame: Optional[SyncedFrame]) -> None:
        if frame is None:
            return
        data = build_model_input(frame, self.cfg, self.calibration)
        outputs = self.runner.infer(data)
        payload = encode_detection_message(outputs[0], self.cfg.score_threshold)
        if payload or self.cfg.publish_empty_frame:
            self.publisher.publish(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BEVFusion as an RSCL Python node.")
    parser.add_argument(
        "--adapter-config",
        default="deploy_rscl/configs/bevfusion_rscl.yaml",
        help="Path to the RSCL adapter yaml/json config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_adapter_config(args.adapter_config)

    import rsclpy

    rsclpy.init(cfg.module_name)
    node = BevFusionRsclNode(cfg)
    node.spin()
    rsclpy.waitforshutdown()


if __name__ == "__main__":
    main()
