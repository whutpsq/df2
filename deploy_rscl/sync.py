from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional


@dataclass
class SensorPacket:
    topic: str
    timestamp_us: int
    payload: Any


@dataclass
class SyncedFrame:
    timestamp_us: int
    cameras: Dict[str, Any]
    lidar: Any


class FrameSynchronizer:
    def __init__(
        self,
        camera_topics: Iterable[str],
        camera_order: Iterable[str],
        lidar_topic: str,
        tolerance_ms: float,
        max_queue_size: int = 30,
    ) -> None:
        self.camera_topics = list(camera_topics)
        self.camera_order = list(camera_order)
        if len(self.camera_topics) != len(self.camera_order):
            raise ValueError("camera_topics and camera_order must have the same length")
        self.topic_to_camera = dict(zip(self.camera_topics, self.camera_order))
        self.lidar_topic = lidar_topic
        self.tolerance_us = int(tolerance_ms * 1000)
        self.max_queue_size = max_queue_size
        self.camera_queues: Dict[str, Deque[SensorPacket]] = {
            name: deque(maxlen=max_queue_size) for name in self.camera_order
        }
        self.lidar_queue: Deque[SensorPacket] = deque(maxlen=max_queue_size)
        self.last_emitted_timestamp_us: Optional[int] = None

    def add_camera(self, topic: str, timestamp_us: int, payload: Any) -> Optional[SyncedFrame]:
        camera_name = self.topic_to_camera[topic]
        self.camera_queues[camera_name].append(SensorPacket(topic, timestamp_us, payload))
        return self.try_sync()

    def add_lidar(self, timestamp_us: int, payload: Any) -> Optional[SyncedFrame]:
        self.lidar_queue.append(SensorPacket(self.lidar_topic, timestamp_us, payload))
        return self.try_sync()

    def try_sync(self) -> Optional[SyncedFrame]:
        if not self.lidar_queue:
            return None
        if any(len(queue) == 0 for queue in self.camera_queues.values()):
            return None

        lidar = self.lidar_queue[-1]
        if self.last_emitted_timestamp_us == lidar.timestamp_us:
            return None
        selected: Dict[str, SensorPacket] = {}
        for camera_name, queue in self.camera_queues.items():
            nearest = min(queue, key=lambda item: abs(item.timestamp_us - lidar.timestamp_us))
            if abs(nearest.timestamp_us - lidar.timestamp_us) > self.tolerance_us:
                return None
            selected[camera_name] = nearest

        self.last_emitted_timestamp_us = lidar.timestamp_us
        self._drop_older_than(lidar.timestamp_us - self.tolerance_us)
        return SyncedFrame(
            timestamp_us=lidar.timestamp_us,
            cameras={name: selected[name].payload for name in self.camera_order},
            lidar=lidar.payload,
        )

    def _drop_older_than(self, timestamp_us: int) -> None:
        while self.lidar_queue and self.lidar_queue[0].timestamp_us < timestamp_us:
            self.lidar_queue.popleft()
        for queue in self.camera_queues.values():
            while queue and queue[0].timestamp_us < timestamp_us:
                queue.popleft()
