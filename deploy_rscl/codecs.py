from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image


def _raw_to_bytes(msg: Any) -> bytes:
    if isinstance(msg, bytes):
        return msg
    if isinstance(msg, bytearray):
        return bytes(msg)
    if isinstance(msg, memoryview):
        return msg.tobytes()
    if isinstance(msg, str):
        maybe_hex = _rscl_echo_hex_string_to_bytes(msg)
        if maybe_hex is not None:
            return maybe_hex
        try:
            return base64.b64decode(msg, validate=True)
        except Exception:
            pass
        try:
            return msg.encode("latin-1")
        except UnicodeEncodeError:
            return msg.encode("utf-8")
    for attr in ("data", "payload", "raw", "buffer"):
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            if callable(value):
                value = value()
            return _raw_to_bytes(value)
    if hasattr(msg, "as_builder"):
        return _raw_to_bytes(msg.as_builder())
    raise TypeError(f"Unsupported RSCL RawMessage payload type: {type(msg)!r}")


def decode_json_raw_message(msg: Any) -> Dict[str, Any]:
    raw = _raw_to_bytes(msg)
    return json.loads(raw.decode("utf-8"))


class VideoFrameNotReady(RuntimeError):
    pass


class _VideoInvalidData(VideoFrameNotReady):
    pass


class StatefulCameraDecoder:
    def __init__(self) -> None:
        self._video_context = None
        self._codec_name: Optional[str] = None

    def decode(self, msg: Any) -> Tuple[int, Image.Image]:
        payload = _message_payload(msg)
        if isinstance(payload, dict):
            return _decode_camera_mapping(payload, wrapper=msg, video_decoder=self)
        return _extract_timestamp_us(payload, wrapper=msg), _decode_image_object(payload, video_decoder=self)

    def decode_video_packet(self, raw: bytes, meta: Any) -> Image.Image:
        codec_name = _infer_video_codec(meta, raw)
        last_error: Optional[Exception] = None
        for candidate in (codec_name, _alternate_video_codec(codec_name)):
            try:
                return self._decode_video_packet_with_codec(raw, meta, candidate)
            except _VideoInvalidData as exc:
                last_error = exc
                continue
        raise VideoFrameNotReady(
            "Video decoder has not produced a frame yet. "
            f"tried={(codec_name, _alternate_video_codec(codec_name))} "
            f"videoFormat={_first_value_nested(meta, ('videoFormat', 'codec', 'codecName'))} "
            f"frameType={_first_value_nested(meta, ('frameType', 'frame_type'))} "
            f"raw_head={raw[:16].hex()}"
        ) from last_error

    def _decode_video_packet_with_codec(self, raw: bytes, meta: Any, codec_name: str) -> Image.Image:
        if self._video_context is None or self._codec_name != codec_name:
            try:
                import av
            except ImportError as exc:
                raise ValueError(
                    "Camera payload is H264/H265 video. Install PyAV in the rsclpy Python "
                    "environment, e.g. `/opt/python3.8/bin/python3.8 -m pip install av`."
                ) from exc
            self._video_context = av.CodecContext.create(codec_name, "r")
            self._codec_name = codec_name

        import av

        try:
            frames = self._video_context.decode(av.Packet(raw))
        except av.error.InvalidDataError as exc:
            self._video_context = None
            self._codec_name = None
            raise _VideoInvalidData(
                "Video decoder rejected packet; waiting for a decodable keyframe or codec config. "
                f"codec={codec_name} videoFormat={_first_value_nested(meta, ('videoFormat', 'codec', 'codecName'))} "
                f"frameType={_first_value_nested(meta, ('frameType', 'frame_type'))} raw_head={raw[:16].hex()}"
            ) from exc
        if not frames:
            raise VideoFrameNotReady(f"{codec_name} decoder has not produced a frame yet")
        return frames[-1].to_image().convert("RGB")


def decode_camera_message(msg: Any) -> Tuple[int, Image.Image]:
    payload = _message_payload(msg)
    if isinstance(payload, dict):
        return _decode_camera_mapping(payload, wrapper=msg)
    return _extract_timestamp_us(payload, wrapper=msg), _decode_image_object(payload)


def decode_lidar_message(msg: Any, point_dim: int = 5) -> Tuple[int, np.ndarray]:
    payload = _message_payload(msg)
    if isinstance(payload, dict):
        return _decode_lidar_mapping(payload, point_dim, wrapper=msg)
    points = _decode_points_object(payload, point_dim)
    return _extract_timestamp_us(payload, wrapper=msg), _reshape_points(points, point_dim)


def decode_message_timestamp_us(msg: Any) -> int:
    payload = _message_payload(msg)
    return _extract_timestamp_us(payload, wrapper=msg)


def encode_detection_message(outputs: Dict[str, Any], score_threshold: float) -> str:
    boxes = outputs.get("boxes_3d")
    scores = outputs.get("scores_3d")
    labels = outputs.get("labels_3d")
    if boxes is None or scores is None or labels is None:
        return json.dumps({"objects": []})

    boxes_np = _to_numpy(boxes.tensor if hasattr(boxes, "tensor") else boxes)
    scores_np = _to_numpy(scores)
    labels_np = _to_numpy(labels)

    objects = []
    for box, score, label in zip(boxes_np, scores_np, labels_np):
        if float(score) < score_threshold:
            continue
        objects.append(
            {
                "label": int(label),
                "score": float(score),
                "box": [float(v) for v in box.tolist()],
            }
        )
    return json.dumps({"objects": objects}, separators=(",", ":"))


def _decode_camera_mapping(
    data: Dict[str, Any], wrapper: Any = None, video_decoder: Optional[StatefulCameraDecoder] = None
) -> Tuple[int, Image.Image]:
    timestamp_us = _extract_timestamp_us(data, wrapper=wrapper)
    if "image_path" in data:
        image = Image.open(data["image_path"]).convert("RGB")
    elif data.get("encoding") in ("base64_jpeg", "base64_png", "base64_image"):
        image = Image.open(BytesIO(base64.b64decode(data["data"]))).convert("RGB")
    elif "data" in data:
        image = _decode_image_bytes(_raw_to_bytes(data["data"]), data, video_decoder=video_decoder)
    else:
        raise ValueError(
            "Camera message must contain image_path, base64 data, or image bytes; "
            f"available keys={sorted(data.keys())}"
        )
    return timestamp_us, image


def _decode_lidar_mapping(data: Dict[str, Any], point_dim: int, wrapper: Any = None) -> Tuple[int, np.ndarray]:
    timestamp_us = _extract_timestamp_us(data, wrapper=wrapper)
    if "points_path" in data:
        path = Path(data["points_path"])
        if path.suffix.lower() == ".npy":
            points = np.load(path).astype(np.float32)
        else:
            points = np.fromfile(path, dtype=np.float32)
    elif "points" in data:
        points = np.asarray(data["points"], dtype=np.float32)
    elif "data" in data:
        raw = _raw_to_bytes(data["data"])
        point_step = int(data.get("pointStep", data.get("point_step", 0)) or 0)
        width = int(data.get("width", 0) or 0)
        if point_step:
            points = _decode_point_cloud_bytes(raw, point_step, width)
        else:
            points = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(
            "Lidar message must contain points_path, points, or data bytes; "
            f"available keys={sorted(data.keys())}"
        )
    return timestamp_us, _reshape_points(points, point_dim)


def _message_payload(msg: Any) -> Any:
    if hasattr(msg, "message_obj") and getattr(msg, "message_obj") is not None:
        return getattr(msg, "message_obj")
    if hasattr(msg, "message_json") and getattr(msg, "message_json") is not None:
        value = getattr(msg, "message_json")
        return value if isinstance(value, dict) else _object_to_mapping(value)
    try:
        return decode_json_raw_message(msg)
    except Exception:
        return msg


def _decode_image_object(obj: Any, video_decoder: Optional[StatefulCameraDecoder] = None) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, np.ndarray):
        return Image.fromarray(obj.astype(np.uint8)).convert("RGB")

    for attr in ("image", "img", "frame", "picture", "payload"):
        value = _maybe_get(obj, attr)
        if value is not None and value is not obj:
            try:
                return _decode_image_object(value, video_decoder=video_decoder)
            except Exception:
                pass

    raw = _first_bytes(obj, ("data", "raw", "buffer", "imageData", "image_data", "jpeg", "jpg"))
    if raw is None:
        raise ValueError(f"Unsupported camera RSCL message fields: {_field_names(obj)}")
    return _decode_image_bytes(raw, obj, video_decoder=video_decoder)


def _decode_image_bytes(
    raw: bytes, meta: Any, video_decoder: Optional[StatefulCameraDecoder] = None
) -> Image.Image:
    try:
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        pass
    try:
        decoded = base64.b64decode(raw, validate=True)
        return Image.open(BytesIO(decoded)).convert("RGB")
    except Exception:
        pass

    if _looks_like_video_packet(raw, meta):
        if video_decoder is None:
            raise ValueError(
                "Camera payload is H264/H265 video and needs a StatefulCameraDecoder; "
                f"available keys={_field_names(meta)} raw_len={len(raw)} raw_head={raw[:16].hex()}"
            )
        return video_decoder.decode_video_packet(raw, meta)

    width = _first_int_nested(meta, ("width", "cols", "imageWidth", "image_width"))
    height = _first_int_nested(meta, ("height", "rows", "imageHeight", "image_height"))
    if not width or not height:
        raise ValueError(
            "Camera raw bytes are not JPEG/PNG/base64 image and width/height were not found; "
            f"available keys={_field_names(meta)} raw_len={len(raw)} raw_head={raw[:16].hex()}"
        )

    encoding = str(
        _first_value_nested(meta, ("encoding", "format", "pixelFormat", "pixel_format")) or "rgb8"
    ).lower()
    arr = np.frombuffer(raw, dtype=np.uint8)
    if encoding in ("rgb", "rgb8"):
        arr = arr.reshape(height, width, 3)
    elif encoding in ("bgr", "bgr8"):
        arr = arr.reshape(height, width, 3)[:, :, ::-1]
    elif encoding in ("gray", "mono8", "y8"):
        arr = arr.reshape(height, width)
    elif encoding in ("rgba", "rgba8", "bgra", "bgra8"):
        arr = arr.reshape(height, width, 4)
        if encoding.startswith("bgra"):
            arr = arr[:, :, [2, 1, 0, 3]]
    elif encoding in ("nv12", "yuv420"):
        try:
            import cv2
        except ImportError as exc:
            raise ValueError(f"OpenCV is required to decode {encoding} camera frames") from exc
        yuv = arr.reshape(height * 3 // 2, width)
        code = cv2.COLOR_YUV2RGB_NV12 if encoding == "nv12" else cv2.COLOR_YUV2RGB_I420
        arr = cv2.cvtColor(yuv, code)
    else:
        raise ValueError(f"Unsupported camera raw encoding: {encoding}")
    return Image.fromarray(arr).convert("RGB")


def _looks_like_video_packet(raw: bytes, meta: Any) -> bool:
    video_format = _first_value_nested(meta, ("videoFormat", "codec", "codecName", "format", "encoding"))
    if video_format is not None:
        text = str(video_format).lower()
        if any(token in text for token in ("h264", "h265", "hevc", "avc", "video")):
            return True
    return raw.startswith(b"\x00\x00\x00\x01") or raw.startswith(b"\x00\x00\x01")


def _infer_video_codec(meta: Any, raw: bytes) -> str:
    video_format = _first_value_nested(meta, ("videoFormat", "codec", "codecName", "format", "encoding"))
    text = str(video_format).lower() if video_format is not None else ""
    if any(token in text for token in ("h265", "hevc", "265")):
        return "hevc"
    if any(token in text for token in ("h264", "avc", "264")):
        return "h264"

    start = 4 if raw.startswith(b"\x00\x00\x00\x01") else 3 if raw.startswith(b"\x00\x00\x01") else 0
    if len(raw) > start:
        h265_type = (raw[start] & 0x7E) >> 1
        h264_type = raw[start] & 0x1F
        if h265_type in range(0, 64) and h264_type in (0, 2):
            return "hevc"
    return "hevc"


def _alternate_video_codec(codec_name: str) -> str:
    return "h264" if codec_name == "hevc" else "hevc"


def _decode_points_object(obj: Any, point_dim: int) -> np.ndarray:
    for attr in ("points", "pointCloud", "point_cloud", "cloud"):
        value = _maybe_get(obj, attr)
        if value is not None:
            points = _points_from_sequence(value)
            if points is not None:
                return points

    raw = _first_bytes(obj, ("data", "raw", "buffer", "pointsData", "point_data", "payload"))
    if raw is None:
        raise ValueError(f"Unsupported lidar RSCL message fields: {_field_names(obj)}")

    point_step = _first_int(obj, ("pointStep", "point_step", "stride"))
    if point_step and point_step % 4 == 0:
        stride = max(point_step // 4, point_dim)
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, stride)[:, :point_dim].copy()
    return _reshape_points(np.frombuffer(raw, dtype=np.float32), point_dim).copy()


def _decode_point_cloud_bytes(raw: bytes, point_step: int, width: int = 0) -> np.ndarray:
    if width > 0:
        raw = raw[: width * point_step]
    if point_step == 16:
        # RSCL echo shows this cloud as 16-byte points. The first three int16
        # values are x/y/z in centimeters; remaining bytes carry timing/padding.
        records = np.frombuffer(raw, dtype=np.uint8).reshape(-1, point_step)
        xyz_i16 = records[:, 0:6].copy().view("<i2").reshape(-1, 3)
        points = np.zeros((records.shape[0], 4), dtype=np.float32)
        points[:, :3] = xyz_i16.astype(np.float32) * 0.01
        return points
    if point_step % 4 == 0:
        dim = max(point_step // 4, 1)
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, dim)
    raise ValueError(f"Unsupported point cloud pointStep={point_step} raw_len={len(raw)}")


def _reshape_points(points: Any, point_dim: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    flat = arr.reshape(-1)
    for dim in (point_dim, 4, 5, 6, 7, 3):
        if dim > 0 and flat.size % dim == 0:
            return flat.reshape(-1, dim)
    raise ValueError(
        f"Cannot reshape lidar float array of size {flat.size}; "
        f"tried dims={(point_dim, 4, 5, 6, 7, 3)}"
    )


def _rscl_echo_hex_string_to_bytes(value: str) -> Optional[bytes]:
    stripped = value.strip()
    if not stripped:
        return None
    tokens = stripped.split()
    if len(tokens) < 4:
        return None
    out = bytearray()
    for token in tokens:
        try:
            out.append(int(token, 16) & 0xFF)
        except ValueError:
            return None
    return bytes(out)


def _points_from_sequence(value: Any) -> Optional[np.ndarray]:
    try:
        seq = list(value)
    except TypeError:
        return None
    if not seq:
        return np.empty((0, 5), dtype=np.float32)
    first = seq[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.asarray(seq, dtype=np.float32)
    rows = []
    for point in seq:
        rows.append(
            [
                float(_first_value(point, ("x", "posX", "positionX")) or 0.0),
                float(_first_value(point, ("y", "posY", "positionY")) or 0.0),
                float(_first_value(point, ("z", "posZ", "positionZ")) or 0.0),
                float(_first_value(point, ("intensity", "i", "reflectance")) or 0.0),
                float(_first_value(point, ("time", "timestamp", "offsetTime", "relativeTime")) or 0.0),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _extract_timestamp_us(data: Any, wrapper: Any = None) -> int:
    for source in (data, wrapper):
        if source is None:
            continue
        value = _first_value(
            source,
            (
                "timestamp_us",
                "timestampUs",
                "time_us",
                "timestamp_ns",
                "timestampNs",
                "time_ns",
                "headerTimestamp",
                "header_time",
                "timestamp_ms",
                "timestampMs",
                "timestamp",
                "time",
            ),
        )
        if value is not None:
            return _normalize_timestamp_us(value)
        header = _maybe_get(source, "header")
        if header is not None:
            try:
                return _extract_timestamp_us(header)
            except ValueError:
                pass
    raise ValueError("Message does not contain a recognizable timestamp field")


def _normalize_timestamp_us(value: Any) -> int:
    if isinstance(value, dict):
        nested = _timestamp_from_mapping(value)
        if nested is not None:
            return nested
        return _extract_timestamp_us(value)
    value_int = int(float(value))
    if value_int > 10_000_000_000_000_000:
        return value_int // 1000
    if value_int > 10_000_000_000_000:
        return value_int
    if value_int > 10_000_000_000:
        return value_int * 1000
    return value_int


def _timestamp_from_mapping(data: Dict[str, Any]) -> Optional[int]:
    for name in ("timestamp_us", "timestampUs", "time_us", "timeUs"):
        if name in data:
            return int(float(data[name]))
    for name in ("timestamp_ns", "timestampNs", "time_ns", "timeNs"):
        if name in data:
            return int(float(data[name])) // 1000

    sec = _first_value(data, ("sec", "secs", "second", "seconds", "tv_sec"))
    nsec = _first_value(data, ("nsec", "nsecs", "nanosec", "nanosecs", "nanosecond", "nanoseconds", "tv_nsec"))
    if sec is not None:
        usec = int(float(sec)) * 1_000_000
        if nsec is not None:
            usec += int(float(nsec)) // 1000
        return usec

    msec = _first_value(data, ("msec", "millisec", "millisecond", "milliseconds"))
    if msec is not None:
        return int(float(msec) * 1000)
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    return np.asarray(value)


def _object_to_mapping(obj: Any) -> Dict[str, Any]:
    return {name: _maybe_get(obj, name) for name in _field_names(obj)}


def _first_bytes(obj: Any, names: Iterable[str]) -> Optional[bytes]:
    for name in names:
        value = _maybe_get(obj, name)
        if value is None:
            continue
        try:
            return _raw_to_bytes(value)
        except TypeError:
            continue
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    return None


def _first_int(obj: Any, names: Iterable[str]) -> Optional[int]:
    value = _first_value(obj, names)
    if value is None:
        return None
    return int(value)


def _first_int_nested(obj: Any, names: Iterable[str]) -> Optional[int]:
    value = _first_value_nested(obj, names)
    if value is None:
        return None
    return int(value)


def _first_value_nested(obj: Any, names: Iterable[str]) -> Any:
    value = _first_value(obj, names)
    if value is not None:
        return value
    if isinstance(obj, dict):
        for item in obj.values():
            if isinstance(item, dict):
                value = _first_value_nested(item, names)
                if value is not None:
                    return value
    return None


def _first_value(obj: Any, names: Iterable[str]) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return None
    for name in names:
        value = _maybe_get(obj, name)
        if value is not None:
            return value
    return None


def _maybe_get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    if hasattr(obj, name):
        value = getattr(obj, name)
        return value() if callable(value) else value
    return None


def _field_names(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return sorted(obj.keys())
    names = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if not callable(value):
            names.append(name)
    return names
