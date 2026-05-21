"""Convert custom clip datasets into BEVFusion nuScenes-style infos.

This converter keeps the training code on the existing NuScenesDataset path:
it reads the custom sample/annotation JSON files from one clip or a directory
of clips, converts compressed PCD point clouds to float32 .bin files, and
writes the info pkl files consumed by the current BEVFusion pipelines.
"""

import argparse
import json
import os
import struct
from decimal import Decimal
from os import path as osp

import numpy as np

try:
    from pyquaternion import Quaternion
except ImportError:
    Quaternion = None

try:
    import mmcv
except ImportError:
    mmcv = None


CAMERA_MAP = {
    "CAM_FRONT": ("center_camera_fov120", "front-camera-fov120"),
    "CAM_FRONT_LEFT": ("left_front_camera", "left-front-camera"),
    "CAM_FRONT_RIGHT": ("right_front_camera", "right-front-camera"),
    "CAM_BACK": ("rear_camera", "rear-camera"),
    "CAM_BACK_LEFT": ("left_rear_camera", "left-rear-camera"),
    "CAM_BACK_RIGHT": ("right_rear_camera", "right-rear-camera"),
}

CUSTOM_OBJECT_CLASSES = [
    "pedestrian",
    "rider",
    "bicycle",
    "motorcycle",
    "tricycle",
    "car",
    "bus",
    "truck",
    "large_vehicle",
    "special_vehicle",
    "vehicle_door",
    "cart",
    "animal",
    "traffic_sign",
    "traffic_cone",
    "bollard",
    "road_barrier",
    "barrier_gate",
    "parking_lock",
    "chock",
    "unknown_obstacle",
]

CLASS_MAP = {
    "Pedestrian": "pedestrian",
    "Pedestrian_else": "pedestrian",
    "Police": "pedestrian",
    "Standed_rider": "rider",
    "Other_rider": "rider",
    "Non_motor_rider": "rider",
    "Motor_rider": "rider",
    "Bicycle": "bicycle",
    "Motorcycle": "motorcycle",
    "Tricycle": "tricycle",
    "Car": "car",
    "Suv": "car",
    "Bus": "bus",
    "Truck": "truck",
    "Huge_vehicle": "large_vehicle",
    "Vehicle_else": "special_vehicle",
    "Firetruck": "special_vehicle",
    "Ambulance": "special_vehicle",
    "Policecar": "special_vehicle",
    "Sprinkler": "special_vehicle",
    "Vehicle_door": "vehicle_door",
    "Cart": "cart",
    "Animal_small": "animal",
    "Animal_big": "animal",
    "Traffic_sign": "traffic_sign",
    "Triangle_mark": "traffic_sign",
    "Cone": "traffic_cone",
    "Bollards": "bollard",
    "Sphere_bollards": "bollard",
    "Water_barrier": "road_barrier",
    "Water_barrier_crowding": "road_barrier",
    "Road_barrier": "road_barrier",
    "Crash_bucket": "road_barrier",
    "Stopping_sign": "traffic_sign",
    "Barrier_gate": "barrier_gate",
    "Parking_lock": "parking_lock",
    "Chock": "chock",
    "Unknown": "unknown_obstacle",
}

IGNORED_CLASSES = set()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-path",
        required=True,
        help="Custom clip root, or a directory containing multiple clip roots.",
    )
    parser.add_argument("--info-prefix", default="custom_dataset")
    parser.add_argument("--version", default="custom-v1.0")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument(
        "--split-by",
        choices=["auto", "clip", "sample"],
        default="auto",
        help=(
            "How to split train/val. 'auto' keeps sample-level split for a "
            "single clip and clip-level split for multiple clips."
        ),
    )
    parser.add_argument(
        "--skip-pcd-convert",
        action="store_true",
        help="Reuse existing points_bin files instead of converting PCDs.",
    )
    return parser.parse_args()


def _lzf_decompress(data, expected_size):
    """Decompress the LZF payload used by PCL binary_compressed PCD files."""
    out = bytearray()
    i = 0
    data_len = len(data)
    while i < data_len:
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            length = ctrl + 1
            out.extend(data[i : i + length])
            i += length
        else:
            length = ctrl >> 5
            if length == 7:
                if i >= data_len:
                    raise ValueError("Malformed LZF stream: missing length byte")
                length += data[i]
                i += 1
            ref_offset = (ctrl & 0x1F) << 8
            if i >= data_len:
                raise ValueError("Malformed LZF stream: missing offset byte")
            ref_offset += data[i]
            i += 1
            length += 2
            ref_pos = len(out) - ref_offset - 1
            if ref_pos < 0:
                raise ValueError("Malformed LZF stream: invalid back reference")
            for _ in range(length):
                out.append(out[ref_pos])
                ref_pos += 1
    if len(out) != expected_size:
        raise ValueError(
            f"LZF decompressed size mismatch: got {len(out)}, expected {expected_size}"
        )
    return bytes(out)


def _read_pcd_header(f):
    header = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PCD file ended before DATA line")
        decoded = line.decode("ascii", errors="strict").strip()
        header.append(decoded)
        if decoded.startswith("DATA"):
            break

    meta = {}
    for line in header:
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        meta[key] = values
    return meta


def _pcd_field_array(raw, meta):
    fields = meta["FIELDS"]
    sizes = [int(x) for x in meta["SIZE"]]
    types = meta["TYPE"]
    counts = [int(x) for x in meta.get("COUNT", ["1"] * len(fields))]
    width = int(meta["WIDTH"][0])
    height = int(meta.get("HEIGHT", ["1"])[0])
    points = int(meta.get("POINTS", [str(width * height)])[0])
    point_step = sum(size * count for size, count in zip(sizes, counts))

    if len(raw) != points * point_step:
        raise ValueError(
            f"Unexpected PCD payload size: got {len(raw)}, "
            f"expected {points * point_step}"
        )

    # PCL stores binary_compressed data in structure-of-arrays order.
    columns = {}
    offset = 0
    for field, size, typ, count in zip(fields, sizes, types, counts):
        nbytes = points * size * count
        chunk = raw[offset : offset + nbytes]
        offset += nbytes
        if typ == "F" and size == 4:
            dtype = np.float32
        elif typ == "F" and size == 8:
            dtype = np.float64
        elif typ == "I" and size == 4:
            dtype = np.int32
        elif typ == "U" and size == 4:
            dtype = np.uint32
        else:
            raise ValueError(f"Unsupported PCD field {field}: TYPE={typ} SIZE={size}")
        arr = np.frombuffer(chunk, dtype=dtype)
        if count != 1:
            arr = arr.reshape(points, count)
        columns[field] = arr
    return columns, points


def load_pcd_xyzi(pcd_path):
    with open(pcd_path, "rb") as f:
        meta = _read_pcd_header(f)
        data_type = meta["DATA"][0]
        if data_type == "binary_compressed":
            sizes = f.read(8)
            if len(sizes) != 8:
                raise ValueError(f"Missing compressed sizes in {pcd_path}")
            compressed_size, uncompressed_size = struct.unpack("<II", sizes)
            compressed = f.read(compressed_size)
            raw = _lzf_decompress(compressed, uncompressed_size)
            columns, points = _pcd_field_array(raw, meta)
        elif data_type == "binary":
            raw = f.read()
            columns, points = _pcd_field_array(raw, meta)
        else:
            raise ValueError(f"Unsupported PCD DATA type: {data_type}")

    xyzi = np.zeros((points, 5), dtype=np.float32)
    for i, name in enumerate(("x", "y", "z", "intensity")):
        xyzi[:, i] = columns[name].astype(np.float32, copy=False)
    # BEVFusion uses the 5th lidar channel as sweep time lag.
    xyzi[:, 4] = 0.0
    finite_mask = np.isfinite(xyzi).all(axis=1)
    return xyzi[finite_mask]


def matrix_to_quaternion(matrix):
    rot = normalize_rotation(np.asarray(matrix, dtype=np.float64)[:3, :3])
    if Quaternion is not None:
        q = Quaternion(matrix=rot)
        return [q.w, q.x, q.y, q.z]

    trace = np.trace(rot)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def normalize_rotation(rot):
    """Project a near-rotation matrix onto SO(3).

    Some exported calibration matrices are only approximately orthogonal.
    pyquaternion rejects those, while BEVFusion only needs a valid rigid
    transform, so use the closest orthonormal rotation.
    """
    u, _, vt = np.linalg.svd(rot)
    fixed = u @ vt
    if np.linalg.det(fixed) < 0:
        u[:, -1] *= -1
        fixed = u @ vt
    return fixed


def sample_token_to_annotation_file(token):
    timestamp_ns = int(Decimal(str(token)) * Decimal("1000000000"))
    return f"{timestamp_ns}.json"


def resolve_clip_path(root_path, relative_path):
    rel = relative_path.replace("\\", "/")
    clip_name = osp.basename(osp.normpath(root_path))
    if rel.startswith(clip_name + "/"):
        rel = rel[len(clip_name) + 1 :]
    return osp.join(root_path, *rel.split("/"))


def camera_file_path(root_path, sample, sample_camera_key):
    stem = sample[sample_camera_key]
    return osp.join(root_path, "camera", sample_camera_key, stem + ".jpg")


def invert_transform(transform):
    transform = np.asarray(transform, dtype=np.float64)
    inv = np.eye(4, dtype=np.float64)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def convert_points(root_path, sample, skip=False):
    stem = sample["concated_pcl"]
    out_dir = osp.join(root_path, "points_bin")
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, stem + ".bin")
    if skip and osp.exists(out_path):
        return out_path

    pcd_path = osp.join(root_path, "concated_pcl", stem + ".pcd")
    points = load_pcd_xyzi(pcd_path)
    points.astype(np.float32).tofile(out_path)
    return out_path


def make_unique_token(clip_id, token):
    if token is None or token == "":
        return ""
    token = str(token)
    return token if clip_id is None else f"{clip_id}:{token}"


def build_cams(root_path, sample, ann, lidar2ego, clip_id=None):
    cams = {}
    ego2global = np.asarray(ann["ego2global_transformation_matrix"], dtype=np.float64)
    ego2global[:3, :3] = normalize_rotation(ego2global[:3, :3])
    for bev_name, (ann_key, sample_key) in CAMERA_MAP.items():
        cam_ann = ann["sensors"]["cams"][ann_key]
        # The custom camera extrinsic is LiDAR -> camera, while BEVFusion's
        # nuScenes-style info expects camera -> lidar and camera -> ego.
        lidar2cam = np.asarray(cam_ann["extrinsic"], dtype=np.float64)
        lidar2cam[:3, :3] = normalize_rotation(lidar2cam[:3, :3])
        cam2lidar = invert_transform(lidar2cam)
        cam2ego = lidar2ego @ cam2lidar
        cam2ego[:3, :3] = normalize_rotation(cam2ego[:3, :3])
        img_path = camera_file_path(root_path, sample, sample_key)
        if not osp.exists(img_path):
            raise FileNotFoundError(img_path)
        cams[bev_name] = {
            "data_path": img_path,
            "type": bev_name,
            "sample_data_token": make_unique_token(clip_id, sample[sample_key]),
            "sensor2ego_translation": cam2ego[:3, 3].tolist(),
            "sensor2ego_rotation": matrix_to_quaternion(cam2ego),
            "ego2global_translation": ego2global[:3, 3].tolist(),
            "ego2global_rotation": matrix_to_quaternion(ego2global),
            "timestamp": int(cam_ann["timestamp"]),
            "cam_intrinsic": np.asarray(cam_ann["cam_intrinsic"], dtype=np.float32),
            "sensor2lidar_rotation": cam2lidar[:3, :3].astype(np.float32),
            "sensor2lidar_translation": cam2lidar[:3, 3].astype(np.float32),
        }
    return cams


def build_annotations(ann):
    boxes = []
    names = []
    velocities = []
    num_lidar_pts = []
    valid = []
    unknown = set()

    for obj in ann.get("objects", []):
        raw_name = obj.get("type")
        if raw_name in IGNORED_CLASSES:
            continue
        name = CLASS_MAP.get(raw_name)
        if name is None:
            unknown.add(raw_name)
            continue

        length, width, height = [float(v) for v in obj["size"]]
        x, y, z = [float(v) for v in obj["location"]]
        yaw = float(obj.get("rotation", [0.0, 0.0, 0.0])[2])
        boxes.append([x, y, z, width, length, height, yaw])
        names.append(name)

        velocity = obj.get("velocity") or [0.0, 0.0, 0.0]
        if len(velocity) < 2:
            velocity = [0.0, 0.0, 0.0]
        vx, vy = float(velocity[0]), float(velocity[1])
        if max(abs(vx), abs(vy)) > 200.0:
            vx /= 1000.0
            vy /= 1000.0
        velocities.append([vx, vy])

        points = int(obj.get("clip_points", 0))
        if points <= 0 and isinstance(obj.get("num_points"), dict):
            points = int(sum(obj["num_points"].values()))
        num_lidar_pts.append(points)
        valid.append(points > 0)

    if unknown:
        print(f"Skipped unknown classes: {sorted(unknown)}")

    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
        np.asarray(names),
        np.asarray(velocities, dtype=np.float32).reshape(-1, 2),
        np.asarray(num_lidar_pts, dtype=np.int32),
        np.asarray(valid, dtype=bool),
    )


def build_info(root_path, sample, skip_pcd_convert=False, clip_id=None):
    ann_path = osp.join(
        root_path,
        "annotations",
        "sample_annotation",
        sample_token_to_annotation_file(sample["sample_annotation"]),
    )
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    lidar_info = ann["sensors"]["lidar"]["perception"]
    lidar2ego = np.asarray(lidar_info["extrinsic"], dtype=np.float64)
    ego2global = np.asarray(ann["ego2global_transformation_matrix"], dtype=np.float64)
    lidar2ego[:3, :3] = normalize_rotation(lidar2ego[:3, :3])
    ego2global[:3, :3] = normalize_rotation(ego2global[:3, :3])
    lidar_path = convert_points(root_path, sample, skip_pcd_convert)
    gt_boxes, gt_names, gt_velocity, num_lidar_pts, valid_flag = build_annotations(ann)

    return {
        "lidar_path": lidar_path,
        "token": make_unique_token(clip_id, sample["sample_annotation"]),
        "sweeps": [],
        "cams": build_cams(root_path, sample, ann, lidar2ego, clip_id),
        "lidar2ego_translation": lidar2ego[:3, 3].tolist(),
        "lidar2ego_rotation": matrix_to_quaternion(lidar2ego),
        "ego2global_translation": ego2global[:3, 3].tolist(),
        "ego2global_rotation": matrix_to_quaternion(ego2global),
        "timestamp": int(ann["timestamp"]),
        "ann_path": ann_path,
        "prev_token": make_unique_token(clip_id, sample.get("prev")),
        "next_token": make_unique_token(clip_id, sample.get("next")),
        "location": "custom" if clip_id is None else clip_id,
        "gt_boxes": gt_boxes,
        "gt_names": gt_names,
        "gt_velocity": gt_velocity,
        "num_lidar_pts": num_lidar_pts,
        "num_radar_pts": np.zeros(len(gt_boxes), dtype=np.int32),
        "valid_flag": valid_flag,
    }


def is_clip_root(path):
    return osp.isfile(osp.join(path, "annotations", "sample.json"))


def discover_clip_roots(root_path):
    root_path = osp.abspath(root_path)
    if is_clip_root(root_path):
        return [root_path], root_path

    clip_roots = []
    for name in sorted(os.listdir(root_path)):
        child = osp.join(root_path, name)
        if osp.isdir(child) and is_clip_root(child):
            clip_roots.append(child)

    if not clip_roots:
        raise FileNotFoundError(
            "No clip roots found. Expected annotations/sample.json either "
            f"under {root_path} or one level below it."
        )
    return clip_roots, root_path


def load_clip_infos(clip_root, clip_id, skip_pcd_convert):
    sample_path = osp.join(clip_root, "annotations", "sample.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    samples = sorted(samples, key=lambda x: Decimal(str(x["sample_annotation"])))

    iterator = mmcv.track_iter_progress(samples) if mmcv is not None else samples
    infos = []
    for index, sample in enumerate(iterator):
        if mmcv is None:
            print(
                f"[{index + 1}/{len(samples)}] "
                f"{clip_id}/{sample['sample_annotation']}"
            )
        infos.append(build_info(clip_root, sample, skip_pcd_convert, clip_id))
    return infos


def split_infos_by_sample(infos, split_ratio):
    split_index = int(len(infos) * split_ratio)
    return infos[:split_index], infos[split_index:]


def split_infos_by_clip(clip_infos, split_ratio):
    split_index = int(len(clip_infos) * split_ratio)
    if len(clip_infos) > 1:
        split_index = min(max(split_index, 1), len(clip_infos) - 1)

    train_infos = []
    val_infos = []
    for _, infos in clip_infos[:split_index]:
        train_infos.extend(infos)
    for _, infos in clip_infos[split_index:]:
        val_infos.extend(infos)
    return train_infos, val_infos


def create_custom_infos(
    root_path,
    info_prefix,
    version,
    split_ratio,
    skip_pcd_convert,
    split_by="auto",
):
    clip_roots, output_root = discover_clip_roots(root_path)
    multiple_clips = len(clip_roots) > 1

    clip_infos = []
    for clip_index, clip_root in enumerate(clip_roots):
        clip_id = osp.basename(osp.normpath(clip_root)) if multiple_clips else None
        print(f"Processing clip {clip_index + 1}/{len(clip_roots)}: {clip_root}")
        infos = load_clip_infos(clip_root, clip_id, skip_pcd_convert)
        clip_infos.append((clip_root, infos))

    if split_by == "auto":
        split_by = "clip" if multiple_clips else "sample"

    if split_by == "clip":
        train_infos, val_infos = split_infos_by_clip(clip_infos, split_ratio)
    else:
        infos = []
        for _, clip_info in clip_infos:
            infos.extend(clip_info)
        infos = sorted(infos, key=lambda x: (x["location"], x["timestamp"]))
        train_infos, val_infos = split_infos_by_sample(infos, split_ratio)

    metadata = {
        "version": version,
        "classes": CUSTOM_OBJECT_CLASSES,
        "clip_count": len(clip_roots),
        "split_by": split_by,
    }

    train_path = osp.join(output_root, f"{info_prefix}_infos_train.pkl")
    val_path = osp.join(output_root, f"{info_prefix}_infos_val.pkl")
    dump_pickle({"infos": train_infos, "metadata": metadata}, train_path)
    dump_pickle({"infos": val_infos, "metadata": metadata}, val_path)

    print(f"Total clips: {len(clip_roots)}")
    print(f"Total samples: {len(train_infos) + len(val_infos)}")
    print(f"Train samples: {len(train_infos)} -> {train_path}")
    print(f"Val samples: {len(val_infos)} -> {val_path}")
    print("Converted point clouds are stored in each clip's points_bin directory.")


def dump_pickle(data, path):
    if mmcv is not None:
        mmcv.dump(data, path)
        return
    import pickle

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    create_custom_infos(
        root_path=osp.abspath(args.root_path),
        info_prefix=args.info_prefix,
        version=args.version,
        split_ratio=args.split_ratio,
        skip_pcd_convert=args.skip_pcd_convert,
        split_by=args.split_by,
    )


if __name__ == "__main__":
    main()
