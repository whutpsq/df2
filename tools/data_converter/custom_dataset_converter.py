"""Convert the clip_S31a custom dataset into BEVFusion nuScenes-style infos.

This converter keeps the training code on the existing NuScenesDataset path:
it reads the custom sample/annotation JSON files, converts compressed PCD
point clouds to float32 .bin files, and writes the info pkl files consumed by
the current BEVFusion pipelines.
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

CLASS_MAP = {
    "Car": "car",
    "Suv": "car",
    "Vehicle_else": "car",
    "Pedestrian": "pedestrian",
    "Bicycle": "cyclist",
    "Motorcycle": "cyclist",
    "Tricycle": "cyclist",
    "Non_motor_rider": "cyclist",
    "Bollards": "obstacle",
    "Sphere_bollards": "obstacle",
    "Cone": "obstacle",
}

IGNORED_CLASSES = {"Vehicle_door"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-path", required=True, help="Custom clip root")
    parser.add_argument("--info-prefix", default="custom_dataset")
    parser.add_argument("--version", default="custom-v1.0")
    parser.add_argument("--split-ratio", type=float, default=0.8)
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


def build_cams(root_path, sample, ann, lidar2ego):
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
            "sample_data_token": sample[sample_key],
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


def build_info(root_path, sample, skip_pcd_convert=False):
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
        "token": str(sample["sample_annotation"]),
        "sweeps": [],
        "cams": build_cams(root_path, sample, ann, lidar2ego),
        "lidar2ego_translation": lidar2ego[:3, 3].tolist(),
        "lidar2ego_rotation": matrix_to_quaternion(lidar2ego),
        "ego2global_translation": ego2global[:3, 3].tolist(),
        "ego2global_rotation": matrix_to_quaternion(ego2global),
        "timestamp": int(ann["timestamp"]),
        "ann_path": ann_path,
        "prev_token": "" if sample.get("prev") is None else str(sample["prev"]),
        "next_token": "" if sample.get("next") is None else str(sample["next"]),
        "location": "custom",
        "gt_boxes": gt_boxes,
        "gt_names": gt_names,
        "gt_velocity": gt_velocity,
        "num_lidar_pts": num_lidar_pts,
        "num_radar_pts": np.zeros(len(gt_boxes), dtype=np.int32),
        "valid_flag": valid_flag,
    }


def create_custom_infos(root_path, info_prefix, version, split_ratio, skip_pcd_convert):
    sample_path = osp.join(root_path, "annotations", "sample.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    samples = sorted(samples, key=lambda x: Decimal(str(x["sample_annotation"])))

    iterator = mmcv.track_iter_progress(samples) if mmcv is not None else samples
    infos = []
    for index, sample in enumerate(iterator):
        if mmcv is None:
            print(f"[{index + 1}/{len(samples)}] {sample['sample_annotation']}")
        infos.append(build_info(root_path, sample, skip_pcd_convert))

    split_index = int(len(infos) * split_ratio)
    train_infos = infos[:split_index]
    val_infos = infos[split_index:]
    metadata = {"version": version, "classes": sorted(set(CLASS_MAP.values()))}

    train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
    val_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
    dump_pickle({"infos": train_infos, "metadata": metadata}, train_path)
    dump_pickle({"infos": val_infos, "metadata": metadata}, val_path)

    print(f"Total samples: {len(infos)}")
    print(f"Train samples: {len(train_infos)} -> {train_path}")
    print(f"Val samples: {len(val_infos)} -> {val_path}")
    print(f"Converted point clouds are in: {osp.join(root_path, 'points_bin')}")


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
    )


if __name__ == "__main__":
    main()
