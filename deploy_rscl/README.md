# BEVFusion RSCL Adapter

This directory keeps the RSCL integration outside `mmdet3d/`, so the trained
BEVFusion model code stays unchanged.

## Inputs

The default config is aligned with the rsclbag channels shown below:

- Cameras: `/sensor/camera/center_camera_fov120/encode`,
  `/sensor/camera/left_front_camera/encode`,
  `/sensor/camera/right_front_camera/encode`,
  `/sensor/camera/rear_camera/encode`,
  `/sensor/camera/left_rear_camera/encode`,
  `/sensor/camera/right_rear_camera/encode`
- Lidar: `/perception/lidar/preproc_points_cloud`

`deploy_rscl/codecs.py` accepts both the previous JSON RawMessage smoke-test
payloads and direct RSCL bag messages. For capnp messages it tries common field
names such as `message_obj`, `message_json`, `data`, `image`, `points`, `width`,
`height`, `encoding`, `timestampNs`, and `header.timestamp`.

If a real schema uses different field names, the decoder prints the available
fields in the error message; update only `decode_camera_message()` or
`decode_lidar_message()` for that schema.

## Offline rsclbag Run

Source RSCL first:

```bash
source /opt/senseauto_active/senseauto-rscl/resource/scripts/setup.sh
cd /path/to/bevfusion
```

First check that messages can be decoded and synchronized without loading the
model:

```bash
python3 -m deploy_rscl.rscl_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /workspace/mybag.000.rsclbag \
  --decode-only \
  --max-frames 3
```

Then check H264/H265 camera decoding without loading the model:

```bash
/opt/python3.8/bin/python3.8 -m pip install av
/opt/python3.8/bin/python3.8 -m deploy_rscl.rscl_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /workspace/mybag.000.rsclbag \
  --decode-images-only \
  --max-frames 3
```

Run BEVFusion inference directly from the bag:

```bash
/opt/python3.8/bin/python3.8 -m pip install av
/opt/python3.8/bin/python3.8 -m deploy_rscl.rscl_bag_runner \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /workspace/mybag.000.rsclbag \
  --max-frames 20 \
  --output-file runs/rsclbag_outputs.jsonl
```

The output file is JSONL. Each line contains `timestamp_us` and the filtered
`objects` returned by the model.

## Visualize Bag Contents

RSCL bags are not ROS bags, so RViz cannot open them directly unless you build a
bridge that republishes RSCL messages as ROS topics. For a quick inspection in a
Docker environment, export synchronized camera images and lidar BEV images:

```bash
/opt/python3.8/bin/python3.8 -m pip install av
/opt/python3.8/bin/python3.8 -m deploy_rscl.visualize_bag \
  --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml \
  --bag /workspace/mybag.000.rsclbag \
  --output-dir runs/rsclbag_visualization \
  --max-frames 5
```

Each exported frame directory contains:

- `front.jpg`, `front_left.jpg`, `front_right.jpg`, `rear.jpg`,
  `rear_left.jpg`, `rear_right.jpg`
- `cameras_mosaic.jpg`
- `lidar_bev.png`

## Online RSCL Node

For online subscription, keep using:

```bash
python3 -m deploy_rscl.rscl_node --adapter-config deploy_rscl/configs/bevfusion_rscl.yaml
```

Or launch through mainboard:

```bash
mainboard -d deploy_rscl/configs/bevfusion_rscl.dag
```

## Calibration

`calibration_file` still defaults to identity matrices. This is useful only for
decoder and pipeline smoke tests. Real inference needs the vehicle calibration
YAML/JSON with `lidar2ego`, per-camera `camera2ego`, and
`camera_intrinsics`.

## JSON RawMessage Smoke-Test Format

Camera:

```json
{"timestamp_us": 1710000000000000, "image_path": "/tmp/front.jpg"}
```

Lidar:

```json
{"timestamp_us": 1710000000000000, "points_path": "/tmp/lidar.bin"}
```

Lidar `.bin` is expected to be float32 with `point_dim` columns, defaulting to
`x, y, z, intensity, time`.
