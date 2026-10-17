#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export nuScenes mini inputs for the CUDA-BEVFusion ResNet50 FP16 pipeline.

Only NumPy is required; no torch, mmcv, nuscenes-devkit or image decoder is used.
Example (copy images when the exported data will move to another host):

    python3 tool/prepare_nuscenes_mini.py --data-root /data/nuscenes \
        --output-dir /data/bevfusion-mini-val --split mini_val --copy-images

The output directory must be empty. Each sample-token directory contains the six
original 1600x900 JPEGs, four FP32 [1,6,4,4] calibration tensors, FP16 [N,5]
points, token.txt and frame.json. data-list.txt contains relative sample paths
for main --data-list. manifest.json records provenance and point/sweep counts.
Default image mode is an absolute symlink; --copy-images makes the export portable.

Preprocessing follows the MIT BEVFusion commit explicitly pinned by NVIDIA's
CUDA-BEVFusion README: db75150717a9462cb60241e36ba28d65f6908607 (2023-02-09).
One intentional, recorded choice: select the nearest nine historical sweeps,
equivalent to LoadPointsFromMultiSweeps(test_mode=True). The original YAML omits
that loader flag and can randomly select nine of ten converter sweeps, so this
export does not reproduce an unknown dump-data.py random seed. Both mini splits
use evaluation transforms, without point shuffle or training augmentation.

JPEG resizing/normalization is performed by CUDA-BEVFusion at inference time.
The stored img_aug_matrix represents resize 0.48 and crop (32,176,736,432), giving
704x256. Points are [x,y,z,intensity,time_lag_seconds], current scan first, then
nearest-to-oldest sweeps. History removes abs(x)<1 AND abs(y)<1 before transform;
the current scan retains close points. Empty history pads nine filtered copies
of the current scan; nonempty short history is not padded. Finally apply strict
XYZ range (-54,-54,-5) < xyz < (54,54,3) in FP32, then cast once to FP16.
No point truncation, voxelization, annotations, accuracy evaluation or mAP is
performed here. This export supports runtime/performance validation.

Exact upstream source URLs are embedded in SOURCES and each output manifest.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


MIT_COMMIT = "db75150717a9462cb60241e36ba28d65f6908607"
NVIDIA_COMMIT = "7c1623f234d5a9794fd862c9456f68f87ae7b900"
NUSCENES_COMMIT = "b40adc467b919192899405d9b77871afee8efa07"
MIT_BASE = "https://github.com/mit-han-lab/bevfusion/blob/" + MIT_COMMIT + "/"
NVIDIA_BASE = "https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/" + NVIDIA_COMMIT + "/CUDA-BEVFusion/"
SOURCES = {
    "converter_camera_order_and_sensor_time_transform": MIT_BASE + "tools/data_converter/nuscenes_converter.py",
    "dataset_calibration_and_timestamp_order": MIT_BASE + "mmdet3d/datasets/nuscenes_dataset.py",
    "points_and_sweeps": MIT_BASE + "mmdet3d/datasets/pipelines/loading.py",
    "test_pipeline": MIT_BASE + "configs/nuscenes/default.yaml",
    "image_augmentation_and_range_filter": MIT_BASE + "mmdet3d/datasets/pipelines/transforms_3d.py",
    "strict_range_boundaries": MIT_BASE + "mmdet3d/core/points/base_points.py",
    "nvidia_pinned_upstream": NVIDIA_BASE + "README.md",
    "resnet50_range": NVIDIA_BASE + "configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/default.yaml",
    "nvidia_tensor_format": NVIDIA_BASE + "tool/tensor.py",
    "nvidia_dump_format": NVIDIA_BASE + "tool/dump-data.py",
    "mini_splits": "https://github.com/nutonomy/nuscenes-devkit/blob/" + NUSCENES_COMMIT + "/python-sdk/nuscenes/utils/splits.py",
}
CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
SPLITS = {
    "mini_train": ("scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-1077", "scene-1094", "scene-1100"),
    "mini_val": ("scene-0103", "scene-0916"),
}
POINT_RANGE = np.array([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], dtype=np.float32)
HISTORY_SWEEPS = 9
RUNTIME_POINT_CAPACITY = 300000
TABLE_NAMES = ("scene", "sample", "sample_data", "sensor", "calibrated_sensor", "ego_pose")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def save_tensor(path, value):
    """NVIDIA tool/tensor.py wire format, with explicit little-endian fields."""
    value = np.asarray(value)
    codes = {np.dtype("float16"): 2, np.dtype("float32"): 3}
    if value.dtype not in codes or not np.isfinite(value).all():
        raise ValueError("Tensor must be finite float16/float32: " + str(path))
    header = np.array([0x33FF1101, value.ndim, codes[value.dtype], *value.shape], dtype="<i4")
    with Path(path).open("wb") as stream:
        stream.write(header.tobytes())
        stream.write(np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<")).tobytes())


def quaternion_rotation(quaternion):
    """nuScenes stores quaternions as [w,x,y,z], mapping sensor to ego/global."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("Invalid nuScenes quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


class MiniMetadata:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.metadata_root = self.root / "v1.0-mini"
        self.tables = {}
        self.metadata_hashes = {}
        for name in TABLE_NAMES:
            path = self.metadata_root / (name + ".json")
            records = json.loads(path.read_text(encoding="utf-8"))
            table = {row["token"]: row for row in records}
            if len(table) != len(records):
                raise ValueError("Duplicate tokens in " + str(path))
            self.tables[name] = table
            self.metadata_hashes[name + ".json"] = sha256_file(path)
        # Standard raw sample.json does not contain devkit's reverse-indexed data.
        self.keyframes = {}
        for record in self.tables["sample_data"].values():
            if record["is_key_frame"]:
                channel = self.channel(record)
                data = self.keyframes.setdefault(record["sample_token"], {})
                if channel in data:
                    raise ValueError("Duplicate keyframe channel for " + record["sample_token"])
                data[channel] = record["token"]

    def channel(self, record):
        calibration = self.tables["calibrated_sensor"][record["calibrated_sensor_token"]]
        return self.tables["sensor"][calibration["sensor_token"]]["channel"]

    def select_samples(self, split, limit=None):
        names = set(SPLITS["mini_train"] + SPLITS["mini_val"] if split == "all" else SPLITS[split])
        available = {scene["name"] for scene in self.tables["scene"].values()}
        if names - available:
            raise ValueError("Missing mini scenes: " + ", ".join(sorted(names - available)))
        samples = [sample for sample in self.tables["sample"].values()
                   if self.tables["scene"][sample["scene_token"]]["name"] in names]
        # Matches NuScenesDataset.load_annotations, independent of table order.
        samples.sort(key=lambda sample: sample["timestamp"])
        if not samples:
            raise ValueError("No samples in requested split")
        return samples[:limit] if limit is not None else samples

    def sample_sensors(self, sample):
        channels = self.keyframes.get(sample["token"], {})
        required = ("LIDAR_TOP",) + CAMERAS
        missing = set(required) - set(channels)
        if missing:
            raise ValueError("Sample {} lacks {}".format(sample["token"], sorted(missing)))
        return {channel: self.tables["sample_data"][channels[channel]] for channel in required}

    def file_path(self, record):
        path = (self.root / record["filename"]).resolve(strict=True)
        if not path.is_file():
            raise ValueError("Missing sensor file: " + str(path))
        return path

    def sensor_to_lidar(self, sensor_record, lidar_record):
        """Same float64 operations as converter.obtain_sensor2top.

        Each side uses its own acquisition-time ego pose. Return a column-vector
        rotation R and translation t: xyz_current_lidar = R @ xyz_sensor + t.
        """
        source_cal = self.tables["calibrated_sensor"][sensor_record["calibrated_sensor_token"]]
        source_pose = self.tables["ego_pose"][sensor_record["ego_pose_token"]]
        target_cal = self.tables["calibrated_sensor"][lidar_record["calibrated_sensor_token"]]
        target_pose = self.tables["ego_pose"][lidar_record["ego_pose_token"]]
        source_rotation = quaternion_rotation(source_cal["rotation"])
        source_global_rotation = quaternion_rotation(source_pose["rotation"])
        target_rotation = quaternion_rotation(target_cal["rotation"])
        target_global_rotation = quaternion_rotation(target_pose["rotation"])
        global_to_target = np.linalg.inv(target_global_rotation).T @ np.linalg.inv(target_rotation).T
        row_rotation = (source_rotation.T @ source_global_rotation.T) @ global_to_target
        translation = (np.asarray(source_cal["translation"]) @ source_global_rotation.T
                       + np.asarray(source_pose["translation"])) @ global_to_target
        translation -= (np.asarray(target_pose["translation"]) @ global_to_target
                        + np.asarray(target_cal["translation"]) @ np.linalg.inv(target_rotation).T)
        return row_rotation.T, translation

    def history(self, lidar_record):
        result = []
        seen = {lidar_record["token"]}
        token = lidar_record["prev"]
        while token and len(result) < HISTORY_SWEEPS:
            if token in seen:
                raise ValueError("Cycle in LiDAR sweep chain")
            seen.add(token)
            record = self.tables["sample_data"][token]
            if self.channel(record) != "LIDAR_TOP" or record["timestamp"] >= lidar_record["timestamp"]:
                raise ValueError("Invalid historical LiDAR record: " + token)
            result.append(record)
            token = record["prev"]
        return result


def image_augmentation():
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = matrix[1, 1] = 0.48
    matrix[0, 3], matrix[1, 3] = -32, -176
    return matrix


def calibration_tensors(metadata, sensors):
    matrices = {name: [] for name in ("camera2lidar", "camera_intrinsics", "lidar2image", "img_aug_matrix")}
    for channel in CAMERAS:
        camera = sensors[channel]
        if (camera["width"], camera["height"]) != (1600, 900):
            raise ValueError("{} must be 1600x900 for the fixed ResNet50 preprocessing".format(channel))
        rotation, translation = metadata.sensor_to_lidar(camera, sensors["LIDAR_TOP"])
        camera2lidar = np.eye(4, dtype=np.float32)
        camera2lidar[:3, :3] = rotation
        camera2lidar[:3, 3] = translation
        # Match get_data_info's cast placement before K @ lidar2camera.
        inverse_rotation = np.linalg.inv(rotation)
        inverse_translation = translation @ inverse_rotation.T
        lidar2camera = np.eye(4, dtype=np.float32)
        lidar2camera[:3, :3] = inverse_rotation
        lidar2camera[:3, 3] = -inverse_translation
        camera_cal = metadata.tables["calibrated_sensor"][camera["calibrated_sensor_token"]]
        intrinsics = np.eye(4, dtype=np.float32)
        intrinsics[:3, :3] = np.asarray(camera_cal["camera_intrinsic"], dtype=np.float32)
        matrices["camera2lidar"].append(camera2lidar)
        matrices["camera_intrinsics"].append(intrinsics)
        matrices["lidar2image"].append(intrinsics @ lidar2camera)
        matrices["img_aug_matrix"].append(image_augmentation())
    return {name: np.stack(values, axis=0)[None] for name, values in matrices.items()}


def read_points(path):
    if path.stat().st_size % (5 * 4):
        raise ValueError("LiDAR binary must contain float32 [N,5]: " + str(path))
    points = np.fromfile(path, dtype="<f4").reshape(-1, 5)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("LiDAR scan is empty or contains nonfinite values: " + str(path))
    return points


def remove_close(points):
    return points[~((np.abs(points[:, 0]) < 1.0) & (np.abs(points[:, 1]) < 1.0))]


def range_filter(points):
    return points[np.all(points[:, :3] > POINT_RANGE[:3], axis=1)
                  & np.all(points[:, :3] < POINT_RANGE[3:], axis=1)]


def aggregate_points(metadata, sample, lidar_record):
    current = read_points(metadata.file_path(lidar_record))
    current[:, 4] = 0.0  # Replace the raw ring index, retaining intensity in col 3.
    chunks = [current]
    sweeps = metadata.history(lidar_record)
    details = []
    if not sweeps:
        padded = remove_close(current)
        chunks.extend([padded] * HISTORY_SWEEPS)
    else:
        for sweep in sweeps:
            points = read_points(metadata.file_path(sweep))
            raw_count = len(points)
            points = remove_close(points)
            rotation, translation = metadata.sensor_to_lidar(sweep, lidar_record)
            # Separate assignments preserve the loader's float32 rounding steps.
            points[:, :3] = points[:, :3] @ rotation.T
            points[:, :3] += translation
            lag = sample["timestamp"] / 1e6 - sweep["timestamp"] / 1e6
            if lag < 0:
                raise ValueError("A historical sweep has a negative time lag")
            points[:, 4] = lag
            chunks.append(points)
            details.append({"sample_data_token": sweep["token"], "filename": sweep["filename"],
                            "timestamp_us": sweep["timestamp"], "time_lag_seconds": lag,
                            "raw_points": raw_count, "after_remove_close": len(points)})
    combined = np.concatenate(chunks, axis=0)
    filtered = range_filter(combined)
    output = filtered.astype(np.float16)
    if not len(output) or not np.isfinite(output).all():
        raise ValueError("Empty / nonfinite FP16 point tensor after preprocessing")
    details_summary = {
        "current_raw_points": len(current), "history_sweep_count": len(sweeps),
        "padded_current_copies": HISTORY_SWEEPS if not sweeps else 0,
        "combined_points_before_range_filter": len(combined), "point_count": len(output),
        "points_removed_by_range_filter": len(combined) - len(output),
        "exceeds_default_runtime_point_capacity": len(output) > RUNTIME_POINT_CAPACITY,
        "sweeps": details,
    }
    return output, details_summary


def convert_sample(metadata, sample, output_root, copy_images):
    sensors = metadata.sample_sensors(sample)
    matrices = calibration_tensors(metadata, sensors)
    points, point_metadata = aggregate_points(metadata, sample, sensors["LIDAR_TOP"])
    directory = output_root / sample["token"]
    directory.mkdir()
    images = []
    for index, channel in enumerate(CAMERAS):
        record = sensors[channel]
        source = metadata.file_path(record)
        filename = "{}-{}.jpg".format(index, channel.removeprefix("CAM_"))
        target = directory / filename
        if copy_images:
            shutil.copyfile(source, target)
        else:
            target.symlink_to(source)
        images.append({"channel": channel, "output": filename, "source": record["filename"],
                       "sample_data_token": record["token"], "timestamp_us": record["timestamp"],
                       "ego_pose_token": record["ego_pose_token"], "sha256": sha256_file(source)})
    for name, values in matrices.items():
        save_tensor(directory / (name + ".tensor"), values)
    save_tensor(directory / "points.tensor", points)
    (directory / "token.txt").write_text(sample["token"] + "\n", encoding="utf-8")
    frame = {
        "token": sample["token"], "scene_token": sample["scene_token"],
        "scene_name": metadata.tables["scene"][sample["scene_token"]]["name"],
        "timestamp_us": sample["timestamp"], "path": sample["token"],
        "lidar_sample_data_token": sensors["LIDAR_TOP"]["token"],
        "lidar_timestamp_us": sensors["LIDAR_TOP"]["timestamp"],
        "images": images, **point_metadata,
        "tensor_sha256": {name: sha256_file(directory / name) for name in
                          ["points.tensor"] + [key + ".tensor" for key in matrices]},
    }
    write_json(directory / "frame.json", frame)
    return frame


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="nuScenes root containing v1.0-mini/, samples/ and sweeps/")
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty export directory")
    parser.add_argument("--split", choices=("mini_val", "mini_train", "all"), default="mini_val")
    parser.add_argument("--limit", type=int, help="export the first N samples after timestamp sorting")
    parser.add_argument("--copy-images", action="store_true", help="copy original JPEG bytes instead of absolute symlinks")
    result = parser.parse_args(argv)
    if result.limit is not None and result.limit < 1:
        parser.error("--limit must be positive")
    return result


def main(argv=None):
    args = arguments(argv)
    metadata = MiniMetadata(args.data_root)
    samples = metadata.select_samples(args.split, args.limit)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid mixing exports: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "status": "exporting", "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "nuScenes", "version": "v1.0-mini", "split": args.split,
        "data_root": str(metadata.root), "output_dir": str(output), "limit": args.limit,
        "sample_count": len(samples), "image_mode": "copy" if args.copy_images else "absolute_symlink",
        "sources": SOURCES, "metadata_sha256": metadata.metadata_hashes,
        "script_sha256": sha256_file(Path(__file__).resolve()), "numpy_version": np.__version__,
        "preprocessing": {
            "camera_order": list(CAMERAS), "calibration_shape": [1, 6, 4, 4], "calibration_dtype": "float32",
            "sensor_time_ego_poses": True, "raw_image_size_wh": [1600, 900], "network_image_size_wh": [704, 256],
            "image_resize": 0.48, "image_crop_xyxy": [32, 176, 736, 432], "image_bytes_transformed": False,
            "point_columns": ["x", "y", "z", "intensity", "time_lag_seconds"], "point_dtype": "float16",
            "historical_sweeps": HISTORY_SWEEPS, "sweep_selection": "nearest (official loader test_mode=True)",
            "original_yaml_difference": "The pinned YAML omits loader test_mode; nine of ten available sweeps can be randomly selected there.",
            "close_removal": "abs(x)<1 AND abs(y)<1 in sweep-local coordinates; current scan exempt",
            "empty_history": "nine close-filtered current-scan copies", "short_nonempty_history": "no padding",
            "xyz_range": POINT_RANGE.tolist(), "range_boundaries": "strict > min and < max in float32 before FP16 cast",
            "point_shuffle": False, "point_truncation": False, "default_runtime_point_capacity": RUNTIME_POINT_CAPACITY,
        },
        "scope": "Inference inputs for runtime and performance checks; not an mAP evaluation or accuracy-equivalence result.",
        "frames": [],
    }
    write_json(output / "manifest.json", manifest)
    try:
        for index, sample in enumerate(samples):
            frame = convert_sample(metadata, sample, output, args.copy_images)
            manifest["frames"].append(frame)
            print("[{}/{}] {} {} points={} sweeps={} padded={}".format(
                index + 1, len(samples), frame["scene_name"], frame["token"], frame["point_count"],
                frame["history_sweep_count"], frame["padded_current_copies"]), flush=True)
            if frame["exceeds_default_runtime_point_capacity"]:
                print("WARNING: exceeds main's default 300000-point capacity; export preserves every filtered point", file=sys.stderr, flush=True)
        # Publish the list only when the entire requested export is complete.
        (output / "data-list.txt").write_text("".join(frame["path"] + "\n" for frame in manifest["frames"]), encoding="utf-8")
        counts = [frame["point_count"] for frame in manifest["frames"]]
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                        point_count_summary={"min": min(counts), "max": max(counts), "mean": float(np.mean(counts)),
                                             "over_runtime_capacity_frames": sum(count > RUNTIME_POINT_CAPACITY for count in counts)})
        write_json(output / "manifest.json", manifest)
        print("Export complete: " + str(output / "data-list.txt"), flush=True)
        return 0
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("ERROR: " + str(error), file=sys.stderr, flush=True)
        sys.exit(1)
