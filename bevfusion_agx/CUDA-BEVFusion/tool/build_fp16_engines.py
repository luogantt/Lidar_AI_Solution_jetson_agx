#!/usr/bin/env python3
"""Build all four ResNet50 BEVFusion TensorRT engines without trtexec.

Requires Linux, an NVIDIA driver, TensorRT 10.x (API target: 10.13.3), and
``onnx`` in the selected Python environment. Example, from CUDA-BEVFusion:

    python3 tool/build_fp16_engines.py --model-dir model/resnet50
    python3 tool/build_fp16_engines.py --model-dir model/resnet50 --reuse

The default output is <model-dir>/build. Every invocation rebuilds every engine
unless --reuse is given. Reuse requires a matching per-engine manifest, source
and external-weight hashes, TensorRT version, GPU UUID, build settings, plugin
hashes, plan hash, and a successful deserialization / I/O validation.

This matches build_trt_engine.sh's --fp16, fp16:chw I/O and 2048 MiB workspace.
FP16 enables mixed-precision tactics; it does not force every internal layer to
FP16. No inference benchmark is performed. Build timings are not inference
latencies. Custom ONNX plugins can be loaded with repeated --plugin arguments;
the same libraries must also be loaded by the inference application.

TensorRT API reference pinned to the matching upstream release:
https://github.com/NVIDIA/TensorRT/tree/v10.13.3/python/src/infer
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
import uuid


SCHEMA_VERSION = 1
# Keep names, order and I/O counts aligned with tool/build_trt_engine.sh.
MODELS = (
    ("camera.backbone", 2, 2),
    ("fuser", 2, 1),
    ("camera.vtransform", 1, 1),
    ("head.bbox", 1, 6),
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, data):
    """Publish a complete file; an interrupted build cannot publish half a plan."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path, value):
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def tensor_protos(message):
    """Visit external weights in initializers, attributes and nested graphs."""
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        yield message
        return
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            repeated = field.is_repeated if hasattr(field, "is_repeated") else field.label == field.LABEL_REPEATED
            if repeated:
                for child in value:
                    yield from tensor_protos(child)
            else:
                yield from tensor_protos(value)


def source_identity(path, onnx):
    model = onnx.load_model(str(path), load_external_data=False)
    external_paths = {}
    for tensor in tensor_protos(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            entries = {entry.key: entry.value for entry in tensor.external_data}
            location = entries.get("location")
            if not location:
                raise ValueError("External ONNX tensor has no location: " + tensor.name)
            external_paths[location] = (path.parent / location).resolve(strict=True)
    return {
        "onnx_sha256": sha256_file(path),
        "external_data": [
            {"location": location, "sha256": sha256_file(external), "bytes": external.stat().st_size}
            for location, external in sorted(external_paths.items())
        ],
    }


class CudaDevice:
    """Own a CUDA primary context without a torch/cuda-python dependency."""

    def __init__(self, ordinal):
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.device = ctypes.c_int()
        self.context = ctypes.c_void_p()
        self.retained = False
        self.pushed = False
        self._bind("cuInit", [ctypes.c_uint])
        self._bind("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)])
        self._bind("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
        self._bind("cuDeviceGetName", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int])
        self._bind("cuDeviceComputeCapability", [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int])
        self._bind("cuDriverGetVersion", [ctypes.POINTER(ctypes.c_int)])
        self._bind("cuDevicePrimaryCtxRetain", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int])
        self._bind("cuDevicePrimaryCtxRelease_v2", [ctypes.c_int])
        self._bind("cuCtxPushCurrent_v2", [ctypes.c_void_p])
        self._bind("cuCtxPopCurrent_v2", [ctypes.POINTER(ctypes.c_void_p)])
        self._bind("cuGetErrorString", [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)])
        self.uuid_function = "cuDeviceGetUuid_v2" if hasattr(self.driver, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
        self._bind(self.uuid_function, [ctypes.c_void_p, ctypes.c_int])
        self._call("cuInit", 0)
        count = ctypes.c_int()
        self._call("cuDeviceGetCount", ctypes.byref(count))
        if not 0 <= ordinal < count.value:
            raise ValueError("--device {} is outside {} visible CUDA device(s)".format(ordinal, count.value))
        self._call("cuDeviceGet", ctypes.byref(self.device), ordinal)
        name = ctypes.create_string_buffer(256)
        gpu_uuid = (ctypes.c_ubyte * 16)()
        major, minor, driver_version = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self._call("cuDeviceGetName", name, len(name), self.device)
        self._call(self.uuid_function, gpu_uuid, self.device)
        self._call("cuDeviceComputeCapability", ctypes.byref(major), ctypes.byref(minor), self.device)
        self._call("cuDriverGetVersion", ctypes.byref(driver_version))
        self.identity = {
            "name": name.value.decode("utf-8"),
            "uuid": "GPU-" + str(uuid.UUID(bytes=bytes(gpu_uuid))),
            "compute_capability": [major.value, minor.value],
            "cuda_driver_version": driver_version.value,
        }

    def _bind(self, name, argument_types):
        function = getattr(self.driver, name)
        function.argtypes = argument_types
        function.restype = ctypes.c_int

    def _call(self, name, *arguments):
        status = getattr(self.driver, name)(*arguments)
        if status != 0:
            description = ctypes.c_char_p()
            self.driver.cuGetErrorString(status, ctypes.byref(description))
            error = description.value.decode("utf-8") if description.value else "unknown CUDA error"
            raise RuntimeError("{} failed ({}): {}".format(name, status, error))

    def __enter__(self):
        self._call("cuDevicePrimaryCtxRetain", ctypes.byref(self.context), self.device)
        self.retained = True
        try:
            self._call("cuCtxPushCurrent_v2", self.context)
            self.pushed = True
        except Exception:
            self._call("cuDevicePrimaryCtxRelease_v2", self.device)
            self.retained = False
            raise
        return self

    def __exit__(self, exception_type, exception, traceback):
        try:
            if self.pushed:
                previous = ctypes.c_void_p()
                self._call("cuCtxPopCurrent_v2", ctypes.byref(previous))
        finally:
            if self.retained:
                self._call("cuDevicePrimaryCtxRelease_v2", self.device)


def static_shape(shape, description):
    result = [int(dimension) for dimension in shape]
    if not result or any(dimension <= 0 for dimension in result):
        raise ValueError("{} must have a nonempty, positive static shape; got {}".format(description, result))
    return result


def configure_network(network, trt, model_name, expected_inputs, expected_outputs):
    actual = (network.num_inputs, network.num_outputs)
    if actual != (expected_inputs, expected_outputs):
        raise ValueError("{} I/O count: expected {}/{}, got {}/{}".format(
            model_name, expected_inputs, expected_outputs, *actual))
    tensors = []
    for mode, count, getter in (("INPUT", network.num_inputs, network.get_input),
                                 ("OUTPUT", network.num_outputs, network.get_output)):
        for index in range(count):
            tensor = getter(index)
            shape = static_shape(tensor.shape, "{} {} {}".format(model_name, mode, tensor.name))
            if tensor.is_shape_tensor or tensor.dtype not in (trt.float32, trt.float16):
                raise ValueError("{} {} is not a floating-point execution tensor".format(model_name, tensor.name))
            tensor.dtype = trt.float16
            # trtexec fp16:chw is an unvectorized, contiguous TensorFormat.LINEAR.
            tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
            tensors.append({"name": tensor.name, "mode": mode, "shape": shape, "dtype": "HALF", "format": "LINEAR"})
    return tensors


def inspect_plan(data, trt, logger, expected_io, expected_inputs, expected_outputs):
    runtime = trt.Runtime(logger)
    engine = None
    inspector = None
    try:
        engine = runtime.deserialize_cuda_engine(data)
        if engine is None:
            raise RuntimeError("TensorRT could not deserialize the engine it built / cached")
        io = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            if mode not in (trt.TensorIOMode.INPUT, trt.TensorIOMode.OUTPUT):
                raise ValueError("Unexpected I/O mode for " + name)
            if engine.get_tensor_dtype(name) != trt.float16:
                raise ValueError("Engine tensor is not FP16: " + name)
            if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError("Engine tensor is not LINEAR / CHW: " + name)
            if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE or engine.is_shape_inference_io(name):
                raise ValueError("Engine tensor is not a device execution tensor: " + name)
            io.append({"name": name, "mode": "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT",
                       "shape": static_shape(engine.get_tensor_shape(name), "Engine tensor " + name),
                       "dtype": "HALF", "format": "LINEAR"})
        counts = (sum(tensor["mode"] == "INPUT" for tensor in io), sum(tensor["mode"] == "OUTPUT" for tensor in io))
        if counts != (expected_inputs, expected_outputs):
            raise ValueError("Engine I/O count mismatch: {}".format(counts))
        if sorted(io, key=lambda tensor: tensor["name"]) != sorted(expected_io, key=lambda tensor: tensor["name"]):
            raise ValueError("Engine tensor names, modes or shapes differ from the parsed ONNX / manifest")
        inspector = engine.create_engine_inspector()
        layers_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        return {"io": io, "device_memory_bytes": int(engine.device_memory_size_v2)}, layers_text
    finally:
        # The engine must be destroyed before its runtime and before CUDA teardown.
        del inspector
        del engine
        del runtime


def reuse_engine(plan_path, manifest_path, signature, trt, logger, expected_inputs, expected_outputs):
    if not plan_path.is_file() or not manifest_path.is_file():
        return None, "plan or manifest is missing"
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("status") != "complete":
            return None, "manifest is incomplete or uses a different schema"
        if metadata.get("signature") != signature:
            return None, "source, TensorRT, GPU, plugin or build settings changed"
        if sha256_file(plan_path) != metadata["engine_sha256"]:
            return None, "plan SHA256 does not match its manifest"
        inspect_plan(plan_path.read_bytes(), trt, logger, metadata["io"], expected_inputs, expected_outputs)
        return metadata, None
    except Exception as error:
        return None, "cached engine validation failed: {}".format(error)


def build_engine(onnx_path, plan_path, manifest_path, signature, trt, logger, expected_inputs, expected_outputs, workspace_mib):
    started_at = timestamp()
    started = time.perf_counter()
    # Invalidate an old manifest before trying a rebuild. A failed build must not
    # make an old plan look like the output of this invocation.
    write_json(manifest_path, {"schema_version": SCHEMA_VERSION, "status": "building",
                              "started_at": started_at, "signature": signature})
    builder = network = config = parser = serialized = None
    try:
        builder = trt.Builder(logger)
        # TensorRT 10 always uses explicit batch. Do not enable STRONGLY_TYPED:
        # the original trtexec pipeline uses mixed precision and FP16 boundary I/O.
        network = builder.create_network(0)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, logger)
        parse_started = time.perf_counter()
        if not parser.parse_from_file(str(onnx_path)):
            errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
            raise RuntimeError("ONNX parser failed for {}:\n{}".format(onnx_path, "\n".join(errors)))
        expected_io = configure_network(network, trt, onnx_path.stem, expected_inputs, expected_outputs)
        parse_seconds = time.perf_counter() - parse_started
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.TF32)  # Also enabled by default in trtexec.
        config.builder_optimization_level = 3
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        build_started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        build_seconds = time.perf_counter() - build_started
        if serialized is None:
            raise RuntimeError("TensorRT returned an empty serialized engine for " + str(onnx_path))
        data = bytes(serialized)
        if not data:
            raise RuntimeError("TensorRT returned an empty serialized engine for " + str(onnx_path))
        # Keep the parser and network alive until serialization finishes.
        serialized = parser = config = network = builder = None
        engine_info, layers_text = inspect_plan(data, trt, logger, expected_io, expected_inputs, expected_outputs)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "signature": signature,
            "source_path": str(onnx_path),
            "plan_path": str(plan_path),
            "started_at": started_at,
            "completed_at": timestamp(),
            "parse_seconds": parse_seconds,
            "build_seconds": build_seconds,
            "total_seconds": time.perf_counter() - started,
            "engine_bytes": len(data),
            "engine_sha256": hashlib.sha256(data).hexdigest(),
            **engine_info,
        }
        atomic_write(plan_path, data)
        atomic_write(plan_path.with_suffix(".layers.json"), layers_text.encode("utf-8"))
        write_json(manifest_path, metadata)
        return metadata
    except Exception as error:
        write_json(manifest_path, {"schema_version": SCHEMA_VERSION, "status": "failed",
                                  "signature": signature, "started_at": started_at,
                                  "failed_at": timestamp(), "error": str(error)})
        raise
    finally:
        # Drop CUDA-owning objects even when an exception traceback is retained.
        serialized = parser = config = network = builder = None


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "model" / "resnet50",
                        help="directory containing the four ONNX files (default: CUDA-BEVFusion/model/resnet50)")
    parser.add_argument("--output-dir", type=Path, help="engine and manifest directory (default: <model-dir>/build)")
    parser.add_argument("--device", type=int, default=0, help="CUDA visible device ordinal (default: 0)")
    parser.add_argument("--workspace-mib", type=int, default=2048, help="workspace limit in MiB (default: 2048)")
    parser.add_argument("--plugin", type=Path, action="append", default=[], help="custom plugin shared library; repeat as needed")
    parser.add_argument("--reuse", action="store_true", help="reuse only engines with matching, validated manifests")
    parser.add_argument("--verbose", action="store_true", help="enable verbose TensorRT console logging")
    result = parser.parse_args(argv)
    if result.workspace_mib <= 0 or result.device < 0:
        parser.error("--workspace-mib must be positive and --device must be nonnegative")
    return result


def main(argv=None):
    args = arguments(argv)
    if platform.system() != "Linux":
        raise RuntimeError("Run this builder on the target Linux NVIDIA GPU host")
    try:
        import onnx
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("The selected Python environment needs onnx and TensorRT 10.x: " + str(error)) from error
    if trt.__version__.split(".")[0] != "10":
        raise RuntimeError("This script targets TensorRT 10.x; found " + trt.__version__)
    model_dir = args.model_dir.resolve(strict=True)
    output_dir = (args.output_dir or model_dir / "build").resolve()
    source_files = {name: (model_dir / (name + ".onnx")).resolve(strict=True) for name, _, _ in MODELS}
    sources = {name: source_identity(path, onnx) for name, path in source_files.items()}
    plugin_paths = [path.resolve(strict=True) for path in args.plugin]
    plugin_signatures = [{"name": path.name, "sha256": sha256_file(path)} for path in plugin_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "build_fp16_manifest.json"
    run = {"schema_version": SCHEMA_VERSION, "status": "building", "started_at": timestamp(),
           "model_dir": str(model_dir), "output_dir": str(output_dir), "engines": []}
    write_json(run_path, run)
    started = time.perf_counter()
    try:
        with CudaDevice(args.device) as device:
            logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
            if not trt.init_libnvinfer_plugins(logger, ""):
                raise RuntimeError("TensorRT's standard plugin initialization failed")
            # Retain handles for the whole build/deserialization lifetime.
            plugins = [ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL) for path in plugin_paths]
            settings = {
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "tensorrt_version": trt.__version__,
                "onnx_version": onnx.__version__,
                "gpu": device.identity,
                "platform": {"system": platform.system(), "machine": platform.machine()},
                "workspace_bytes": args.workspace_mib * 1024 * 1024,
                "builder_flags": ["FP16", "TF32"],
                "builder_optimization_level": 3,
                "io_dtype": "HALF", "io_format": "LINEAR", "profiling_verbosity": "DETAILED",
                "plugins": plugin_signatures,
            }
            run["build_environment"] = settings
            print("Building on {} ({}, SM{}{}) with TensorRT {}".format(
                device.identity["name"], device.identity["uuid"], *device.identity["compute_capability"], trt.__version__), flush=True)
            for name, expected_inputs, expected_outputs in MODELS:
                plan_path = output_dir / (name + ".plan")
                manifest_path = output_dir / (name + ".manifest.json")
                signature = {**settings, "model": name, "source": sources[name],
                             "expected_inputs": expected_inputs, "expected_outputs": expected_outputs}
                metadata = None
                reused = False
                stage_started = time.perf_counter()
                if args.reuse:
                    metadata, reason = reuse_engine(plan_path, manifest_path, signature, trt, logger, expected_inputs, expected_outputs)
                    reused = metadata is not None
                    if not reused:
                        print("{}: rebuilding ({})".format(name, reason), flush=True)
                if not reused:
                    print("{}: building FP16, static I/O, {} MiB workspace".format(name, args.workspace_mib), flush=True)
                    metadata = build_engine(source_files[name], plan_path, manifest_path, signature, trt, logger,
                                            expected_inputs, expected_outputs, args.workspace_mib)
                run["engines"].append({"name": name, "reused": reused, "plan_path": str(plan_path),
                                       "manifest_path": str(manifest_path), "engine_sha256": metadata["engine_sha256"],
                                       "build_seconds": metadata["build_seconds"], "io": metadata["io"],
                                       "invocation_seconds": time.perf_counter() - stage_started})
                write_json(run_path, run)
                print("{}: {} ({:.3f} s original build), {} bytes".format(
                    name, "validated cache" if reused else "built and validated", metadata["build_seconds"], metadata["engine_bytes"]), flush=True)
                for tensor in metadata["io"]:
                    print("  {mode} {name}: {shape} {dtype} {format}".format(**tensor), flush=True)
            del plugins
        run.update(status="complete", completed_at=timestamp(), total_seconds=time.perf_counter() - started)
        write_json(run_path, run)
        print("All four engines are ready. Manifest: " + str(run_path), flush=True)
        return 0
    except Exception as error:
        run.update(status="failed", failed_at=timestamp(), error=str(error), total_seconds=time.perf_counter() - started)
        write_json(run_path, run)
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("ERROR: " + str(error), file=sys.stderr, flush=True)
        sys.exit(1)
