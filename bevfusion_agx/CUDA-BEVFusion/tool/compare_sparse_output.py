#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate dense sparse-backbone results, using NumPy only.

Example:
  python tool/compare_sparse_output.py output.dense reference.dense \
      --repeat output.dense.first --json-report metrics.json

Shapes must match exactly; no sorting, flatten-only shape matching or implicit
axis permutation is allowed. Thresholds are regression criteria, not a claim
about downstream detection accuracy. Exit 0 means all checks passed, 1 means a
numerical mismatch, and 2 means invalid input or arguments.
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np


DTYPES = {1: "<i4", 2: "<f2", 3: "<f4", 4: "<i8", 5: "<u8", 6: "<u4", 7: "i1", 8: "u1"}


def load_tensor(path):
    """Read the shared nv::Tensor/spconv::Tensor format, or a .npy array."""
    path = Path(path)
    if path.suffix == ".npy":
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        with path.open("rb") as source:
            header = source.read(12)
            if len(header) != 12:
                raise ValueError(f"Truncated tensor header: {path}")
            magic, ndim, dtype_id = struct.unpack("<iii", header)
            if magic != 0x33FF1101 or not 1 <= ndim <= 16 or dtype_id not in DTYPES:
                raise ValueError(f"Invalid tensor header: {path}")
            dimensions = source.read(4 * ndim)
            if len(dimensions) != 4 * ndim:
                raise ValueError(f"Truncated tensor shape: {path}")
            shape = struct.unpack(f"<{ndim}i", dimensions)
            if any(dimension <= 0 for dimension in shape):
                raise ValueError(f"Tensor dimensions must be positive: {shape}")
            dtype = np.dtype(DTYPES[dtype_id])
            offset = 12 + 4 * ndim
            expected = offset + math.prod(shape) * dtype.itemsize
            if os.fstat(source.fileno()).st_size != expected:
                raise ValueError(f"Tensor payload size mismatch: {path}")
        value = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)
    if value.dtype.kind != "f" or value.dtype.itemsize not in (2, 4, 8):
        raise ValueError(f"Expected floating-point output, got {value.dtype}: {path}")
    if value.size == 0:
        raise ValueError(f"Empty tensor: {path}")
    return value


def compare(actual, reference, chunk_size=1_000_000):
    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape}, reference {reference.shape}")
    actual_flat = actual.reshape(-1)
    reference_flat = reference.reshape(-1)
    sum_abs = sum_squared = actual_squared = reference_squared = dot = max_abs = 0.0
    nonfinite_actual = nonfinite_reference = 0
    for start in range(0, actual.size, chunk_size):
        x = np.asarray(actual_flat[start:start + chunk_size], dtype=np.float64)
        y = np.asarray(reference_flat[start:start + chunk_size], dtype=np.float64)
        bad_x = int(np.count_nonzero(~np.isfinite(x)))
        bad_y = int(np.count_nonzero(~np.isfinite(y)))
        nonfinite_actual += bad_x
        nonfinite_reference += bad_y
        if bad_x or bad_y:
            continue
        difference = x - y
        max_abs = max(max_abs, float(np.max(np.abs(difference))))
        sum_abs += float(np.sum(np.abs(difference)))
        sum_squared += float(np.dot(difference, difference))
        actual_squared += float(np.dot(x, x))
        reference_squared += float(np.dot(y, y))
        dot += float(np.dot(x, y))
    result = {
        "shape": list(actual.shape), "actual_dtype": str(actual.dtype), "reference_dtype": str(reference.dtype),
        "numel": int(actual.size), "nonfinite_actual": nonfinite_actual, "nonfinite_reference": nonfinite_reference,
    }
    if nonfinite_actual or nonfinite_reference:
        result.update(max_abs=None, mae=None, rmse=None, relative_l2=None, cosine=None)
        return result
    relative_l2 = math.sqrt(sum_squared / reference_squared) if reference_squared else (0.0 if sum_squared == 0 else math.inf)
    if actual_squared == 0 or reference_squared == 0:
        cosine = 1.0 if actual_squared == reference_squared else 0.0
    else:
        cosine = max(-1.0, min(1.0, dot / math.sqrt(actual_squared * reference_squared)))
    result.update(max_abs=max_abs, mae=sum_abs / actual.size, rmse=math.sqrt(sum_squared / actual.size),
                  relative_l2=relative_l2, cosine=cosine)
    return result


def evaluate(metrics, thresholds):
    failures = []
    if metrics["nonfinite_actual"] or metrics["nonfinite_reference"]:
        return ["Non-finite values present; metrics are not valid"]
    for name in ("max_abs", "mae", "rmse", "relative_l2"):
        if not math.isfinite(metrics[name]) or metrics[name] > thresholds[name]:
            failures.append(f"{name}={metrics[name]:.9g} > {thresholds[name]:.9g}")
    if metrics["cosine"] < thresholds["cosine"]:
        failures.append(f"cosine={metrics['cosine']:.9g} < {thresholds['cosine']:.9g}")
    return failures


def print_result(label, metrics, failures):
    print(f"{label}: {'FAIL' if failures else 'PASS'} shape={metrics['shape']} "
          f"actual_dtype={metrics['actual_dtype']} reference_dtype={metrics['reference_dtype']}")
    for name in ("max_abs", "mae", "rmse", "relative_l2", "cosine"):
        print(f"  {name}: {metrics[name]}")
    print(f"  nonfinite: actual={metrics['nonfinite_actual']}, reference={metrics['nonfinite_reference']}")
    for failure in failures:
        print(f"  {failure}")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actual", help="Backend dense output (.tensor/.dense or .npy)")
    parser.add_argument("reference", help="Upstream/PyTorch dense reference in exactly the same layout")
    parser.add_argument("--max-abs", type=float, default=0.1)
    parser.add_argument("--max-mae", type=float, default=0.001)
    parser.add_argument("--max-rmse", type=float, default=0.01)
    parser.add_argument("--max-relative-l2", type=float, default=0.01)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--repeat", action="append", default=[], help="Earlier output(s), checked against actual; may repeat")
    parser.add_argument("--repeat-atol", type=float, default=0.0, help="Maximum absolute change across runs; default exact")
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args(argv)
    thresholds = {"max_abs": args.max_abs, "mae": args.max_mae, "rmse": args.max_rmse,
                  "relative_l2": args.max_relative_l2, "cosine": args.min_cosine}
    if any(not math.isfinite(value) or value < 0 for key, value in thresholds.items() if key != "cosine"):
        parser.error("Error thresholds must be finite and non-negative")
    if not math.isfinite(args.min_cosine) or not -1 <= args.min_cosine <= 1:
        parser.error("--min-cosine must be finite and in [-1, 1]")
    if not math.isfinite(args.repeat_atol) or args.repeat_atol < 0:
        parser.error("--repeat-atol must be finite and non-negative")
    try:
        actual = load_tensor(args.actual)
        metrics = compare(actual, load_tensor(args.reference))
        failures = evaluate(metrics, thresholds)
        print_result("reference", metrics, failures)
        report = {"actual": args.actual, "reference": args.reference, "thresholds": thresholds,
                  "metrics": metrics, "failures": failures, "repeats": []}
        passed = not failures
        for path in args.repeat:
            repeat_metrics = compare(actual, load_tensor(path))
            repeat_failures = []
            if repeat_metrics["nonfinite_actual"] or repeat_metrics["nonfinite_reference"]:
                repeat_failures.append("Non-finite values present")
            elif repeat_metrics["max_abs"] > args.repeat_atol:
                repeat_failures.append(f"max_abs={repeat_metrics['max_abs']:.9g} > repeat_atol={args.repeat_atol:.9g}")
            print_result(f"repeat {path}", repeat_metrics, repeat_failures)
            report["repeats"].append({"path": path, "metrics": repeat_metrics, "failures": repeat_failures})
            passed = passed and not repeat_failures
        report["passed"] = passed
        if args.json_report:
            args.json_report.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return 0 if passed else 1
    except (OSError, ValueError, OverflowError) as error:
        print(f"compare_sparse_output: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
