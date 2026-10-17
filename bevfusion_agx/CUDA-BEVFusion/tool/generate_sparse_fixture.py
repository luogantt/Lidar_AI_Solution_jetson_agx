#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate small non-square sparse ONNX models and independent NumPy references.

Requires NumPy and ONNX. Example:
  python tool/generate_sparse_fixture.py --output fixtures/non_square

Each manifest entry provides arguments for sparse_backend_check and the dense
reference for compare_sparse_output.py. Models use the repository's custom
sparse ONNX dialect; they are not standard ONNX Runtime convolution models.
"""

import argparse
import itertools
import json
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


DOMAIN = "bevfusion.sparse"
GRID_XYZ = (9, 7, 5)
TAILS = ("dense5d", "transpose5d", "reshape4d", "bev4d")


def save_tensor(path, array):
    """Write the little-endian nv::Tensor/spconv::Tensor file format."""
    array = np.ascontiguousarray(array)
    dtype_ids = {np.dtype("float16"): 2, np.dtype("int32"): 1}
    dtype_id = dtype_ids[array.dtype]
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    with Path(path).open("wb") as output:
        output.write(struct.pack("<iii", 0x33FF1101, array.ndim, dtype_id))
        output.write(struct.pack(f"<{array.ndim}i", *array.shape))
        output.write(array.tobytes(order="C"))


def sparse_convolution(features, coordinates, grid, weight, bias, stride, subm):
    """Cross-correlate by looking up each output's receptive field on the CPU.

    Coordinates and kernel axes have the same order. Submanifold convolution
    evaluates only the original active coordinates, preserving their order.
    Regular convolution activates an output if any receptive-field coordinate
    exists in the input, even when that input feature happens to be zero.
    Accumulation uses float32 and each layer's output is rounded to float16.
    """
    kernel = weight.shape[1:4]
    padding = tuple(size // 2 for size in kernel)
    output_grid = grid if subm else tuple(
        (size + 2 * pad - width) // step + 1
        for size, pad, width, step in zip(grid, padding, kernel, stride)
    )
    lookup = {tuple(coordinate): index for index, coordinate in enumerate(coordinates)}
    candidates = coordinates if subm else itertools.product(*(range(size) for size in output_grid))
    kernel_coordinates = list(itertools.product(*(range(size) for size in kernel)))
    weight32 = weight.astype(np.float32)
    features32 = features.astype(np.float32)
    output_coordinates, output_features = [], []
    for candidate in candidates:
        candidate = tuple(int(value) for value in candidate)
        total = bias.astype(np.float32).copy()
        active = subm
        for offset in kernel_coordinates:
            neighbor = tuple(
                point * step + delta - pad
                for point, step, delta, pad in zip(candidate, stride, offset, padding)
            )
            index = lookup.get(neighbor)
            if index is None:
                continue
            active = True
            total += weight32[(slice(None), *offset, slice(None))] @ features32[index]
        if active:
            output_coordinates.append(candidate)
            output_features.append(total.astype(np.float16))
    return (np.asarray(output_features, dtype=np.float16),
            np.asarray(output_coordinates, dtype=np.int32), tuple(output_grid))


def dense_reference(features, coordinates, grid):
    output = np.zeros((1, features.shape[1], *grid), dtype=np.float16)
    for feature, coordinate in zip(features, coordinates):
        output[(0, slice(None), *coordinate)] = feature
    return output


def node(op_type, inputs, outputs, name, **attributes):
    return helper.make_node(op_type, inputs, outputs, name=name, domain=DOMAIN, **attributes)


def conv_node(name, input_name, output_name, weight, bias, grid, output_grid, subm):
    stride = [1, 1, 1] if subm else [2, 2, 2]
    operation = node(
        "SparseConvolution", [input_name, f"{name}.weight", f"{name}.bias"], [output_name], name,
        kernel_size=[3, 3, 3], stride=stride, padding=[1, 1, 1], dilation=[1, 1, 1],
        subm=int(subm), inverse=0, output_bound=512, rulebook=name,
        precision="fp16", output_precision="fp16", activation="None",
        input_dynamic_range=0.0, input_spatial_shape=list(grid), output_spatial_shape=list(output_grid),
    )
    parameters = [numpy_helper.from_array(np.ascontiguousarray(weight), f"{name}.weight"),
                  numpy_helper.from_array(np.ascontiguousarray(bias), f"{name}.bias")]
    return operation, parameters


def make_tail(input_name, dense, grid, order, tail):
    operations = [node("ScatterDense", [input_name], ["dense"], "scatter",
                       input_spatial_shape=list(grid), output_shape=list(dense.shape),
                       output_layout="NCHW", format=order, input_dynamic_range=0.0)]
    result = dense
    output_name = "dense"
    if tail in ("transpose5d", "bev4d"):
        permutation = [0, 1, 4, 2, 3] if order == "xyz" else [0, 1, 2, 4, 3]
        operations.append(node("Transpose", [output_name], ["permuted"], "transpose", dims=permutation))
        output_name = "permuted"
        result = result.transpose(permutation)
    if tail in ("reshape4d", "bev4d"):
        shape = (result.shape[0], result.shape[1] * result.shape[2], result.shape[3], result.shape[4])
        operations.append(node("Reshape", [output_name], ["output"], "reshape", dims=list(shape)))
        output_name = "output"
        result = result.reshape(shape)
    return operations, output_name, np.ascontiguousarray(result)


def generate(output_dir, seed, include_residual, tails):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    all_coordinates = np.asarray(list(itertools.product(*(range(size) for size in GRID_XYZ))), dtype=np.int32)
    # Include corners to exercise boundary padding, then deliberately scramble input order.
    corners = list(itertools.product(*((0, size - 1) for size in GRID_XYZ)))
    remaining = [tuple(value) for value in all_coordinates if tuple(value) not in corners]
    # Keep holes after stride-two convolution to also check that dense scatter
    # clears inactive locations instead of leaking previous buffer contents.
    selected = [remaining[index] for index in rng.choice(len(remaining), 24, replace=False)] + corners
    coordinates_xyz = np.asarray(selected, dtype=np.int32)[rng.permutation(len(selected))]
    features = (rng.integers(-8, 9, size=(len(selected), 8)) / 16).astype(np.float16)
    weight_subm = (rng.integers(-2, 3, size=(8, 3, 3, 3, 8)) / 32).astype(np.float16)
    bias_subm = (rng.integers(-2, 3, size=8) / 64).astype(np.float16)
    weight_stride = (rng.integers(-2, 3, size=(16, 3, 3, 3, 8)) / 32).astype(np.float16)
    bias_stride = (rng.integers(-2, 3, size=16) / 64).astype(np.float16)
    manifest = {"seed": seed, "physical_grid_xyz": list(GRID_XYZ), "residual_add_relu": include_residual,
                "reference": "Independent NumPy coordinate lookup, float32 accumulation, FP16 layer outputs",
                "cases": []}
    bev_references = {}
    for order in ("xyz", "zyx"):
        grid = GRID_XYZ if order == "xyz" else GRID_XYZ[::-1]
        coordinates = coordinates_xyz.copy() if order == "xyz" else coordinates_xyz[:, ::-1].copy()
        first_weight = weight_subm if order == "xyz" else weight_subm.transpose(0, 3, 2, 1, 4)
        second_weight = weight_stride if order == "xyz" else weight_stride.transpose(0, 3, 2, 1, 4)
        first_features, first_coordinates, first_grid = sparse_convolution(
            features, coordinates, grid, first_weight, bias_subm, (1, 1, 1), True)
        operations, initializers = [], []
        operation, parameters = conv_node("subm", "input", "subm_features", first_weight, bias_subm,
                                           grid, first_grid, True)
        operations.append(operation)
        initializers.extend(parameters)
        sparse_input = "subm_features"
        if include_residual:
            first_features = np.maximum((first_features + features).astype(np.float16), np.float16(0))
            operations.extend([
                node("Add", [sparse_input, "input"], ["summed"], "residual_add",
                     precision="fp16", output_precision="fp16", input0_dynamic_range=0.0, input1_dynamic_range=0.0),
                node("Relu", ["summed"], ["activated"], "residual_relu")])
            sparse_input = "activated"
        final_features, final_coordinates, final_grid = sparse_convolution(
            first_features, first_coordinates, first_grid, second_weight, bias_stride, (2, 2, 2), False)
        operation, parameters = conv_node("stride2", sparse_input, "sparse_output", second_weight, bias_stride,
                                           first_grid, final_grid, False)
        operations.append(operation)
        initializers.extend(parameters)
        dense = dense_reference(final_features, final_coordinates, final_grid)
        voxels_path = output_dir / f"{order}.voxels"
        coordinates_path = output_dir / f"{order}.coors"
        save_tensor(voxels_path, features)
        save_tensor(coordinates_path, np.column_stack((np.zeros(len(coordinates), dtype=np.int32), coordinates)))
        for tail in tails:
            tail_nodes, output_name, reference = make_tail("sparse_output", dense, final_grid, order, tail)
            case_name = f"{order}_{tail}"
            model_path = output_dir / f"{case_name}.onnx"
            reference_path = output_dir / f"{case_name}.reference.dense"
            graph = helper.make_graph(
                operations + tail_nodes, case_name,
                [helper.make_tensor_value_info("input", TensorProto.FLOAT16, ["num_voxels", 8])],
                [helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, list(reference.shape))],
                initializer=initializers)
            model = helper.make_model(graph, producer_name="bevfusion_numpy_sparse_fixture",
                                      opset_imports=[helper.make_opsetid(DOMAIN, 1)])
            # This old, portable container version is sufficient for the custom dialect.
            model.ir_version = 7
            onnx.checker.check_model(model)
            onnx.save(model, model_path)
            save_tensor(reference_path, reference)
            case = {"name": case_name, "model": model_path.name, "voxels": voxels_path.name,
                    "coors": coordinates_path.name, "grid": list(grid), "reference": reference_path.name,
                    "output_shape": list(reference.shape), "input_points": len(features),
                    "output_points": len(final_features),
                    "inactive_dense_points": int(np.prod(final_grid)) - len(final_features)}
            manifest["cases"].append(case)
            print(f"{case_name}: grid={grid} output_shape={reference.shape} active_points={len(final_features)}")
            if tail == "bev4d":
                bev_references[order] = reference
    if len(bev_references) == 2:
        # Axis reversal should be a semantic identity, not a transpose of the final BEV.
        np.testing.assert_array_equal(bev_references["xyz"], bev_references["zyx"])
        manifest["xyz_zyx_bev_reference_equal"] = True
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Fixture destination directory")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--tail", choices=("all", *TAILS), default="all")
    parser.add_argument("--no-residual", action="store_true", help="Omit the residual Add/ReLU pair")
    args = parser.parse_args()
    generate(args.output, args.seed, not args.no_residual, TAILS if args.tail == "all" else (args.tail,))


if __name__ == "__main__":
    main()
