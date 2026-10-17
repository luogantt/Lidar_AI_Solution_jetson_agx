# Source provenance and local changes

The `libspconv/include` and `libspconv/src` files were copied from
[riteshkhrn/Lidar_AI_Solution, New3DSparseConvolution](https://github.com/riteshkhrn/Lidar_AI_Solution/tree/3f823115cb16d9ddca9561394b9fa49242ab8315/libraries/New3DSparseConvolution)
at commit `3f823115cb16d9ddca9561394b9fa49242ab8315`.
No upstream binaries, generated protobuf files, models, or datasets were copied.

This implementation uses the source-generated backend from
`traveller59/spconv` tag `v2.3.8` and `FindDefinition/cumm` tag `v0.7.11`.
The resulting backend library is also named `libspconv.so`; it is built from
those sources and is distinct from the NVIDIA precompiled library.

Local changes fix dangling shape-formatting strings, missing enum-string
returns, and transpose shape inference. Runtime checks reject invalid sparse
tensor shapes, dtypes, grids, fixed-buffer capacities, unsupported inverse
convolutions, and unsupported dense layouts before kernel execution.
This integration intentionally enables FP16 execution only. The upstream INT8
paths remain unvalidated and are rejected by the engine builder.

ScatterDense now reports the actual permuted/reshaped dimensions, including
non-square XYZ/ZYX grids. Fusion checks graph edges, consumer counts and marked
outputs before combining nodes; unsupported unfused dense transforms fail
explicitly instead of silently changing the graph.

Existing copyright and license notices are retained verbatim. Upstream contains
mixed notices, including NVIDIA proprietary notices in some headers, and does
not provide an unambiguous license for every added implementation file. Copying
the source here does not change or resolve those terms.
