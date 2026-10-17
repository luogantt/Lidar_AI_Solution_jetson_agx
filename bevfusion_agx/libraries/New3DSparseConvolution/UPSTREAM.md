# Source provenance and local changes

The `libspconv/include` and `libspconv/src` files were copied from
[riteshkhrn/Lidar_AI_Solution, New3DSparseConvolution](https://github.com/riteshkhrn/Lidar_AI_Solution/tree/3f823115cb16d9ddca9561394b9fa49242ab8315/libraries/New3DSparseConvolution)
at commit `3f823115cb16d9ddca9561394b9fa49242ab8315`.
No upstream binaries, generated protobuf files, models, or datasets were copied.

The complete upstream source archives for the compute backend are vendored in
this project, including their documentation, examples, embedded third-party
headers and existing license notices. No Git metadata is included:

| Local source | Version | Pinned upstream commit | Root license |
| --- | --- | --- | --- |
| [third_party/spconv](third_party/spconv) | 2.3.8 | [`263d6b47425ef843c82f997b12d8b714013d216c`](https://github.com/traveller59/spconv/tree/263d6b47425ef843c82f997b12d8b714013d216c) | [Apache-2.0](third_party/spconv/LICENSE) |
| [third_party/cumm](third_party/cumm) | 0.7.11 | [`b8d83b6511b70746d04485d0d2ec00896b735430`](https://github.com/FindDefinition/cumm/tree/b8d83b6511b70746d04485d0d2ec00896b735430) | [Apache-2.0](third_party/cumm/LICENSE) |

[third_party/source-manifest.json](third_party/source-manifest.json) records the
archive URLs, archive SHA-256 checksums, commits and original file/byte counts.
Individual files retain their own notices; the root licenses do not replace
embedded third-party terms or the compatibility layer's separate notices.

`build_deps.sh` copies these reviewed local trees to a writable dependency root,
generates C++/CUDA and builds the backend library named `libspconv.so`. It neither
downloads replacement source archives nor consumes NVIDIA's precompiled sparse
library. The default application backend is `open`; `nvidia` remains an explicit
build option. CUDA and TensorRT themselves remain NVIDIA dependencies.

Local patches belong in `third_party/spconv` or `third_party/cumm` and can be
reviewed in the project diff. `prepare_sources.py` fingerprints current file paths
and contents separately from the original archive provenance. It records these
identities in `_deps/source-state.json` and refuses an old dependency root after
source or manifest changes. It also verifies originally copied files before
reuse, allowing generated additions but rejecting edits to those copied inputs.
Use a new `NEWSCN_DEPS_ROOT` after changing sources; the script never silently
ignores a vendored patch or overwrites existing build work.

Python bytecode caches are excluded from source fingerprints. Git metadata,
compiled artifacts and common in-tree build directories are rejected in the
vendored trees. Python code-generation/build dependencies are still installed
with pip, so vendoring these sources alone does not make the entire build offline.

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
