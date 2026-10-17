#!/usr/bin/env bash
# Compare XYZ and ZYX BEV features against pinned upstream reference tensors.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:-"${script_dir}/build"}
fixtures=${2:-"${script_dir}/fixtures/bevfusion"}
output_dir=${3:-"${build_dir}/validation"}
python=${PYTHON:-python3}
mkdir -p "${output_dir}"
for order in xyz zyx; do
  grid=1440,1440,41
  [[ ${order} != zyx ]] || grid=41,1440,1440
  "${build_dir}/sparse_backend_check" \
    --model "${fixtures}/bevfusion.scn.${order}.onnx" \
    --voxels "${fixtures}/infer.${order}.voxels" \
    --coors "${fixtures}/infer.${order}.coors" --grid "${grid}" \
    --output "${output_dir}/${order}.dense" --warmup 2 --runs 10 \
    | tee "${output_dir}/${order}.log"
  "${python}" "${script_dir}/../../CUDA-BEVFusion/tool/compare_sparse_output.py" \
    "${output_dir}/${order}.dense" "${fixtures}/infer.${order}.dense" \
    --repeat "${output_dir}/${order}.dense.first" \
    --max-abs 0.1 --max-mae 0.001 --max-rmse 0.01 \
    --max-relative-l2 0.01 --min-cosine 0.9999 \
    --json-report "${output_dir}/${order}.metrics.json"
done
ldd "${build_dir}/libscn.so" | tee "${output_dir}/linked-libraries.txt"
