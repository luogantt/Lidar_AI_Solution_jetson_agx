#!/usr/bin/env bash
# Build the vendored spconv/cumm sources as an isolated standalone C++ library.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
deps_root=${NEWSCN_DEPS_ROOT:-"${script_dir}/_deps"}
cuda_home=${CUDA_HOME:-/usr/local/cuda}
jobs=${BUILD_JOBS:-2}
python=${PYTHON:-python3}
# Keep unrelated Python installations from overriding the local generator/cumm.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
# Validate and copy local reviewed sources before invoking CUDA, pip or CMake.
# A legacy _deps, changed vendored patch, or edited copied source is rejected.
"${python}" "${script_dir}/prepare_sources.py" \
  --source-root "${script_dir}/third_party" --deps-root "${deps_root}"
deps_root=$(cd "${deps_root}" && pwd)
export PATH="${cuda_home}/bin:${PATH}"
export CUDACXX="${cuda_home}/bin/nvcc"
export CUMM_CUDA_VERSION=${CUMM_CUDA_VERSION:-$(${CUDACXX} --version | sed -n 's/.*release \([0-9]*\.[0-9]*\),.*/\1/p')}
export CUMM_CUDA_ARCH_LIST=${CUMM_CUDA_ARCH_LIST:-8.6}
cmake_archs=${CMAKE_CUDA_ARCHITECTURES:-${CUMM_CUDA_ARCH_LIST//./}}
cmake_archs=${cmake_archs//+PTX/}
export CUMM_DISABLE_JIT=1 SPCONV_DISABLE_JIT=1

# Exact upstream revisions are also verified against third_party/source-manifest.json.
spconv_rev=263d6b47425ef843c82f997b12d8b714013d216c # v2.3.8
cumm_rev=b8d83b6511b70746d04485d0d2ec00896b735430   # v0.7.11

if [[ ! -x "${deps_root}/venv/bin/python" ]] || \
   ! "${deps_root}/venv/bin/python" -m pip --version >/dev/null 2>&1; then
  # A failed ensurepip (for example before installing python3-venv) can leave
  # bin/python behind. Repair that partial environment on the next invocation.
  "${python}" -m venv "${deps_root}/venv"
fi
python="${deps_root}/venv/bin/python"
"${python}" -m pip install 'numpy==1.26.4' 'pccm==0.4.16' 'ccimport==0.4.4' 'pybind11==3.1.0' 'fire==0.7.1' 'sympy==1.14.0' wheel setuptools
"${python}" -m pip install --no-build-isolation --no-deps --force-reinstall "${deps_root}/cumm"

generated="${deps_root}/spconv/example/libspconv"
printf "__version__ = '2.3.8'\n" > "${deps_root}/spconv/spconv/__version__.py"
(
  cd "${deps_root}/spconv"
  "${python}" -m spconv.gencode --include="${generated}/spconv/include" --src="${generated}/spconv/src" --inference_only=True
)
# The upstream example expects the cumm source at this location.
if [[ ! -e "${generated}/cumm" ]]; then
  ln -s "${deps_root}/cumm" "${generated}/cumm"
fi
cmake -S "${generated}" -B "${generated}/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
  -DCMAKE_CUDA_ARCHITECTURES="${cmake_archs}" \
  -DCMAKE_CXX_FLAGS=-DTV_CUDA
cmake --build "${generated}/build" --target spconv --parallel "${jobs}"
{
  printf 'spconv=%s\ncumm=%s\ncuda=%s\narchitectures=%s\n' "${spconv_rev}" "${cumm_rev}" "${CUMM_CUDA_VERSION}" "${CUMM_CUDA_ARCH_LIST}"
  printf 'source_state=%s\n' "${deps_root}/source-state.json"
  sha256sum "${deps_root}/source-state.json" "${script_dir}/build_deps.sh" "${script_dir}/prepare_sources.py"
  "${python}" -m pip freeze
} > "${deps_root}/build-manifest.txt"
echo "Open spconv built. Configure New3DSparseConvolution with -DNEWSCN_DEPS_ROOT=${deps_root}"
