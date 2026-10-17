#!/usr/bin/env bash
# Build traveller59/spconv as a standalone C++ library, in an isolated directory.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
deps_root=${NEWSCN_DEPS_ROOT:-"${script_dir}/_deps"}
cuda_home=${CUDA_HOME:-/usr/local/cuda}
jobs=${BUILD_JOBS:-2}
python=${PYTHON:-python3}
export PATH="${cuda_home}/bin:${PATH}"
export CUDACXX="${cuda_home}/bin/nvcc"
export CUMM_CUDA_VERSION=${CUMM_CUDA_VERSION:-$(${CUDACXX} --version | sed -n 's/.*release \([0-9]*\.[0-9]*\),.*/\1/p')}
export CUMM_CUDA_ARCH_LIST=${CUMM_CUDA_ARCH_LIST:-8.6}
cmake_archs=${CMAKE_CUDA_ARCHITECTURES:-${CUMM_CUDA_ARCH_LIST//./}}
cmake_archs=${cmake_archs//+PTX/}
export CUMM_DISABLE_JIT=1 SPCONV_DISABLE_JIT=1
mkdir -p "${deps_root}"
deps_root=$(cd "${deps_root}" && pwd)

# Exact upstream revisions, matching New3DSparseConvolution's build_deps.sh.
spconv_rev=263d6b47425ef843c82f997b12d8b714013d216c # v2.3.8
cumm_rev=b8d83b6511b70746d04485d0d2ec00896b735430   # v0.7.11
fetch_source() {
  local repo=$1 rev=$2 dest=$3
  if [[ -f "${dest}/.source-revision" ]]; then
    [[ $(cat "${dest}/.source-revision") == "${rev}" ]] || { echo "Revision mismatch: ${dest}" >&2; exit 1; }
    return
  fi
  if [[ -e "${dest}" ]]; then
    echo "Unmanaged source directory exists: ${dest}; choose a fresh NEWSCN_DEPS_ROOT." >&2
    exit 1
  fi
  local archive="${deps_root}/${rev}.tar.gz"
  curl --fail --location --retry 3 --connect-timeout 20 "https://codeload.github.com/${repo}/tar.gz/${rev}" -o "${archive}.part"
  tar -tzf "${archive}.part" >/dev/null
  mv "${archive}.part" "${archive}"
  mkdir -p "${dest}"
  tar -xzf "${archive}" --strip-components=1 -C "${dest}"
  printf '%s\n' "${rev}" > "${dest}/.source-revision"
}
fetch_source traveller59/spconv "${spconv_rev}" "${deps_root}/spconv"
fetch_source FindDefinition/cumm "${cumm_rev}" "${deps_root}/cumm"

if [[ ! -x "${deps_root}/venv/bin/python" ]]; then
  "${python}" -m venv "${deps_root}/venv"
fi
python="${deps_root}/venv/bin/python"
"${python}" -m pip install 'numpy==1.26.4' 'pccm==0.4.16' 'ccimport==0.4.4' 'pybind11==3.1.0' 'fire==0.7.1' 'sympy==1.14.0' wheel setuptools
"${python}" -m pip install --no-build-isolation "${deps_root}/cumm"

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
  "${python}" -m pip freeze
} > "${deps_root}/build-manifest.txt"
echo "Open spconv built. Configure New3DSparseConvolution with -DNEWSCN_DEPS_ROOT=${deps_root}"
