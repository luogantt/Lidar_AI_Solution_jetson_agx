cmake_minimum_required(VERSION 3.18)
project(New3DSparseConvolution LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(NEWSCN_DEPS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/_deps" CACHE PATH "Isolated open spconv dependencies")
set(SPCONV_GENERATED_ROOT "${NEWSCN_DEPS_ROOT}/spconv/example/libspconv" CACHE PATH "Generated spconv C++ project")
set(CUMM_SOURCE_ROOT "${NEWSCN_DEPS_ROOT}/cumm" CACHE PATH "cumm source with tensorview headers")
set(OPEN_SPCONV_LIBRARY "${SPCONV_GENERATED_ROOT}/build/spconv/src/libspconv.so" CACHE FILEPATH "Open source generated spconv shared library (not NVIDIA's library)")

foreach(required_path "${OPEN_SPCONV_LIBRARY}" "${SPCONV_GENERATED_ROOT}/spconv/include/spconvlib" "${CUMM_SOURCE_ROOT}/include/tensorview")
  if(NOT EXISTS "${required_path}")
    message(FATAL_ERROR "Missing open backend dependency: ${required_path}\nRun libraries/New3DSparseConvolution/build_deps.sh first.")
  endif()
endforeach()

add_library(open_spconv SHARED IMPORTED GLOBAL)
set_target_properties(open_spconv PROPERTIES IMPORTED_LOCATION "${OPEN_SPCONV_LIBRARY}")

add_library(newscn SHARED
  libspconv/src/engine.cpp
  libspconv/src/common.cpp
  libspconv/src/tensor.cu
  libspconv/src/sparseConvImplicit.cu
  libspconv/src/sparseAdd.cu
  libspconv/src/sparseFusedAddRelu.cu
  libspconv/src/sparseRelu.cu
  libspconv/src/sparseToDense.cu)
set_target_properties(newscn PROPERTIES
  OUTPUT_NAME scn
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(newscn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/libspconv/include"
  PRIVATE "${SPCONV_GENERATED_ROOT}/spconv/include" "${CUMM_SOURCE_ROOT}/include")
target_compile_definitions(newscn PRIVATE TV_CUDA)
target_compile_options(newscn PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=-fopenmp>)
target_link_libraries(newscn PRIVATE open_spconv CUDA::cudart CUDA::cublasLt OpenMP::OpenMP_CXX)
get_filename_component(OPEN_SPCONV_LIBRARY_DIR "${OPEN_SPCONV_LIBRARY}" DIRECTORY)
set_target_properties(newscn PROPERTIES BUILD_RPATH "${OPEN_SPCONV_LIBRARY_DIR}")

# Standalone verification intentionally needs neither TensorRT nor camera models.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(Protobuf REQUIRED)
  set(BEV_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../CUDA-BEVFusion/src")
  protobuf_generate_cpp(ONNX_SOURCES ONNX_HEADERS "${BEV_SOURCE}/onnx/onnx-ml.proto" "${BEV_SOURCE}/onnx/onnx-operators-ml.proto")
  add_executable(sparse_backend_check "${BEV_SOURCE}/sparse_backend_check.cpp"
    "${BEV_SOURCE}/bevfusion/lidar-scn-onnx-parser.cpp" ${ONNX_SOURCES})
  target_include_directories(sparse_backend_check PRIVATE "${BEV_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}")
  target_compile_definitions(sparse_backend_check PRIVATE BEVFUSION_SPARSE_BACKEND_OPEN=1)
  target_compile_features(sparse_backend_check PRIVATE cxx_std_17)
  target_link_libraries(sparse_backend_check PRIVATE newscn CUDA::cudart protobuf::libprotobuf)
endif()
