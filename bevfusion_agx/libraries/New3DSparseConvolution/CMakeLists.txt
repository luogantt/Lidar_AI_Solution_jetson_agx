cmake_minimum_required(VERSION 3.18)
project(New3DSparseConvolution LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(NEWSCN_DEPS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/_deps" CACHE PATH "Isolated open spconv dependencies")
# Recompute these paths on every configure. Independent cache entries would
# silently retain the old backend when NEWSCN_DEPS_ROOT changes after a patch.
set(SPCONV_GENERATED_ROOT "${NEWSCN_DEPS_ROOT}/spconv/example/libspconv")
set(CUMM_SOURCE_ROOT "${NEWSCN_DEPS_ROOT}/cumm")
set(OPEN_SPCONV_LIBRARY "${SPCONV_GENERATED_ROOT}/build/spconv/src/libspconv.so")

find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_sources.py"
          --source-root "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
          --deps-root "${NEWSCN_DEPS_ROOT}" --check-only
  RESULT_VARIABLE NEWSCN_SOURCE_CHECK_RESULT
  OUTPUT_VARIABLE NEWSCN_SOURCE_CHECK_OUTPUT
  ERROR_VARIABLE NEWSCN_SOURCE_CHECK_ERROR)
if(NOT NEWSCN_SOURCE_CHECK_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Open backend source verification failed.\n${NEWSCN_SOURCE_CHECK_OUTPUT}${NEWSCN_SOURCE_CHECK_ERROR}\n"
    "Run libraries/New3DSparseConvolution/build_deps.sh with the same NEWSCN_DEPS_ROOT first.")
endif()

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
