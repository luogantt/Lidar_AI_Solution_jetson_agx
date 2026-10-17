# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.10)
project(bevfusion LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(_sparse_backend_default open)
if(DEFINED ENV{BEVFUSION_SPARSE_BACKEND})
  set(_sparse_backend_default "$ENV{BEVFUSION_SPARSE_BACKEND}")
endif()
set(BEVFUSION_SPARSE_BACKEND "${_sparse_backend_default}" CACHE STRING "Sparse backend: nvidia or open")
set_property(CACHE BEVFUSION_SPARSE_BACKEND PROPERTY STRINGS nvidia open)
if(NOT BEVFUSION_SPARSE_BACKEND MATCHES "^(nvidia|open)$")
  message(FATAL_ERROR "BEVFUSION_SPARSE_BACKEND must be nvidia or open")
endif()
if(BEVFUSION_SPARSE_BACKEND STREQUAL "open" AND CMAKE_VERSION VERSION_LESS 3.18)
  message(FATAL_ERROR "The open sparse backend requires CMake 3.18 or newer")
endif()
message(STATUS "BEVFusion sparse backend: ${BEVFUSION_SPARSE_BACKEND}")

option(CMAKE_BUILD_PYTHON "Build the libpybev Python module" "$ENV{USE_Python}")
set(_spconv_cuda_default 11.4)
if(DEFINED ENV{SPCONV_CUDA_VERSION})
  set(_spconv_cuda_default "$ENV{SPCONV_CUDA_VERSION}")
endif()
set(SPCONV_CUDA_VERSION "${_spconv_cuda_default}" CACHE STRING "NVIDIA prebuilt libspconv CUDA directory suffix")
set(arch "${CMAKE_SYSTEM_PROCESSOR}")

find_package(CUDA REQUIRED)
set(Protobuf_USE_STATIC_LIBS OFF)
find_package(Protobuf REQUIRED)
if(NOT Protobuf_PROTOC_EXECUTABLE)
  message(FATAL_ERROR "protoc is required to generate ONNX protobuf sources")
endif()

if(BEVFUSION_SPARSE_BACKEND STREQUAL "nvidia" AND SPCONV_CUDA_VERSION STREQUAL "11.4")
  set(CMAKE_CXX_STANDARD 14)
else()
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD "${CMAKE_CXX_STANDARD}")
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -Wno-unused-parameter")
list(APPEND CUDA_NVCC_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
set(BEVFUSION_CUDA_ARCH "$ENV{CUDASM}" CACHE STRING "CUDA compute capability without dot, for example 86")
if(BEVFUSION_CUDA_ARCH)
  if(NOT BEVFUSION_CUDA_ARCH MATCHES "^[0-9]+$")
    message(FATAL_ERROR "BEVFUSION_CUDA_ARCH must be a numeric compute capability such as 86")
  endif()
  list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${BEVFUSION_CUDA_ARCH},code=sm_${BEVFUSION_CUDA_ARCH}"
                             "-gencode=arch=compute_${BEVFUSION_CUDA_ARCH},code=compute_${BEVFUSION_CUDA_ARCH}")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${BEVFUSION_CUDA_ARCH}" CACHE STRING "CUDA architectures")
  endif()
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS "$ENV{TensorRT_Inc}")
find_library(TENSORRT_LIBRARY nvinfer HINTS "$ENV{TensorRT_Lib}")
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin HINTS "$ENV{TensorRT_Lib}")
find_library(CUBLASLT_LIBRARY cublasLt HINTS "$ENV{CUDA_Lib}" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_PLUGIN_LIBRARY)
  message(FATAL_ERROR "TensorRT headers/libraries are missing; set TensorRT_Inc and TensorRT_Lib")
endif()
if(NOT CUBLASLT_LIBRARY)
  message(FATAL_ERROR "cuBLASLt was not found; set CUDA_Lib to the CUDA library directory")
endif()

# Generate with the installed protobuf toolchain instead of using stale checked-in .pb.cc files.
set(PROTOBUF_GENERATE_CPP_APPEND_PATH ON)
protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
  src/onnx/onnx-ml.proto src/onnx/onnx-operators-ml.proto)
add_custom_target(bevfusion_onnx_proto DEPENDS ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})

if(BEVFUSION_SPARSE_BACKEND STREQUAL "open")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../libraries/New3DSparseConvolution" "open-sparse")
  set(BEVFUSION_SPARSE_TARGET newscn)
else()
  set(spconv_include "${CMAKE_CURRENT_SOURCE_DIR}/../libraries/3DSparseConvolution/libspconv/include")
  set(spconv_lib "${CMAKE_CURRENT_SOURCE_DIR}/../libraries/3DSparseConvolution/libspconv/lib/${arch}_cuda${SPCONV_CUDA_VERSION}")
  unset(NVIDIA_SPCONV_LIBRARY CACHE)
  find_library(NVIDIA_SPCONV_LIBRARY spconv PATHS "${spconv_lib}" NO_DEFAULT_PATH)
  if(NOT NVIDIA_SPCONV_LIBRARY)
    message(FATAL_ERROR "NVIDIA libspconv was not found in ${spconv_lib}")
  endif()
  add_library(bevfusion_nvidia_spconv SHARED IMPORTED)
  set_target_properties(bevfusion_nvidia_spconv PROPERTIES
    IMPORTED_LOCATION "${NVIDIA_SPCONV_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${spconv_include}")
  set(BEVFUSION_SPARSE_TARGET bevfusion_nvidia_spconv)
endif()

include_directories(
  "${CMAKE_CURRENT_BINARY_DIR}"
  ${Protobuf_INCLUDE_DIRS}
  "${CMAKE_CURRENT_SOURCE_DIR}/include/stb"
  ${CUDA_INCLUDE_DIRS}
  "${TENSORRT_INCLUDE_DIR}"
  ../dependencies/stb
  ../dependencies/dlpack/include
  ../dependencies/pybind11/include
  ../libraries/cuOSD/src
  src/common
  src)
if(NOT "$ENV{Python_Inc}" STREQUAL "")
  include_directories("$ENV{Python_Inc}")
endif()
if(NOT "$ENV{Python_Lib}" STREQUAL "")
  link_directories("$ENV{Python_Lib}")
endif()
add_definitions(-DENABLE_TEXT_BACKEND_STB)

file(GLOB_RECURSE CORE_FILES src/bevfusion/*.cu src/bevfusion/*.cpp
  src/common/tensor.cu src/common/tensorrt.cpp)
file(GLOB_RECURSE VISUALIZE_FILES ../libraries/cuOSD/src/*.cpp
  ../libraries/cuOSD/src/*.cu src/common/visualize.cu)

cuda_add_library(bevfusion_core SHARED ${CORE_FILES} ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
add_dependencies(bevfusion_core bevfusion_onnx_proto)
if(BEVFUSION_SPARSE_BACKEND STREQUAL "open")
  target_compile_definitions(bevfusion_core PRIVATE BEVFUSION_SPARSE_BACKEND_OPEN=1)
endif()
target_link_libraries(bevfusion_core
  ${CUDA_CUDART_LIBRARY} ${CUBLASLT_LIBRARY}
  ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY}
  ${BEVFUSION_SPARSE_TARGET} ${Protobuf_LIBRARIES})

cuda_add_executable(bevfusion src/main.cpp ${VISUALIZE_FILES})
if(BEVFUSION_SPARSE_BACKEND STREQUAL "open")
  target_compile_definitions(bevfusion PRIVATE BEVFUSION_SPARSE_BACKEND_OPEN=1)
endif()
target_link_libraries(bevfusion bevfusion_core ${CMAKE_DL_LIBS})

cuda_add_library(custom_layernorm SHARED src/plugins/custom_layernorm.cu)
target_link_libraries(custom_layernorm ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY})

if(CMAKE_BUILD_PYTHON)
  cuda_add_library(pybev SHARED src/python.cpp)
  target_link_libraries(pybev bevfusion_core "$ENV{Python_Soname}" ${CMAKE_DL_LIBS})
endif()
