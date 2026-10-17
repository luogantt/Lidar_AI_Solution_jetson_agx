// SPDX-License-Identifier: MIT
// Standalone FP16 sparse-backbone check. No TensorRT dependency.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bevfusion/lidar-scn-onnx-parser.hpp"

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void report_memory(const char* stage) {
  size_t available = 0, total = 0;
  check_cuda(cudaMemGetInfo(&available, &total), "query device memory");
  std::cout << "memory stage=" << stage << " device_used_mib=" << (total - available) / (1024.0 * 1024.0)
            << " device_total_mib=" << total / (1024.0 * 1024.0)
            << " (device-wide snapshot, not peak allocation)\n";
}

struct Stream {
  cudaStream_t value = nullptr;
  Stream() { check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
  ~Stream() { cudaStreamDestroy(value); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};

struct Event {
  cudaEvent_t value = nullptr;
  Event() { check_cuda(cudaEventCreate(&value), "create event"); }
  ~Event() { cudaEventDestroy(value); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

struct Options {
  std::string model, voxels, coordinates, output;
  std::vector<int> grid;
  int warmup = 2;
  int runs = 5;
  int output_index = 0;
  int device = 0;
  bool save_all = false;
};

void usage(const char* program) {
  std::cout << "Usage: " << program
            << " --model MODEL.onnx --voxels INPUT.voxels --coors INPUT.coors\n"
               "       --grid 1440,1440,41 --output OUTPUT.dense [options]\n"
               "Options: --warmup 2 --runs 5 --output-index 0 --device 0 --save-all\n"
               "Input files use the nv::Tensor/spconv::Tensor binary format. Features\n"
               "must be FP16 [N,C], coordinates Int32 [N,4], batch index zero.\n"
               "Grid dimensions follow the model/input coordinate order (XYZ or ZYX).\n"
               "Always uses a non-default, non-blocking CUDA stream and FP16 inference.\n"
               "Saves the first measured result as OUTPUT.dense.first and the final\n"
               "result as OUTPUT.dense. --save-all also saves OUTPUT.dense.runN.\n";
}

int parse_integer(const std::string& value, const std::string& option, int minimum) {
  size_t consumed = 0;
  long parsed = std::stol(value, &consumed);
  if (consumed != value.size() || parsed < minimum || parsed > std::numeric_limits<int>::max())
    throw std::runtime_error("Invalid value for " + option + ": " + value);
  return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--save-all") {
      options.save_all = true;
      continue;
    }
    if (i + 1 == argc) throw std::runtime_error("Missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--model") options.model = value;
    else if (key == "--voxels") options.voxels = value;
    else if (key == "--coors") options.coordinates = value;
    else if (key == "--output") options.output = value;
    else if (key == "--warmup") options.warmup = parse_integer(value, key, 0);
    else if (key == "--runs") options.runs = parse_integer(value, key, 2);
    else if (key == "--output-index") options.output_index = parse_integer(value, key, 0);
    else if (key == "--device") options.device = parse_integer(value, key, 0);
    else if (key == "--grid") {
      std::istringstream parts(value);
      std::string dimension;
      options.grid.clear();
      while (std::getline(parts, dimension, ','))
        options.grid.push_back(parse_integer(dimension, key, 1));
      if (options.grid.size() != 3 || value.back() == ',')
        throw std::runtime_error("--grid must contain three positive comma-separated dimensions");
    } else throw std::runtime_error("Unknown option: " + key);
  }
  if (options.model.empty() || options.voxels.empty() || options.coordinates.empty() || options.output.empty() ||
      options.grid.size() != 3)
    throw std::runtime_error("--model, --voxels, --coors, --grid and --output are required");
  return options;
}

// Validate sizes before calling the backend's tensor loader, which assumes a valid file.
void validate_input_file(const std::string& path, int expected_dtype, size_t element_bytes) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot open input: " + path);
  const auto file_bytes = file.tellg();
  file.seekg(0);
  int32_t header[3] = {};
  int32_t dimensions[2] = {};
  file.read(reinterpret_cast<char*>(header), sizeof(header));
  file.read(reinterpret_cast<char*>(dimensions), sizeof(dimensions));
  if (!file || header[0] != 0x33ff1101 || header[1] != 2 || header[2] != expected_dtype ||
      dimensions[0] <= 0 || dimensions[1] <= 0)
    throw std::runtime_error("Invalid tensor header/shape/dtype: " + path);
  const uint64_t count = static_cast<uint64_t>(dimensions[0]) * dimensions[1];
  const uint64_t expected_bytes = sizeof(header) + sizeof(dimensions) + count * element_bytes;
  if (file_bytes < 0 || static_cast<uint64_t>(file_bytes) != expected_bytes)
    throw std::runtime_error("Tensor payload size mismatch: " + path);
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < shape.size(); ++i) out << (i ? "," : "") << shape[i];
  return out.str() + ']';
}

void save_dense(spconv::Tensor& output, const std::string& path, cudaStream_t stream) {
  if (output.empty() || output.shape.size() < 3)
    throw std::runtime_error("Selected output is empty or is not a dense spatial tensor");
  if (output.dtype() != spconv::DataType::Float16 && output.dtype() != spconv::DataType::Float32)
    throw std::runtime_error("Selected dense output must be FP16 or FP32");
  if (!output.save(path.c_str(), stream)) throw std::runtime_error("Cannot save output: " + path);
  check_cuda(cudaStreamSynchronize(stream), "save output");
  std::cout << "output=" << path << " shape=" << shape_string(output.shape)
            << " dtype=" << spconv::dtype_string(output.dtype()) << '\n';
}

void run(const Options& options) {
  check_cuda(cudaSetDevice(options.device), "set device");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, options.device), "get device properties");
  std::cout << "device=" << options.device << " name=" << properties.name << " sm=" << properties.major
            << properties.minor << " precision=fp16 stream=nonblocking_nondefault\n";
  Stream stream;
  report_memory("before_engine");
  validate_input_file(options.voxels, static_cast<int>(spconv::DataType::Float16), 2);
  validate_input_file(options.coordinates, static_cast<int>(spconv::DataType::Int32), 4);
  auto features_host = spconv::Tensor::load(options.voxels.c_str(), false);
  auto indices_host = spconv::Tensor::load(options.coordinates.c_str(), false);
  if (features_host.empty() || indices_host.empty() || indices_host.shape[1] != 4 ||
      features_host.shape[0] != indices_host.shape[0])
    throw std::runtime_error("Expected matching FP16 [N,C] features and Int32 [N,4] coordinates");
  const int32_t* indices = indices_host.ptr<int32_t>();
  for (int64_t row = 0; row < indices_host.shape[0]; ++row) {
    if (indices[row * 4] != 0) throw std::runtime_error("Only batch size 1 is supported");
    for (size_t axis = 0; axis < 3; ++axis)
      if (indices[row * 4 + axis + 1] < 0 || indices[row * 4 + axis + 1] >= options.grid[axis])
        throw std::runtime_error("Coordinate outside --grid at row " + std::to_string(row));
  }
  auto features = features_host.to_device(stream.value);
  auto coordinates = indices_host.to_device(stream.value);
  check_cuda(cudaStreamSynchronize(stream.value), "upload inputs");
  std::cout << "features=" << shape_string(features.shape) << " coordinates=" << shape_string(coordinates.shape)
            << " grid=" << options.grid[0] << ',' << options.grid[1] << ',' << options.grid[2] << '\n';

  auto engine = spconv::load_engine_from_onnx(options.model, spconv::Precision::Float16, false, false, false,
                                              10000, stream.value);
  if (!engine) throw std::runtime_error("Failed to load sparse engine: " + options.model);
  if (engine->num_input() != 1 || options.output_index >= static_cast<int>(engine->num_output()))
    throw std::runtime_error("Expected one sparse input and a valid --output-index");
  engine->input(0)->features().reference(features.ptr(), features.shape, spconv::DataType::Float16);
  engine->input(0)->indices().reference(coordinates.ptr(), coordinates.shape, spconv::DataType::Int32);
  engine->input(0)->set_grid_size(options.grid);
  check_cuda(cudaStreamSynchronize(stream.value), "initialize engine");
  report_memory("after_engine_load");

  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    engine->forward(stream.value);
    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaStreamSynchronize(stream.value), "warmup completion");
  }
  report_memory("after_warmup");
  Event begin, end;
  std::vector<float> milliseconds;
  for (int iteration = 0; iteration < options.runs; ++iteration) {
    const auto wall_begin = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(begin.value, stream.value), "record begin");
    engine->forward(stream.value);
    check_cuda(cudaGetLastError(), "inference launch");
    check_cuda(cudaEventRecord(end.value, stream.value), "record end");
    check_cuda(cudaEventSynchronize(end.value), "inference completion");
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_begin).count();
    float elapsed = 0;
    check_cuda(cudaEventElapsedTime(&elapsed, begin.value, end.value), "elapsed time");
    milliseconds.push_back(elapsed);
    std::cout << "run=" << iteration + 1 << " gpu_ms=" << elapsed << " wall_ms=" << wall_ms << '\n';
    auto& output = engine->output(options.output_index)->features();
    if (iteration == 0) save_dense(output, options.output + ".first", stream.value);
    if (options.save_all) save_dense(output, options.output + ".run" + std::to_string(iteration + 1), stream.value);
  }
  save_dense(engine->output(options.output_index)->features(), options.output, stream.value);
  std::sort(milliseconds.begin(), milliseconds.end());
  double total = 0;
  for (float value : milliseconds) total += value;
  const size_t middle = milliseconds.size() / 2;
  const double median = milliseconds.size() % 2 ? milliseconds[middle]
                                               : (milliseconds[middle - 1] + milliseconds[middle]) / 2.0;
  std::cout << "timing runs=" << options.runs << " warmup=" << options.warmup << " min_ms=" << milliseconds.front()
            << " median_ms=" << median << " mean_ms=" << total / milliseconds.size()
            << " max_ms=" << milliseconds.back() << " (forward only; excludes model load, upload and output save)\n";
  std::cout << "Inference completed; numerical validation still requires tool/compare_sparse_output.py.\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    usage(argv[0]);
    return 0;
  }
  try {
    std::cout << std::fixed << std::setprecision(6);
    run(parse_options(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "sparse_backend_check: " << error.what() << '\n';
    return 2;
  }
}
