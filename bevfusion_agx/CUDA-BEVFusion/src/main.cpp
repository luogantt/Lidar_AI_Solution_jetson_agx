/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_runtime.h>
#include <string.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"

static std::vector<unsigned char*> load_images(const std::string& root) {
  const char* file_names[] = {"0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
                              "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"};

  std::vector<unsigned char*> images;
  for (int i = 0; i < 6; ++i) {
    const std::string path = root + "/" + file_names[i];
    int width = 0, height = 0, channels = 0;
    auto* image = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!image || width != 1600 || height != 900) {
      if (image) stbi_image_free(image);
      for (auto* previous : images) stbi_image_free(previous);
      throw std::runtime_error("Cannot load a 1600x900 RGB image from " + path);
    }
    images.push_back(image);
  }
  return images;
}

static void free_images(std::vector<unsigned char*>& images) {
  for (size_t i = 0; i < images.size(); ++i) stbi_image_free(images[i]);

  images.clear();
}

static void visualize(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, const nv::Tensor& lidar_points,
                      const std::vector<unsigned char*> images, const nv::Tensor& lidar2image, const std::string& save_path,
                      cudaStream_t stream) {
  std::vector<nv::Prediction> predictions(bboxes.size());
  if (!bboxes.empty()) memcpy(predictions.data(), bboxes.data(), bboxes.size() * sizeof(nv::Prediction));

  int padding = 300;
  int lidar_size = 1024;
  int content_width = lidar_size + padding * 3;
  int content_height = 1080;
  nv::SceneArtistParameter scene_artist_param;
  scene_artist_param.width = content_width;
  scene_artist_param.height = content_height;
  scene_artist_param.stride = scene_artist_param.width * 3;

  nv::Tensor scene_device_image(std::vector<int>{scene_artist_param.height, scene_artist_param.width, 3}, nv::DataType::UInt8);
  scene_device_image.memset(0x00, stream);

  scene_artist_param.image_device = scene_device_image.ptr<unsigned char>();
  auto scene = nv::create_scene_artist(scene_artist_param);

  nv::BEVArtistParameter bev_artist_param;
  bev_artist_param.image_width = content_width;
  bev_artist_param.image_height = content_height;
  bev_artist_param.rotate_x = 70.0f;
  bev_artist_param.norm_size = lidar_size * 0.5f;
  bev_artist_param.cx = content_width * 0.5f;
  bev_artist_param.cy = content_height * 0.5f;
  bev_artist_param.image_stride = scene_artist_param.stride;

  auto points = lidar_points.to_device(stream);
  auto bev_visualizer = nv::create_bev_artist(bev_artist_param);
  bev_visualizer->draw_lidar_points(points.ptr<nvtype::half>(), points.size(0));
  bev_visualizer->draw_prediction(predictions, false);
  bev_visualizer->draw_ego();
  bev_visualizer->apply(scene_device_image.ptr<unsigned char>(), stream);

  nv::ImageArtistParameter image_artist_param;
  image_artist_param.num_camera = images.size();
  image_artist_param.image_width = 1600;
  image_artist_param.image_height = 900;
  image_artist_param.image_stride = image_artist_param.image_width * 3;
  image_artist_param.viewport_nx4x4.resize(images.size() * 4 * 4);
  memcpy(image_artist_param.viewport_nx4x4.data(), lidar2image.ptr<float>(),
         sizeof(float) * image_artist_param.viewport_nx4x4.size());

  int gap = 0;
  int camera_width = 500;
  int camera_height = static_cast<float>(camera_width / (float)image_artist_param.image_width * image_artist_param.image_height);
  int offset_cameras[][3] = {
      {-camera_width / 2, -content_height / 2 + gap, 0},
      {content_width / 2 - camera_width - gap, -content_height / 2 + camera_height / 2, 0},
      {-content_width / 2 + gap, -content_height / 2 + camera_height / 2, 0},
      {-camera_width / 2, +content_height / 2 - camera_height - gap, 1},
      {-content_width / 2 + gap, +content_height / 2 - camera_height - camera_height / 2, 0},
      {content_width / 2 - camera_width - gap, +content_height / 2 - camera_height - camera_height / 2, 1}};

  auto visualizer = nv::create_image_artist(image_artist_param);
  for (size_t icamera = 0; icamera < images.size(); ++icamera) {
    int ox = offset_cameras[icamera][0] + content_width / 2;
    int oy = offset_cameras[icamera][1] + content_height / 2;
    bool xflip = static_cast<bool>(offset_cameras[icamera][2]);
    visualizer->draw_prediction(icamera, predictions, xflip);

    nv::Tensor device_image(std::vector<int>{900, 1600, 3}, nv::DataType::UInt8);
    device_image.copy_from_host(images[icamera], stream);

    if (xflip) {
      auto clone = device_image.clone(stream);
      scene->flipx(clone.ptr<unsigned char>(), clone.size(1), clone.size(1) * 3, clone.size(0), device_image.ptr<unsigned char>(),
                   device_image.size(1) * 3, stream);
      checkRuntime(cudaStreamSynchronize(stream));
    }
    visualizer->apply(device_image.ptr<unsigned char>(), stream);

    scene->resize_to(device_image.ptr<unsigned char>(), ox, oy, ox + camera_width, oy + camera_height, device_image.size(1),
                     device_image.size(1) * 3, device_image.size(0), 0.8f, stream);
    checkRuntime(cudaStreamSynchronize(stream));
  }

  printf("Save to %s\n", save_path.c_str());
  if (!stbi_write_jpg(save_path.c_str(), scene_device_image.size(1), scene_device_image.size(0), 3,
                     scene_device_image.to_host(stream).ptr(), 100))
    throw std::runtime_error("Cannot write visualization: " + save_path);
}

std::shared_ptr<bevfusion::Core> create_core(const std::string& model, const std::string& precision) {

  printf("Create by %s, %s\n", model.c_str(), precision.c_str());
  bevfusion::camera::NormalizationParameter normalization;
  normalization.image_width = 1600;
  normalization.image_height = 900;
  normalization.output_width = 704;
  normalization.output_height = 256;
  normalization.num_camera = 6;
  normalization.resize_lim = 0.48f;
  normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;

  float mean[3] = {0.485, 0.456, 0.406};
  float std[3] = {0.229, 0.224, 0.225};
  normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

  bevfusion::lidar::VoxelizationParameter voxelization;
  voxelization.min_range = nvtype::Float3(-54.0f, -54.0f, -5.0);
  voxelization.max_range = nvtype::Float3(+54.0f, +54.0f, +3.0);
  voxelization.voxel_size = nvtype::Float3(0.075f, 0.075f, 0.2f);
  voxelization.grid_size =
      voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
  voxelization.max_points_per_voxel = 10;
  voxelization.max_points = 300000;
  voxelization.max_voxels = 160000;
  voxelization.num_feature = 5;

  bevfusion::lidar::SCNParameter scn;
  scn.voxelization = voxelization;
  scn.model = nv::format("model/%s/lidar.backbone.xyz.onnx", model.c_str());
  scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

  if (precision == "int8") {
    scn.precision = bevfusion::lidar::Precision::Int8;
  } else {
    scn.precision = bevfusion::lidar::Precision::Float16;
  }

  bevfusion::camera::GeometryParameter geometry;
  geometry.xbound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.ybound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
  geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);
  geometry.image_width = 704;
  geometry.image_height = 256;
  geometry.feat_width = 88;
  geometry.feat_height = 32;
  geometry.num_camera = 6;
  geometry.geometry_dim = nvtype::Int3(360, 360, 80);

  bevfusion::head::transbbox::TransBBoxParameter transbbox;
  transbbox.out_size_factor = 8;
  transbbox.pc_range = {-54.0f, -54.0f};
  transbbox.post_center_range_start = {-61.2, -61.2, -10.0};
  transbbox.post_center_range_end = {61.2, 61.2, 10.0};
  transbbox.voxel_size = {0.075, 0.075};
  transbbox.model = nv::format("model/%s/build/head.bbox.plan", model.c_str());

  // if you got an inaccurate boundingbox result please turn on the layernormplugin plan.
  // transbbox.model = nv::format("model/%s/build/head.bbox.layernormplugin.plan", model.c_str());
  transbbox.confidence_threshold = 0.12f;
  transbbox.sorted_bboxes = true;

  bevfusion::CoreParameter param;
  param.camera_model = nv::format("model/%s/build/camera.backbone.plan", model.c_str());
  param.normalize = normalization;
  param.lidar_scn = scn;
  param.geometry = geometry;
  param.transfusion = nv::format("model/%s/build/fuser.plan", model.c_str());
  param.transbbox = transbbox;
  param.camera_vtransform = nv::format("model/%s/build/camera.vtransform.plan", model.c_str());
  return bevfusion::create_core(param);
}

namespace {

using BoundingBox = bevfusion::head::transbbox::BoundingBox;

struct RunOptions {
  std::string data = "example-data";
  std::string model = "resnet50int8";
  std::string precision = "int8";
  bool benchmark = false;
  int warmup = 10;
  int runs = 100;
  int profile_runs = 3;
  std::string benchmark_json;
  std::string visualization;
  std::string data_list;
  int limit = 0;
};

void usage(const char* program) {
  std::cout << "Usage: " << program << " [data [model [fp16|int8]]] [options]\n"
            << "       " << program << " [model [fp16|int8]] --data-list FILE [options]\n"
               "  --benchmark                 Enable repeated single-frame, batch-1 benchmark\n"
               "  --data-list FILE            Benchmark each listed data directory once, with one shared core\n"
               "  --limit N                   Use the first N list entries (only with --data-list)\n"
               "  --warmup N                  Warmup forwards (default 10, enables benchmark)\n"
               "  --runs N                    Single-frame measured forwards (default 100; rejected with --data-list)\n"
               "  --profile-runs N            Separate stage-profile forwards (default 3, 0 disables)\n"
               "  --benchmark-json PATH       Write timing, GPU, memory and boxes JSON\n"
               "  --visualization PATH        Write a JPEG after inference (outside measured time)\n"
               "Without benchmark options, preserve the original 1+5 profiled forwards and\n"
               "build/cuda-bevfusion.jpg output. Benchmark mode only writes an image when requested.\n"
               "Benchmark performs one untimed raw-head finite-value preflight before the requested warmup.\n"
               "Dataset mode checks every frame before timing, warms up only the first frame, and profiles/\n"
               "visualizes only the last frame. List paths are relative to FILE's directory; blank/# lines are skipped.\n"
               "The positional precision controls the LiDAR engine; camera/fuser/head precision\n"
               "is determined by the TensorRT plans in model/<model>/build.\n";
}

int integer_option(const std::string& value, const std::string& option, int minimum) {
  size_t consumed = 0;
  const long number = std::stol(value, &consumed);
  if (consumed != value.size() || number < minimum || number > std::numeric_limits<int>::max())
    throw std::runtime_error("Invalid value for " + option + ": " + value);
  return static_cast<int>(number);
}

RunOptions parse_options(int argc, char** argv) {
  RunOptions options;
  std::vector<std::string> positional;
  bool explicit_runs = false;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--benchmark") {
      options.benchmark = true;
      continue;
    }
    if (key.compare(0, 2, "--") == 0) {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
      const std::string value = argv[++i];
      if (key == "--visualization") options.visualization = value;
      else {
        options.benchmark = true;
        if (key == "--warmup") options.warmup = integer_option(value, key, 0);
        else if (key == "--runs") {
          options.runs = integer_option(value, key, 1);
          explicit_runs = true;
        }
        else if (key == "--profile-runs") options.profile_runs = integer_option(value, key, 0);
        else if (key == "--benchmark-json") options.benchmark_json = value;
        else if (key == "--data-list") options.data_list = value;
        else if (key == "--limit") options.limit = integer_option(value, key, 1);
        else throw std::runtime_error("Unknown option: " + key);
      }
      if (value.empty()) throw std::runtime_error("Empty value for " + key);
    } else {
      positional.push_back(key);
    }
  }
  if (options.data_list.empty()) {
    if (positional.size() > 3) throw std::runtime_error("Too many positional arguments");
    if (!positional.empty()) options.data = positional[0];
    if (positional.size() > 1) options.model = positional[1];
    if (positional.size() > 2) options.precision = positional[2];
    if (options.limit) throw std::runtime_error("--limit requires --data-list");
  } else {
    if (explicit_runs) throw std::runtime_error("--runs cannot be used with --data-list; each selected frame is measured once");
    if (positional.size() > 2)
      throw std::runtime_error("With --data-list use [model [fp16|int8]], without a positional data directory");
    if (!positional.empty()) options.model = positional[0];
    if (positional.size() > 1) options.precision = positional[1];
  }
  if (options.precision != "fp16" && options.precision != "int8")
    throw std::runtime_error("Precision must be fp16 or int8");
  if (!options.benchmark && options.visualization.empty()) options.visualization = "build/cuda-bevfusion.jpg";
  return options;
}

std::vector<std::string> load_data_list(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open data list: " + path);
  const auto slash = path.find_last_of("/\\");
  const std::string parent = slash == std::string::npos ? "./" : path.substr(0, slash + 1);
  std::vector<std::string> directories;
  std::string line;
  bool first_line = true;
  while (std::getline(input, line)) {
    if (first_line && line.compare(0, 3, "\xef\xbb\xbf") == 0) line.erase(0, 3);
    first_line = false;
    const auto first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '#') continue;
    const auto last = line.find_last_not_of(" \t\r\n");
    line = line.substr(first, last - first + 1);
    if (line.find('\0') != std::string::npos) throw std::runtime_error("NUL byte in data list: " + path);
    directories.push_back(line.front() == '/' ? line : parent + line);
  }
  if (input.bad()) throw std::runtime_error("Failed reading data list: " + path);
  if (directories.empty()) throw std::runtime_error("Data list has no frame directories: " + path);
  return directories;
}

struct ImageSet {
  std::vector<unsigned char*> values;
  explicit ImageSet(const std::string& root) : values(load_images(root)) {}
  ~ImageSet() { free_images(values); }
  ImageSet(const ImageSet&) = delete;
  ImageSet& operator=(const ImageSet&) = delete;
};

struct StreamGuard {
  cudaStream_t value = nullptr;
  StreamGuard() { checkRuntime(cudaStreamCreate(&value)); }
  ~StreamGuard() { if (value) cudaStreamDestroy(value); }
  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;
};

// Validate the complete file before entering nv::Tensor::load, whose header reader
// assumes a well-formed tensor and accumulates the element count in a signed int.
nv::Tensor load_host_tensor(const std::string& path, nv::DataType expected_type) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot open tensor: " + path);
  const auto file_size = file.tellg();
  file.seekg(0);
  int32_t header[3] = {};
  file.read(reinterpret_cast<char*>(header), sizeof(header));
  if (!file || header[0] != 0x33ff1101 || header[1] < 1 || header[1] > 16 ||
      header[2] != static_cast<int>(expected_type))
    throw std::runtime_error("Invalid tensor header or unexpected dtype: " + path);
  std::vector<int32_t> dimensions(header[1]);
  file.read(reinterpret_cast<char*>(dimensions.data()), dimensions.size() * sizeof(int32_t));
  if (!file) throw std::runtime_error("Truncated tensor shape: " + path);
  uint64_t count = 1;
  for (int32_t dimension : dimensions) {
    if (dimension <= 0 || count > static_cast<uint64_t>(std::numeric_limits<int>::max()) / dimension)
      throw std::runtime_error("Invalid or oversized tensor shape: " + path);
    count *= dimension;
  }
  const uint64_t expected_bytes = sizeof(header) + dimensions.size() * sizeof(int32_t) + count * nv::dtype_bytes(expected_type);
  if (file_size < 0 || static_cast<uint64_t>(file_size) != expected_bytes)
    throw std::runtime_error("Tensor payload size mismatch: " + path);
  file.close();
  auto tensor = nv::Tensor::load(path, false);
  if (tensor.empty() || tensor.device() || tensor.dtype() != expected_type)
    throw std::runtime_error("Failed to load host tensor: " + path);
  return tensor;
}

nv::Tensor load_matrix(const std::string& root, const std::string& name) {
  const std::string path = root + "/" + name + ".tensor";
  auto tensor = load_host_tensor(path, nv::DataType::Float32);
  if (tensor.shape != std::vector<int64_t>{6, 4, 4} && tensor.shape != std::vector<int64_t>{1, 6, 4, 4})
    throw std::runtime_error("Expected [6,4,4] or [1,6,4,4] calibration matrix: " + path);
  for (size_t i = 0; i < tensor.numel; ++i)
    if (!std::isfinite(tensor.ptr<float>()[i])) throw std::runtime_error("Non-finite calibration value: " + path);
  return tensor;
}

// Own one frame at a time in dataset mode. Construction, validation and
// calibration updates always happen outside the measured forward region.
struct FrameInput {
  std::string data_directory;
  ImageSet images;
  nv::Tensor camera2lidar, camera_intrinsics, lidar2image, img_aug_matrix, lidar_points;
  int point_count = 0;

  explicit FrameInput(const std::string& root)
      : data_directory(root), images(root), camera2lidar(load_matrix(root, "camera2lidar")),
        camera_intrinsics(load_matrix(root, "camera_intrinsics")), lidar2image(load_matrix(root, "lidar2image")),
        img_aug_matrix(load_matrix(root, "img_aug_matrix")),
        lidar_points(load_host_tensor(root + "/points.tensor", nv::DataType::Float16)) {
    if (lidar_points.shape.size() != 2 || lidar_points.shape[1] != 5)
      throw std::runtime_error("points.tensor must be FP16 [N,5]: " + root);
    for (size_t i = 0; i < lidar_points.numel; ++i)
      if ((lidar_points.ptr<nvtype::half>()[i].__x & 0x7c00u) == 0x7c00u)
        throw std::runtime_error("points.tensor contains NaN or infinity: " + root);
    point_count = static_cast<int>(std::min<int64_t>(lidar_points.size(0), 300000));
    if (lidar_points.size(0) > point_count)
      std::cout << "Frame " << root << " has " << lidar_points.size(0) << " points; processing first " << point_count << ".\n";
  }
};

void validate_boxes(const std::vector<BoundingBox>& boxes, const std::string& phase, int iteration) {
  for (size_t index = 0; index < boxes.size(); ++index) {
    const auto& box = boxes[index];
    const float values[] = {box.position.x, box.position.y, box.position.z, box.size.w, box.size.l, box.size.h,
                            box.velocity.vx, box.velocity.vy, box.z_rotation, box.score};
    for (float value : values)
      if (!std::isfinite(value))
        throw std::runtime_error("Non-finite bounding box in " + phase + " iteration " + std::to_string(iteration) +
                                 ", box " + std::to_string(index));
  }
}

struct MemorySnapshot {
  std::string name;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
};

MemorySnapshot memory_snapshot(const char* name) {
  MemorySnapshot result;
  result.name = name;
  checkRuntime(cudaMemGetInfo(&result.free_bytes, &result.total_bytes));
  return result;
}

struct FrameTiming {
  double milliseconds;
  size_t bbox_count;
  std::string data_directory;
  int64_t input_points = 0;
  int processed_points = 0;
};

struct TimingSummary {
  double mean, median, p90, p95, minimum, maximum, fps;
};

double percentile(const std::vector<double>& sorted, double fraction) {
  const double position = (sorted.size() - 1) * fraction;
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1, sorted.size() - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

TimingSummary summarize(const std::vector<FrameTiming>& frames) {
  std::vector<double> sorted;
  for (const auto& frame : frames) sorted.push_back(frame.milliseconds);
  std::sort(sorted.begin(), sorted.end());
  const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
  return {mean, percentile(sorted, 0.5), percentile(sorted, 0.9), percentile(sorted, 0.95), sorted.front(), sorted.back(),
          1000.0 / mean};
}

std::string json_string(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (unsigned char character : value) {
    if (character == '"' || character == '\\') output << '\\' << character;
    else if (character < 0x20) output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(character);
    else output << character;
  }
  output << '"';
  return output.str();
}

void write_benchmark_json(const RunOptions& options, const cudaDeviceProp& gpu, int device, int runtime_version,
                          int driver_version, int64_t input_points, int64_t processed_points, const TimingSummary& summary,
                          const std::vector<FrameTiming>& frames, const std::vector<BoundingBox>& boxes,
                          const std::vector<MemorySnapshot>& memory, size_t listed_frames = 0) {
  if (options.benchmark_json.empty()) return;
  const bool dataset_mode = !options.data_list.empty();
  std::ofstream out(options.benchmark_json);
  if (!out) throw std::runtime_error("Cannot write benchmark JSON: " + options.benchmark_json);
  out.imbue(std::locale::classic());
  out << std::setprecision(12)
      << "{\n  \"schema_version\": 1,\n  \"mode\": "
      << json_string(dataset_mode ? "dataset_sequence" : "single_frame_repeat") << ",\n"
      << "  \"data_directory\": " << (dataset_mode ? "null" : json_string(options.data)) << ",\n";
  if (dataset_mode) {
    out << "  \"data_list\": " << json_string(options.data_list) << ",\n"
        << "  \"listed_frames\": " << listed_frames << ", \"selected_frames\": " << frames.size()
        << ", \"limit\": " << (options.limit ? std::to_string(options.limit) : "null") << ",\n"
        << "  \"point_count_scope\": \"Totals across selected measured frames; per-frame counts are in frames\",\n"
        << "  \"warmup_scope\": \"Requested warmup runs only on the first selected frame\",\n"
        << "  \"dataset_sequence_note\": \"One measured forward per selected frame after an untimed raw-head preflight "
           "on that same frame. Timings therefore use warmed inputs, exclude data loading and per-frame calibration, "
           "and do not represent the wall throughput of the dataset runner.\",\n";
  }
  out
      << "  \"model\": " << json_string(options.model) << ",\n"
      << "  \"lidar_precision\": " << json_string(options.precision) << ",\n"
      << "  \"tensorrt_engine_directory\": " << json_string("model/" + options.model + "/build") << ",\n"
      << "  \"tensorrt_precision_note\": \"Determined by loaded engine files, not the lidar precision argument\",\n"
      << "  \"batch_size\": 1, \"camera_count\": 6, \"input_points\": " << input_points
      << ", \"processed_points\": " << processed_points << ",\n"
      << "  \"warmup_runs\": " << options.warmup << ", \"measured_runs\": " << frames.size() << ",\n"
      << "  \"timing_scope\": \"Synchronized host wall time for normal Core::forward: host input copy, preprocessing, "
         "camera/lidar inference, fusion and decoded host boxes. Excludes disk IO, engine load, calibration update, "
         "preflight, warmup, box validation, visualization and separate stage profiling.\",\n"
      << "  \"percentile_method\": \"linear interpolation at (N-1)*q\",\n"
      << "  \"profile\": {\"runs\": " << options.profile_runs
      << ", \"included_in_statistics\": false, \"stage_times_destination\": \"stdout\", "
      << "\"data_directory\": " << json_string(dataset_mode ? frames.back().data_directory : options.data) << ", "
         "\"note\": \"Instrumented forward has stage synchronizations and a different execution order. "
         "Its Total excludes CopyLidar and ImageNorm; stage sums are not end-to-end latency.\"},\n"
      << "  \"gpu\": {\"device\": " << device << ", \"name\": " << json_string(gpu.name)
      << ", \"compute_capability\": " << json_string(std::to_string(gpu.major) + "." + std::to_string(gpu.minor))
      << ", \"total_global_memory_bytes\": " << gpu.totalGlobalMem << ", \"cuda_runtime_version\": " << runtime_version
      << ", \"cuda_driver_version\": " << driver_version << "},\n"
      << "  \"memory_note\": \"Device-wide cudaMemGetInfo snapshots; not process-only usage and not peak memory\",\n"
      << "  \"memory_snapshots\": [\n";
  for (size_t i = 0; i < memory.size(); ++i) {
    const auto& snapshot = memory[i];
    out << "    {\"stage\": " << json_string(snapshot.name) << ", \"free_bytes\": " << snapshot.free_bytes
        << ", \"used_bytes\": " << snapshot.total_bytes - snapshot.free_bytes << ", \"total_bytes\": " << snapshot.total_bytes
        << '}' << (i + 1 < memory.size() ? ",\n" : "\n");
  }
  out << "  ],\n  \"latency_ms\": {\"mean\": " << summary.mean << ", \"median\": " << summary.median
      << ", \"p90\": " << summary.p90 << ", \"p95\": " << summary.p95 << ", \"min\": " << summary.minimum
      << ", \"max\": " << summary.maximum << "},\n  \"fps\": " << summary.fps
      << ",\n  \"fps_note\": \"1000 / mean single-frame latency; no batching or multi-frame pipelining\",\n"
      << "  \"all_boxes_finite\": true,\n  \"raw_head_outputs_finite\": true,\n"
      << "  \"raw_head_validation\": {\"runs\": " << (dataset_mode ? frames.size() : size_t{1}) << ", \"phase\": "
      << json_string(dataset_mode ? "untimed preflight on every selected frame; first frame then warms up"
                                  : "untimed preflight before warmup") << ", "
         "\"scope\": \"All elements of reg, height, dim, rot, vel and score before decoding; "
         "disabled during measured and profiled forwards\"},\n  \"visualization\": "
      << (options.visualization.empty() ? "null" : json_string(options.visualization)) << ",\n  \"frames\": [\n";
  for (size_t i = 0; i < frames.size(); ++i) {
    out << "    {\"iteration\": " << i + 1 << ", \"latency_ms\": " << frames[i].milliseconds
        << ", \"bbox_count\": " << frames[i].bbox_count;
    if (dataset_mode)
      out << ", \"data_directory\": " << json_string(frames[i].data_directory)
          << ", \"input_points\": " << frames[i].input_points << ", \"processed_points\": " << frames[i].processed_points;
    out << '}' << (i + 1 < frames.size() ? ",\n" : "\n");
  }
  out << "  ],\n  \"latest_boxes_source\": \"last measured forward, before separate stage profiling\",\n"
      << "  \"latest_data_directory\": " << json_string(dataset_mode ? frames.back().data_directory : options.data) << ",\n"
      << "  \"latest_bbox_count\": " << boxes.size() << ",\n  \"latest_boxes\": [\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    const auto& box = boxes[i];
    out << "    {\"x\": " << box.position.x << ", \"y\": " << box.position.y << ", \"z\": " << box.position.z
        << ", \"w\": " << box.size.w << ", \"l\": " << box.size.l << ", \"h\": " << box.size.h
        << ", \"vx\": " << box.velocity.vx << ", \"vy\": " << box.velocity.vy
        << ", \"yaw\": " << box.z_rotation << ", \"score\": " << box.score << ", \"class_id\": " << box.id
        << '}' << (i + 1 < boxes.size() ? ",\n" : "\n");
  }
  out << "  ]\n}\n";
  out.close();
  if (!out) throw std::runtime_error("Failed writing benchmark JSON: " + options.benchmark_json);
  std::cout << "Benchmark JSON: " << options.benchmark_json << '\n';
}

int run_dataset(const RunOptions& options) {
  auto directories = load_data_list(options.data_list);
  const size_t listed_frames = directories.size();
  if (options.limit && directories.size() > static_cast<size_t>(options.limit)) directories.resize(options.limit);
  if (directories.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("Too many selected dataset frames");

  std::unique_ptr<FrameInput> frame(new FrameInput(directories.front()));
  StreamGuard stream;
  int device = 0, runtime_version = 0, driver_version = 0;
  cudaDeviceProp gpu{};
  checkRuntime(cudaGetDevice(&device));
  checkRuntime(cudaGetDeviceProperties(&gpu, device));
  checkRuntime(cudaRuntimeGetVersion(&runtime_version));
  checkRuntime(cudaDriverGetVersion(&driver_version));
  std::vector<MemorySnapshot> memory;
  memory.push_back(memory_snapshot("before_engine"));
  dlopen("libcustom_layernorm.so", RTLD_NOW);
  auto core = create_core(options.model, options.precision);
  if (!core) throw std::runtime_error("Failed to create BEVFusion core");
  core->print();
  core->set_timer(false);
  core->set_validation(false);
  auto forward = [&]() {
    return core->forward((const unsigned char**)frame->images.values.data(), frame->lidar_points.ptr<nvtype::half>(),
                          frame->point_count, stream.value);
  };

  std::cout << "Dataset benchmark: " << directories.size() << '/' << listed_frames << " selected frames, GPU=" << gpu.name
            << ", first-frame warmup=" << options.warmup << ", one untimed preflight and one measured forward per frame\n";
  std::vector<FrameTiming> frames;
  frames.reserve(directories.size());
  std::vector<BoundingBox> latest_boxes;
  int64_t total_input_points = 0, total_processed_points = 0;
  for (size_t index = 0; index < directories.size(); ++index) {
    if (index) {
      // The previous forward was synchronized. Release its host inputs before
      // loading the next frame, keeping dataset memory usage bounded.
      frame.reset();
      frame.reset(new FrameInput(directories[index]));
    }
    core->update(frame->camera2lidar.ptr<float>(), frame->camera_intrinsics.ptr<float>(), frame->lidar2image.ptr<float>(),
                 frame->img_aug_matrix.ptr<float>(), stream.value);
    checkRuntime(cudaStreamSynchronize(stream.value));
    const std::string phase = "dataset frame " + std::to_string(index + 1) + " (" + frame->data_directory + ")";
    core->set_validation(true);
    validate_boxes(forward(), phase + " raw head preflight", 1);
    core->set_validation(false);
    if (index == 0) {
      for (int warmup = 0; warmup < options.warmup; ++warmup)
        validate_boxes(forward(), phase + " warmup", warmup + 1);
      checkRuntime(cudaStreamSynchronize(stream.value));
      memory.push_back(memory_snapshot("after_first_frame_warmup"));
    }

    checkRuntime(cudaStreamSynchronize(stream.value));
    const auto begin = std::chrono::steady_clock::now();
    auto boxes = forward();
    checkRuntime(cudaStreamSynchronize(stream.value));
    const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (!std::isfinite(milliseconds) || milliseconds <= 0) throw std::runtime_error("Invalid measured latency: " + phase);
    validate_boxes(boxes, phase + " measured", 1);
    frames.push_back({milliseconds, boxes.size(), frame->data_directory, frame->lidar_points.size(0), frame->point_count});
    total_input_points += frame->lidar_points.size(0);
    total_processed_points += frame->point_count;
    latest_boxes = std::move(boxes);
    std::cout << "Frame " << index + 1 << '/' << directories.size() << ": " << frame->data_directory
              << ", points=" << frame->point_count << ", latency_ms=" << milliseconds << ", boxes=" << latest_boxes.size() << '\n';
  }
  memory.push_back(memory_snapshot("after_measured"));
  const auto summary = summarize(frames);
  std::cout << std::fixed << std::setprecision(3)
            << "Dataset normal forward wall latency (ms): mean=" << summary.mean << " median=" << summary.median
            << " p90=" << summary.p90 << " p95=" << summary.p95 << " min=" << summary.minimum
            << " max=" << summary.maximum << ", FPS=" << summary.fps << ", frames=" << frames.size() << '\n';
  std::cout << "Timing includes host input copies and preprocessing through decoded host boxes. Every frame has an "
               "untimed preflight before measurement; disk IO, per-frame calibration, preflight, first-frame warmup, "
               "validation, visualization and separate profiling are excluded.\n";
  if (options.profile_runs > 0) {
    std::cout << "Separate stage profile on the last dataset frame only; excluded from latency statistics.\n";
    core->set_timer(true);
    for (int i = 0; i < options.profile_runs; ++i) {
      std::cout << "Profile iteration " << i + 1 << '/' << options.profile_runs << '\n';
      validate_boxes(forward(), "last dataset frame stage profile", i + 1);
    }
    core->set_timer(false);
  }
  if (!options.visualization.empty())
    visualize(latest_boxes, frame->lidar_points, frame->images.values, frame->lidar2image, options.visualization, stream.value);
  write_benchmark_json(options, gpu, device, runtime_version, driver_version, total_input_points, total_processed_points,
                        summary, frames, latest_boxes, memory, listed_frames);
  return 0;
}

int run(const RunOptions& options) {
  if (!options.data_list.empty()) return run_dataset(options);
  // File IO and validation are outside every measured region.
  ImageSet images(options.data);
  auto camera2lidar = load_matrix(options.data, "camera2lidar");
  auto camera_intrinsics = load_matrix(options.data, "camera_intrinsics");
  auto lidar2image = load_matrix(options.data, "lidar2image");
  auto img_aug_matrix = load_matrix(options.data, "img_aug_matrix");
  auto lidar_points = load_host_tensor(options.data + "/points.tensor", nv::DataType::Float16);
  if (lidar_points.shape.size() != 2 || lidar_points.shape[1] != 5)
    throw std::runtime_error("points.tensor must be FP16 [N,5]");
  for (size_t i = 0; i < lidar_points.numel; ++i)
    if ((lidar_points.ptr<nvtype::half>()[i].__x & 0x7c00u) == 0x7c00u)
      throw std::runtime_error("points.tensor contains NaN or infinity");
  const int point_count = static_cast<int>(std::min<int64_t>(lidar_points.size(0), 300000));
  if (lidar_points.size(0) > point_count)
    std::cout << "Input point count " << lidar_points.size(0) << " exceeds core capacity; processing first " << point_count << ".\n";

  StreamGuard stream;
  int device = 0, runtime_version = 0, driver_version = 0;
  cudaDeviceProp gpu{};
  checkRuntime(cudaGetDevice(&device));
  checkRuntime(cudaGetDeviceProperties(&gpu, device));
  checkRuntime(cudaRuntimeGetVersion(&runtime_version));
  checkRuntime(cudaDriverGetVersion(&driver_version));
  std::vector<MemorySnapshot> memory;
  memory.push_back(memory_snapshot("before_engine"));
  dlopen("libcustom_layernorm.so", RTLD_NOW);
  auto core = create_core(options.model, options.precision);
  if (!core) throw std::runtime_error("Failed to create BEVFusion core");
  core->print();
  core->update(camera2lidar.ptr<float>(), camera_intrinsics.ptr<float>(), lidar2image.ptr<float>(), img_aug_matrix.ptr<float>(),
               stream.value);
  checkRuntime(cudaStreamSynchronize(stream.value));
  auto forward = [&]() {
    return core->forward((const unsigned char**)images.values.data(), lidar_points.ptr<nvtype::half>(), point_count, stream.value);
  };

  if (!options.benchmark) {
    core->set_timer(true);
    auto boxes = forward();
    validate_boxes(boxes, "warmup", 1);
    for (int i = 0; i < 5; ++i) validate_boxes(forward(), "legacy profile", i + 1);
    visualize(boxes, lidar_points, images.values, lidar2image, options.visualization, stream.value);
    return 0;
  }

  std::cout << "Benchmark: repeated single frame, batch=1, GPU=" << gpu.name << ", warmup=" << options.warmup
            << ", measured=" << options.runs << ", separate_profile=" << options.profile_runs << '\n';
  core->set_timer(false);
  core->set_validation(true);
  validate_boxes(forward(), "raw head preflight", 1);
  core->set_validation(false);
  std::cout << "Untimed preflight passed: all six raw head outputs are finite before decoding.\n";
  for (int i = 0; i < options.warmup; ++i) validate_boxes(forward(), "warmup", i + 1);
  checkRuntime(cudaStreamSynchronize(stream.value));
  memory.push_back(memory_snapshot("after_warmup"));

  std::vector<FrameTiming> frames;
  frames.reserve(options.runs);
  std::vector<BoundingBox> latest_boxes;
  for (int i = 0; i < options.runs; ++i) {
    checkRuntime(cudaStreamSynchronize(stream.value));
    const auto begin = std::chrono::steady_clock::now();
    auto boxes = forward();
    checkRuntime(cudaStreamSynchronize(stream.value));
    const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (!std::isfinite(milliseconds) || milliseconds <= 0) throw std::runtime_error("Invalid measured latency");
    validate_boxes(boxes, "measured", i + 1);
    frames.push_back({milliseconds, boxes.size()});
    latest_boxes = std::move(boxes);
  }
  memory.push_back(memory_snapshot("after_measured"));
  const auto summary = summarize(frames);
  std::cout << std::fixed << std::setprecision(3)
            << "Normal forward wall latency (ms): mean=" << summary.mean << " median=" << summary.median
            << " p90=" << summary.p90 << " p95=" << summary.p95 << " min=" << summary.minimum
            << " max=" << summary.maximum << ", FPS=" << summary.fps << ", final_boxes=" << latest_boxes.size() << '\n';
  std::cout << "Timing includes host-to-device input copies and preprocessing through decoded host boxes; "
               "excludes disk IO, engine load, calibration update, preflight, warmup, validation and visualization.\n";
  if (options.profile_runs > 0) {
    std::cout << "Separate stage profile follows; excluded from benchmark statistics. Instrumentation adds "
                 "stage synchronizations and changes execution order. Its Total excludes CopyLidar/ImageNorm "
                 "and is not end-to-end latency.\n";
    core->set_timer(true);
    for (int i = 0; i < options.profile_runs; ++i) {
      std::cout << "Profile iteration " << i + 1 << '/' << options.profile_runs << '\n';
      validate_boxes(forward(), "stage profile", i + 1);
    }
    core->set_timer(false);
  }
  if (!options.visualization.empty())
    visualize(latest_boxes, lidar_points, images.values, lidar2image, options.visualization, stream.value);
  write_benchmark_json(options, gpu, device, runtime_version, driver_version, lidar_points.size(0), point_count, summary,
                        frames, latest_boxes, memory);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
      usage(argv[0]);
      return 0;
    }
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "BEVFusion: " << error.what() << '\n';
    return 2;
  }
}
