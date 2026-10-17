#include <cuda_fp16.h>

#include <iostream>
#include <numeric>
#include <sstream>
#include <utility>

#include "sparseToDense.h"
#include "spconv/check.hpp"
#include "spconv/launch.cuh"

template <typename T>
__global__ void sparseToDense_kernel(size_t num_indices, T* features_data,
                                     int* indices_data, T* feature_map,
                                     bool isXYZ, int strideb, int stridec,
                                     int stridex, int stridey, int stridez,
                                     int out_channels) {
  int idx = cuda_linear_index;
  if (idx >= num_indices) return;
  int b, x, y, z;
  if (isXYZ) {
    b = indices_data[idx * 4 + 0];
    x = indices_data[idx * 4 + 1];
    y = indices_data[idx * 4 + 2];
    z = indices_data[idx * 4 + 3];
  } else {
    b = indices_data[idx * 4 + 0];
    z = indices_data[idx * 4 + 1];
    y = indices_data[idx * 4 + 2];
    x = indices_data[idx * 4 + 3];
  }
  for (int i = 0; i < out_channels; i++) {
    feature_map[b * strideb + i * stridec + z * stridez + x * stridex +
                y * stridey] = features_data[idx * out_channels + i];
  }
}

SparseToDense::SparseToDense(std::string input_name, std::string output_name,
                             std::string format, std::vector<int> spatial_shape,
                             std::vector<int> output_spatial_shape,
                             spconv::TensorLayout output_layout)
    : input_name_(input_name),
      output_name_(output_name),
      format_(format),
      spatial_shape_(spatial_shape),
      output_spatial_shape_(output_spatial_shape),
      output_layout_(output_layout) {
  if ((format_ != "xyz" && format_ != "zyx") || spatial_shape_.size() != 3 ||
      output_spatial_shape_.size() != 5 || output_spatial_shape_[0] != 1 ||
      output_layout_ != spconv::TensorLayout::NCHW) {
    throw std::invalid_argument("ScatterDense requires batch-one, 3D, FP16 NCHW output");
  }
  int64_t total_volume = 1;
  for (int dimension : output_spatial_shape_) {
    if (dimension <= 0 || total_volume > std::numeric_limits<int>::max() / dimension) {
      throw std::invalid_argument("ScatterDense shape exceeds the positive int32 indexing range");
    }
    total_volume *= dimension;
  }
  for (size_t axis = 0; axis < spatial_shape_.size(); ++axis) {
    if (spatial_shape_[axis] != output_spatial_shape_[axis + 2]) {
      throw std::invalid_argument("ScatterDense spatial and output shapes disagree");
    }
  }
  out_channels_ = output_spatial_shape[1];
  output_size_ = total_volume * sizeof(half);
  checkRuntime(cudaMalloc(&output_, output_size_));
  stride_.resize(output_spatial_shape_.size());
  for (int i = 0; i < output_spatial_shape_.size(); i++) {
    auto len = output_spatial_shape_[i];
    total_volume /= len;
    stride_[i] = total_volume;
  }
}

SparseToDense::~SparseToDense() {
  if (output_) checkRuntime(cudaFree(output_));
}

void SparseToDense::configure(
    spconv::Precision precision,
    std::unordered_map<std::string, float>& tensor_name_to_scale,
    std::unordered_map<std::string, std::shared_ptr<void>> parameters) {
  // Apply permutation before reshape regardless of unordered_map iteration order.
  // Strides stay indexed by the original five sparse axes; only the exposed
  // tensor shape changes when the following reshape is fused.
  auto configured_shape = output_spatial_shape_;
  auto configured_strides = stride_;
  const int64_t elements = output_size_ / sizeof(half);
  const auto permute_entry = parameters.find("dims");
  if (permute_entry != parameters.end()) {
    auto permute = std::static_pointer_cast<std::vector<int64_t>>(permute_entry->second);
    if (!permute || configured_shape.size() != 5 || permute->size() != 5) {
      throw std::invalid_argument("ScatterDense fusion requires a five-axis transpose");
    }
    std::vector<bool> seen(5, false);
    std::vector<int> transposed_shape;
    int64_t remaining = elements;
    for (auto axis : *permute) {
      if (axis < 0 || axis >= 5 || seen[axis]) {
        throw std::invalid_argument("ScatterDense transpose dimensions are not a permutation");
      }
      seen[axis] = true;
      const int dimension = configured_shape[axis];
      remaining /= dimension;
      configured_strides[axis] = static_cast<int>(remaining);
      transposed_shape.push_back(dimension);
    }
    configured_shape = std::move(transposed_shape);
  }
  const auto reshape_entry = parameters.find("reshape");
  if (reshape_entry != parameters.end()) {
    auto reshape_dims = std::static_pointer_cast<std::vector<int64_t>>(reshape_entry->second);
    if (!reshape_dims || reshape_dims->empty()) {
      throw std::invalid_argument("ScatterDense fused reshape must have explicit dimensions");
    }
    std::vector<int> reshaped;
    int inferred_axis = -1;
    int64_t known_elements = 1;
    for (auto dimension : *reshape_dims) {
      if (dimension == -1) {
        if (inferred_axis >= 0) {
          throw std::invalid_argument("ScatterDense fused reshape has multiple inferred dimensions");
        }
        inferred_axis = static_cast<int>(reshaped.size());
        reshaped.push_back(-1);
      } else {
        if (dimension <= 0 || dimension > elements || known_elements > elements / dimension) {
          throw std::invalid_argument("ScatterDense fused reshape has invalid dimensions");
        }
        known_elements *= dimension;
        reshaped.push_back(static_cast<int>(dimension));
      }
    }
    if (inferred_axis >= 0 && elements % known_elements == 0) {
      reshaped[inferred_axis] = static_cast<int>(elements / known_elements);
    } else if (inferred_axis >= 0 || known_elements != elements) {
      throw std::invalid_argument("ScatterDense fused reshape changes its element count");
    }
    configured_shape = std::move(reshaped);
  }
  output_spatial_shape_ = std::move(configured_shape);
  stride_ = std::move(configured_strides);
}

void SparseToDense::forward(
    std::unordered_map<std::string, std::shared_ptr<spconv::SparseDTensor>> &io_dict,
    void* stream) {
  // find the inputs
  auto input_iterator = io_dict.find(input_name_);
  THROW_COND_EXCEPTION((input_iterator != io_dict.end()), std::runtime_error,
                       "Cannot find input", input_name_);

  auto input = input_iterator->second;
  spconv::validate_sparse_fp16(*input, std::numeric_limits<int>::max(), out_channels_);
  if (input->grid_size() != spatial_shape_) {
    throw std::invalid_argument("ScatterDense input grid differs from the exported model grid");
  }

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  checkRuntime(cudaMemsetAsync(output_, 0, output_size_, _stream));
  auto input_shape = input->features().shape;
  int64_t num_indices = input_shape[0];
  // check coordinate order
  bool isXYZ = format_ == "xyz";
  int strideb, stridec, stridex, stridey, stridez;
  std::vector<int64_t> feature_map_shape(output_spatial_shape_.begin(),
                                       output_spatial_shape_.end());
  if (isXYZ) {
    strideb = stride_[0];
    stridec = stride_[1];
    stridex = stride_[2];
    stridey = stride_[3];
    stridez = stride_[4];
  } else {
    strideb = stride_[0];
    stridec = stride_[1];
    stridez = stride_[2];
    stridey = stride_[3];
    stridex = stride_[4];
  }
  // launch kernel
  cuda_linear_launch(sparseToDense_kernel<half>, _stream, num_indices,
                     reinterpret_cast<half*>(input->features().ptr()),
                     reinterpret_cast<int*>(input->indices().ptr()),
                     reinterpret_cast<half*>(output_), isXYZ, strideb, stridec,
                     stridex, stridey, stridez, out_channels_);
//  auto out = std::make_shared<spconv::SparseDTensorImplement>(output_name_);
  auto output_iterator = io_dict.find(output_name_);
  THROW_COND_EXCEPTION((output_iterator != io_dict.end()), std::out_of_range,
                           "cannot find output in io_dict", output_name_);
  auto out = output_iterator->second;
  out->features().reference((void*)output_, feature_map_shape, input->features().dtype());
  out->indices().reference(input->indices().ptr(), input->indices().shape, input->indices().dtype());
  out->set_grid_size(spatial_shape_);
  // out->set_device(input->device());
  // io_dict[output_name_] = out ;
}
