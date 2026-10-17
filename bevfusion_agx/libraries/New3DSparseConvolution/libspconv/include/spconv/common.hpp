#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "spconv/engine.hpp"

#define LOG(...)                                                         \
  do {                                                                     \
    if(spconv::verbose_) spconv::ssprint(__FILE__, __LINE__, __VA_ARGS__); \
  } while (false)

#define THROW_EXCEPTION(e, ...)                                  \
  do {                                                           \
    std::stringstream msg;                                       \
    spconv::sstream_print(msg, __FILE__, __LINE__, __VA_ARGS__); \
    throw e(msg.str());                                          \
  } while (false)

#define THROW_COND_EXCEPTION(cond, e,  ...)                         \
  do {                                                             \
    if (!cond) {                                                   \
      std::stringstream msg;                                       \
      spconv::sstream_print(msg, __FILE__, __LINE__, __VA_ARGS__); \
      throw e(msg.str());                                          \
    }                                                              \
  } while (false)

#define ASSERT(cond, ...)                                        \
  do {                                                           \
    if (!cond) spconv::ssprint(__FILE__, __LINE__, __VA_ARGS__); \
    assert(cond);                                                \
  } while (false)

namespace spconv {

extern bool verbose_;

// Operators keep fixed-capacity buffers. Reject incompatible inputs before
// passing their pointers to CUDA kernels or the generated sparse backend.
inline void validate_sparse_fp16(SparseDTensor& tensor,
                                int64_t point_capacity, int64_t channels) {
  const auto& features = tensor.features();
  const auto& indices = tensor.indices();
  const auto grid = tensor.grid_size();
  if (features.shape.size() != 2 || features.shape[0] <= 0 ||
      features.shape[0] > point_capacity || features.shape[1] != channels ||
      channels <= 0 || features.shape[0] > std::numeric_limits<int>::max() / channels ||
      features.dtype() != DataType::Float16 || !features.device() || !features.ptr() ||
      indices.shape.size() != 2 || indices.shape[0] != features.shape[0] ||
      indices.shape[1] != 4 || indices.dtype() != DataType::Int32 ||
      !indices.device() || !indices.ptr() || grid.size() != 3) {
    throw std::invalid_argument("Invalid FP16 sparse tensor shape, type, device, or capacity");
  }
  int64_t volume = 1;
  for (int dimension : grid) {
    if (dimension <= 0 || volume > std::numeric_limits<int>::max() / dimension) {
      throw std::invalid_argument("Sparse grid exceeds the supported positive int32 range");
    }
    volume *= dimension;
  }
}

inline size_t checked_fp16_buffer_bytes(int64_t point_capacity, int64_t channels) {
  if (point_capacity <= 0 || channels <= 0 ||
      point_capacity > std::numeric_limits<int>::max() / channels) {
    throw std::invalid_argument("Invalid sparse output capacity or channel count");
  }
  return static_cast<size_t>(point_capacity) * static_cast<size_t>(channels) * 2;
}

template <char Sep = ' ', class SStream, class T>
void sstream_print(SStream& ss, T val) {
  ss << val;
}

template <char Sep = ' ', class SStream, class T, class... TArgs>
void sstream_print(SStream& ss, T val, TArgs... args) {
  ss << val << Sep;
  sstream_print<Sep>(ss, args...);
}

template <char Sep = ' ', class... TArgs>
std::string ssprint(TArgs... args) {
  std::stringstream ss;
  sstream_print<Sep>(ss, args...);
  std::cout << ss.str() << std::endl;
  return ss.str();
}

// overload the ostream << opertor for precision
std::ostream& operator<<(std::ostream& out, const spconv::Precision precision);

class Operation {
 public:
  virtual void configure(
      spconv::Precision precision,
      std::unordered_map<std::string, float>& tensor_name_to_scale,
      std::unordered_map<std::string, std::shared_ptr<void>> parameters) = 0;
  virtual void forward(
      std::unordered_map<std::string, std::shared_ptr<SparseDTensor>>&  io_dict,
      void* stream) = 0;
};

class UnspportedOperation : public Operation {
 public:
  virtual void configure(
      spconv::Precision precision,
      std::unordered_map<std::string, float>& tensor_name_to_scale,
      std::unordered_map<std::string, std::shared_ptr<void>> parameters) {
    THROW_EXCEPTION(std::runtime_error, "Operation not implemented yet");
  };
  virtual void forward(
      std::unordered_map<std::string, std::shared_ptr<SparseDTensor>>&  io_dict,
      void* stream) {
    THROW_EXCEPTION(std::runtime_error, "Operation not implemented yet");
  }
};

class IdenityOperation : public Operation {
  public:
   IdenityOperation(std::string input_name, std::string output_name)
    :input_name_(input_name),
     output_name_(output_name) {}
     virtual void configure(
        spconv::Precision precision,
        std::unordered_map<std::string, float>& tensor_name_to_scale,
        std::unordered_map<std::string, std::shared_ptr<void>> parameters) {};
   virtual void forward(
       std::unordered_map<std::string, std::shared_ptr<SparseDTensor>>&  io_dict,
       void* stream) {
        auto input_iterator = io_dict.find(input_name_);
        THROW_COND_EXCEPTION((input_iterator != io_dict.end()), std::out_of_range,
                            input_name_, "input not found");
        auto input = input_iterator->second;
        auto output_iterator = io_dict.find(output_name_);
        THROW_COND_EXCEPTION((output_iterator != io_dict.end()), std::out_of_range,
                                 "cannot find output in io_dict", output_name_);
        auto out = output_iterator->second;
        out->features().reference((void*)input->features().ptr(),
        input->features().shape, input->features().dtype());
        out->indices().reference((void*)input->indices().ptr(),
        input->indices().shape, input->indices().dtype());
        out->set_grid_size(input->grid_size());
   }
  private:
  std::string input_name_;
  std::string output_name_;
};
};  // namespace spconv

#endif  // __COMMON_HPP__
