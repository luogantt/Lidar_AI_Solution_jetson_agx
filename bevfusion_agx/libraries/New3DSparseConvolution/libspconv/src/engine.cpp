#include <cuda_runtime.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <spconv/engine.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sparseAdd.h"
#include "sparseConvImplicit.h"
#include "sparseFusedAddRelu.h"
#include "sparseRelu.h"
#include "sparseToDense.h"
#include "spconv/common.hpp"
#include "spconv/tensor.hpp"

namespace spconv {

class ITensorImplement: public ITensor, public std::enable_shared_from_this<ITensorImplement>{
public:
  ITensorImplement(std::string name): name_(name){};
  virtual const char* name() override {return name_.c_str();};
  virtual void  set_shape(std::vector<int64_t> shape) { shape_ = shape;};
  virtual const std::vector<int64_t> shape() {return shape_;};

  std::string name_;
  std::vector<int64_t> shape_;
};

class INodeImplement : public INode,  public std::enable_shared_from_this<INodeImplement>{
public:
  INodeImplement(std::string node_name, std::string node_optype):
      node_name_(node_name),node_optype_(node_optype) {};
  virtual void add_input(std::shared_ptr<ITensor> input) {
    inputs_.push_back(input);
  }
  virtual void add_output(std::shared_ptr<ITensor> output) {
    outputs_.push_back(output);
  }
  virtual void set_operation(std::shared_ptr<Operation> operation) {
    operation_ = operation;
  }
  virtual void set_input_precision(spconv::Precision precision) {
    precision_ = precision;
  }
  virtual void set_output_precision(spconv::Precision precision) {
    output_precision_ = precision;
  }
  virtual void set_attribute(std::string key, std::shared_ptr<void> value) {
    attribute_map_[key] = value;
  }
  virtual std::shared_ptr<void> get_attribute(std::string key) {
    auto iterator = attribute_map_.find(key);
    if(iterator != attribute_map_.end()) return iterator->second;
    return std::shared_ptr<void>(); // do not use make_shared<void>();
  }
  virtual std::shared_ptr<Operation> operation() { return operation_;}
  virtual spconv::Precision input_precision() { return precision_;}
  virtual spconv::Precision output_precision() { return output_precision_;}

  virtual const char* name() override {return node_name_.c_str();};
  virtual const char* optype() override {return node_optype_.c_str();};
  virtual ITensor* input(unsigned int index) override {return inputs_[index].get();};
  virtual ITensor* output(unsigned int index) override { return outputs_[index].get();};
  virtual unsigned int num_output() override {return outputs_.size();};
  virtual unsigned int num_input() override {return inputs_.size();};

private:
  std::string node_name_;
  std::string node_optype_;
  std::vector<std::shared_ptr<ITensor>> inputs_;
  std::vector<std::shared_ptr<ITensor>> outputs_;
  std::shared_ptr<Operation> operation_;
  spconv::Precision precision_, output_precision_;
  std::unordered_map<std::string, std::shared_ptr<void>> attribute_map_;
};

class SparseDTensorImplement : public SparseDTensor {
 public:
  SparseDTensorImplement(std::string name): name_(name){}
  virtual Tensor& features() override {return features_;};
  virtual Tensor& indices()  override {return indices_;};
  virtual void set_grid_size(const std::vector<int>& grid_size) override {
    grid_size_ = grid_size;
  }
  virtual std::vector<int> grid_size() const override {return grid_size_;};
  virtual void set_device(int device) {device_ = device;}
  virtual int device() const override {return device_;};

  virtual const char* name() const override {return name_.c_str();};

 private:
  std::string name_;
  Tensor features_;
  Tensor indices_;
  std::vector<int> grid_size_;
  int device_ = 0;
};

class EngineImplement : public Engine {
 public:
  EngineImplement(std::vector<std::shared_ptr<ITensor>> inputs,
                  std::vector<std::shared_ptr<ITensor>> outputs,
                  std::vector<std::shared_ptr<INodeImplement>> nodes) {
    nodes_ = nodes;
    // create io_dict
    for (auto node : nodes_) {
      for (int i = 0; i < node->num_input(); i++) {
        if (io_dict_.count(node->input(i)->name()) > 0) continue;
        auto input = std::make_shared<SparseDTensorImplement>(node->input(i)->name());
        io_dict_[node->input(i)->name()] = input;
      }
      for (int i = 0; i < node->num_output(); i++) {
        if (io_dict_.count(node->output(i)->name()) > 0) continue;
        auto output = std::make_shared<SparseDTensorImplement>(node->output(i)->name());
        io_dict_[node->output(i)->name()] = output;
      }
    }
    // create inputs tensor
    for (auto input : inputs) {
      auto input_iterator = io_dict_.find(input->name());
      THROW_COND_EXCEPTION((input_iterator != io_dict_.end()), std::out_of_range,
                           "cannot find input", input->name(), "in nodes");
      inputs_.push_back(input_iterator->second);
    }
    // create output tensor
    for (auto output : outputs) {
      auto output_iterator = io_dict_.find(output->name());
      THROW_COND_EXCEPTION((output_iterator != io_dict_.end()), std::out_of_range,
                           "cannot find output", output->name(), "in nodes");
      outputs_.push_back(output_iterator->second);
    }
  }
  ~EngineImplement() {}
  virtual void forward(void* stream = nullptr) {
    // for (auto& [key, value]: io_dict_) {
    //   LOG("Key", key, value.use_count());
    // }
    for (auto node : nodes_) {
      // LOG(node->name(), node.use_count());
      node->operation()->forward(io_dict_, stream);
    }
  };
  virtual size_t num_input() const { return inputs_.size();};
  virtual SparseDTensor* input(unsigned int index) {return inputs_[index].get();};
  virtual size_t num_output() const {return outputs_.size();};
  virtual SparseDTensor* output(unsigned int index) {return outputs_[index].get();};

 private:
  std::vector<std::shared_ptr<INodeImplement>> nodes_;
  std::vector<std::shared_ptr<SparseDTensor>> inputs_;
  std::vector<std::shared_ptr<SparseDTensor>> outputs_;
  std::unordered_map<std::string, std::shared_ptr<SparseDTensor>> io_dict_;
};

class EngineBuilderImplement : public EngineBuilder {
 public:
  std::vector<std::shared_ptr<ITensor>> inputs_;
  std::vector<std::shared_ptr<ITensor>> outputs_;
  std::vector<std::shared_ptr<INodeImplement>> nodes_;
  std::unordered_map<std::string, float> tensor_name_to_scale_;

  virtual ~EngineBuilderImplement() {}

  virtual INode* push_reshape(
      const char* name, ITensor* x,
      const std::vector<int64_t>& shape,
      const char* output_name) override {
      // set input properties
      auto inp_0 = dynamic_cast<ITensorImplement *>(x)->shared_from_this();
      std::string input_name_0 = inp_0->name();
      // set output properties
      auto out_0 = std::make_shared<ITensorImplement>(output_name);
      std::string output_name_0 = out_0->name();
      out_0->set_shape(shape);
      // create operation
      auto operation = std::make_shared<UnspportedOperation>();
      // create node
      LOG("Adding node", name, "of type Reshape",
        "with input shape", inp_0->shape(), "with output shape", out_0->shape());
      auto new_node = std::make_shared<INodeImplement>(name, "Reshape");
      new_node->add_input(inp_0);
      new_node->add_output(out_0);
      new_node->set_operation(operation);
      // add node
      nodes_.push_back(new_node);

      return new_node.get();
  }

  virtual INode* push_transpose(
      const char* name, ITensor* x,
      const std::vector<int64_t>& dims,
      const char* output_name) override {
      // set input properties
      auto inp_0 = dynamic_cast<ITensorImplement *>(x)->shared_from_this();
      std::string input_name_0 = inp_0->name();
      if (inp_0->shape().size() != dims.size()) {
        throw std::invalid_argument("Transpose rank does not match its input");
      }
      // set output properties
      auto out_0 = std::make_shared<ITensorImplement>(output_name);
      std::string output_name_0 = out_0->name();
      std::vector<int64_t> output_shape_0;
      std::vector<bool> seen(dims.size(), false);
      for (auto i : dims) {
        if (i < 0 || static_cast<size_t>(i) >= dims.size() || seen[i]) {
          throw std::invalid_argument("Transpose dimensions must be a permutation");
        }
        seen[i] = true;
        output_shape_0.push_back(inp_0->shape()[i]);
      }
      out_0->set_shape(output_shape_0);
      // create operation
      auto operation = std::make_shared<UnspportedOperation>();
      // create node
      LOG("Adding node", name, "of type Transpose",
        "with input shape", inp_0->shape(), "with output shape", out_0->shape());
      auto new_node = std::make_shared<INodeImplement>(name, "Transpose");
      new_node->add_input(inp_0);
      new_node->add_output(out_0);
      new_node->set_operation(operation);
      // add attribute
      auto _dims =
          std::make_shared<std::vector<int64_t>>(dims.begin(), dims.end());
      new_node->set_attribute("dims", _dims);
      // add node
      nodes_.push_back(new_node);

      return new_node.get();
  }

  virtual INode* push_dense(
        const char* name, ITensor* x,
        const char* format,
        const char* output_name,
        const std::vector<int>& input_spatial_shape,
        const std::vector<int>& output_shape,
        TensorLayout output_layout = TensorLayout::NCHW,
        float input_dynamic_range = 0.0f              // Enabled if int8 out
    ) override {
      // set input properties
      auto inp_0 = dynamic_cast<ITensorImplement *>(x)->shared_from_this();
      std::string input_name_0 = inp_0->name();
      // set output properties
      auto out_0 = std::make_shared<ITensorImplement>(output_name);
      std::string output_name_0 = out_0->name();
      std::vector<int64_t> output_shape_long(output_shape.begin(), output_shape.end());
      out_0->set_shape(output_shape_long);
      // get last node outbound and out channels
      int out_channels = out_0->shape()[1];
      // create operation
      auto operation = std::make_shared<SparseToDense>(input_name_0,
                                                        output_name_0, format,
                                                        input_spatial_shape,
                                                        output_shape,
                                                        output_layout);
      // create node
      LOG("Adding node", name, "of type SparseToDense",
          "with input shape", inp_0->shape(), "with output shape", out_0->shape());
      auto new_node = std::make_shared<INodeImplement>(name, "ScatterDense");
      new_node->add_input(inp_0);
      new_node->add_output(out_0);
      new_node->set_operation(operation);
      // add node
      nodes_.push_back(new_node);

      return new_node.get();
  }

  virtual INode* push_relu(
      const char* name,
      ITensor* x,
      const char* output_name) override {
    // set input properties
    auto inp_0 = dynamic_cast<ITensorImplement *>(x)->shared_from_this();
    std::string input_name_0 = inp_0->name();
    // set output properties
    auto out_0 = std::make_shared<ITensorImplement>(output_name);
    std::string output_name_0 = out_0->name();
    out_0->set_shape(inp_0->shape());
    // set last node outbound and out channels
    int output_bound = out_0->shape()[0];
    int out_channels = out_0->shape()[1];
    // create operation
    auto operation = std::make_shared<SparseRelu>(input_name_0, output_name_0,
                                                   output_bound, out_channels);
    // create node
    LOG("Adding node", name, "of type Relu",
        "with input shape", inp_0->shape(), "with output shape", out_0->shape());
    auto new_node = std::make_shared<INodeImplement>(name, "Relu");
    new_node->add_input(inp_0);
    new_node->add_output(out_0);
    new_node->set_operation(operation);
    // add node
    nodes_.push_back(new_node);

    return new_node.get();
  }

  virtual INode* push_add(
      const char* name,
      ITensor* a,
      ITensor* b,
      float a_dynamic_range,
      float b_dynamic_range,
      const char* output_name,
      Precision precision, Precision output_precision) {
    // TODO: do some checks
    THROW_COND_EXCEPTION(
        (precision == spconv::Precision::Int8 ||
         precision == spconv::Precision::Float16),
         std::runtime_error,
        "The input precision", precision,
        "is not valid, the supported choices are int8 or fp16");
    THROW_COND_EXCEPTION(
        (output_precision == spconv::Precision::Int8 ||
        output_precision == spconv::Precision::Float16),
        std::runtime_error,
        "The output precision", output_precision,
        "is not valid, the supported choices are int8 or fp16");
    // set input properties
    auto inp_0 = dynamic_cast<ITensorImplement *>(a)->shared_from_this();
    auto inp_1 = dynamic_cast<ITensorImplement *>(b)->shared_from_this();
    std::string input_name_0 = inp_0->name();
    std::string input_name_1 = inp_1->name();
    // set output properties
    auto out_0 = std::make_shared<ITensorImplement>(output_name);
    std::string output_name_0 = out_0->name();
    out_0->set_shape(inp_0->shape());
    // set last node outbound and out channels
    int output_bound = out_0->shape()[0];
    int out_channels = out_0->shape()[1];
    // set the tensor scales
    tensor_name_to_scale_[input_name_0] = a_dynamic_range / 127;
    tensor_name_to_scale_[input_name_1] = b_dynamic_range / 127;
    // create operation
    auto operation =  std::make_shared<SparseAdd>(
        input_name_0, input_name_1, precision, output_name_0,
        output_precision, output_bound, out_channels);
    // create node
    LOG("Adding node", name, "of type Add",
        "with input shape", inp_0->shape(), "with output shape", out_0->shape(),
        "with I/O precision", precision, "/", output_precision);
    auto new_node = std::make_shared<INodeImplement>(name, "Add");
    new_node->add_input(inp_0);
    new_node->add_input(inp_1);
    new_node->add_output(out_0);
    new_node->set_operation(operation);
    new_node->set_input_precision(precision);
    new_node->set_output_precision(output_precision);
    // add node
    nodes_.push_back(new_node);

    return new_node.get();
  }

  std::shared_ptr<INodeImplement> create_identity(
      const char* name,
      ITensor* a,
      ITensor* x) {
    // set input properties
    auto inp_0 = dynamic_cast<ITensorImplement *>(a)->shared_from_this();
    std::string input_name_0 = inp_0->name();
    // set output properties
    auto out_0 = dynamic_cast<ITensorImplement *>(x)->shared_from_this();
    std::string output_name_0 = out_0->name();
    // create operation
    auto operation = std::make_shared<IdenityOperation>(
        input_name_0, output_name_0);
    // create node
    LOG("Adding node", name, "of type identity",
        "with input shape", inp_0->shape(), "with output shape", out_0->shape());
    auto new_node = std::make_shared<INodeImplement>(name, "Identity");
    new_node->add_input(inp_0);
    new_node->add_output(out_0);
    new_node->set_operation(operation);

    return new_node;
  }

  std::shared_ptr<INodeImplement> create_add_relu(
      const char* name,
      ITensor* a,
      ITensor* b,
      ITensor* x,
      Precision precision, Precision output_precision) {
    // TODO: do some checks
    THROW_COND_EXCEPTION(
        (precision == spconv::Precision::Int8 ||
         precision == spconv::Precision::Float16),
         std::runtime_error,
        "The input precision", precision,
        "is not valid, the supported choices are int8 or fp16");
    THROW_COND_EXCEPTION(
        (output_precision == spconv::Precision::Int8 ||
         output_precision == spconv::Precision::Float16),
         std::runtime_error,
        "The output precision", output_precision,
        "is not valid, the supported choices are int8 or fp16");
    // set input properties
    auto inp_0 = dynamic_cast<ITensorImplement *>(a)->shared_from_this();
    auto inp_1 = dynamic_cast<ITensorImplement *>(b)->shared_from_this();
    std::string input_name_0 = inp_0->name();
    std::string input_name_1 = inp_1->name();
    // set output properties
    auto out_0 = dynamic_cast<ITensorImplement *>(x)->shared_from_this();
    std::string output_name_0 = out_0->name();
    // set last node outbound and out channels
    int output_bound = out_0->shape()[0];
    int out_channels = out_0->shape()[1];
    // create operation
    auto operation = std::make_shared<SparseFusedAddRelu>(
        input_name_0, input_name_1, precision, output_name_0,
        output_precision, output_bound, out_channels);
    // create node
    LOG("Adding node", name, "of type Add Relu",
        "with input shape", inp_0->shape(), "with output shape", out_0->shape(),
        "with I/O precision", precision, "/", output_precision);
    auto new_node = std::make_shared<INodeImplement>(name, "AddRelu");
    new_node->add_input(inp_0);
    new_node->add_input(inp_1);
    new_node->add_output(out_0);
    new_node->set_operation(operation);
    new_node->set_input_precision(precision);
    new_node->set_output_precision(output_precision);

    return new_node;
  }

  virtual INode* push_sparse_conv(
      const char* name,
      ITensor* x,
      const std::vector<unsigned short>& weight,
      const std::vector<int>& weight_shape,
      const std::vector<float>& weight_dynamic_ranges,
      const std::vector<unsigned short>& bias,
      const std::vector<int>& bias_shape,
      const char* activation,
      const std::vector<int>& kernel_size,
      const std::vector<int>& stride,
      const std::vector<int>& padding,
      const std::vector<int>& dilation,
      float input_dynamic_range,
      bool submanifold,
      int max_output_points,
      const char* rulebook,
      Precision precision,
      Precision output_precision,
      const char* output_name,
      bool inverse) override {
    if (inverse) {
      throw std::invalid_argument("Inverse sparse convolution is not supported by this backend");
    }
    THROW_COND_EXCEPTION(
        (precision == spconv::Precision::Int8 ||
         precision == spconv::Precision::Float16),
         std::runtime_error,
        "The input precision", precision,
        "is not valid, the supported choices are int8 or fp16");
    THROW_COND_EXCEPTION(
        (output_precision == spconv::Precision::Int8 ||
         output_precision == spconv::Precision::Float16),
         std::runtime_error,
        "The output precision", output_precision,
        "is not valid, the supported choices are int8 or fp16");
    // Initialize variables
    int ndim = 3;
    THROW_COND_EXCEPTION((weight_shape.size() == ndim+2), std::runtime_error,
                         "change the ndim accordingly");
    // TODO: do some checks to confirm the weight layout is KRSC, [out channels, *kernel_size, in channels]
    int out_channels = weight_shape[0];
    int in_channels = weight_shape[4];
    std::vector<int64_t> input_shape = {max_output_points, in_channels};
    std::vector<int64_t> output_shape = {max_output_points, out_channels};
    // set input properties
    auto inp_0 = dynamic_cast<ITensorImplement *>(x)->shared_from_this();
    std::string input_name_0 = inp_0->name();
    inp_0->set_shape(input_shape);
    // set output properties
    auto out_0 = std::make_shared<ITensorImplement>(output_name);
    std::string output_name_0 = out_0->name();
    out_0->set_shape(output_shape);
    // dummy values
    int transposed = 0;
    int groups = 0;
    std::vector<int> output_padding;
    std::vector<int> input_spatial_shape = {0, 0, 0};
    std::vector<int> output_spatial_shape = {0, 0, 0};
    // set the scale
    tensor_name_to_scale_[input_name_0] = input_dynamic_range / 127;
    // create operation
    auto operation = std::make_shared<SparseConvolution>(input_name_0,
        precision, output_name_0, output_precision,
        weight.data(), weight_shape, bias.data(), bias_shape, ndim,
        input_spatial_shape, output_spatial_shape, in_channels, out_channels,
        kernel_size, max_output_points, stride, dilation, padding, transposed,
        inverse, output_padding, groups, submanifold, rulebook, activation,
        input_shape, output_shape, input_dynamic_range, weight_dynamic_ranges);
    // create node
    LOG("Adding node", name, "of type Sparse Convolution",
        "with input shape", input_shape, "with output shape", output_shape,
        "with I/O precision", precision, "/", output_precision);
    auto new_node = std::make_shared<INodeImplement>(name, "SparseConvolution");
    new_node->add_input(inp_0);
    new_node->add_output(out_0);
    new_node->set_operation(operation);
    new_node->set_input_precision(precision);
    new_node->set_output_precision(output_precision);
    // add node
    nodes_.push_back(new_node);

    return new_node.get();
  }

  virtual ITensor* push_input(const char* name) override {
    auto inp = std::make_shared<ITensorImplement>(name);
    inputs_.push_back(inp);
    return inp.get();
  }

  virtual void push_output(ITensor* value) override {
    outputs_.push_back(dynamic_cast<ITensorImplement *>(value)->shared_from_this());
  }

  virtual std::shared_ptr<Engine> build(Precision precision, void* stream) {
    if (precision != Precision::Float16) {
      throw std::invalid_argument("The source sparse backend currently supports FP16 execution only");
    }
    // Only erase an intermediate tensor when it has one actual consumer and
    // is not a requested graph output. Adjacency alone does not establish an edge.
    std::unordered_map<ITensor*, size_t> consumer_counts;
    std::unordered_set<ITensor*> graph_outputs;
    for (const auto& output : outputs_) graph_outputs.insert(output.get());
    for (const auto& node : nodes_) {
      for (unsigned int i = 0; i < node->num_input(); ++i) {
        ++consumer_counts[node->input(i)];
      }
    }
    const auto can_fuse = [&](const std::shared_ptr<INodeImplement>& producer,
                              const std::shared_ptr<INodeImplement>& consumer) {
      if (producer->num_output() != 1 || consumer->num_input() != 1) return false;
      auto* intermediate = producer->output(0);
      return consumer->input(0) == intermediate && consumer_counts[intermediate] == 1 &&
             graph_outputs.count(intermediate) == 0;
    };
    std::vector<std::shared_ptr<INodeImplement>> tmp_nodes;
    std::unordered_map<std::string, std::shared_ptr<void>> parameters;
    for (size_t index = 0; index < nodes_.size(); ++index) {
      auto node = nodes_[index];
      if (!strcmp(node->optype(), "Add") && index + 1 < nodes_.size()) {
        const auto& next = nodes_[index + 1];
        if (!strcmp(next->optype(), "Relu") && can_fuse(node, next)) {
          LOG("Merging 2 Nodes types ADD and Relu");
          const std::string node_name = std::string(node->name()) + next->name();
          node = create_add_relu(node_name.c_str(), node->input(0), node->input(1),
                                 next->output(0), node->input_precision(),
                                 node->output_precision());
          ++index;
        }
      } else if (!strcmp(node->optype(), "ScatterDense")) {
        size_t next_index = index + 1;
        auto producer = node;
        std::unordered_map<std::string, std::shared_ptr<void>> fused_parameters;
        if (next_index < nodes_.size() && !strcmp(nodes_[next_index]->optype(), "Transpose")) {
          const auto& transpose = nodes_[next_index];
          if (!can_fuse(node, transpose)) {
            throw std::invalid_argument("ScatterDense/Transpose fusion requires an unmarked single-consumer edge");
          }
          fused_parameters["dims"] = transpose->get_attribute("dims");
          producer = transpose;
          ++next_index;
        }
        if (next_index < nodes_.size() && !strcmp(nodes_[next_index]->optype(), "Reshape")) {
          const auto& reshape = nodes_[next_index];
          if (!can_fuse(producer, reshape)) {
            throw std::invalid_argument("ScatterDense/Reshape fusion requires an unmarked single-consumer edge");
          }
          const auto reshape_dims = dynamic_cast<ITensorImplement*>(reshape->output(0))->shape();
          fused_parameters["reshape"] = std::make_shared<std::vector<int64_t>>(reshape_dims);
          node->operation()->configure(precision, tensor_name_to_scale_, fused_parameters);
          tmp_nodes.push_back(node);
          const std::string node_name = std::string(node->name()) + reshape->name();
          node = create_identity(node_name.c_str(), node->output(0), reshape->output(0));
          index = next_index;
        } else if (!fused_parameters.empty()) {
          // A directly adjacent five-dimensional transpose can also be
          // represented by the scatter strides without a following reshape.
          node->operation()->configure(precision, tensor_name_to_scale_, fused_parameters);
          tmp_nodes.push_back(node);
          const std::string node_name = std::string(node->name()) + producer->name();
          node = create_identity(node_name.c_str(), node->output(0), producer->output(0));
          index = next_index - 1;
        }
      } else if (!strcmp(node->optype(), "Transpose") || !strcmp(node->optype(), "Reshape")) {
        throw std::invalid_argument("Only Transpose/Reshape directly fused with ScatterDense are supported");
      }
      node->operation()->configure(precision, tensor_name_to_scale_, parameters);
      tmp_nodes.push_back(node);
    }
    return std::make_shared<EngineImplement>(inputs_, outputs_, tmp_nodes);
  }
};

std::shared_ptr<EngineBuilder> create_engine_builder() {
  std::shared_ptr<EngineBuilderImplement> impl(new EngineBuilderImplement());
  return impl;
}

bool verbose_ = false;
void set_verbose(bool enable) { verbose_ = enable; }
bool get_verbose() { return verbose_;};
const char* get_precision_string(Precision precision) {
  switch (precision) {
    case Precision::Float16: return "fp16";
    case Precision::Int8: return "int8";
    case Precision::None: return "none";
    default: return "unknown";
  }
}
const char* get_tensor_layout_string(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::NCHW: return "NCHW";
    case TensorLayout::NCHW32: return "NCHW32";
    case TensorLayout::None: return "none";
    default: return "unknown";
  }
}
};  // namespace spconv
