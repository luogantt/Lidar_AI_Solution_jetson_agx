# BEVFusion 的源码稀疏卷积后端

本目录把 `riteshkhrn/New3DSparseConvolution` 接入现有 BEVFusion SCN 接口。
点云体素化、相机分支和 TensorRT 融合网络保留原流程：

```text
points → CUDA voxelization → New3DSparseConvolution/libscn.so
                                ↓
                       traveller59/spconv（源码生成的 libspconv.so）
                                ↓
                       FP16 LiDAR BEV → 原 TensorRT fuser/head
```

`BEVFUSION_SPARSE_BACKEND=nvidia` 是原有默认值；`open` 选择新后端。
两个 `libspconv.so` 名字相同，API 不同。请使用独立构建目录，并检查 `ldd`
实际解析的路径。不要把 NVIDIA 的库目录放入 open 后端的 `LD_LIBRARY_PATH`。

当前接入范围为 Linux、CUDA、FP16、batch=1、XYZ/ZYX 坐标，以及原有
BEVFusion 的 SparseConvolution/SubMConv、Add、ReLU 和融合 ScatterDense 尾部。
INT8、inverse convolution、NCHW32/NHWzC、NVIDIA CUDA Graph/辅助流选项未启用。
构建器会拒绝不支持的配置。各层 `output_bound` 也用作本层输入容量，因此应
覆盖实际输入和输出活跃点数；上游 BEVFusion fixture 使用统一上限。

## 1. 构建计算内核

需要 Linux、CMake >= 3.20、GCC、CUDA Toolkit、Python >= 3.10、git/curl 和
Protobuf 开发包。Ubuntu 可安装：

```bash
sudo apt-get install build-essential cmake curl python3-venv libprotobuf-dev protobuf-compiler
```

从 `bevfusion_agx` 目录执行：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUMM_CUDA_ARCH_LIST=8.6  # RTX 3070/3070 Ti/3090
export BUILD_JOBS=2           # 按主机 CPU/RAM 调整，首次编译需要时间
bash libraries/New3DSparseConvolution/build_deps.sh
```

脚本在 `_deps` 中建立独立 Python 环境，固定 spconv 2.3.8/cumm 0.7.11
的源码 commit，生成目标 GPU 的 C++/CUDA 并编译；不需要 PyTorch，也不会
使用候选仓库附带的二进制。`_deps/build-manifest.txt` 记录构建依赖。
可设置 `NEWSCN_DEPS_ROOT` 改变依赖位置，并把同一路径传给后续 CMake。
首次下载失败可以重试；已存在但没有版本记录的源码目录会被拒绝覆盖。

## 2. 独立数值验证（不需要 TensorRT）

```bash
cmake -S libraries/New3DSparseConvolution \
  -B libraries/New3DSparseConvolution/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build libraries/New3DSparseConvolution/build --parallel 2
python3 libraries/New3DSparseConvolution/fetch_fixtures.py
python3 -m pip install numpy
bash libraries/New3DSparseConvolution/validate.sh
```

也可以让 `PYTHON` 指向 `_deps/venv/bin/python`，复用已经安装的 NumPy。
fixture 下载脚本使用固定上游 commit 和 SHA-256 校验。验证器使用非默认、
非阻塞 CUDA stream，每种坐标顺序预热 2 次、运行 10 次，保存首轮与末轮结果。
Python 比较器检查完整 shape、有限值、max abs、MAE、RMSE、relative L2、cosine，
并默认要求首末输出逐元素一致。失败返回非零状态；结果写到 `build/validation`。

单个模型可以直接运行 `build/sparse_backend_check --help`，数值阈值可通过
`CUDA-BEVFusion/tool/compare_sparse_output.py --help` 调整。
数值验证通过仅证明所测稀疏主干样本，不能替代全量 nuScenes 精度评估。

非方形小模型回归使用独立 NumPy 卷积参考，检查 8 种 XYZ/ZYX 和
Scatter/Transpose/Reshape 组合：

```bash
python3 -m pip install numpy onnx
python3 CUDA-BEVFusion/tool/generate_sparse_fixture.py \
  --output libraries/New3DSparseConvolution/fixtures/non_square
python3 CUDA-BEVFusion/tool/run_sparse_regression.py \
  --checker libraries/New3DSparseConvolution/build/sparse_backend_check \
  --fixtures libraries/New3DSparseConvolution/fixtures/non_square \
  --output libraries/New3DSparseConvolution/build/regression
```

## 3. 编译现有 BEVFusion 主程序

安装与运行环境相匹配的 TensorRT 开发头文件、库及 `trtexec`，准备原项目
`model/resnet50` 和 `example-data`。TensorRT engine 必须为目标 GPU/版本重新构建。

```bash
cd CUDA-BEVFusion
export BEVFUSION_SPARSE_BACKEND=open
export DEBUG_MODEL=resnet50 DEBUG_PRECISION=fp16 CUDASM=86
export TensorRT_Inc=/path/to/TensorRT/include
export TensorRT_Lib=/path/to/TensorRT/lib
export TensorRT_Bin=/path/to/TensorRT/bin
# 根据原项目说明先构建相机、fuser 和 head 的 TensorRT engines。
bash tool/run.sh
```

`run.sh` 将 open 后端放在 `build/open`；原 NVIDIA 后端继续使用 `build`。
如果自定义了依赖位置，可直接配置：

```bash
cmake -S . -B build/open -DBEVFUSION_SPARSE_BACKEND=open \
  -DBEVFUSION_CUDA_ARCH=86 -DNEWSCN_DEPS_ROOT=/absolute/path/to/deps
cmake --build build/open --parallel 2
```

## 来源与 Autoware 参考

- [New3DSparseConvolution](https://github.com/riteshkhrn/Lidar_AI_Solution/tree/3f823115cb16d9ddca9561394b9fa49242ab8315/libraries/New3DSparseConvolution)：兼容图引擎来源。
- [traveller59/spconv](https://github.com/traveller59/spconv/blob/v2.3.8/docs/PURE_CPP_BUILD.md)：纯 C++/CUDA 生成和部署方式。
- [Autoware spconv_cpp](https://github.com/autowarefoundation/spconv_cpp)：参考其独立 C++ 库和部署方式。这里为 SM86 重新生成内核，没有引入 ROS2 或 Autoware 的另一套 ONNX/plugin 接口。

源码版本、补丁和保留的许可标记见 [UPSTREAM.md](UPSTREAM.md)。底层
spconv/cumm 的开源许可不自动覆盖兼容层；上游兼容层混有 NVIDIA 原许可标记，
本分支保留原文，没有把整个兼容层重新声明为 Apache/MIT。
