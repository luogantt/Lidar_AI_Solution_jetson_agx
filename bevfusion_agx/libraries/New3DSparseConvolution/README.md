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

`BEVFUSION_SPARSE_BACKEND=open` 是默认值；`nvidia` 可显式选择原有后端。
两个 `libspconv.so` 名字相同，API 不同。请使用独立构建目录，并检查 `ldd`
实际解析的路径。不要把 NVIDIA 的库目录放入 open 后端的 `LD_LIBRARY_PATH`。

兼容图引擎在 `libspconv/`，计算内核和代码生成器的完整源码已纳入
`third_party/spconv/`、`third_party/cumm/`，可以直接修改、审查和随项目提交。
open 后端从这些文件生成并编译 `libspconv.so`，构建脚本不从 GitHub 下载
spconv/cumm，也不使用 NVIDIA 预编译稀疏卷积库。

当前接入范围为 Linux、CUDA、FP16、batch=1、XYZ/ZYX 坐标，以及原有
BEVFusion 的 SparseConvolution/SubMConv、Add、ReLU 和融合 ScatterDense 尾部。
INT8、inverse convolution、NCHW32/NHWzC、NVIDIA CUDA Graph/辅助流选项未启用。
构建器会拒绝不支持的配置。各层 `output_bound` 也用作本层输入容量，因此应
覆盖实际输入和输出活跃点数；上游 BEVFusion fixture 使用统一上限。

## 1. 构建计算内核

需要 Linux、CMake >= 3.20、GCC、CUDA Toolkit、Python >= 3.10 和
Protobuf 开发包。Ubuntu 可安装：

```bash
sudo apt-get install build-essential cmake python3-dev python3-venv libprotobuf-dev protobuf-compiler
```

若通过 `PYTHON` 选择自定义 Python，需要安装该版本的开发头文件（`Python.h`）。

从 `bevfusion_agx` 目录执行：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUMM_CUDA_ARCH_LIST=8.6  # RTX 3070/3070 Ti/3090
export BUILD_JOBS=2           # 按主机 CPU/RAM 调整，首次编译需要时间
bash libraries/New3DSparseConvolution/build_deps.sh
```

脚本先核对 `third_party/source-manifest.json` 中固定的 spconv 2.3.8/cumm
0.7.11 commit 和许可文件，再把当前源码复制到可写的 `_deps`。生成文件、
独立 Python 环境和编译结果都留在 `_deps`，不会写回 `third_party`。
`_deps/source-state.json` 记录实际源码树 SHA-256、commit 和许可哈希；
`_deps/build-manifest.txt` 记录 CUDA、架构、构建脚本哈希及 Python 依赖版本。

源码修改后需要新构建目录。脚本会拒绝没有 `source-state.json` 的旧 `_deps`、
与当前 vendored 源码指纹不符的目录，以及被直接修改过的已复制源码，避免
悄悄沿用旧代码。请在 `third_party` 中改源码，用新目录重新生成，然后让
BEVFusion CMake 使用同一路径，例如：

```bash
export NEWSCN_DEPS_ROOT="$PWD/libraries/New3DSparseConvolution/_deps-patch1"
bash libraries/New3DSparseConvolution/build_deps.sh
# 在 CUDA-BEVFusion 目录执行 tool/run.sh 时会传递此环境变量。
# 直接调用 CMake 则加 -DNEWSCN_DEPS_ROOT="$NEWSCN_DEPS_ROOT"。
```

原始归档哈希用于记录来源，不会阻止本地源码补丁；当前源码内容另行计算指纹。
Python 的 `__pycache__`/`.pyc`/`.pyo` 不参与指纹，其余已复制源码会逐文件核对。
请勿把编译输出放进 `third_party`。CMake 配置时也通过 `--check-only` 只读核对
源码及依赖目录来源，未准备、旧版本或不匹配的依赖会被拒绝。
如果只想检查本地来源和复制行为，不编译 CUDA：

```bash
python3 libraries/New3DSparseConvolution/prepare_sources.py \
  --deps-root /tmp/bevfusion-source-check
```

这不等于完全离线构建：脚本仍通过 pip 安装 NumPy、pccm、ccimport 等通用
生成/打包依赖；无网络环境需事先准备相应 Python 包。CUDA Toolkit/驱动和
完整推理使用的 TensorRT 仍依赖 NVIDIA 软件。代码生成和构建不需要 PyTorch。

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

安装与运行环境相匹配的 TensorRT 开发头文件及库，准备原项目
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

`run.sh` 将默认 open 后端放在 `build/open`；显式选择的 NVIDIA 后端使用 `build`。
如果自定义了依赖位置，可直接配置：

```bash
cmake -S . -B build/open -DBEVFUSION_SPARSE_BACKEND=open \
  -DBEVFUSION_CUDA_ARCH=86 -DNEWSCN_DEPS_ROOT=/absolute/path/to/deps
cmake --build build/open --parallel 2
```

## 4. 完整 FP16 推理测速

TensorRT 10.x 可通过 Python API 构建四个 dense engine，无需 `trtexec`。
在 `CUDA-BEVFusion` 目录执行，选用已安装 `tensorrt`、`onnx` 的 Python：

```bash
python3 tool/build_fp16_engines.py --model-dir model/resnet50
./build/open/bevfusion example-data resnet50 fp16 \
  --warmup 10 --runs 100 --profile-runs 3 \
  --benchmark-json build/open/benchmark.json \
  --visualization build/open/benchmark.jpg
```

构建脚本默认重建。`--reuse` 仅复用模型/外部权重、GPU、TensorRT、配置和
引擎哈希均匹配且反序列化检查通过的缓存。FP16 允许内部混合精度 tactic；
各 engine 的边界 I/O 固定为 FP16、LINEAR。

测速使用正常 `Core::forward` 的同步 CPU 墙钟时间，包含 host 输入拷贝、
预处理、相机和 LiDAR、融合、检测框解码及结果回传；不包含文件读取、
引擎加载、标定更新、预热、数值检查和图片保存。单帧 FPS 为平均延迟的倒数，
没有跨帧并行。阶段 profile 独立执行，其原有 `Total` 不含输入拷贝和图像
归一化，不能代替端到端延迟。

正式测速前独立检查全部原始检测头输出的 NaN/Inf；执行失败立即退出。
这项检查和输出图片用于发现运行异常，不等于 nuScenes mAP/NDS 精度评估。
显存记录是 `cudaMemGetInfo` 的整卡快照，不是进程峰值。

### nuScenes mini 多帧输入

使用原始 `v1.0-mini` JSON、六路相机和 LiDAR 文件，纯 NumPy 转换即可：

```bash
python3 tool/prepare_nuscenes_mini.py --data-root /data/nuscenes \
  --output-dir /data/bevfusion-mini-val --split mini_val --copy-images
./build/open/bevfusion resnet50 fp16 \
  --data-list /data/bevfusion-mini-val/data-list.txt \
  --warmup 10 --profile-runs 3 --benchmark-json build/open/mini-val.json
```

转换器保留 JPEG 原始字节；计算各传感器采样时刻的坐标变换，合并当前和
最多九个历史 LiDAR 扫描，再过滤点云范围。默认选择最近九帧历史，等价于
上游 loader 的 `test_mode=True`；原训练仓库 YAML 未设置该参数时可能随机
选择九个历史帧。该差异和固定源码版本记录在 manifest，转换不代表精度一致性。

多帧模式一次加载引擎，默认遍历列表所有帧；`--limit N` 可限制帧数，不接受
`--runs`。每帧先做独立原始检测头预检查，再测一次正常推理；只有第一帧执行
指定预热次数。逐帧文件读取和标定更新在计时外，结果是预检查后的推理延迟，
不是实时传感器输入到结果输出的完整应用吞吐。点数超过当前 300,000 容量时，
转换器会报警并保留原点数，主程序会明确记录实际处理点数。

## 来源与 Autoware 参考

- [New3DSparseConvolution](https://github.com/riteshkhrn/Lidar_AI_Solution/tree/3f823115cb16d9ddca9561394b9fa49242ab8315/libraries/New3DSparseConvolution)：兼容图引擎来源。
- [traveller59/spconv](https://github.com/traveller59/spconv/blob/263d6b47425ef843c82f997b12d8b714013d216c/docs/PURE_CPP_BUILD.md)：纯 C++/CUDA 生成和部署方式，源码已在 `third_party/spconv`。
- [FindDefinition/cumm](https://github.com/FindDefinition/cumm/tree/b8d83b6511b70746d04485d0d2ec00896b735430)：矩阵计算和 tensorview 源码，已在 `third_party/cumm`。
- [Autoware spconv_cpp](https://github.com/autowarefoundation/spconv_cpp)：参考其独立 C++ 库和部署方式。这里为 SM86 重新生成内核，没有引入 ROS2 或 Autoware 的另一套 ONNX/plugin 接口。

源码版本、补丁和保留的许可标记见 [UPSTREAM.md](UPSTREAM.md)。底层
spconv/cumm 的开源许可不自动覆盖兼容层；上游兼容层混有 NVIDIA 原许可标记，
本分支保留原文，没有把整个兼容层重新声明为 Apache/MIT。
