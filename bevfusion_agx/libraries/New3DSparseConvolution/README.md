# 从内置源码构建稀疏卷积与完整 BEVFusion

本指南对应 `lg/open-spconv-3070` 分支。默认后端为 `open`，支持 FP16；INT8 会被明确拒绝。底层 spconv、cumm 源码已纳入本项目，构建不下载它们的 GitHub 源码，也不使用 NVIDIA 预编译的稀疏卷积库。CUDA、cuBLAS 和完整 BEVFusion 所需的 TensorRT 仍是 NVIDIA 软件；Python 代码生成依赖仍通过 pip 安装，因此这不是全离线构建。

生成的底层库也叫 `libspconv.so`，但它与旧 NVIDIA 同名库的 API/ABI **不兼容**。请搭配本分支的 `libscn.so`、解析器和 BEVFusion 程序使用，不能把它直接复制覆盖旧程序的 NVIDIA 库。

当前范围为 Linux、CUDA、FP16、batch=1、XYZ/ZYX 坐标，以及 BEVFusion 使用的 SparseConvolution/SubMConv、Add、ReLU 和融合 ScatterDense 尾部。INT8、inverse convolution、NCHW32/NHWzC 和 NVIDIA CUDA Graph/辅助流选项未启用，构建器会拒绝不支持的配置。各层 `output_bound` 还承担输入容量上限，修改模型时必须同时覆盖实际输入和输出活跃点数。

## 本轮源码重建实测

2026-10-17 使用 `f3c9793` 内置源码加当前 `build_deps.sh` 的部分 venv 恢复修复，在 `/root/bevfusion-vendored-rebuild-20261017` 的全新依赖/构建目录完成重建。实际环境为 RTX 3090、CUDA 12.8.93、GCC 11.4、CMake 3.22.1；代码生成使用系统 Python 3.10.12，引擎构建使用 Python 3.11.15、TensorRT 10.13.3.9、ONNX 1.23.0。

底层 `libspconv.so`、兼容层和完整程序均重新编译，动态链接路径核对通过；四个 TensorRT 引擎的首次构建记录均为 `reused: false`。XYZ/ZYX 参考数值检查、8/8 非方形回归，以及官方样例和 mini_val 81/81 帧完整推理均通过运行检查。

| 完整 ResNet50 FP16 推理 | 平均延迟 | P95 | 单帧串行 FPS |
| --- | ---: | ---: | ---: |
| 官方样例，预热 10 次、测量 100 次 | 19.226500 ms | 19.548640 ms | 52.011548 |
| mini_val，首帧预热 10 次、每帧测量一次 | 22.242601 ms | 23.898073 ms | 44.958771 |

每帧测量前会执行不计时的 raw-head 预检，文件读取、标定、预热和可视化均不计入上述耗时；mini 不是新帧首次执行的冷路径。结果仅代表 RTX 3090，不推断 3070 Ti 性能，也不构成 mAP/NDS 或性能提升结论。源码来源、误差、显存和原始记录位置见 [分批验证记录](VALIDATION.md#本轮从内置源码全新重建)。

## 1. 获取项目并固定工作目录

以下命令在 Linux Bash 下执行。除显式括号中的子 shell 外，始终停留在项目根目录 `bevfusion_agx`，并在同一 shell 中依次执行。不要使用旧 NVIDIA 仓库的 clone 命令替代这里的地址。

```bash
git clone --branch lg/open-spconv-3070 --single-branch \
  https://github.com/luogantt/Lidar_AI_Solution_jetson_agx.git
cd Lidar_AI_Solution_jetson_agx/bevfusion_agx
export PROJECT_ROOT="$PWD"
export SCN_ROOT="$PROJECT_ROOT/libraries/New3DSparseConvolution"
export BEV_ROOT="$PROJECT_ROOT/CUDA-BEVFusion"
export NEWSCN_DEPS_ROOT="$SCN_ROOT/_deps-sm86"
export SCN_BUILD="$SCN_ROOT/build-sm86"
export BEV_BUILD="$BEV_ROOT/build/open"
export RESULT_ROOT="$PROJECT_ROOT/artifacts/results"
mkdir -p "$RESULT_ROOT"
git rev-parse HEAD
```

`third_party/spconv` 与 `third_party/cumm` 是完整的固定版本源码快照，无需初始化它们的 Git 子模块。版本、归档校验和及许可证位置记录在 [source-manifest.json](third_party/source-manifest.json)；兼容层来源和修改说明见 [UPSTREAM.md](UPSTREAM.md)。

## 2. 系统依赖与 CUDA 架构

下面以 Ubuntu 22.04、CUDA 12.8、GCC 11、SM 86 为例。RTX 3070 Ti 和 RTX 3090 的计算能力均为 8.6，见 [NVIDIA GPU 架构表](https://developer.nvidia.com/cuda/gpus)；其他 GPU 必须设置对应架构，并检查该 CUDA Toolkit 的支持范围。此流程要求 NVIDIA CUDA GPU，不能直接用于瑞芯微 NPU 或华为昇腾。

```bash
sudo apt-get update
sudo apt-get install -y build-essential gcc-11 g++-11 git cmake \
  python3 python3-venv python3-dev protobuf-compiler libprotobuf-dev unzip

export PYTHON=python3
export CC=gcc-11
export CXX=g++-11
export CUDA_HOME=/usr/local/cuda
export CUDA_Lib="$CUDA_HOME/lib64"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CUMM_CUDA_ARCH_LIST=8.6
export CMAKE_CUDA_ARCHITECTURES=86
export CUDASM=86
export BUILD_JOBS=2

"$PYTHON" --version
cmake --version
"$CUDACXX" --version
nvidia-smi
```

CUDA Toolkit 和驱动需另行安装；`nvidia-smi` 显示的是驱动支持的 CUDA 上限，实际编译版本以 `nvcc --version` 为准。建议 Python 3.10–3.12、CMake 3.20–3.x：固定的 NumPy 1.26.4 不支持 Python 3.13，主项目仍使用旧 `FindCUDA`，不要直接使用 CMake 4。Ubuntu 22.04 自带 Python 3.10 和 CMake 3.22 满足此流程。

cumm 的 Linux 构建内部还会引用 `/usr/local/cuda`。请确认该路径与 `CUDA_HOME` 指向同一套 Toolkit，不要混用多个 CUDA 版本。`BUILD_JOBS` 控制最终 CMake 编译并行度；代码生成中的 Python 原生扩展构建可能另有并行开销。

## 3. 从内置源码生成底层 libspconv.so

```bash
bash "$SCN_ROOT/build_deps.sh"
export SPCONV_LIB_DIR="$NEWSCN_DEPS_ROOT/spconv/example/libspconv/build/spconv/src"
test -f "$SPCONV_LIB_DIR/libspconv.so"
cat "$NEWSCN_DEPS_ROOT/build-manifest.txt"
```

脚本将项目内源码复制到可写的 `NEWSCN_DEPS_ROOT`，建立独立 Python venv，安装固定代码生成依赖，生成 SM 86 的推理内核，再编译动态库。上游源码版本为 spconv 2.3.8 / `263d6b47425ef843c82f997b12d8b714013d216c` 和 cumm 0.7.11 / `b8d83b6511b70746d04485d0d2ec00896b735430`。

`$NEWSCN_DEPS_ROOT/source-state.json` 记录当前源码树 SHA-256、commit 和许可哈希，`build-manifest.txt` 记录 CUDA、架构、构建脚本哈希及 Python 依赖版本。生成文件和编译结果不写回 `third_party`；Python 字节码缓存不计入源码指纹。使用自定义 Python 时，需同时安装该版本的 venv 和开发头文件包。

可以直接审查、修改 `third_party` 中的源码。脚本会记录源码内容校验和；已有目录缺少来源状态、项目内源码发生变化、或已复制的上游文件被修改时会拒绝继续，防止沿用旧构建而忽略补丁。修改源码、CUDA 或架构后，请显式选择新的依赖和构建目录，例如：

```bash
# 需要重新构建时才执行本段；后续所有命令沿用这些新目录。
export NEWSCN_DEPS_ROOT="$SCN_ROOT/_deps-sm86-patch1"
export SCN_BUILD="$SCN_ROOT/build-sm86-patch1"
export BEV_BUILD="$BEV_ROOT/build/open-patch1"
bash "$SCN_ROOT/build_deps.sh"
export SPCONV_LIB_DIR="$NEWSCN_DEPS_ROOT/spconv/example/libspconv/build/spconv/src"
```

CMake 配置时也会以只读的 `prepare_sources.py --check-only` 核对来源状态和源码内容；不会隐式创建副本。构建生成的附加文件允许保留，不会被当成上游源码。

## 4. 编译 libscn.so 和独立稀疏卷积检查程序

这一阶段不需要 TensorRT 或相机模型。

```bash
cmake -S "$SCN_ROOT" -B "$SCN_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
  -DNEWSCN_DEPS_ROOT="$NEWSCN_DEPS_ROOT" \
  -DPython3_EXECUTABLE="$(command -v "$PYTHON")"
cmake --build "$SCN_BUILD" --parallel "$BUILD_JOBS"
test -f "$SCN_BUILD/libscn.so"
test -x "$SCN_BUILD/sparse_backend_check"
```

`libscn.so` 是本项目的图执行兼容层，链接上一节生成的底层内核库。Protobuf 源码由 CMake 自动生成，无需执行旧 `make_pb.sh`。

下载固定校验和的公开测试数据并执行独立对照：

```bash
"$PYTHON" "$SCN_ROOT/fetch_fixtures.py" --output "$SCN_ROOT/fixtures/bevfusion"
PYTHON="$NEWSCN_DEPS_ROOT/venv/bin/python" \
  bash "$SCN_ROOT/validate.sh" "$SCN_BUILD" \
  "$SCN_ROOT/fixtures/bevfusion" "$SCN_BUILD/validation"
```

该检查同时运行 XYZ 与 ZYX 布局，比较参考输出、有限值和重复执行一致性。它验证稀疏分支，不代替下文的整套 BEVFusion 推理或检测精度评测。

## 5. 安装 TensorRT 10 的构建与 Python 环境

完整推理需要 **C++ 头文件、动态库和匹配版本的 Python API**。以下固定 TensorRT 10.13.3.9 / CUDA 12，Python 包为 NVIDIA 发布的 [tensorrt-cu12 10.13.3.9](https://pypi.org/project/tensorrt-cu12/10.13.3.9/)。先从 [NVIDIA TensorRT 下载页](https://developer.nvidia.com/tensorrt/download) 获取对应 Linux SDK，解压到实际路径，并设置：

```bash
# 将此路径改为实际解压目录。
export TENSORRT_ROOT=/opt/TensorRT-10.13.3.9
export TensorRT_Inc="$TENSORRT_ROOT/include"
export TensorRT_Lib="$TENSORRT_ROOT/lib"
test -f "$TensorRT_Inc/NvInfer.h"
test -f "$TensorRT_Lib/libnvinfer.so.10"
test -f "$TensorRT_Lib/libnvinfer_plugin.so.10"

"$PYTHON" -m venv "$PROJECT_ROOT/artifacts/infer-venv"
export INFER_PYTHON="$PROJECT_ROOT/artifacts/infer-venv/bin/python"
"$INFER_PYTHON" -m pip install --upgrade pip
"$INFER_PYTHON" -m pip install \
  'numpy==1.26.4' 'onnx==1.17.0' 'tensorrt-cu12==10.13.3.9'
"$INFER_PYTHON" -c 'import tensorrt as trt; print(trt.__version__)'
```

### 只有 Python wheel、没有完整 SDK 时

上面的 NVIDIA Python 包提供运行库，但不提供 C++ 头文件，也不能假定带有 `trtexec`。先完成 Python 环境安装，再用以下命令替代 SDK 的两个路径设置。头文件来自官方 [v10.13.3 标签](https://github.com/NVIDIA/TensorRT/releases/tag/v10.13.3)，校验到 commit `94e2b9ef6d2cce74c76cdad499cca36cc4949197`，版本宏为 10.13.3.9。

```bash
git clone --depth 1 --branch v10.13.3 --filter=blob:none --sparse \
  https://github.com/NVIDIA/TensorRT.git "$PROJECT_ROOT/artifacts/tensorrt-headers"
git -C "$PROJECT_ROOT/artifacts/tensorrt-headers" sparse-checkout set include
test "$(git -C "$PROJECT_ROOT/artifacts/tensorrt-headers" rev-parse HEAD)" = \
  94e2b9ef6d2cce74c76cdad499cca36cc4949197
export TensorRT_Inc="$PROJECT_ROOT/artifacts/tensorrt-headers/include"
export TensorRT_Lib="$("$INFER_PYTHON" - <<'PY'
from importlib.metadata import distribution
from pathlib import Path
dist = distribution('tensorrt-cu12-libs')
paths = {dist.locate_file(item).resolve().parent for item in dist.files
         if Path(str(item)).name == 'libnvinfer.so.10'}
assert len(paths) == 1, paths
print(paths.pop())
PY
)"
test -f "$TensorRT_Inc/NvInfer.h"
test -f "$TensorRT_Lib/libnvinfer.so.10"
test -f "$TensorRT_Lib/libnvinfer_plugin.so.10"
```

下面的 CMake 命令显式传递 `.so.10`，因此不依赖 SDK 才可能提供的无版本 `.so` 链接。不要将一个 TensorRT 版本的头文件与另一个版本的库混用。

## 6. 编译完整 BEVFusion 程序

```bash
cmake -S "$BEV_ROOT" -B "$BEV_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBEVFUSION_SPARSE_BACKEND=open \
  -DNEWSCN_DEPS_ROOT="$NEWSCN_DEPS_ROOT" \
  -DBEVFUSION_CUDA_ARCH="$CUDASM" \
  -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DCMAKE_BUILD_PYTHON=OFF \
  -DPython3_EXECUTABLE="$(command -v "$PYTHON")" \
  -DTENSORRT_INCLUDE_DIR="$TensorRT_Inc" \
  -DTENSORRT_LIBRARY="$TensorRT_Lib/libnvinfer.so.10" \
  -DTENSORRT_PLUGIN_LIBRARY="$TensorRT_Lib/libnvinfer_plugin.so.10"
cmake --build "$BEV_BUILD" --parallel "$BUILD_JOBS"

export LD_LIBRARY_PATH="$TensorRT_Lib:$CUDA_Lib:$BEV_BUILD:$BEV_BUILD/open-sparse:$SPCONV_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test -x "$BEV_BUILD/bevfusion"
ldd "$BEV_BUILD/libbevfusion_core.so"
```

确认 `ldd` 中没有 `not found`，且 `libspconv.so` 来自所选 `NEWSCN_DEPS_ROOT`。不要在 `LD_LIBRARY_PATH` 前面放旧 NVIDIA 稀疏卷积库。

| 文件 | 实际输出路径 |
| --- | --- |
| 底层稀疏 CUDA 内核 | `$SPCONV_LIB_DIR/libspconv.so` |
| 独立兼容层 | `$SCN_BUILD/libscn.so` |
| 独立检查程序 | `$SCN_BUILD/sparse_backend_check` |
| 完整程序使用的兼容层 | `$BEV_BUILD/open-sparse/libscn.so` |
| BEVFusion 主库 | `$BEV_BUILD/libbevfusion_core.so` |
| LayerNorm 插件 | `$BEV_BUILD/libcustom_layernorm.so` |
| 推理程序 | `$BEV_BUILD/bevfusion` |

## 7. 获取模型与样例，构建四个 TensorRT 引擎

模型和样例不包含在 Git 仓库。使用 [原官方 CUDA-BEVFusion README 的 Model and Data](../../CUDA-BEVFusion/README.md#model-and-data) 中提供的 NVIDIA Box 或百度网盘入口，获取 `model.zip` 与 `nuScenes-example-data.zip`。若原下载入口失效，需要取得相同的官方模型资源；此文档不把其他来源的模型当作等价替代。

```bash
# 改为已下载归档的实际路径。
export MODEL_ARCHIVE="$HOME/Downloads/model.zip"
export SAMPLE_ARCHIVE="$HOME/Downloads/nuScenes-example-data.zip"
unzip "$MODEL_ARCHIVE" -d "$BEV_ROOT"
unzip "$SAMPLE_ARCHIVE" -d "$BEV_ROOT"
for file in camera.backbone.onnx camera.vtransform.onnx fuser.onnx head.bbox.onnx lidar.backbone.xyz.onnx; do
  test -f "$BEV_ROOT/model/resnet50/$file" || exit 1
done
test -f "$BEV_ROOT/example-data/points.tensor"

"$INFER_PYTHON" "$BEV_ROOT/tool/build_fp16_engines.py" \
  --model-dir "$BEV_ROOT/model/resnet50" \
  --output-dir "$BEV_ROOT/model/resnet50/build" \
  --device 0 --workspace-mib 2048
```

解压后必须形成 `CUDA-BEVFusion/model/resnet50/` 和 `CUDA-BEVFusion/example-data/`；若归档另包了一层目录，应先整理到此结构。脚本直接调用 TensorRT 10 Python API，不需要 `trtexec`。四个 `.plan` 分别为 `camera.backbone.plan`、`camera.vtransform.plan`、`fuser.plan`、`head.bbox.plan`；`lidar.backbone.xyz.onnx` 则由 `libscn.so` 直接解析执行。

引擎使用 FP16、LINEAR/CHW 输入输出和 2048 MiB workspace 上限；内部 TensorRT 层可能混用 FP16/FP32。脚本检查静态形状与输入输出数量，构建失败即退出，并输出逐引擎 metadata、层信息和 `build_fp16_manifest.json`。默认重新构建；只有显式添加 `--reuse` 才会检查 ONNX/外部权重哈希、TensorRT 版本、GPU、配置、插件及引擎完整性后复用。不要直接沿用别人机器生成的 `.plan`。

标准 ResNet50 模型按上述命令构建。若另用带额外自定义算子的模型，构建脚本可重复接收 `--plugin /absolute/path/plugin.so`，但它只负责构建进程加载。C++ 程序目前仅主动加载 `libcustom_layernorm.so`；其他插件还需在推理进程通过 `LD_PRELOAD` 或新增加载代码载入同一库，否则引擎可能无法反序列化。

## 8. 单帧完整推理与计时

程序目前按工作目录解析 `model/resnet50`，因此下面用子 shell 切换到 `CUDA-BEVFusion`；外层仍停留在统一项目根目录。

```bash
(
  cd "$BEV_ROOT"
  "$BEV_BUILD/bevfusion" "$BEV_ROOT/example-data" resnet50 fp16 \
    --visualization "$RESULT_ROOT/example.jpg"
)

(
  cd "$BEV_ROOT"
  "$BEV_BUILD/bevfusion" "$BEV_ROOT/example-data" resnet50 fp16 \
    --warmup 10 --runs 100 --profile-runs 3 \
    --benchmark-json "$RESULT_ROOT/single-frame.json" \
    --visualization "$RESULT_ROOT/single-frame.jpg"
)
```

普通 forward 的同步墙钟时间覆盖点云复制、图像预处理、相机和 LiDAR 分支、融合、检测头与结果回传；模型加载、文件读取、标定更新、预热和可视化写文件不计入该数字。benchmark 还会在预热前执行一次不计时的完整 forward，检查解码前六路检测头输出均为有限值；因此记录的是预检、预热后的运行耗时，不是首次冷启动。

逐阶段 profiling 会插入同步，原来的阶段 `Total` 还排除 `CopyLidar` 和 `ImageNorm`，不能与完整 forward 墙钟时间混用。报告的 FPS 是单帧耗时倒数，不是相机采集或并行流水线吞吐。显存采样来自设备级 `cudaMemGetInfo`，不是进程独占峰值。

## 9. nuScenes mini 多帧输入与测试

准备标准 nuScenes mini 数据目录，包含 `v1.0-mini/*.json`、`samples/` 和 `sweeps/`。转换仅依赖 NumPy，无需 PyTorch、MMCV 或 nuScenes Python SDK。输出目录必须是新目录或空目录。

```bash
export NUSCENES_ROOT=/data/nuscenes
export MINI_INPUT="$PROJECT_ROOT/artifacts/mini-val-input"
"$INFER_PYTHON" "$BEV_ROOT/tool/prepare_nuscenes_mini.py" \
  --data-root "$NUSCENES_ROOT" --output-dir "$MINI_INPUT" \
  --split mini_val --copy-images

(
  cd "$BEV_ROOT"
  "$BEV_BUILD/bevfusion" resnet50 fp16 \
    --data-list "$MINI_INPUT/data-list.txt" \
    --warmup 10 --profile-runs 3 \
    --benchmark-json "$RESULT_ROOT/mini-val.json"
)
```

`mini_val` 为官方两个场景的 81 个关键帧；也可选 `mini_train` 或 `all`，转换及推理均支持 `--limit N`。`--data-list` 每个非空行是一个样本目录，相对路径按列表所在目录解析，`#` 开头整行注释被忽略。多帧模式每帧先执行一次不计时的完整 forward 校验六路检测头，再执行一次计时 forward，不能再传 `--runs`；额外预热只使用首帧。逐帧加载、标定检查和该预检均在计时区间之外，FPS 对应每帧已预检后的测量，不代表新帧第一次运行的冷路径。

转换器使用当前扫描和最近 9 个历史 sweep，依照各传感器采样时刻的 ego pose 变换，历史扫描移除近距离点，按测试范围严格过滤后再转 FP16。与原训练/导出配置可能随机挑选 9 个 sweep 的行为差异会记入 manifest；此流程用于推理健康和性能测试，不声称检测精度等价，也不计算 mAP。

### 每帧输入格式

| 文件 | 内容 |
| --- | --- |
| `0-FRONT.jpg`、`1-FRONT_RIGHT.jpg`、`2-FRONT_LEFT.jpg` | 前方三路原始相机 JPEG |
| `3-BACK.jpg`、`4-BACK_LEFT.jpg`、`5-BACK_RIGHT.jpg` | 后方三路原始相机 JPEG |
| `points.tensor` | FP16 `[N,5]`，列为 `x,y,z,intensity,time_lag_seconds` |
| `camera2lidar.tensor` | FP32 `[1,6,4,4]`，各相机到当前 LiDAR 变换 |
| `camera_intrinsics.tensor` | FP32 `[1,6,4,4]`，扩展相机内参 |
| `lidar2image.tensor` | FP32 `[1,6,4,4]`，LiDAR 到各相机像素坐标的投影 |
| `img_aug_matrix.tensor` | FP32 `[1,6,4,4]`，缩放 0.48、裁剪平移 `(-32,-176)` |

标定也接受 `[6,4,4]`。JPEG 保留原始 1600×900 图像，CUDA 在运行时缩放裁剪到 704×256 并归一化；不要预先裁剪后再使用同一组标定。点云范围在 float32 阶段按 `(-54,-54,-5) < (x,y,z) < (54,54,3)` 严格过滤，第五列是秒。当前推理入口最多接收 300,000 点，超出时会截断并在运行记录中反映；准备数据时应检查 manifest 中的点数。

`.tensor` 是本项目带 magic `0x33ff1101`、维数和 dtype 的二进制格式，不能用改名后的 `.npy` 代替。转换脚本直接写入该格式，并输出 `manifest.json` 与可直接消费的 `data-list.txt`。`--copy-images` 适合跨主机传输；默认符号链接只适用于原始图片仍可访问的环境。

## 可选：非方形小模型回归

完成上面的独立检查程序和 Python 环境后，还可用 NumPy 卷积参考检查 8 种 XYZ/ZYX 与 Scatter/Transpose/Reshape 组合：

```bash
"$INFER_PYTHON" "$BEV_ROOT/tool/generate_sparse_fixture.py" \
  --output "$SCN_ROOT/fixtures/non_square"
"$INFER_PYTHON" "$BEV_ROOT/tool/run_sparse_regression.py" \
  --checker "$SCN_BUILD/sparse_backend_check" \
  --fixtures "$SCN_ROOT/fixtures/non_square" \
  --output "$SCN_BUILD/regression"
```

这是独立小模型数值回归，与完整 BEVFusion 的运行健康检查互为补充。

## 旧 NVIDIA 后端

原后端仍可显式选择 `-DBEVFUSION_SPARSE_BACKEND=nvidia`，但需要单独取得对应 CPU/CUDA 版本的 NVIDIA 预编译库，并使用隔离的构建目录。上面的内置源码步骤均针对 `open` 后端。原项目文档中的 Jetson 配置、INT8 和历史性能表不代表本分支源码后端的当前实测结果。

底层 spconv/cumm 的开源许可不自动覆盖图执行兼容层；兼容层保留上游各文件原许可标记，没有将整个兼容层重新声明为 Apache/MIT。来源、修改与许可说明见 [UPSTREAM.md](UPSTREAM.md)。
