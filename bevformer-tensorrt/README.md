# BEVFormer TensorRT

一个可独立安装、构建 TensorRT engine、执行推理和导出测速报告的项目。无需安装原编译器项目、MLIR、Vortex、MMCV 或 MMDetection。

**2026-10-20 已在 RTX 3070 Ti Laptop 8 GB 上用独立项目代码成功构建并执行完整图**：真实六相机输入下，混合 INT8/FP16 中位延时 **420.78 ms（2.38 FPS）**，FP32 **924.93 ms（1.08 FPS）**。FP32 数值对照通过，混合精度仍未通过严格逐元素门限；详见 [本次独立验证](docs/validation-3070.md)。

**当前交付的是 BEVFormer-base 固定形状、无历史帧的完整网络到检测头输出。** 包含 ResNet-101＋DCNv2、FPN、BEV Encoder、检测 Decoder 和检测头。图内相机投影与 CAN-bus 元数据在导出时固定；本项目尚未提供真实视频时序输入、框解码和 nuScenes mAP/NDS 评估，不能直接作为车辆感知应用使用。

```text
固定形状 BEVFormer ONNX + 权重
                 ↓ adapt（首次适配）
TensorRT 插件兼容 ONNX
                 ↓ build（在目标 GPU 上构建）
TensorRT engine
                 ↓ run（预处理后的六相机张量）
BEV 特征 + 分类 logits + 3D 框编码 + 性能报告
```

“构建”指 ONNX → TensorRT engine。神经网络计算由 TensorRT 和 NVIDIA 插件执行；Python 负责构建、参数绑定与运行控制，PyTorch 提供 CUDA buffer、stream 和 event。

## 1. 环境

目标平台为 Linux x86_64＋NVIDIA GPU。已用过的环境组合：

| 组件 | 版本 |
|---|---|
| Python | 3.11 |
| TensorRT CUDA 12 | **10.13.3.9** |
| PyTorch | **2.7.0+cu126** |
| NumPy / ONNX | 1.26.4 / 1.23.0 |
| 驱动 | 历史测试为 580.126.18、580.178.04 |

PyTorch 是 `run`、`check`、`profile` 的必需依赖，但已有 CUDA 版本可复用。构建和校准通过 CUDA Driver 管理内存，不导入 Torch，以减少构建阶段额外的显存占用。TensorRT 自带 DCNv2 和 MSDeformAttn 插件，不需要额外编译第三方插件或安装 `nvcc`。TensorRT engine 应在目标 GPU/软件环境构建，不把其他 GPU 的 engine 当作可移植模型。

```bash
# 在本项目根目录执行。新建隔离环境并安装 CUDA Torch；下载量较大。
bash scripts/setup.sh --python python3.11 --install-torch
source .venv/bin/activate

# 国内普通 Python 依赖可选清华源或中科大源：
# bash scripts/setup.sh --python python3.11 --install-torch --mirror tsinghua
# bash scripts/setup.sh --python python3.11 --install-torch --mirror ustc

python -m bevformer_trt check --report results/environment.json
```

已有匹配的 CUDA Torch 环境时，先激活该环境，然后直接安装，避免重新下载 Torch：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-tensorrt.txt
python -m pip install --no-deps -e .
python -m bevformer_trt check --report results/environment.json
```

脚本的 `--reuse-system-packages` 仅暴露基础 Python 的包，不会继承另一个 venv 的包；已有 venv 以上述直接安装方式复用。`--venv` 可指定新目录；已有目录不会被脚本删除或重新初始化。

`check` 验证 CUDA、TensorRT 插件和一个小网络，不代表完整 BEVFormer 已验证。安装脚本不修改显卡驱动。Torch 使用 PyTorch 官方 CUDA 12.6 源；TensorRT 使用 PyPI，镜像选项只作用于普通 Python 依赖。

## 2. 准备模型包

源码包与模型资产包分开。将配套模型包解压到本项目根目录，应得到：

```text
models/bevformer-base/
├── adapted.onnx          # 已适配的完整图，包含权重
├── ranges.json           # 与此模型绑定的 backbone 校准范围
├── images.npy            # 固定合成输入，用于可复现数值验证
├── reference.npz         # 与 images.npy 对应的 PyTorch 参考输出
├── export-report.json    # 导出配置和来源
├── fixed-metadata.json   # 固定几何/元数据的范围说明
└── manifest.json         # 文件大小、SHA-256 与模型说明
```

这个 `images.npy` **不是 nuScenes 真实校准样本**。不能用它重新做 INT8 校准或评估检测精度。已有 `ranges.json` 来源于此前真实图像校准；范围能否跨图复用由程序检查绑定关系。

模型与数据的获取、输入格式和适配规则见 [模型与数据契约](docs/model-and-data.md)。源码可独立上传到代码仓库；大模型资产和数据遵循各自许可，通过单独资产包交付。

## 3. 构建和推理

以下命令使用配套的已适配 ONNX。所有输出选用新路径；程序会拒绝覆盖已有结果。

### FP32 基线

```bash
CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0 python -m bevformer_trt build \
  --onnx models/bevformer-base/adapted.onnx \
  --engine results/fp32/base.engine --precision fp32 --plugins-fp32 \
  --workspace-mib 4096 --builder-optimization-level 3 \
  --tactic-dram-mib 4096 --max-aux-streams 0 \
  --report results/fp32/build.json

CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0 python -m bevformer_trt run \
  --engine results/fp32/base.engine \
  --inputs models/bevformer-base/images.npy \
  --reference models/bevformer-base/reference.npz \
  --output results/fp32/run --warmup 5 --repeat 20
```

上述参数已在本次 8 GB 测试机完成构建。`--workspace-mib` 是候选实现的工作区上限，**不是总显存限制**；`--tactic-dram-mib` 也不是进程总显存上限。构建显存可能远大于推理显存，缩小工作区有时反而排除可用实现，不能保证其他环境同样成功。

### 混合 INT8/FP16

```bash
CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0 python -m bevformer_trt build \
  --onnx models/bevformer-base/adapted.onnx \
  --engine results/mixed/base.engine --precision mixed-int8 \
  --calibrated-ranges models/bevformer-base/ranges.json \
  --workspace-mib 1024 --builder-optimization-level 3 \
  --tactic-dram-mib 4096 --max-aux-streams 0 \
  --report results/mixed/build.json

CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0 python -m bevformer_trt run \
  --engine results/mixed/base.engine \
  --inputs models/bevformer-base/images.npy \
  --output results/mixed/run --warmup 5 --repeat 20
```

这里 backbone 普通卷积使用 INT8；入口、DCNv2 offset/mask 预测、FPN 和检测头等使用浮点精度。并非整个 backbone 的每个算子都为 INT8。公开输入输出保持 FP32。详细规则和真实数据校准命令见 [混合精度与校准](docs/precision-and-calibration.md)。

混合精度示例不加 `--reference`，报告会明确记录数值比较 `not_checked`。需要比较时显式传入同一输入的参考文件，并根据任务定义误差门限；程序不会把“输出有限”当作“精度达标”。默认严格逐元素门限通常不适合直接接受量化结果，检测任务最终需要 mAP/NDS 验收。

### 看结果

```text
results/fp32/
├── base.engine
├── build.json
├── build.layers.json     # TensorRT 实际层、格式、tactic
└── run/
    ├── outputs.npz
    └── report.json       # 环境、哈希、延时、输出检查
```

| 输出 | 形状 | 含义 |
|---|---|---|
| `bev_embed` | `[40000,1,256]` | 200×200 BEV 特征 |
| `all_cls_scores` | `[6,1,900,10]` | 六层 Decoder 的分类 logits |
| `all_bbox_preds` | `[6,1,900,10]` | 六层 Decoder 的框编码 |

这些是原始检测头张量，尚未变成最终车辆/行人的列表。

## 4. 性能与验证范围

**本次独立项目实测：2026-10-20，RTX 3070 Ti Laptop 8 GB。**

| 完整固定图、同一真实六相机输入 | 中位延时 | P95 | 延时倒数 | 严格数值对照 |
|---|---:|---:|---:|---|
| 混合 INT8/FP16 | 420.782 ms | 423.458 ms | 2.377 FPS | 未通过，输出均有限 |
| FP32，关闭 TF32 | 924.934 ms | 986.490 ms | 1.081 FPS | 通过 |

复用测试机已有的 CUDA 依赖环境，安装独立 wheel 后从源码目录外执行；模型计算不依赖原项目的 `gac` 包。**97 项 CPU 测试和 6 组 DCNv2/MSDA 插件测试全部通过。** 合成输入的 FP32 对 PyTorch 参考也通过，中位延时 809.661 ms；输入和运行时段不同，不与真实输入结果混合比较。

真实输入使用此前独立 native-MLIR FP32 输出作为参考；混合图的 BEV/分类/框相对 RMSE 为 **2.404%/0.528%/0.468%**。参考输出是预先保存的文件，运行本项目不需要该编译器。没有 nuScenes mAP/NDS 验收，也没有达到 30 FPS。构建参数、失败尝试和可下载证据见 [独立验证报告](docs/validation-3070.md)。

**以下仅为拆分前的历史 3090 实测。**

| 历史设备与模型 | 精度 | 中位延时 | 延时倒数 |
|---|---|---:|---:|
| RTX 3090 24 GB，完整固定图 BEVFormer-base | FP32，关闭 TF32 | 373.821 ms | 2.675 FPS |
| 同上 | 混合 INT8/FP16 | 204.073 ms | 4.900 FPS |

六相机输入 `[1,6,3,928,1600]`，预热 5 次、测量 20 次。计时包含 engine 提交至完成的设备时间，不含预处理、H2D/D2H、加载、构建和框解码；无 CUDA Graph replay。图使用固定合成投影、零 CAN-bus、无历史帧。详细边界见 [历史性能记录](docs/historical-results.md)。

同一计时边界也用于本次独立验证。真实测试图像不随源码/模型资产包分发；配套 `images.npy` 是固定合成输入，适合复现数值回归，不能据此要求速度与上述真实图像完全相同。

## 5. 代码入口与扩展

| 命令 | 用途 |
|---|---|
| `check` | 环境、插件和小网络验证 |
| `adapt` | 自定义 ONNX 节点适配到 TensorRT 插件 ABI |
| `build` | ONNX 解析、精度约束、engine 构建、层信息导出 |
| `run` | 绑定 I/O、执行、测速、数值比较 |
| `prepare-calibration` | 从 nuScenes mini 生成真实校准输入 |
| `backbone` | 提取与整图精确绑定的校准子图 |
| `calibration-ranges convert` | 校准缓存转独立范围文件 |
| `profile` | TensorRT 层耗时和可追踪执行 |
| `verify-plugins` | 给定算子 fixture 的插件数值验证 |

命令参数可通过 `python -m bevformer_trt <命令> --help` 查看。安装后也可使用 `bevformer-trt`。

```bash
python -m unittest discover -s tests -v

python -m bevformer_trt verify-plugins \
  --fixtures tests/fixtures --output results/plugin-validation

python -m bevformer_trt profile \
  --engine results/mixed/base.engine \
  --inputs models/bevformer-base/images.npy \
  --output-dir results/profile --mode layers --warmup 5 --repeat 20
```

CPU 测试用于检查图适配、精度规则、数据契约和缓存绑定，不会代替 GPU 构建、插件数值验证或整网精度评估。Profiler 会引入测量开销，逐层耗时不应直接代替正常 `run` 延时。

本项目以固定 ONNX 为输入。更换模型结构、分辨率、batch、checkpoint、投影方式或时序输入，需要重新导出，并重新验证插件契约、校准范围和输出；不能只修改输入文件形状。
