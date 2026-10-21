# Robot_infer：自动驾驶推理项目

这个仓库维护三套面向 NVIDIA GPU 的自动驾驶感知与视觉语言驾驶推理实现：

| 项目 | 传感器 | 当前输出 | 部署方式 | 当前实测设备 |
| --- | --- | --- | --- | --- |
| [BEVFusion](bevfusion_agx/README.md) | 六路相机 + LiDAR | 解码后的三维检测框和可视化 | C++、CUDA、TensorRT，以及源码构建的稀疏卷积后端 | RTX 3090 |
| [BEVFormer TensorRT](bevformer-tensorrt/README.md) | 六路相机 | BEV 特征、分类 logits 和三维框编码 | Python、TensorRT，支持 FP32 与混合 INT8/FP16 | RTX 3070 Ti Laptop |
| [SimLingo 量化](simlingo-quantization/README.md) | 前视相机 + 车速 + 目标点 | Qwen logits smoke、路线与速度 waypoint | PyTorch；默认 BF16 边界 + NF4，可选 FP16 边界实验 | RTX 3070 Ti Laptop |

三个目录可以独立安装和运行。BEVFusion 提供完整多传感器三维检测链路；BEVFormer TensorRT 当前交付固定形状的完整网络到检测头原始输出；SimLingo 子项目从官方 PyTorch checkpoint 生成 BF16/NF4 artifact，并重组为完整视觉语言驾驶模型。

```text
Robot_infer/
├── bevfusion_agx/       # 相机 + LiDAR BEVFusion C++/CUDA 推理
├── bevformer-tensorrt/  # 相机 BEVFormer TensorRT 构建、推理与验证工具
└── simlingo-quantization/ # SimLingo PyTorch/NF4 量化、导出与验证
```

## BEVFusion：相机与 LiDAR 融合推理

BEVFusion 将六路相机图像与 LiDAR 点云投影到统一 BEV 空间，融合两路特征后通过 TransFusion 检测头输出三维检测框。

![BEVFusion 与源码稀疏卷积后端架构](bevfusion_agx/docs/images/open-sparse-architecture.svg)

### 源码构建的 Sparse Convolution

当前 `master` 默认不再加载 NVIDIA 项目附带的预编译闭源稀疏 `libspconv.so`。稀疏主干改为由仓库内固定版本的 spconv 2.3.8、cumm 0.7.11 和 C++/CUDA 兼容层源码构建：

- `libspconv.so`：从项目内 vendored spconv/cumm 源码生成并编译，负责索引生成、Mask Implicit GEMM 和底层 CUDA 稀疏卷积；
- `libscn.so`：源码可见的执行与适配层，负责稀疏张量、计算图、卷积调度、残差操作和 SparseToDense；
- BEVFusion 主程序使用本分支的解析器和以上两个动态库完成 LiDAR 稀疏主干推理。

这是一套可审查、可修改、可针对目标 GPU 重新编译的替代实现，不是 NVIDIA 闭源动态库的源码或反编译结果。新旧 `libspconv.so` 虽然同名，但 API/ABI 不兼容，不能把新库单独复制进旧程序。当前源码稀疏后端支持 FP16，不支持 INT8；完整推理仍依赖 NVIDIA 驱动、CUDA、cuBLAS/cuBLASLt 和 TensorRT。

### 当前验证结果

2026-10-17 在 RTX 3090、CUDA 12.8、TensorRT 10.13.3.9、FP16、SM 86 环境完成源码重建和完整推理验证：

| 输入 | 平均延迟 | P95 | 单帧串行 FPS |
| --- | ---: | ---: | ---: |
| 官方六相机 + 242,180 点样例，预热 10 次、测量 100 次 | 19.226500 ms | 19.548640 ms | 52.011548 |
| nuScenes mini_val，81/81 帧 | 22.242601 ms | 23.898073 ms | 44.958771 |

该计时覆盖正常 `Core::forward`，包含输入复制、预处理、相机与 LiDAR 分支、融合、检测头、解码和检测框返回，不包含文件读取、模型加载、预热和可视化。

![BEVFusion 官方样例完整推理结果](bevfusion_agx/docs/images/official-sample.jpg)

使用入口：

- [BEVFusion 项目首页](bevfusion_agx/README.md)
- [源码稀疏后端专题文章](bevfusion_agx/docs/BEVFUSION_OPEN_SPARSE_BACKEND.md)
- [从源码构建动态库和完整推理](bevfusion_agx/libraries/New3DSparseConvolution/README.md)
- [RTX 3090 验证记录](bevfusion_agx/libraries/New3DSparseConvolution/VALIDATION.md)

原 Jetson AGX 实现保留在 [`orin-AgX`](https://github.com/luogantt/Robot_infer/tree/orin-AgX) 分支。

## BEVFormer：六相机 TensorRT 推理

`bevformer-tensorrt` 是一个可独立安装的 Python 项目，用于适配 BEVFormer ONNX、构建 TensorRT engine、执行推理、校准混合精度并导出验证和性能报告。它从已经导出的固定 ONNX 开始，不包含 checkpoint 到 ONNX 的导出器，也不要求安装原编译器项目、MLIR、Vortex、MMCV 或 MMDetection。

当前交付范围是固定形状的 BEVFormer-base：

```text
六相机输入 [1, 6, 3, 928, 1600]
             ↓
ResNet-101 + DCNv2 + FPN
             ↓
六层 BEV Encoder + 六层检测 Decoder
             ↓
bev_embed + all_cls_scores + all_bbox_preds
```

TensorRT 执行完整固定图，Python 负责构建、输入输出绑定、运行控制和报告生成。TensorRT 10.13.3.9 自带的 DCNv2 与 MSDeformAttn 插件用于相关算子，不需要另外编译第三方插件。

### 当前验证结果

2026-10-20 在 RTX 3070 Ti Laptop 8 GB、TensorRT 10.13.3.9、PyTorch 2.7.0+cu126 上完成独立项目验证：

| 完整固定图、同一真实六相机输入 | 中位延迟 | P95 | 延时倒数 | 数值状态 |
| --- | ---: | ---: | ---: | --- |
| 混合 INT8/FP16 | 420.782 ms | 423.458 ms | 2.377 FPS | 可执行，严格逐元素比较未通过 |
| FP32，关闭 TF32 | 924.934 ms | 986.490 ms | 1.081 FPS | 严格比较通过 |

同时完成 97 项 CPU 测试和 6/6 组 DCNv2/MSDeformAttn 插件 fixture。当前混合精度结果尚未通过检测精度验收，也没有达到 30 FPS。

BEVFormer 的计时只覆盖 TensorRT engine 提交到 GPU 完成的设备时间，排除预处理、H2D/D2H、加载、构建和最终框解码；它与上方 BEVFusion 的完整 `Core::forward` 计时范围不同，因此不能直接横向比较。

使用入口：

- [BEVFormer TensorRT 项目首页](bevformer-tensorrt/README.md)
- [模型和数据契约](bevformer-tensorrt/docs/model-and-data.md)
- [混合精度与校准](bevformer-tensorrt/docs/precision-and-calibration.md)
- [RTX 3070 Ti Laptop 独立验证](bevformer-tensorrt/docs/validation-3070.md)

源码仓库不包含完整 ONNX 权重、TensorRT engine 或 nuScenes 数据。模型资产应按 [模型说明](bevformer-tensorrt/models/README.md) 单独取得，并在目标 GPU 和软件环境上重新构建 engine。

## SimLingo：RTX 3070 PyTorch/NF4 量化

`simlingo-quantization` 使用官方 SimLingo PyTorch checkpoint 和 InternVL2-1B 基础模型，在 CPU 合并 Qwen2 LoRA 后，只量化语言 transformer 内部的 168 个线性层。默认且推荐的配置保持 InternViT、token embeddings、`lm_head`、waypoint encoder 和驾驶输出头为 BF16；另提供将这些浮点边界改为 FP16、Qwen 168 层继续使用 NF4 的 opt-in 实验配置。

官方当前没有发布 SimLingo ONNX、TensorRT engine 或量化权重。这个子项目提供固定 revision 下载与 SHA-256 校验、最小上游源码快照、BF16/NF4/实验性 INT8 基准、持久化 artifact 导出、从磁盘重载对照，以及将拆分权重重组为完整 `DrivingModel` 的加载器。

2026-10-21 在 RTX 3070 Ti Laptop 8 GB 上完成真实模型与最终 NF4 artifact 验证：

| 合成 RGB 整链路：InternViT → Qwen → 驾驶头 | 平均延迟 | P90 | 峰值 CUDA 显存 | 状态 |
| --- | ---: | ---: | ---: | --- |
| BF16 | 114.35 ms | 114.93 ms | 2011.42 MiB | 路线与速度 waypoint 全部有限 |
| NF4 | 133.68 ms | 134.37 ms | 1498.98 MiB | 同形状、全部有限 |

NF4 的训练后状态导出 payload 为 1286.06 MiB，比对应 BF16 payload 小约 28.2%；运行时还需要固定的 InternVL2-1B 基础快照。整链路峰值显存下降约 25.5%，延迟增加约 16.9%。同一合成输入上，NF4 路线 waypoint 相对 BF16 的 mean absolute difference 为 0.0347、cosine 为 0.999978。INT8 artifact 虽然能够执行，但持久化重载后的 logits cosine 为 -0.4855、top-5 无重合，已明确拒绝。

这些是无 CARLA、无数据集的连通性和相对数值 smoke test，不是驾驶精度或闭环成绩。整链路 CUDA 计时只覆盖模型 forward，排除模型加载、CPU 图像处理和 H2D。使用入口：

- [SimLingo 量化项目首页](simlingo-quantization/README.md)
- [RTX 3070 Ti 验证记录](simlingo-quantization/docs/validation-rtx3070ti.md)
- [最终整链路原始结果](simlingo-quantization/results/rtx3070ti/full-pipeline-smoke.json)

### FP16 边界 + Qwen 168 层 NF4（实验）

同一台 RTX 3070 Ti Laptop 上补充测试了四种 560-token 语言模型组合；每组均预热 10 次、测量 50 次：

| seq560 配置 | Mean | P90 | 峰值 CUDA 显存 |
| --- | ---: | ---: | ---: |
| Dense BF16 边界 | 40.0197 ms | 40.3828 ms | 1997.49 MiB |
| Dense FP16 边界 | 36.8185 ms | 37.0839 ms | 1997.49 MiB |
| Qwen 168 层 NF4 + BF16 边界 | 46.8083 ms | 47.2070 ms | 1483.96 MiB |
| Qwen 168 层 NF4 + FP16 边界 | 48.0029 ms | 48.3000 ms | 1483.96 MiB |

FP16/NF4 的持久化语言 artifact 验证通过：hidden cosine 为 0.941288、logits cosine 为 0.960320，top-1 相同、top-5 重合 4/5。FP16 训练后 payload 为 1792.23 MiB，FP16/NF4 payload 为 1286.06 MiB，下降 28.2422%。相对 dense FP16，seq560 峰值显存下降 25.7088%。

但完整合成 RGB 链路中，FP16/NF4 的 route waypoint MAE 为 **0.051497**，超过预先设定的 smoke 阈值 **0.05**；虽然所有浮点边界 dtype 审计、有限值检查和其余阈值均通过，该完整链路仍判定为未通过。它只作为显式选择的实验配置，默认推荐仍是原 BF16 边界 + NF4，不把这组结果解释为真实驾驶精度。

```bash
cd simlingo-quantization
python export_quantized_artifacts.py \
  --model-dir models/simlingo \
  --output-dir artifacts/simlingo-3070-fp16-nf4 \
  --boundary-dtype fp16
```

实验详情和原始数据：

- [FP16 边界完整验证说明](simlingo-quantization/docs/validation-rtx3070ti.md#fp16-边界--qwen-168-层-nf4实验补充)
- [实验结论摘要 JSON](simlingo-quantization/results/rtx3070ti/mixed-fp16-nf4/evaluation-summary.json)
- [持久化语言 artifact 验证 JSON](simlingo-quantization/results/rtx3070ti/mixed-fp16-nf4/mixed-fp16-nf4-artifact-validation.json)
- [FP16/NF4 完整链路 JSON](simlingo-quantization/results/rtx3070ti/mixed-fp16-nf4/mixed-fp16-nf4-full-pipeline.json)

## 获取代码

```bash
git clone --recursive --branch master \
  https://github.com/luogantt/Robot_infer.git
cd Robot_infer
```

运行 BEVFusion 时进入 `bevfusion_agx/`，从其[完整构建指南](bevfusion_agx/libraries/New3DSparseConvolution/README.md)开始。运行 BEVFormer 时进入 `bevformer-tensorrt/`，按照其[项目 README](bevformer-tensorrt/README.md)准备环境与模型资产。运行 SimLingo 时进入 `simlingo-quantization/`，按其 README 下载固定 PyTorch 权重、导出 NF4 并执行重载与整链路 smoke test。

## 验证边界

- BEVFusion 当前公开性能来自物理 RTX 3090；RTX 3070 Ti 与 RTX 3090 都是 SM 86，但不能据此假定二者性能相同。
- BEVFusion 已跑通官方样例与 nuScenes mini_val 81 帧，但本轮没有计算 nuScenes mAP/NDS。
- BEVFormer 当前图内相机投影和 CAN-bus 元数据在导出时固定，无历史 BEV，输出仍是检测头原始张量。
- SimLingo 的 NF4 结果来自固定合成提示与合成 RGB 图像，没有真实样本标签、Bench2Drive 或 CARLA Driving Score；量化节省显存，但没有在 3070 Ti 上加速。FP16 边界实验的语言 artifact 通过，但完整链路 route MAE 超过预设门槛，因此没有替换默认 BF16/NF4。
- 三个项目都面向 NVIDIA CUDA 环境，不能直接部署到瑞芯微 NPU 或华为昇腾。
- 模型、数据集和第三方代码继续遵循各自许可证；仓库中的源码许可证不自动覆盖模型权重或数据集。
