# GPU 验证记录：2026-10-17

验证分支：`codex/open-spconv-3070`。
用户提供的远端实际为 **RTX 3090 24 GB（SM 8.6）**，不是 3070/3070 Ti。
下面的性能和显存数据仅代表这台 3090。3070/3070 Ti 同为 SM 8.6，
本次生成了该架构的内核，但尚未在其物理设备上测量性能或完整模型显存。

## 环境与构建

| 项目 | 实测版本 |
| --- | --- |
| OS / CPU 架构 | Ubuntu 22.04 / x86_64 |
| GPU / 驱动 | RTX 3090 / 580.126.18 |
| CUDA Toolkit | 12.8.93 |
| GCC / CMake | 11.4 / 3.22.1 |
| Protobuf | 3.12.4 |
| TensorRT（主工程编译） | 10.13.3.9 |
| spconv | 2.3.8，`263d6b47425ef843c82f997b12d8b714013d216c` |
| cumm | 0.7.11，`b8d83b6511b70746d04485d0d2ec00896b735430` |
| New3DSparseConvolution 来源 | `3f823115cb16d9ddca9561394b9fa49242ab8315`，含本分支修复 |

使用单架构 `CUMM_CUDA_ARCH_LIST=8.6` 生成纯 C++/CUDA，再编译底层
`libspconv.so`、兼容层 `libscn.so` 和独立验证器。
`cuobjdump` 确认兼容层包含 `sm_86` cubin；`ldd` 确认加载的底层库来自
`deps/spconv/example/libspconv/build/spconv/src/libspconv.so`，没有使用
`libraries/3DSparseConvolution/libspconv/lib` 中的 NVIDIA 预编译库。

完整 C++ 主工程的 `bevfusion`、`bevfusion_core`、`custom_layernorm`
均编译成功，链接到新后端。TensorRT 头文件来自官方 `v10.13.3`
（`94e2b9ef6d2cce74c76cdad499cca36cc4949197`），与运行库 10.13.3.9 匹配。
本分支同时修复了 TensorRT 10 的 64 位 `Dims` 拷贝问题。

## 真实 BEVFusion 稀疏主干样例

使用固定上游 commit 的 ONNX、体素、坐标及参考 dense tensors；下载内容
经 SHA-256 校验。输入为 17,748 个体素，每体素 5 个 FP16 特征；
输出均为 FP16 `[1,256,180,180]`。XYZ/ZYX 各使用匹配的模型、网格和参考。

| 指标 | XYZ | ZYX |
| --- | ---: | ---: |
| 最大绝对误差 | 0.02734375 | 0.03125 |
| 平均绝对误差 MAE | 0.0000631346 | 0.0000633167 |
| RMSE | 0.000442169 | 0.000445180 |
| relative L2 | 0.00373850 | 0.00376396 |
| cosine similarity | 0.999993027 | 0.999992937 |
| 10 次运行中位耗时 | 3.329024 ms | 3.397632 ms |
| 10 次运行平均耗时 | 3.344131 ms | 3.422989 ms |
| 首次/末次输出最大差异 | 0 | 0 |

两组均通过 `validate.sh` 中预先设置的阈值：max abs ≤ 0.1、MAE ≤ 0.001、
RMSE ≤ 0.01、relative L2 ≤ 0.01、cosine ≥ 0.9999。没有 NaN/Inf。
每组预热 2 次、测量 10 次，使用非默认、非阻塞 CUDA stream。
计时只包括稀疏引擎 `forward`，不含模型加载、点云体素化、相机、融合、
检测头、输入传输及结果保存；未在这台机器上与 NVIDIA 闭源库做同条件测速。

预热后的整卡显存快照为 **1917.125 MiB**；引擎创建前为 265.125 MiB。
这是设备总体占用快照，不是峰值分配，也不是完整 BEVFusion 的显存需求。

## 非方形数值回归

`generate_sparse_fixture.py` 使用独立 NumPy 坐标查找实现卷积参考，
覆盖物理网格 `9×7×5`、32 个输入点、SubM 3³、残差 Add/ReLU、stride2 3³。
下采样后 60 个网格位置中 49 个有效，检查稀疏位置和 dense 清零。
XYZ/ZYX 各验证原始五维输出、Transpose、Reshape、Transpose+Reshape，共 **8/8 通过**。
标准 BEV 输出 `[1,48,5,4]`，XYZ/ZYX 参考逐元素相同。
各测试在非默认 stream 上预热 1 次、执行 3 次，首末结果完全相同。
8 个测试中最大绝对误差为 `0.00006103515625`，最低 cosine 为
`0.9999999986552003`。

## 验证边界

- 已验证 FP16 稀疏主干的编译、数值、非方形坐标布局和重复推理。
- NVIDIA parser 分支做了使用原头文件的 C++ 语法检查；未重新测试其闭源运行库。
- 本次没有运行相机 + LiDAR 的完整数据推理，也没有完成 nuScenes mAP/NDS 评估。
- INT8、瑞芯微、昇腾以及其他 CUDA/CPU ABI 组合不在本次验证范围。

复现步骤见 [README.md](README.md)。运行日志、逐项指标和二进制保存在远端
`/root/bevfusion-open-spconv-20261017/` 的 `results`、`build-sparse`、
`build-bevfusion` 和 `deps` 目录；这些大文件没有提交到 Git。
