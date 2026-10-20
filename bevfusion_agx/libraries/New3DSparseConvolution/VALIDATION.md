# GPU 验证记录：2026-10-17（按批次保留）

验证时分支：`lg/open-spconv-3070`（原名 `codex/open-spconv-3070`）。
自 2026-10-20 起，该实现由 `master` 维护；原 Jetson AGX 版本保留在 `orin-AgX`。
用户提供的远端实际为 **RTX 3090 24 GB（SM 8.6）**，不是 3070/3070 Ti。
下面的性能和显存数据仅代表这台 3090。3070/3070 Ti 同为 SM 8.6，
本次生成了该架构的内核，但尚未在其物理设备上测量性能或完整模型显存。

本文保留首次接入验证、源码纳入时的本地检查，以及随后完成的全新源码重建，
不把不同批次的结果合并为同一次测试。

| 批次 | 构建来源与状态 | 结果位置 |
| --- | --- | --- |
| 首次 GPU 验证 | 下载固定版本源码后构建，完成稀疏与完整推理测试 | 本文“首次 GPU 验证”；本地 `artifacts/full-benchmark/` |
| 源码纳入时的检查 | 内置源码逐文件核对与构建入口静态检查，当时未重编译 CUDA | 本文“源码纳入项目时的本地检查” |
| 本轮全新源码重建 | `f3c9793` 内置源码加 venv 恢复修复，全新依赖/构建目录，重新生成四个 TensorRT 引擎 | 本文“本轮：从内置源码全新重建”；本地 `artifacts/vendor-source/results/` |

## 首次 GPU 验证

以下至“首次验证边界”的指标均来自首次测试；本轮重建结果在文末单独记录。

### 环境与构建

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

### 真实 BEVFusion 稀疏主干样例

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

### 非方形数值回归

`generate_sparse_fixture.py` 使用独立 NumPy 坐标查找实现卷积参考，
覆盖物理网格 `9×7×5`、32 个输入点、SubM 3³、残差 Add/ReLU、stride2 3³。
下采样后 60 个网格位置中 49 个有效，检查稀疏位置和 dense 清零。
XYZ/ZYX 各验证原始五维输出、Transpose、Reshape、Transpose+Reshape，共 **8/8 通过**。
标准 BEV 输出 `[1,48,5,4]`，XYZ/ZYX 参考逐元素相同。
各测试在非默认 stream 上预热 1 次、执行 3 次，首末结果完全相同。
8 个测试中最大绝对误差为 `0.00006103515625`，最低 cosine 为
`0.9999999986552003`。

### 完整 BEVFusion：官方 ResNet50 样例

使用 NVIDIA 官方 NVBox 中的 ResNet50 五个 ONNX 和完整相机/LiDAR 样例，
不是用随机输入模拟相机网络。下载逐成员校验 ZIP CRC，并记录 SHA-256；
四个 dense engine 在当前 RTX 3090、TensorRT 10.13.3.9 上重新构建，
边界 I/O 为 FP16 LINEAR，workspace 2048 MiB，内部允许 FP16/FP32/TF32 tactic。
引擎构建后重新反序列化检查形状、I/O 类型，严格缓存复用也通过。

输入为六张 1600×900 RGB 图像（网络输入各 704×256），加 **242,180 个点**。
正常 `Core::forward` 关闭逐阶段 timer，独立预检查后预热 10 次、测量 100 次：

| 指标 | 实测 |
| --- | ---: |
| 平均延迟 | 19.899014 ms |
| 中位延迟 | 19.749326 ms |
| P90 / P95 | 20.092498 / 20.417813 ms |
| 最小 / 最大 | 19.363298 / 25.894485 ms |
| 单帧串行 FPS（1000 / mean） | 50.254 |
| 所有测量轮的检测框数 | 33 |
| 预热后 / 测量后整卡显存快照 | 2453.125 MiB |

计时覆盖已加载的 host 图像/点云拷贝、归一化与体素化、相机和 LiDAR 主干、
BEV pool/变换、融合、检测头、解码及 host 检测框返回，并在 CUDA stream
完成后停止 CPU 墙钟。**不含磁盘读取、引擎加载、标定更新、预检查、预热及
可视化。** 没有跨帧流水线；这里的 FPS 不是读取数据集和保存图片的吞吐率。
显存为整卡快照，不是峰值，也不能直接保证其他设备显存足够。

原始检测头的六个输出在独立预检查中全部有限，100 次正式输出均无 NaN/Inf，
另一次单帧运行成功生成六视角/BEV 检测图。TensorRT 构建时提示 FP16 LayerNorm
溢出风险，本样例运行未发现非有限值；没有据此声称完成精度评估。

另外执行 3 次独立阶段 profile（插入同步、执行顺序略有差异）：

| 阶段 | 平均 ms |
| --- | ---: |
| 点云拷贝 | 0.711 |
| 图像上传与归一化 | 5.803 |
| LiDAR 体素化 + sparse backbone | 5.055 |
| 相机深度输入生成 | 0.026 |
| 相机 backbone | 3.759 |
| BEV pool | 0.342 |
| View transform | 0.469 |
| 融合 | 1.792 |
| 检测头 + 解码/结果回传 | 1.867 |

阶段 profile 的原 `Total` 约 13.31 ms，**排除了前两项**；报告的端到端
结果采用正常 forward 的 19.90 ms / 50.3 FPS。

### 完整 BEVFusion：nuScenes mini_val 81 帧

使用用户提供的原始 mini 数据，选取官方 `mini_val` 的两个场景：
`scene-0103`（40 帧）、`scene-0916`（41 帧），共 **81/81 帧成功**。
使用同一组官方 ResNet50 模型和上述 TensorRT engines，进程内一次创建 core。
每帧更新标定、独立预检查全部六个原始 head 输出，再测一次正常 forward；
首帧额外预热 10 次。预检查和标定更新均在计时外，因此为热输入推理延迟，
不代表数据集 runner 含读取/预检查的总吞吐。

| 指标 | 实测 |
| --- | ---: |
| 平均延迟 | 21.963639 ms |
| 中位延迟 | 22.163120 ms |
| P90 / P95 | 23.082881 / 23.294111 ms |
| 最小 / 最大 | 17.626745 / 23.439853 ms |
| 单帧串行 FPS（1000 / mean） | 45.530 |
| scene-0103 平均 / FPS | 21.448571 ms / 46.623 |
| scene-0916 平均 / FPS | 22.466145 ms / 44.511 |
| 每帧点数最小 / 最大 | 207,998 / 264,273 |
| 每帧点数平均 | 243,886.86 |
| 输入点数总计 / 实际处理总计 | 19,754,836 / 19,754,836 |
| 每帧检测框数范围 | 33–70 |
| 首帧预热后 / 全部测量后整卡显存快照 | 2453.125 / 2473.125 MiB |

81 次独立 raw-head 有限值检查、81 帧输出检测框检查均通过，没有点云截断。
末帧输出 36 个框，六视角和 BEV 可视化已生成并查看。
这里验证运行与性能，**未计算 nuScenes mAP/NDS，也未与 PyTorch 完整模型
或 NVIDIA 闭源后端逐帧比较检测框数值**。

输入转换由 `prepare_nuscenes_mini.py` 完成，依据 NVIDIA 指定的 MIT BEVFusion
commit `db75150717a9462cb60241e36ba28d65f6908607`，仅依赖 NumPy：
保留六路 JPEG、按传感器各自时刻的 ego pose 计算相机/历史扫描到当前 LiDAR
的变换，合并当前扫描和最近九个历史扫描，FP32 范围过滤后转 FP16。
两场景首帧按上游空历史规则补九份过滤后的当前扫描，其余帧均有九个历史扫描。
采用确定性的最近九帧（上游 loader 的 `test_mode=True`）；原 YAML 未设置
该参数时可能随机选择九帧。此差异写入 manifest，不能把本次结果当作未知
随机种子的训练预处理数值等价证明。

486 个 JPEG、405 个 tensor 的哈希、形状、类型和有限值均检查通过。
矩阵与独立 4×4 全局坐标链交叉检查；空/短/完整历史处理与固定上游 loader
比较通过。跨主机导出包 SHA-256 为
`1373507d47fe15a3b1924e7355ad365a93ae6d6a9a2eb019826ec18feb7ae5f8`。

完整结果 JSON、检测图、模型来源和日志位于本地 `artifacts/full-benchmark/`；
远端为 `results/full/`，mini 输入位于 `data/mini-val-81/`。这些数据文件不进入 Git。
复现命令见 [README.md](README.md) 的完整测速部分。

### 首次验证边界

- 已验证 FP16 稀疏主干的编译、数值、非方形坐标布局和重复推理。
- NVIDIA parser 分支做了使用原头文件的 C++ 语法检查；未重新测试其闭源运行库。
- 已运行相机 + LiDAR + 融合 + 检测头的完整推理；未完成 nuScenes mAP/NDS 评估。
- INT8、瑞芯微、昇腾以及其他 CUDA/CPU ABI 组合不在本次验证范围。

## 源码纳入项目时的本地检查

`third_party/spconv` 和 `third_party/cumm` 现已保存上述固定 commit 的完整
源码归档内容，共 426 个文件、14,728,810 字节。逐文件对照下载归档的原始
字节全部一致，原许可证和可执行文件标记保留；没有加入预编译库。
归档 SHA-256 及来源记录在 `third_party/source-manifest.json`。

构建改为从本地源码复制、生成并编译，默认后端改为 `open`；直接运行
open 可执行程序默认选择 ResNet50/FP16。Python 生成工具仍通过 pip 安装。
源码准备脚本的首次复制、重复校验、只读校验通过；对原始源码补丁、
构建副本篡改、旧目录缺少来源记录、源码中混入构建目录均能明确拒绝复用。
源码补丁可在新依赖目录成功复制，Python 缓存及构建新增文件按规则处理。
三个构建/运行 shell 脚本语法检查、Python AST 检查通过。
CMake 派生路径已验证：保留旧缓存时从依赖目录 A 切换到 B，三个头文件/库
路径均随之更新，包含空格的路径也通过；配置时额外执行只读源码来源检查。

该阶段当时没有重新编译或运行 CUDA：GPU 主机两次 SSH 连接均在握手阶段超时，
本地没有 CUDA 编译器。上面的 GPU 数值和速度结果仍是先前同版本源码的
测试记录。当时未完成的源码打包后重新测试，已在下面的独立批次完成。

复现步骤见 [README.md](README.md)。运行日志、逐项指标和二进制保存在远端
`/root/bevfusion-open-spconv-20261017/` 的 `results`、`build-sparse`、
`build-bevfusion` 和 `deps` 目录；这些大文件没有提交到 Git。

## 本轮：从内置源码全新重建

本轮使用 `f3c9793` 中内置的 spconv/cumm 源码，并加入当前
`build_deps.sh` 的部分 venv 恢复修复：若前次 `ensurepip` 失败留下了
`venv/bin/python`，但环境没有 pip，后续运行会重新执行 `python -m venv`。
因此这是“`f3c9793` 加该修复”的验证，不是未修改 commit 的构建。

远端新工作根目录为 `/root/bevfusion-vendored-rebuild-20261017`，以下记为
`BASE`。源码位于 `$BASE/repo`；本轮从空的依赖与构建目录重新复制源码、
生成内核、编译底层库和主工程，没有沿用首次验证的生成代码或稀疏动态库。
原始 ONNX、稀疏参考 fixture、官方样例和已转换的 mini 输入继续使用此前的
数据资产；四个 TensorRT `.plan` 则在本轮重新构建。

### 环境、构建与动态链接证据

| 项目 | 本轮实际值 |
| --- | --- |
| OS / GPU / 驱动 | Ubuntu 22.04 x86_64 / RTX 3090 24 GB / 580.126.18 |
| CUDA / 架构 | Toolkit 12.8.93，SM 86；`/usr/local/cuda` 指向 `/usr/local/cuda-12.8` |
| GCC / CMake / Protobuf | 11.4.0 / 3.22.1 / 3.12.4 |
| 源码生成及 CMake 校验 Python | 系统 Python 3.10.12 |
| TensorRT 引擎构建 Python | 3.11.15，TensorRT 10.13.3.9，ONNX 1.23.0 |
| TensorRT 头文件 | 官方 `v10.13.3`，commit `94e2b9ef6d2cce74c76cdad499cca36cc4949197` |
| 内置底层源码 | spconv 2.3.8 / cumm 0.7.11，与前述固定 commit 一致 |

首次尝试发现系统 Python 缺少 venv 支持；安装 `python3.10-venv` 和开发包后，
使用修复后的脚本继续完成构建。最终流程退出码为 0，源码只读校验通过。
新底层库、独立兼容层与检查程序、完整 `bevfusion` 及其主库/插件均编译成功。
`cuobjdump` 检查本轮 `libspconv.so` 的 858 个 ELF 条目和 `libscn.so` 的
6 个 ELF 条目，均为 SM 86，未发现其他 SM。

`linked-libraries.txt` 确认完整程序实际加载以下本轮产物：

| 文件 | 本轮路径（相对 `$BASE`） |
| --- | --- |
| 底层 `libspconv.so` | `repo/libraries/New3DSparseConvolution/_deps/spconv/example/libspconv/build/spconv/src/libspconv.so` |
| 完整程序使用的 `libscn.so` | `repo/CUDA-BEVFusion/build/open/open-sparse/libscn.so` |
| `libbevfusion_core.so` | `repo/CUDA-BEVFusion/build/open/libbevfusion_core.so` |

没有加载旧 NVIDIA 预编译稀疏库，也没有加载首次测试目录中的稀疏 `.so`。
TensorRT 运行库仍使用既有的同版本 Python wheel 中的 `.so.10`；这与本轮
重新构建稀疏库和 TensorRT 引擎是不同事项。

### 稀疏主干与非方形回归

XYZ/ZYX 使用与首次测试相同的 17,748 体素输入和匹配参考，输出为
FP16 `[1,256,180,180]`。每种布局预热 2 次、测量 10 次；以下稀疏耗时
为 GPU forward 时间，不含体素化、上传或完整 BEVFusion。

| 指标 | XYZ | ZYX |
| --- | ---: | ---: |
| 最大绝对误差 | 0.02734375 | 0.02734375 |
| MAE | 0.0000635603 | 0.0000635407 |
| RMSE | 0.0004459331 | 0.0004473437 |
| relative L2 | 0.0037703322 | 0.0037822512 |
| cosine similarity | 0.9999929156 | 0.9999928668 |
| 10 次运行平均耗时 | 3.352365 ms | 3.348710 ms |
| 10 次运行中位耗时 | 3.338240 ms | 3.338240 ms |
| 首次/末次输出最大差异 | 0 | 0 |

两组 `metrics.json` 均为 `passed: true`，通过原有阈值且无 NaN/Inf。
独立 NumPy 参考的非方形回归 **8/8 通过**，最大绝对误差
`0.00006103515625`，最低 cosine `0.9999999986552003`；所有首末输出一致。

### 四个 TensorRT 引擎重新生成

保存的 `engine-build-manifest.json` 状态为 `complete`，四个引擎均为
`reused: false`。边界为 FP16 LINEAR，workspace 为 2048 MiB，内部允许
FP16/FP32/TF32 tactic；没有额外自定义插件。构建后形状和输入输出类型检查通过，
随后单独执行的 `--reuse` 严格缓存校验也通过。

| 引擎 | 本轮构建耗时 |
| --- | ---: |
| camera.backbone | 91.469451 s |
| camera.vtransform | 3.857459 s |
| fuser | 15.308550 s |
| head.bbox | 75.762253 s |

这些是引擎构建时间，不是推理耗时。包含解析、校验和写入的整个引擎构建调用
耗时为 196.836695 s。

### 完整 ResNet50 FP16 推理

官方单帧包含六路图像和 242,180 点，独立 raw-head 预检查后预热 10 次、
测量 100 次。mini_val 复用同一 core 运行 81 帧；每帧先做一次不计时的
完整 raw-head 预检查，再测一次 forward，只有首帧额外预热 10 次。

| 指标 | 官方样例，100 次 | mini_val，81 帧 |
| --- | ---: | ---: |
| 平均延迟 | 19.226500 ms | 22.242601 ms |
| 中位延迟 | 19.191417 ms | 22.311182 ms |
| P90 | 19.372165 ms | 23.573383 ms |
| P95 | 19.548640 ms | 23.898073 ms |
| 最小 / 最大 | 18.979413 / 19.879265 ms | 17.268667 / 24.715826 ms |
| 单帧串行 FPS（1000 / mean） | 52.011548 | 44.958771 |
| 检测框数量 | 每轮 33 | 每帧 33–70，末帧 36 |
| 预热后整卡显存快照 | 2453.125 MiB | 2453.125 MiB |
| 测量后整卡显存快照 | 2453.125 MiB | 2473.125 MiB |

81 帧的输入点数和实际处理点数均为 19,754,836，没有点云截断。
官方样例的 1 次、mini 的 81 次原始检测头有限值预检查，以及所有正式输出
检测框的有限值检查均通过。独立的手动运行包装脚本无参数调用也成功退出，
生成 `manual-inference.jpg`；另有官方样例和 mini 末帧可视化。
本轮 `official-sample.jpg` 与 `mini-val-last-frame.jpg` 已打开检查，六相机
和 BEV 检测图均正常生成；可视化检查不等于检测精度评估。

计时采用正常 `Core::forward` 的同步 CPU 墙钟，覆盖 host 输入复制、预处理、
相机和 LiDAR 分支、融合、检测头、解码与 host 检测框返回；不含文件读取、
引擎加载、标定更新、预检查、预热、输出检查、图像保存和独立阶段 profile。
mini 的数字是每帧已经预检后的推理延迟，不是新帧第一次运行的冷路径，
也不是含数据读取的应用吞吐。显存记录仍是整卡快照，不是进程峰值。

### 本轮验证边界与原始记录

- 本轮确认内置源码构建链路、数值回归和完整 FP16 推理可运行；不据此宣称相较首次测试获得性能提升。
- TensorRT 构建仍提示 FP16 LayerNorm 溢出风险；所测输入未出现非有限值，但未计算 nuScenes mAP/NDS，也未与 PyTorch 或 NVIDIA 后端逐帧对比完整输出。
- mini 转换使用确定性的最近 9 个历史 sweep，保留前文说明的预处理差异；本轮复用同一份已校验输入数据。
- 所有速度与显存指标均来自 RTX 3090；3070 Ti、INT8、瑞芯微和昇腾未做本轮实测。

本地原始记录位于 `artifacts/vendor-source/results/`，包括
`official-sample.json`、`mini-val-81.json`、`engine-build-manifest.json`、
`bevfusion/*.metrics.json`、`non_square/summary.json`、动态链接和环境日志。
远端位于 `$BASE/results/`；这些大文件不进入 Git。运行步骤见 [README.md](README.md)。
