# SimLingo RTX 3070 Ti 量化验证

## 结论

本轮在物理 RTX 3070 Ti Laptop GPU 上完成官方 SimLingo checkpoint 的严格加载、LoRA 合并、bitsandbytes 量化、artifact 保存和从磁盘重新加载。

- **速度优先：BF16。** 560-token 完整语言模型 forward 平均 40.09 ms。
- **显存优先：NF4。** 峰值显存从 1997.49 MiB 降到 1483.96 MiB，下降 25.7%；平均延迟增加到 46.81 ms。
- **INT8 未通过。** 虽然能执行和保存，但持久化重载后相对 BF16 的 logits cosine 为 -0.4855，top-5 无重合，不能作为可用模型。

NF4 已通过合成提示和合成 RGB 图像整链路 smoke test，尚未达到真实驾驶精度或 CARLA 闭环验收。

## 环境

| 项目 | 值 |
| --- | --- |
| 主机 | `lg-robot` |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU，8 GB |
| Compute Capability | 8.6 |
| 驱动 | 580.178.04 |
| Python | 3.8.20 |
| PyTorch | 2.2.0+cu121 |
| torchvision | 0.17.0+cu121 |
| Transformers | 4.46.3 |
| PEFT | 0.13.2 |
| bitsandbytes | 0.44.1 |
| Attention | PyTorch SDPA，无 FlashAttention |

模型来源固定为 SimLingo commit `743b243afd6cf5ff51b9fa1f8cac86f22d569684`、官方模型 revision `26c7c89e797d4e25bbf640013317af8da26a5454` 和 InternVL2-1B revision `0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187`。

## 模型加载与量化范围

官方 `pytorch_model.pt` 的 SHA-256 校验通过。使用 `torch.load(weights_only=True)` 读取 992 个 tensor，严格加载结果为 `missing_keys=[]`、`unexpected_keys=[]`。

原始 PEFT 模型为 957,195,520 参数；LoRA 安全合并后为 939,599,104 参数。量化只替换 Qwen transformer 中 168 个线性层：

- InternViT 视觉编码器：BF16；
- token embeddings 与 `lm_head`：BF16；
- waypoint encoder 与驾驶输出头：BF16；
- Qwen transformer linears：NF4 或实验性 INT8。

`lm_head` 不能按当前上游路径直接量化，因为 SimLingo 的生成代码通过 `F.linear(hidden, lm_head.weight)` 读取权重，而不是调用量化模块的 `forward()`。

## 内存中完整 LLM forward

输入为 batch 1、sequence length 560 的固定合成 embeddings；预热 10 次，测量 50 次。一次 forward 包含 Qwen causal LM 和全部序列的词表 logits，不包含视觉编码、逐 token 生成、驾驶头和 CARLA。

| 模式 | 参数存储 | Mean | P50 | P90 | CUDA allocated | 峰值 CUDA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1792.14 MiB | 40.09 ms | 40.01 ms | 40.44 ms | 1812.52 MiB | 1997.49 MiB |
| INT8 | 1450.89 MiB | 153.16 ms | 153.14 ms | 153.69 ms | 1482.44 MiB | 1667.41 MiB |
| NF4 | 1280.27 MiB | 46.81 ms | 46.77 ms | 47.04 ms | 1298.99 MiB | 1483.96 MiB |

原始结果：

- [`bf16-seq560.json`](../results/rtx3070ti/bf16-seq560.json)
- [`int8-seq560.json`](../results/rtx3070ti/int8-seq560.json)
- [`nf4-seq560.json`](../results/rtx3070ti/nf4-seq560.json)

三组结果都校验了当前 9 文件 vendored 源码清单。NF4 相对 BF16 的峰值 CUDA 显存下降 25.7%，延迟增加 16.8%。INT8 峰值显存下降 16.5%，但平均延迟约为 BF16 的 3.82 倍。在这台 3070 Ti 上，量化的收益是显存容量，而不是吞吐速度。

## 持久化 artifact

| 组成 | 文件大小 |
| --- | ---: |
| 非语言部分 BF16 | 591.21 MiB |
| 合并 LoRA 的 BF16 LLM | 1201.02 MiB |
| bitsandbytes NF4 LLM | 694.86 MiB |

BF16 训练后状态 payload 为 1792.23 MiB；组合 BF16 非语言状态与 NF4 LLM 后为 1286.06 MiB，导出 payload 大小下降约 28.2%。这不是完整运行时磁盘占用：加载器还需要固定的 InternVL2-1B 基础快照，其中 `model.safetensors` 为 1789.47 MiB，用于实例化视觉模型及 remote-code 类。最终默认 NF4 导出元数据见 [`artifact-manifest-nf4.json`](../results/rtx3070ti/artifact-manifest-nf4.json)。

## 最终默认 NF4 从磁盘重新加载

验证从最终 BF16 与 NF4 持久化目录重新创建 Hugging Face 模型。输入是一个包含 512 个图像占位 token 的固定提示，总序列长度 557；预热 10 次，测量 50 次。计时范围是 Qwen2 decoder prefill 加最后一个 hidden state 的一次 `lm_head` 投影。

| artifact | Mean | 峰值 CUDA | Hidden cosine | Logits cosine | Top-1 | Top-5 重合 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| BF16 | 24.92 ms | 1239.01 MiB | — | — | 基线 | — | 通过 |
| NF4 | 44.24 ms | 728.61 MiB | 0.9443 | 0.9634 | 相同 | 4/5 | smoke test 通过 |

最终结果见 [`nf4-artifact-validation.json`](../results/rtx3070ti/nf4-artifact-validation.json)。

## 实验性 INT8 拒绝对照

INT8 是此前显式执行 `--formats nf4 int8` 的独立对照，不属于最终默认 NF4 导出。该次三模型重载的结果如下：

| artifact | Mean | 峰值 CUDA | Hidden cosine | Logits cosine | Top-1 | Top-5 重合 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| BF16 | 24.97 ms | 1239.01 MiB | — | — | 基线 | — | 通过 |
| INT8 | 157.91 ms | 926.87 MiB | 0.0303 | -0.4855 | 不同 | 0/5 | **拒绝** |
| NF4 | 44.73 ms | 728.61 MiB | 0.9448 | 0.9634 | 相同 | 4/5 | smoke test 通过 |

原始结果见 [`artifact-reload-validation.json`](../results/rtx3070ti/artifact-reload-validation.json)。

INT8 的错误不是 NaN：输出有限，但数值语义与 BF16 严重偏离。这说明“量化层存在、forward 可运行、logits 有限”不足以作为验收，必须保存后重新加载并对照 BF16。当前 `export_quantized_artifacts.py` 因此默认只生成 NF4；INT8 仅可显式选择用于继续调查。

## 合成图像整链路 smoke test

测试使用固定生成的 1024×512 RGB gradient/checkerboard，经上游裁剪与 `dynamic_preprocess` 生成两个 448×448 tile。路径实际覆盖：InternViT 图像编码、512 个图像 token、两个目标点 embedding、Qwen prefill、20 个路线 query、10 个速度轨迹 query 和驾驶输出头。

| 模式 | Mean | P90 | 峰值 CUDA | 输出状态 |
| --- | ---: | ---: | ---: | --- |
| BF16 | 114.35 ms | 114.93 ms | 2011.42 MiB | `(1,20,2)` route 与 `(1,10,2)` speed waypoints，全部有限 |
| NF4 | 133.68 ms | 134.37 ms | 1498.98 MiB | 同形状，全部有限 |

| NF4 相对 BF16 | Mean abs | Max abs | RMSE | Cosine |
| --- | ---: | ---: | ---: | ---: |
| Route waypoints | 0.03471 | 0.10938 | 0.04990 | 0.999978 |
| Speed waypoints | 0.28955 | 1.06250 | 0.43979 | 0.996041 |

NF4 在这条路径中峰值 CUDA 显存下降约 25.5%，延迟增加约 16.9%。CUDA event 只包围模型 forward；计时排除模型与 artifact 加载、CPU 合成图生成/裁剪/切 tile/归一化和 H2D。结果见 [`full-pipeline-smoke.json`](../results/rtx3070ti/full-pipeline-smoke.json)。

固定上游的 `DrivingModel.forward()` 在 `predict_language=False` 分支把 `(features, logits)` tuple 直接传给 tensor splitter，还会在相关路径混用训练提示与推理提示。本项目通过 `forward_driving_without_language()` 明确使用 `prompt_inference` 的 token 与 placeholder，并在 tensor splitter 前解包 tuple，不修改 vendored 上游文件及其哈希。本轮 smoke 将训练提示设为不同的哨兵值，确认走的是推理提示路径。

这张程序生成图像没有驾驶语义，waypoint 差值只能检查 BF16/NF4 对同一输入的相对行为；不能作为 ADE/FDE、决策正确性或安全性结论。

## 尚未完成的精度验收

当前没有下载 1.17 TB 的完整训练/验证数据，也没有在 CARLA 中运行 Bench2Drive。NF4 在进入真实部署前仍需：

1. 使用未参与调试的真实前视图和标签，比较 BF16 与 NF4 的路线 waypoint 和速度 waypoint；
2. 检查至少数百帧无 NaN/Inf，并统计 ADE/FDE、刹车决策一致率；
3. 先跑分层的短路线，再跑完整 Bench2Drive 闭环；
4. 分别记录视觉编码、LLM prefill、逐 token decode、驾驶 query 和端到端延迟。

SimLingo 当前生成循环没有 KV cache，会在每个 token 上重复计算长序列。量化完成后，端到端速度优化应优先修复该生成路径，而不是根据本页的单次 forward 结果推断整车帧率。
