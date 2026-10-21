# SimLingo RTX 3070 PyTorch 量化

这个子项目在 RTX 3070/3070 Ti 上重建官方 SimLingo PyTorch 模型，合并 Qwen2 的 PEFT LoRA，并导出可重复加载的 bitsandbytes NF4 权重。视觉编码器、语言嵌入、`lm_head` 和驾驶输出头保持 BF16；只量化 Qwen transformer 内部的 168 个 `Linear` 层。

截至本项目固定的官方版本，上游只发布了 PyTorch `pytorch_model.pt`，没有发布 ONNX、TensorRT engine 或官方量化模型。整网包含动态视觉 token、Python 数据结构和自回归生成流程，不适合直接一键导出 ONNX，因此当前交付选择了可验证、改动范围清楚的 PyTorch 权重量化路线。

## 3070 Ti 实测结论

2026-10-21 在 NVIDIA GeForce RTX 3070 Ti Laptop GPU 8 GB、CUDA 12.1、PyTorch 2.2.0 和 bitsandbytes 0.44.1 上完成真实模型加载、LoRA 合并、量化、持久化、重新加载和基准测试。最终 NF4 artifact 还完成了合成 RGB 图像到路线与速度 waypoint 的整链路 smoke test。

| 方案 | 内存模型参数存储 | 560-token 完整 LLM forward | 峰值 CUDA 显存 | 结论 |
| --- | ---: | ---: | ---: | --- |
| BF16 | 1792.14 MiB | 40.09 ms | 1997.49 MiB | 最快，速度优先 |
| INT8 | 1450.89 MiB | 153.16 ms | 1667.41 MiB | 重载后数值严重偏离，拒绝 |
| NF4 | 1280.27 MiB | 46.81 ms | 1483.96 MiB | 显存下降 25.7%，延迟增加 16.8% |

持久化 NF4 LLM 为 694.86 MiB；加上保持 BF16 的视觉与驾驶训练后状态后，导出 payload 为 1286.06 MiB，比对应 BF16 payload 小 28.2%。这不是全部运行时磁盘占用：加载器还需要固定的 InternVL2-1B 基础快照（其中 `model.safetensors` 为 1789.47 MiB）来构造视觉模型。重新加载后的合成提示 smoke test 中，NF4 与 BF16 的 logits cosine 为 0.9634，top-1 相同，top-5 重合 4/5。INT8 的 logits cosine 为 -0.4855，top-5 无重合，所以不能使用该 INT8 artifact。

完整合成图像链路中，BF16 平均 114.35 ms、峰值 2011.42 MiB，NF4 平均 133.68 ms、峰值 1498.98 MiB。NF4 路线 waypoint 相对 BF16 的 mean absolute difference 为 0.0347、cosine 为 0.999978；这些数值只证明相同输入上的整网连通与接近程度，不是相对真实标签的驾驶精度。

第一张表测量语言骨干，整链路数据使用合成图像；两者都不是 CARLA 闭环帧率。原始 JSON、测试口径和限制见 [RTX 3070 Ti 验证记录](docs/validation-rtx3070ti.md)。

## 目录

```text
simlingo-quantization/
├── download_models.py                # 固定 revision 下载并校验官方权重
├── quantize_and_benchmark.py          # BF16、INT8、NF4 内存基准
├── export_quantized_artifacts.py      # 导出 BF16 非语言部分与持久化 NF4/INT8 LLM
├── validate_artifacts.py              # 重新加载 artifact 并与 BF16 比较
├── load_quantized_model.py            # 将拆分 artifact 重组为完整 DrivingModel
├── smoke_test_full_pipeline.py         # 合成图像到驾驶 waypoint 的整链路 smoke test
├── requirements.txt
├── docs/validation-rtx3070ti.md
├── results/rtx3070ti/                 # 3070 Ti 原始 JSON
├── UPSTREAM.md
├── vendor-manifest.json
└── vendor/simlingo/                   # 固定 commit 的最小源码快照与许可证
```

项目没有合并上游 Git 历史，也没有提交模型权重、TensorRT engine、数据集或 Hugging Face token。

## 命令工作目录

本文以下命令都从子项目目录运行，使下载的权重、导出的 artifact 和临时结果由本目录的 `.gitignore` 管理：

```bash
cd /path/to/Robot_infer/simlingo-quantization
```

## 环境

在 RTX 3070 的 Linux/WSL2 环境中创建 Python 3.8 环境：

```bash
conda create -n simlingo-quant python=3.8 -y
conda activate simlingo-quant

python -m pip install --upgrade pip
python -m pip install \
  torch==2.2.0 torchvision==0.17.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

已验证的关键版本是 `transformers==4.46.3`、`peft==0.13.2`、`accelerate==1.0.1` 和 `bitsandbytes==0.44.1`。`flash-attn`、DeepSpeed 和 CARLA 不属于这个最小量化环境。

## 下载官方 PyTorch 模型

本项目固定并直接引用以下官方来源：

- [SimLingo 官方源码](https://github.com/RenzKa/simlingo)及其 [Apache-2.0 许可证](https://github.com/RenzKa/simlingo/blob/main/LICENSE)；
- [SimLingo 官方 Hugging Face 模型](https://huggingface.co/RenzKa/simlingo)，其中发布的是 PyTorch checkpoint；
- [OpenGVLab InternVL2-1B 基础模型](https://huggingface.co/OpenGVLab/InternVL2-1B)，模型页标注 MIT 许可证。

先查看下载计划：

```bash
python download_models.py \
  --output-dir models/simlingo \
  --dry-run
```

再下载约 4.45 GB 的推理必需文件：

```bash
python download_models.py \
  --output-dir models/simlingo
```

下载脚本固定并验证：

- SimLingo revision `26c7c89e797d4e25bbf640013317af8da26a5454`；
- `pytorch_model.pt` SHA-256 `ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28`；
- InternVL2-1B revision `0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187`；
- InternVL2-1B `model.safetensors` SHA-256 `9420916a7fab7d2009f7907cdffa341c9cb6be7c5e0cf4ee193de16fde647dea`。

它不会下载约 7 GB 的 DeepSpeed optimizer/training state。网络需要镜像时，可在环境中设置标准的 `HF_ENDPOINT`；需要认证时设置 `HF_TOKEN`，不要把 token 写进脚本、命令参数或仓库。

## 检查模型

```bash
python quantize_and_benchmark.py \
  --mode inspect \
  --model-dir models/simlingo \
  --result results/simlingo-inspect.json
```

正式运行默认使用 `torch.load(weights_only=True)`，严格加载全部 992 个 checkpoint tensors，并要求 missing/unexpected keys 都为空。只有可信的官方 checkpoint 在旧版 PyTorch 无法安全加载时，才应显式使用 `--allow-pickle`。

## 运行 BF16 与 NF4 基准

分别启动独立进程，使 CUDA 峰值显存可比较：

```bash
python quantize_and_benchmark.py \
  --mode run --quantization none \
  --model-dir models/simlingo \
  --sequence-length 560 --warmup 10 --iterations 50 \
  --result results/simlingo-bf16.json

python quantize_and_benchmark.py \
  --mode run --quantization nf4 \
  --model-dir models/simlingo \
  --sequence-length 560 --warmup 10 --iterations 50 \
  --result results/simlingo-nf4.json
```

这个基准在完整 Qwen causal LM 上运行合成 embeddings，并强制物化全部 logits。它适合测量语言骨干速度、显存和运行稳定性，不代表视觉预处理、语言逐 token 生成、驾驶输出或 CARLA 的端到端延迟。

## 导出可重复加载的 NF4 artifact

默认只导出已经通过本轮 smoke test 的 NF4：

```bash
python export_quantized_artifacts.py \
  --model-dir models/simlingo \
  --output-dir artifacts/simlingo-3070
```

输出结构：

```text
artifacts/simlingo-3070/
├── non_language_bf16.safetensors  # InternViT、waypoint encoder、驾驶头
├── llm_merged_bf16/               # 合并 LoRA 的 BF16 Qwen 基线
├── llm_bnb_nf4/                   # 可重新加载的 NF4 Qwen
├── processor/
├── simlingo-config.yaml
└── artifact-manifest.json
```

导出后必须从磁盘重新加载验证：

```bash
python validate_artifacts.py \
  --artifact-dir artifacts/simlingo-3070 \
  --warmup 10 --iterations 50 \
  --result results/simlingo-artifact-validation.json
```

验证通过后，把拆分 artifact 重组为完整的 SimLingo `DrivingModel`：

```bash
python load_quantized_model.py \
  --artifact-dir artifacts/simlingo-3070 \
  --model-dir models/simlingo \
  --quantization nf4
```

该加载器不再读取 2.57 GB 的 SimLingo checkpoint；它从 `non_language_bf16.safetensors` 恢复训练后的视觉与驾驶模块，并注入已持久化的 Qwen artifact。固定的 InternVL2-1B 本地快照仍用于构造上游 remote-code 类，所以不能在导出后删除该快照。

只运行驾驶输出时，请调用 `load_quantized_model.forward_driving_without_language(model, driving_input)`。固定的上游 `DrivingModel.forward()` 在 `predict_language=False` 分支没有解包 `forward_model()` 返回的 `(features, logits)`，而且训练提示与推理提示的 placeholder 路径不一致。本项目的外层 helper 明确使用 `prompt_inference` 的 token 和 placeholder，并在拆分输出前解包 tuple，同时保持 vendored 上游源码及哈希不变。

最后运行不依赖 CARLA 或数据集的整链路 smoke test：

```bash
python smoke_test_full_pipeline.py \
  --artifact-dir artifacts/simlingo-3070 \
  --model-dir models/simlingo \
  --result results/simlingo-full-pipeline-smoke.json
```

它依次加载 BF16 与 NF4，用同一张确定性合成图像执行 InternViT、图像与目标点 token 替换、Qwen prefill、20 个路线 query、10 个速度轨迹 query 和两个驾驶输出头。测试把训练提示与推理提示设为不同值，以覆盖 `prompt_inference` 路由。CUDA event 计时只包围模型 forward，排除模型加载、CPU 图像生成/裁剪/切 tile/归一化和 H2D。该结果能检查整条模型连接和有限输出，仍不能代替真实驾驶精度测试。

若要调查 INT8，可在导出命令加 `--formats nf4 int8`。当前已验证软件栈中的 INT8 重载结果未通过数值门槛，所以它只是实验选项；不要把“能够保存和运行”当作精度通过。

## 验证边界

- NF4 已通过固定合成提示和固定合成 RGB 图像的整链路 smoke test，还没有用真实驾驶帧计算相对标签的路线 waypoint 误差，也没有运行 Bench2Drive/CARLA Driving Score。
- 量化节省显存但没有在 3070 Ti 上加速。此模型批量为 1、语言骨干较小，反量化和 kernel 启动开销高于 BF16 Tensor Core 的收益。
- 当前上游自回归生成没有使用 KV cache，每生成一个 token 都重复处理长序列。若目标是降低端到端延迟，下一步应先加入 KV cache，再评估 TensorRT-LLM 或拆分后的视觉 TensorRT。
- bitsandbytes artifact 运行时仍依赖 CUDA、PyTorch、Transformers 和 bitsandbytes；它不是独立 ONNX 或 TensorRT 文件。
- 源码、模型和数据集分别遵循各自许可。数据集未包含在仓库中；SimLingo 数据集受[仅限非商业用途的数据集许可](https://github.com/RenzKa/simlingo/blob/main/LICENCE_Dataset)约束，真实车辆或机器人部署前必须单独核对权重与数据许可。
