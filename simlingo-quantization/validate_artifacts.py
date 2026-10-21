#!/usr/bin/env python3
"""Reload exported SimLingo language artifacts and compare them with BF16."""

import argparse
import datetime as _datetime
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload persisted BF16/NF4/INT8 Qwen artifacts on CUDA."
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/simlingo-3070"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/simlingo-artifact-validation.json"),
    )
    return parser.parse_args()


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tensor_metrics(candidate: Any, reference: Any) -> Dict[str, float]:
    import torch

    a = candidate.float().reshape(-1)
    b = reference.float().reshape(-1)
    delta = (a - b).abs()
    return {
        "mean_abs": float(delta.mean()),
        "max_abs": float(delta.max()),
        "rmse": float(torch.sqrt(torch.mean((a - b) ** 2))),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for artifact validation")
    artifact_dir = args.artifact_dir.expanduser().resolve()
    bf16_dir = artifact_dir / "llm_merged_bf16"
    processor_dir = artifact_dir / "processor"
    if not bf16_dir.is_dir() or not processor_dir.is_dir():
        raise FileNotFoundError("Missing BF16 or processor artifact under %s" % artifact_dir)

    tokenizer = AutoTokenizer.from_pretrained(processor_dir, local_files_only=True)
    prompt = (
        "<img>" + "<IMG_CONTEXT>" * 512 + "</img>\n"
        "Current speed: 4.0 m/s. Target waypoints: "
        "(10.0, 0.0), (20.0, 0.0). What should the ego do next?"
    )
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids_cpu = encoded["input_ids"]
    attention_cpu = encoded["attention_mask"]
    device = torch.device(args.device)

    candidates = [("bf16", bf16_dir)]
    for name in ("int8", "nf4"):
        path = artifact_dir / ("llm_bnb_%s" % name)
        if path.is_dir():
            candidates.append((name, path))

    results: Dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sequence_length": int(input_ids_cpu.shape[1]),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "scope": "persisted Qwen2 decoder prefill plus one final lm_head projection",
        "acceptance": {
            "finite": True,
            "hidden_cosine_min": 0.90,
            "logits_cosine_min": 0.90,
            "top1_match": True,
            "note": "Synthetic-prompt smoke threshold; not a driving-accuracy threshold.",
        },
        "models": {},
    }
    reference_hidden = None
    reference_logits = None

    for name, directory in candidates:
        gc.collect()
        torch.cuda.empty_cache()
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            directory,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map={"": device.index or 0},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        load_seconds = time.perf_counter() - load_started
        model.eval()
        input_ids = input_ids_cpu.to(device)
        attention = attention_cpu.to(device)

        def invoke() -> Any:
            decoded = model.model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
            hidden = decoded.last_hidden_state[:, -1]
            logits = model.get_output_embeddings()(hidden)
            return hidden, logits

        with torch.inference_mode():
            hidden, logits = invoke()
            finite = bool(torch.isfinite(hidden).all() and torch.isfinite(logits).all())
            for _ in range(args.warmup):
                invoke()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                hidden, logits = invoke()
                end.record()
                end.synchronize()
                timings.append(float(start.elapsed_time(end)))

        hidden_cpu = hidden.detach().cpu()
        logits_cpu = logits.detach().cpu()
        top5 = torch.topk(logits_cpu.float(), 5, dim=-1).indices[0].tolist()
        item: Dict[str, Any] = {
            "artifact": str(directory),
            "artifact_bytes": directory_size(directory),
            "load_seconds": load_seconds,
            "finite": finite,
            "loaded_in_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
            "loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
            "cuda_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
            "cuda_peak_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "latency_ms": {
                "mean": statistics.mean(timings),
                "p50": percentile(timings, 0.50),
                "p90": percentile(timings, 0.90),
                "min": min(timings),
                "max": max(timings),
            },
            "top1_token_id": top5[0],
            "top5_token_ids": top5,
        }
        if name == "bf16":
            reference_hidden = hidden_cpu
            reference_logits = logits_cpu
            item["comparison_to_bf16"] = None
            item["accepted"] = finite
        else:
            hidden_comparison = tensor_metrics(hidden_cpu, reference_hidden)
            logits_comparison = tensor_metrics(logits_cpu, reference_logits)
            reference_top5 = set(
                torch.topk(reference_logits.float(), 5, dim=-1).indices[0].tolist()
            )
            top1_match = top5[0] == int(torch.argmax(reference_logits.float(), dim=-1)[0])
            item["comparison_to_bf16"] = {
                "hidden": hidden_comparison,
                "logits": logits_comparison,
                "top1_match": top1_match,
                "top5_overlap": len(reference_top5.intersection(top5)),
            }
            item["accepted"] = bool(
                finite
                and hidden_comparison["cosine"] >= 0.90
                and logits_comparison["cosine"] >= 0.90
                and top1_match
            )
        results["models"][name] = item
        del model, hidden, logits, hidden_cpu, logits_cpu, input_ids, attention
        gc.collect()
        torch.cuda.empty_cache()

    result_path = args.result.expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if all(item["accepted"] for item in results["models"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
