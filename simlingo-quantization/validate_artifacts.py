#!/usr/bin/env python3
"""Reload exported SimLingo language artifacts and compare with their FP baseline."""

import argparse
import datetime as _datetime
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import quantize_and_benchmark as workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload persisted BF16/FP16/NF4/INT8 Qwen artifacts on CUDA."
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/simlingo-3070"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--boundary-dtype",
        choices=workflow.BOUNDARY_DTYPE_CHOICES,
        default="bf16",
        help="Floating dtype used by embeddings, lm_head and non-quantized modules.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("nf4", "int8"),
        default=["nf4"],
        help="Quantized artifact formats that must exist and pass validation.",
    )
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


def resolve_artifact_path(
    artifact_dir: Path,
    artifacts: Mapping[str, Any],
    key: str,
    fallback_name: str,
    strict_manifest: bool,
) -> Path:
    entry = artifacts.get(key)
    if strict_manifest and not isinstance(entry, Mapping):
        raise RuntimeError("Artifact manifest is missing required entry: %s" % key)
    value = entry.get("path") if isinstance(entry, Mapping) else None
    if value is None:
        if strict_manifest:
            raise RuntimeError("Artifact manifest entry has no path: %s" % key)
        value = fallback_name
    if not isinstance(value, str) or not value:
        raise RuntimeError("Artifact manifest contains an invalid path for %s" % key)
    relative = Path(value)
    if relative.is_absolute():
        raise RuntimeError("Artifact manifest path must be relative: %s" % relative)
    resolved = (artifact_dir / relative).resolve()
    if resolved != artifact_dir and artifact_dir not in resolved.parents:
        raise RuntimeError("Artifact manifest path escapes artifact directory: %s" % relative)
    return resolved


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
    layout = workflow.artifact_layout(args.boundary_dtype)
    boundary_dtype = workflow.boundary_torch_dtype(torch, args.boundary_dtype)
    boundary_name = workflow.boundary_dtype_name(args.boundary_dtype)
    processor_dir = artifact_dir / "processor"
    manifest_path = artifact_dir / "artifact-manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise RuntimeError("Artifact manifest must contain a JSON object")
        declared = manifest.get("dtype_policy", {}).get("boundary_dtype")
        legacy_bf16 = manifest.get("schema_version") == "1.0" and declared is None
        if legacy_bf16 and args.boundary_dtype != "bf16":
            raise RuntimeError(
                "Schema 1.0 artifact does not declare a boundary dtype and is therefore "
                "BF16; requested %s" % args.boundary_dtype
            )
        if declared is not None and declared != boundary_name:
            raise RuntimeError(
                "Artifact boundary dtype is %s, requested %s" % (declared, boundary_name)
            )
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("Artifact manifest 'artifacts' must be an object")
    strict_manifest = bool(manifest) and manifest.get("schema_version") != "1.0"
    baseline_key = "llm_merged_%s" % args.boundary_dtype
    baseline_dir = resolve_artifact_path(
        artifact_dir,
        artifacts,
        baseline_key,
        layout["baseline_llm"],
        strict_manifest,
    )
    if not baseline_dir.is_dir() or not processor_dir.is_dir():
        raise FileNotFoundError(
            "Missing %s baseline or processor artifact under %s"
            % (args.boundary_dtype, artifact_dir)
        )

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

    candidates = [(args.boundary_dtype, baseline_dir, "dense")]
    for name in args.formats:
        artifact_key = layout["%s_llm" % name]
        path = resolve_artifact_path(
            artifact_dir,
            artifacts,
            artifact_key,
            artifact_key,
            strict_manifest,
        )
        if not path.is_dir():
            raise FileNotFoundError("Missing required %s artifact: %s" % (name, path))
        label = name if args.boundary_dtype == "bf16" else "%s_fp16_boundaries" % name
        candidates.append((label, path, name))

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
        "boundary_dtype": boundary_name,
        "baseline": args.boundary_dtype,
        "requested_formats": list(args.formats),
        "acceptance": {
            "finite": True,
            "hidden_cosine_min": 0.90,
            "logits_cosine_min": 0.90,
            "top1_match": True,
            "top5_overlap_min": 4,
            "note": "Synthetic-prompt smoke threshold; not a driving-accuracy threshold.",
        },
        "models": {},
    }
    reference_hidden = None
    reference_logits = None

    for name, directory, artifact_kind in candidates:
        gc.collect()
        torch.cuda.empty_cache()
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            directory,
            local_files_only=True,
            torch_dtype=boundary_dtype,
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
        input_dtype = str(model.get_input_embeddings().weight.dtype).replace("torch.", "")
        output_dtype = str(model.get_output_embeddings().weight.dtype).replace("torch.", "")
        hidden_dtype = str(hidden.dtype).replace("torch.", "")
        logits_dtype = str(logits.dtype).replace("torch.", "")
        compute_dtypes = workflow.bnb_compute_dtypes(model)
        bnb_layer_count = workflow.count_bnb_layers(model)
        loaded_in_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
        loaded_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
        if (
            input_dtype != boundary_name
            or output_dtype != boundary_name
            or hidden_dtype != boundary_name
            or logits_dtype != boundary_name
        ):
            raise AssertionError(
                "Boundary dtype mismatch: input=%s output=%s hidden=%s logits=%s expected=%s"
                % (input_dtype, output_dtype, hidden_dtype, logits_dtype, boundary_name)
            )
        expected_bnb_layers = (
            0 if artifact_kind == "dense" else workflow.EXPECTED_QWEN_LINEAR_LAYERS
        )
        if bnb_layer_count != expected_bnb_layers:
            raise AssertionError(
                "Expected %d Qwen bitsandbytes layers for %s, found %d"
                % (expected_bnb_layers, artifact_kind, bnb_layer_count)
            )
        expected_compute_dtypes = [boundary_name] if artifact_kind == "nf4" else []
        if compute_dtypes != expected_compute_dtypes:
            raise AssertionError(
                "bitsandbytes compute dtype mismatch: expected %s, found %s"
                % (expected_compute_dtypes, compute_dtypes)
            )
        expected_loaded_flags = {
            "dense": (False, False),
            "nf4": (True, False),
            "int8": (False, True),
        }[artifact_kind]
        if (loaded_in_4bit, loaded_in_8bit) != expected_loaded_flags:
            raise AssertionError(
                "Unexpected load flags for %s: 4bit=%s 8bit=%s"
                % (artifact_kind, loaded_in_4bit, loaded_in_8bit)
            )
        item: Dict[str, Any] = {
            "artifact": str(directory),
            "artifact_bytes": directory_size(directory),
            "load_seconds": load_seconds,
            "finite": finite,
            "loaded_in_8bit": loaded_in_8bit,
            "loaded_in_4bit": loaded_in_4bit,
            "input_embedding_dtype": input_dtype,
            "lm_head_dtype": output_dtype,
            "hidden_dtype": hidden_dtype,
            "logits_dtype": logits_dtype,
            "bnb_layer_count": bnb_layer_count,
            "bnb_compute_dtypes": compute_dtypes,
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
        comparison_key = "comparison_to_%s" % args.boundary_dtype
        if artifact_kind == "dense":
            reference_hidden = hidden_cpu
            reference_logits = logits_cpu
            item[comparison_key] = None
            item["accepted"] = finite
        else:
            hidden_comparison = tensor_metrics(hidden_cpu, reference_hidden)
            logits_comparison = tensor_metrics(logits_cpu, reference_logits)
            reference_top5 = set(
                torch.topk(reference_logits.float(), 5, dim=-1).indices[0].tolist()
            )
            top1_match = top5[0] == int(torch.argmax(reference_logits.float(), dim=-1)[0])
            item[comparison_key] = {
                "hidden": hidden_comparison,
                "logits": logits_comparison,
                "top1_match": top1_match,
                "top5_overlap": len(reference_top5.intersection(top5)),
            }
            top5_overlap = item[comparison_key]["top5_overlap"]
            item["accepted"] = bool(
                finite
                and hidden_comparison["cosine"] >= 0.90
                and logits_comparison["cosine"] >= 0.90
                and top1_match
                and top5_overlap >= 4
            )
        results["models"][name] = item
        del model, hidden, logits, hidden_cpu, logits_cpu, input_ids, attention
        gc.collect()
        torch.cuda.empty_cache()

    results["passed"] = all(item["accepted"] for item in results["models"].values())
    result_path = args.result.expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if results["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
