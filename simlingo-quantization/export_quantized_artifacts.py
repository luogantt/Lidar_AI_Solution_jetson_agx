#!/usr/bin/env python3
"""Export portable floating-point and bitsandbytes SimLingo inference artifacts.

The complete SimLingo checkpoint is reconstructed first, then its Qwen LoRA
weights are merged.  Boundary modules remain BF16 or FP16 while the merged Qwen
transformer may be serialized as NF4 or experimental INT8.
"""

import argparse
import datetime as _datetime
import gc
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping

import quantize_and_benchmark as workflow


GENERATED_ARTIFACT_NAMES = {
    "artifact-manifest.json",
    "llm_bnb_int8",
    "llm_bnb_int8_fp16",
    "llm_bnb_nf4",
    "llm_bnb_nf4_fp16",
    "llm_merged_bf16",
    "llm_merged_fp16",
    "non_language_bf16.safetensors",
    "non_language_fp16.safetensors",
    "processor",
    "simlingo-config.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SimLingo LoRA and export reusable BF16/FP16/NF4/INT8 artifacts."
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/simlingo"))
    parser.add_argument("--upstream", type=Path, default=workflow.VENDORED_SOURCE)
    parser.add_argument("--internvl-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/simlingo-3070"))
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("nf4", "int8"),
        default=("nf4",),
        help="Quantized LLM formats to export. NF4 is the default candidate and still requires validation.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--boundary-dtype",
        choices=workflow.BOUNDARY_DTYPE_CHOICES,
        default="bf16",
        help="Floating dtype for vision/input/output/head modules.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--int8-threshold", type=float, default=6.0)
    parser.add_argument("--no-double-quant", action="store_true")
    parser.add_argument("--allow-nonstrict", action="store_true")
    parser.add_argument("--allow-pickle", action="store_true")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated artifact subdirectories.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    model_dir = args.model_dir.expanduser().resolve()
    return {
        "upstream": args.upstream.expanduser().resolve(),
        "model_dir": model_dir,
        "config": (
            args.config.expanduser().resolve()
            if args.config is not None
            else model_dir / ".hydra" / "config.yaml"
        ),
        "checkpoint": (
            args.checkpoint.expanduser().resolve()
            if args.checkpoint is not None
            else model_dir
            / "simlingo"
            / "checkpoints"
            / "epoch=013.ckpt"
            / "pytorch_model.pt"
        ),
        "internvl": (
            args.internvl_dir.expanduser().resolve()
            if args.internvl_dir is not None
            else model_dir / "pretrained" / "InternVL2-1B"
        ),
    }


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def prepare_directory(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError("Artifact already exists; pass --force to replace: %s" % path)
        shutil.rmtree(path)
    path.mkdir(parents=True)


def prepare_output_root(path: Path, force: bool) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    entries = list(path.iterdir())
    if not entries:
        return
    if not force:
        raise FileExistsError(
            "Output directory is not empty; pass --force to replace it: %s" % path
        )
    unexpected = sorted(item.name for item in entries if item.name not in GENERATED_ARTIFACT_NAMES)
    if unexpected:
        raise RuntimeError(
            "Refusing --force because the output directory contains unknown files: %s"
            % unexpected
        )
    for item in entries:
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def quantization_config(name: str, args: argparse.Namespace) -> Any:
    import torch
    from transformers import BitsAndBytesConfig

    if name == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=not args.no_double_quant,
            bnb_4bit_compute_dtype=workflow.boundary_torch_dtype(
                torch, args.boundary_dtype
            ),
        )
    if name == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=args.int8_threshold,
        )
    raise ValueError("Unsupported format: %s" % name)


def unique(items: Iterable[str]) -> list:
    return list(dict.fromkeys(items))


def main() -> int:
    args = parse_args()
    args.formats = unique(args.formats)
    paths = resolve_paths(args)
    workflow.assert_inputs(paths)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bitsandbytes artifact export")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("A CUDA device is required, got %s" % device)

    output_dir = args.output_dir.expanduser().resolve()
    prepare_output_root(output_dir, args.force)
    layout = workflow.artifact_layout(args.boundary_dtype)
    boundary_dtype = workflow.boundary_torch_dtype(torch, args.boundary_dtype)
    boundary_name = workflow.boundary_dtype_name(args.boundary_dtype)
    load_args = SimpleNamespace(
        allow_nonstrict=args.allow_nonstrict,
        allow_pickle=args.allow_pickle,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
        seed=args.seed,
        boundary_dtype=args.boundary_dtype,
    )

    started = time.perf_counter()
    model, model_report = workflow.load_simlingo(load_args, paths)
    model.eval()
    model.requires_grad_(False)

    non_language = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if not name.startswith(("language_model.", "adaptors.language."))
    }
    non_language_path = output_dir / layout["non_language"]
    if non_language_path.exists() and not args.force:
        raise FileExistsError(
            "Artifact already exists; pass --force to replace: %s" % non_language_path
        )
    save_file(
        non_language,
        str(non_language_path),
        metadata={
            "source": "RenzKa/simlingo",
            "upstream_commit": workflow.VENDORED_COMMIT,
            "checkpoint_sha256": (
                "ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28"
            ),
            "dtype_policy": "vision/input/output/head modules remain %s" % boundary_name,
        },
    )

    processor_dir = output_dir / "processor"
    prepare_directory(processor_dir, args.force)
    model.processor.save_pretrained(processor_dir)
    shutil.copy2(paths["config"], output_dir / "simlingo-config.yaml")

    merge_report = workflow.merge_lora(model)
    merged = model.language_model.model
    merged.eval()
    merged.requires_grad_(False)
    merged.config.use_cache = True
    # SimLingo creates a top-level alias to model.embed_tokens. Keeping it makes
    # safetensors report two names for the same storage during save_pretrained.
    if "embed_tokens" in merged._modules:
        del merged._modules["embed_tokens"]

    merged_dir = output_dir / layout["baseline_llm"]
    prepare_directory(merged_dir, args.force)
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")

    manifest: Dict[str, Any] = {
        "schema_version": "1.1",
        "created_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "source": {
            "repository": "https://github.com/RenzKa/simlingo",
            "commit": workflow.VENDORED_COMMIT,
            "checkpoint_sha256": (
                "ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28"
            ),
            "internvl_revision": workflow.INTERNVL_REVISION,
        },
        "environment": workflow.environment_report(include_torch=True),
        "model": model_report,
        "lora": merge_report,
        "dtype_policy": {
            "boundary_dtype": boundary_name,
            "vision_encoder_and_driving_heads": boundary_name,
            "language_embeddings_and_lm_head": boundary_name,
            "language_transformer_linears": args.formats,
        },
        "artifacts": {
            "non_language_%s" % args.boundary_dtype: {
                "path": non_language_path.name,
                "bytes": non_language_path.stat().st_size,
            },
            "llm_merged_%s" % args.boundary_dtype: {
                "path": merged_dir.name,
                "bytes": directory_size(merged_dir),
            },
        },
        "notes": [
            "Every exported precision profile must pass validate_artifacts.py and the complete-pipeline smoke test before deployment.",
            "INT8 export is experimental and must pass validate_artifacts.py before use.",
            "No ONNX or TensorRT engine is produced by this bitsandbytes workflow.",
        ],
        "total_seconds": None,
    }
    manifest_path = output_dir / "artifact-manifest.json"
    write_manifest(manifest_path, manifest)

    del non_language, model, merged
    gc.collect()

    for name in args.formats:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        quant_started = time.perf_counter()
        quantized = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            local_files_only=True,
            torch_dtype=boundary_dtype,
            quantization_config=quantization_config(name, args),
            device_map={"": device.index or 0},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        quantized.eval()
        test_ids = torch.tensor([[1, 2, 3, 4]], device=args.device, dtype=torch.long)
        with torch.inference_mode():
            output = quantized(input_ids=test_ids, use_cache=False, return_dict=True)
        finite = bool(torch.isfinite(output.logits).all().item())
        if not finite:
            raise RuntimeError("%s produced non-finite logits during export" % name)

        artifact_name = layout["%s_llm" % name]
        artifact_dir = output_dir / artifact_name
        prepare_directory(artifact_dir, args.force)
        quantized.save_pretrained(
            artifact_dir,
            safe_serialization=True,
            max_shard_size="2GB",
        )
        input_dtype = str(quantized.get_input_embeddings().weight.dtype).replace(
            "torch.", ""
        )
        output_dtype = str(quantized.get_output_embeddings().weight.dtype).replace(
            "torch.", ""
        )
        if input_dtype != boundary_name or output_dtype != boundary_name:
            raise AssertionError(
                "Quantized artifact boundary dtype mismatch: input=%s output=%s expected=%s"
                % (input_dtype, output_dtype, boundary_name)
            )
        compute_dtypes = workflow.bnb_compute_dtypes(quantized)
        bnb_layer_count = workflow.count_bnb_layers(quantized)
        if bnb_layer_count != workflow.EXPECTED_QWEN_LINEAR_LAYERS:
            raise AssertionError(
                "Expected %d quantized Qwen Linear layers, found %d"
                % (workflow.EXPECTED_QWEN_LINEAR_LAYERS, bnb_layer_count)
            )
        if name == "nf4" and compute_dtypes != [boundary_name]:
            raise AssertionError(
                "NF4 compute dtype mismatch: expected %s, found %s"
                % (boundary_name, compute_dtypes)
            )
        manifest["artifacts"][artifact_name] = {
            "path": artifact_dir.name,
            "bytes": directory_size(artifact_dir),
            "seconds": round(time.perf_counter() - quant_started, 3),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "preflight_finite": finite,
            "input_embedding_dtype": input_dtype,
            "lm_head_dtype": output_dtype,
            "bnb_layer_count": bnb_layer_count,
            "bnb_compute_dtypes": compute_dtypes,
            "validation_status": "requires validate_artifacts.py",
        }
        write_manifest(manifest_path, manifest)
        del quantized, output, test_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest["total_seconds"] = round(time.perf_counter() - started, 3)
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
