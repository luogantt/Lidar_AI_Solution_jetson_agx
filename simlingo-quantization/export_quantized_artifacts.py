#!/usr/bin/env python3
"""Export portable BF16 and bitsandbytes SimLingo inference artifacts.

The complete SimLingo checkpoint is reconstructed first, then its Qwen LoRA
weights are merged.  The vision encoder and driving heads remain BF16 while the
merged Qwen transformer may be serialized as NF4 or experimental INT8.
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
    "llm_bnb_nf4",
    "llm_merged_bf16",
    "non_language_bf16.safetensors",
    "processor",
    "simlingo-config.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SimLingo LoRA and export reusable BF16/NF4/INT8 artifacts."
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
        help="Quantized LLM formats to export. NF4 is the validated default.",
    )
    parser.add_argument("--device", default="cuda:0")
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
            bnb_4bit_compute_dtype=torch.bfloat16,
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
    load_args = SimpleNamespace(
        allow_nonstrict=args.allow_nonstrict,
        allow_pickle=args.allow_pickle,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
        seed=args.seed,
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
    non_language_path = output_dir / "non_language_bf16.safetensors"
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
            "dtype_policy": "vision encoder and driving heads remain BF16",
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

    merged_dir = output_dir / "llm_merged_bf16"
    prepare_directory(merged_dir, args.force)
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")

    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
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
            "vision_encoder_and_driving_heads": "bfloat16",
            "language_embeddings_and_lm_head": "bfloat16",
            "language_transformer_linears": args.formats,
        },
        "artifacts": {
            "non_language_bf16": {
                "path": non_language_path.name,
                "bytes": non_language_path.stat().st_size,
            },
            "llm_merged_bf16": {
                "path": merged_dir.name,
                "bytes": directory_size(merged_dir),
            },
        },
        "notes": [
            "NF4 is the validated default for this environment.",
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
            torch_dtype=torch.bfloat16,
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

        artifact_name = "llm_bnb_%s" % name
        artifact_dir = output_dir / artifact_name
        prepare_directory(artifact_dir, args.force)
        quantized.save_pretrained(
            artifact_dir,
            safe_serialization=True,
            max_shard_size="2GB",
        )
        manifest["artifacts"][artifact_name] = {
            "path": artifact_dir.name,
            "bytes": directory_size(artifact_dir),
            "seconds": round(time.perf_counter() - quant_started, 3),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "preflight_finite": finite,
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
