#!/usr/bin/env python3
"""Load, merge and quantize SimLingo without importing CARLA.

The script keeps the vision encoder, output head and driving heads in BF16.  It
replaces only transformer ``nn.Linear`` modules below
``model.language_model.model`` with bitsandbytes INT8 or NF4 modules.
"""

import argparse
import datetime as _datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SPECIAL_TOKENS = [
    "<WAYPOINTS>",
    "<WAYPOINTS_DIFF>",
    "<ORG_WAYPOINTS_DIFF>",
    "<ORG_WAYPOINTS>",
    "<WAYPOINT_LAST>",
    "<ROUTE>",
    "<ROUTE_DIFF>",
    "<TARGET_POINT>",
]
VENDORED_SOURCE = Path(__file__).resolve().parent / "vendor" / "simlingo"
VENDORED_COMMIT = "743b243afd6cf5ff51b9fa1f8cac86f22d569684"
INTERNVL_REVISION = "0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize SimLingo's Qwen language backbone on an NVIDIA GPU."
    )
    parser.add_argument("--mode", choices=("dry-run", "inspect", "run"), default="inspect")
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path(os.environ.get("SIMLINGO_SOURCE", str(VENDORED_SOURCE))),
        help="Path to the allowlisted SimLingo source snapshot bundled with this project.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/simlingo"))
    parser.add_argument("--internvl-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--quantization", choices=("none", "int8", "nf4"), default="nf4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--int8-threshold", type=float, default=6.0)
    parser.add_argument("--no-double-quant", action="store_true")
    parser.add_argument("--allow-nonstrict", action="store_true")
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow torch.load(weights_only=False) if the tensor-only load fails.",
    )
    parser.add_argument("--result", type=Path, default=Path("results/simlingo-nf4.json"))
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).with_name("results.schema.json"),
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    model_dir = args.model_dir.expanduser().resolve()
    config = (
        args.config.expanduser().resolve()
        if args.config is not None
        else model_dir / ".hydra" / "config.yaml"
    )
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else model_dir / "simlingo" / "checkpoints" / "epoch=013.ckpt" / "pytorch_model.pt"
    )
    internvl = (
        args.internvl_dir.expanduser().resolve()
        if args.internvl_dir is not None
        else model_dir / "pretrained" / "InternVL2-1B"
    )
    return {
        "upstream": args.upstream.expanduser().resolve(),
        "model_dir": model_dir,
        "config": config,
        "checkpoint": checkpoint,
        "internvl": internvl,
        "result": args.result.expanduser().resolve(),
        "schema": args.schema.expanduser().resolve(),
    }


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision(path: Path) -> Optional[str]:
    if not (path / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def verify_vendored_source(path: Path) -> Optional[Dict[str, Any]]:
    """Verify the bundled allowlist; return None for an explicit external source."""
    try:
        is_vendor = path.resolve() == VENDORED_SOURCE.resolve()
    except OSError:
        is_vendor = False
    if not is_vendor:
        return None

    manifest_path = Path(__file__).resolve().parent / "vendor-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_paths = {entry["path"] for entry in manifest["files"]}
    actual_paths = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix not in (".pyc", ".pyo")
    }
    extra_paths = sorted(actual_paths - expected_paths)
    missing_paths = sorted(expected_paths - actual_paths)
    if extra_paths or missing_paths:
        raise RuntimeError(
            "Vendored source allowlist mismatch; extra=%s missing=%s"
            % (extra_paths, missing_paths)
        )
    checked = 0
    for entry in manifest["files"]:
        file_path = path / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError("Vendored source file is missing: %s" % file_path)
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(
                "Vendored source hash mismatch for %s: got %s, expected %s"
                % (entry["path"], digest, entry["sha256"])
            )
        checked += 1
    return {
        "verified": True,
        "source_commit": manifest["source"]["commit"],
        "license": manifest["source"]["license"],
        "files_checked": checked,
        "manifest": str(manifest_path),
    }


def nvidia_smi_summary() -> Optional[Dict[str, str]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
        return {"name": values[0], "driver": values[1], "memory_total_mb": values[2]}
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def read_variant_without_dependencies(config: Path) -> Optional[str]:
    if not config.is_file():
        return None
    text = config.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"variant:\s*['\"]?([^'\"\s#]+)", text)
    return match.group(1) if match else None


def environment_report(include_torch: bool = False) -> Dict[str, Any]:
    names = (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "bitsandbytes",
        "hydra-core",
        "omegaconf",
        "pytorch-lightning",
    )
    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in names},
        "nvidia_smi": nvidia_smi_summary(),
    }
    if include_torch:
        import torch

        report.update(
            {
                "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
        )
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            report["gpu"] = {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_mb": round(
                    torch.cuda.get_device_properties(device).total_memory / (1024.0 * 1024.0), 2
                ),
            }
    return report


def base_result(args: argparse.Namespace, paths: Mapping[str, Path]) -> Dict[str, Any]:
    revision = git_revision(paths["upstream"])
    if revision is None and paths["upstream"] == VENDORED_SOURCE.resolve():
        revision = VENDORED_COMMIT
    return {
        "schema_version": "1.0",
        "created_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "status": args.mode,
        "run": {
            "mode": args.mode,
            "upstream_source": str(paths["upstream"]),
            "upstream_revision": revision,
            "config": str(paths["config"]),
            "checkpoint": str(paths["checkpoint"]),
            "quantization": args.quantization,
            "device": args.device,
            "internvl_revision": INTERNVL_REVISION,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "environment": environment_report(include_torch=False),
    }


def validate_arguments(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.sequence_length < 1 or args.iterations < 1:
        raise ValueError("batch-size, sequence-length and iterations must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must not be negative")


def assert_inputs(paths: Mapping[str, Path]) -> None:
    required = {
        "upstream source": paths["upstream"],
        "Hydra config": paths["config"],
        "checkpoint": paths["checkpoint"],
        "pinned InternVL2-1B snapshot": paths["internvl"] / "model.safetensors",
    }
    missing = ["%s: %s" % (name, path) for name, path in required.items() if not path.exists()]
    package = paths["upstream"] / "simlingo_training" / "models" / "driving.py"
    if not package.is_file():
        missing.append("upstream SimLingo package: %s" % package)
    if missing:
        raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing))
    verify_vendored_source(paths["upstream"])


def write_result(result: Mapping[str, Any], path: Path, schema: Optional[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if schema is not None and schema.is_file():
        try:
            import jsonschema

            definition = json.loads(schema.read_text(encoding="utf-8"))
            jsonschema.validate(instance=result, schema=definition)
        except ImportError:
            pass
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_state_dict(torch: Any, checkpoint: Path, allow_pickle: bool) -> MutableMapping[str, Any]:
    try:
        loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        if not allow_pickle:
            raise RuntimeError(
                "This PyTorch version does not support weights_only=True. "
                "Upgrade PyTorch or pass --allow-pickle for the trusted official checkpoint."
            )
        loaded = torch.load(str(checkpoint), map_location="cpu")
    except Exception:
        if not allow_pickle:
            raise RuntimeError(
                "Tensor-only checkpoint loading failed. If this is the trusted official "
                "RenzKa/simlingo checkpoint, retry with --allow-pickle."
            )
        loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)

    if isinstance(loaded, Mapping) and "state_dict" in loaded and isinstance(loaded["state_dict"], Mapping):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, MutableMapping):
        raise TypeError("Checkpoint does not contain a state_dict mapping")
    return loaded


def load_simlingo(args: argparse.Namespace, paths: Mapping[str, Path]) -> Tuple[Any, Dict[str, Any]]:
    if args.hf_cache_dir is not None:
        cache = str(args.hf_cache_dir.expanduser().resolve())
        os.environ.setdefault("HF_HOME", cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    sys.path.insert(0, str(paths["upstream"]))

    import hydra
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bitsandbytes GPU quantization")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This workflow requires a CUDA device, got %s" % device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = OmegaConf.load(str(paths["config"]))
    cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
    configured_variant = str(cfg.model.vision_model.variant)
    internvl_path = str(paths["internvl"])
    if "InternVL2-1B" not in configured_variant:
        raise ValueError(
            "The published quantization recipe expects OpenGVLab/InternVL2-1B, got %s"
            % configured_variant
        )
    # Resolve remote-code and base weights exclusively from the pinned local
    # snapshot downloaded by download_models.py.  The vendored SimLingo model
    # constructors receive this local path and therefore make no network call.
    cfg.model.vision_model.variant = internvl_path
    cfg.model.language_model.variant = internvl_path
    processor = AutoProcessor.from_pretrained(
        internvl_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    tokenizer.padding_side = "left"

    cache_dir = str((paths["model_dir"] / "hf-cache").resolve())
    previous_dtype = torch.get_default_dtype()
    started = time.perf_counter()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = hydra.utils.instantiate(
            cfg.model,
            cfg_data_module=cfg.data_module,
            processor=processor,
            cache_dir=cache_dir,
            _recursive_=False,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    instantiate_seconds = time.perf_counter() - started

    state = extract_state_dict(torch, paths["checkpoint"], args.allow_pickle)
    started = time.perf_counter()
    incompatible = model.load_state_dict(state, strict=not args.allow_nonstrict)
    load_seconds = time.perf_counter() - started
    state_tensor_count = len(state)
    del state
    gc.collect()

    model.eval()
    model.requires_grad_(False)
    model.to(dtype=torch.bfloat16)
    report = {
        "configured_variant": configured_variant,
        "internvl_local_snapshot": internvl_path,
        "internvl_revision": INTERNVL_REVISION,
        "transformers_offline": True,
        "checkpoint_bytes": paths["checkpoint"].stat().st_size,
        "state_tensor_count": state_tensor_count,
        "instantiate_seconds": round(instantiate_seconds, 3),
        "checkpoint_load_seconds": round(load_seconds, 3),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    return model, report


def unique_parameter_report(module: Any) -> Dict[str, Any]:
    seen: set = set()
    count = 0
    storage_bytes = 0
    dtypes: Dict[str, int] = {}
    for parameter in module.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        count += parameter.numel()
        storage_bytes += parameter.numel() * parameter.element_size()
        name = str(parameter.dtype).replace("torch.", "")
        dtypes[name] = dtypes.get(name, 0) + parameter.numel()
    return {
        "parameters": count,
        "parameter_storage_mb": round(storage_bytes / (1024.0 * 1024.0), 2),
        "dtype_parameter_counts": dtypes,
    }


def merge_lora(model: Any) -> Dict[str, Any]:
    language = model.language_model
    target = language.model
    lora_modules_before = [name for name, _ in target.named_modules() if "lora_" in name]
    can_merge = hasattr(target, "merge_and_unload")
    if not can_merge:
        raise RuntimeError("The official SimLingo language model is expected to contain PEFT LoRA")
    if not lora_modules_before:
        raise RuntimeError("PEFT model found, but no LoRA modules were discovered before merge")

    started = time.perf_counter()
    try:
        merged = target.merge_and_unload(safe_merge=True, progressbar=False)
    except TypeError:
        merged = target.merge_and_unload(safe_merge=True)
    language.model = merged
    language.lora = False

    input_embeddings = merged.get_input_embeddings()
    output_embeddings = merged.get_output_embeddings()
    if not hasattr(merged, "embed_tokens"):
        merged.embed_tokens = input_embeddings
    model.adaptors.language.embed_tokens = input_embeddings
    if output_embeddings is not None:
        model.adaptors.language.lm_head = output_embeddings

    remaining = [name for name, _ in merged.named_modules() if "lora_" in name]
    if remaining:
        raise RuntimeError("LoRA modules remain after merge: %s" % remaining[:5])
    return {
        "merged": True,
        "seconds": round(time.perf_counter() - started, 3),
        "lora_module_count_before": len(lora_modules_before),
        "lora_module_count_after": 0,
    }


def _make_bnb_linear(
    torch: Any,
    bnb: Any,
    source: Any,
    mode: str,
    int8_threshold: float,
    double_quant: bool,
) -> Any:
    has_bias = source.bias is not None
    if mode == "int8":
        replacement = bnb.nn.Linear8bitLt(
            source.in_features,
            source.out_features,
            bias=has_bias,
            has_fp16_weights=False,
            threshold=int8_threshold,
        )
        replacement.weight = bnb.nn.Int8Params(
            source.weight.detach().contiguous(),
            requires_grad=False,
            has_fp16_weights=False,
        )
    elif mode == "nf4":
        replacement = bnb.nn.Linear4bit(
            source.in_features,
            source.out_features,
            bias=has_bias,
            compute_dtype=torch.bfloat16,
            compress_statistics=double_quant,
            quant_type="nf4",
            quant_storage=torch.uint8,
        )
        replacement.weight = bnb.nn.Params4bit(
            source.weight.detach().contiguous(),
            requires_grad=False,
            compress_statistics=double_quant,
            quant_type="nf4",
            quant_storage=torch.uint8,
            module=replacement,
        )
    else:
        raise ValueError("Unsupported quantization mode: %s" % mode)

    if has_bias:
        replacement.bias = torch.nn.Parameter(source.bias.detach().contiguous(), requires_grad=False)
    replacement.source_cls = type(source)
    replacement.train(source.training)
    return replacement


def quantize_language_linears(model: Any, args: argparse.Namespace) -> Dict[str, Any]:
    import bitsandbytes as bnb
    import torch

    if args.quantization == "none":
        return {
            "backend": "pytorch",
            "mode": "none",
            "scope": "language_model.model transformer linear layers",
            "linear_layers_replaced": 0,
            "skipped_output_layers": [],
        }

    root = model.language_model.model
    output_module = model.adaptors.language.lm_head
    replaced: List[str] = []
    skipped: List[str] = []

    def recurse(parent: Any, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            path = "%s.%s" % (prefix, child_name) if prefix else child_name
            is_output = child is output_module or path == "lm_head" or path.endswith(".lm_head")
            if isinstance(child, torch.nn.Linear):
                if is_output:
                    skipped.append(path)
                    continue
                replacement = _make_bnb_linear(
                    torch,
                    bnb,
                    child,
                    args.quantization,
                    args.int8_threshold,
                    not args.no_double_quant,
                )
                setattr(parent, child_name, replacement)
                replaced.append(path)
            else:
                recurse(child, path)

    started = time.perf_counter()
    recurse(root)
    if not replaced:
        raise RuntimeError("No Qwen transformer Linear layers were found for quantization")
    return {
        "backend": "bitsandbytes",
        "bitsandbytes_version": package_version("bitsandbytes"),
        "mode": args.quantization,
        "scope": "language_model.model transformer linear layers",
        "linear_layers_replaced": len(replaced),
        "replaced_layer_names": replaced,
        "skipped_output_layers": skipped,
        "lm_head_policy": "BF16; SimLingo greedy_sample reads lm_head.weight via F.linear",
        "nf4_double_quant": not args.no_double_quant if args.quantization == "nf4" else None,
        "int8_threshold": args.int8_threshold if args.quantization == "int8" else None,
        "replacement_seconds": round(time.perf_counter() - started, 3),
    }


def count_bnb_layers(module: Any) -> int:
    try:
        import bitsandbytes as bnb
    except ImportError:
        return 0
    classes = (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)
    return sum(1 for child in module.modules() if isinstance(child, classes))


def floating_parameter_dtypes(module: Any) -> List[str]:
    dtypes = {
        str(parameter.dtype).replace("torch.", "")
        for parameter in module.parameters()
        if parameter.is_floating_point()
    }
    return sorted(dtypes)


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def benchmark_language_model(model: Any, args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    device = torch.device(args.device)
    language = model.language_model.model
    embeddings = model.adaptors.language.embed_tokens
    vocab_size = int(embeddings.num_embeddings)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.batch_size, args.sequence_length),
        generator=generator,
        device=device,
    )
    attention_mask = torch.ones_like(token_ids, dtype=torch.long)
    input_embeds = embeddings(token_ids).to(dtype=torch.bfloat16)

    def invoke() -> Any:
        output = language(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        # Force materialization of the paths used by SimLingo.
        _ = output.hidden_states[-1]
        _ = output[0]
        return output

    with torch.inference_mode():
        preflight = invoke()
        hidden_finite = bool(torch.isfinite(preflight.hidden_states[-1][:, -1]).all().item())
        logits_finite = bool(torch.isfinite(preflight[0][:, -1]).all().item())
        del preflight
        if not hidden_finite or not logits_finite:
            raise RuntimeError(
                "Non-finite values detected in the quantized language-model preflight"
            )
        for _ in range(args.warmup):
            invoke()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings: List[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            invoke()
            torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - started) * 1000.0)

    mean_ms = statistics.mean(timings)
    return {
        "scope": "language model forward",
        "note": "Synthetic token embeddings; excludes vision preprocessing and CARLA loop",
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "preflight_hidden_finite": hidden_finite,
        "preflight_logits_finite": logits_finite,
        "latency_ms": {
            "mean": round(mean_ms, 4),
            "p50": round(percentile(timings, 0.50), 4),
            "p90": round(percentile(timings, 0.90), 4),
            "min": round(min(timings), 4),
            "max": round(max(timings), 4),
        },
        "throughput_tokens_per_second": round(
            args.batch_size * args.sequence_length / (mean_ms / 1000.0), 3
        ),
        "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024.0**2), 2),
        "cuda_memory_reserved_mb": round(torch.cuda.memory_reserved(device) / (1024.0**2), 2),
        "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024.0**2), 2),
    }


def run_quantization(args: argparse.Namespace, paths: Mapping[str, Path], result: Dict[str, Any]) -> None:
    import torch

    result["environment"] = environment_report(include_torch=True)
    result["source_integrity"] = verify_vendored_source(paths["upstream"])
    model, model_report = load_simlingo(args, paths)
    result["model"] = model_report
    result["model"]["before_merge_and_quantization"] = unique_parameter_report(model)

    lora_report = merge_lora(model)
    quant_report = quantize_language_linears(model, args)
    quant_report["lora_merged"] = bool(lora_report["merged"])
    quant_report["lora"] = lora_report

    device = torch.device(args.device)
    started = time.perf_counter()
    model.to(device=device)
    torch.cuda.synchronize(device)
    quant_report["move_to_cuda_and_quantize_seconds"] = round(time.perf_counter() - started, 3)
    model.eval()

    quant_report["language_bnb_layers"] = count_bnb_layers(model.language_model.model)
    quant_report["vision_bnb_layers"] = count_bnb_layers(model.vision_model)
    quant_report["driving_head_bnb_layers"] = count_bnb_layers(model.adaptors.driving)
    if quant_report["vision_bnb_layers"] != 0 or quant_report["driving_head_bnb_layers"] != 0:
        raise AssertionError("Quantized modules escaped the language-model scope")
    if args.quantization != "none" and quant_report["language_bnb_layers"] == 0:
        raise AssertionError("No active bitsandbytes layers found after moving to CUDA")

    component_dtypes = {
        "vision_encoder": floating_parameter_dtypes(model.vision_model),
        "driving_adaptor": floating_parameter_dtypes(model.adaptors.driving),
        "waypoint_encoder": floating_parameter_dtypes(model.wp_encoder),
        "language_embeddings": [
            str(model.adaptors.language.embed_tokens.weight.dtype).replace("torch.", "")
        ],
        "language_lm_head": [
            str(model.adaptors.language.lm_head.weight.dtype).replace("torch.", "")
        ],
    }
    for component, dtypes in component_dtypes.items():
        if dtypes != ["bfloat16"]:
            raise AssertionError("%s must remain BF16, found %s" % (component, dtypes))
    quant_report["preserved_component_dtypes"] = component_dtypes

    result["model"]["after_quantization"] = unique_parameter_report(model)
    result["model"]["dtype_policy"] = {
        "vision_encoder": "bfloat16",
        "driving_heads": "bfloat16",
        "language_transformer_linears": args.quantization,
        "language_embeddings_and_lm_head": "bfloat16",
    }
    result["quantization"] = quant_report
    result["benchmark"] = benchmark_language_model(model, args)
    result["status"] = "success"


def main() -> int:
    args = parse_args()
    validate_arguments(args)
    paths = resolve_paths(args)
    result = base_result(args, paths)

    if args.mode == "dry-run":
        result["status"] = "dry-run"
        result["plan"] = {
            "actions": [
                "load official Hydra config and InternVL2-1B components on CPU",
                "load official SimLingo pytorch_model.pt",
                "load InternVL2-1B only from the pinned local revision",
                "merge PEFT LoRA into the Qwen language model",
                "replace only Qwen transformer Linear layers",
                "move the mixed BF16/quantized model to CUDA",
                "benchmark a language-model forward pass and write JSON",
            ],
            "inputs_exist": {name: path.exists() for name, path in paths.items() if name != "result"},
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.mode == "inspect":
        assert_inputs(paths)
        result["status"] = "inspect"
        result["inspection"] = {
            "variant": read_variant_without_dependencies(paths["config"]),
            "checkpoint_bytes": paths["checkpoint"].stat().st_size,
            "checkpoint_size_gib": round(paths["checkpoint"].stat().st_size / (1024.0**3), 3),
            "carla_required_for_model_load": False,
            "upstream_model_import": "simlingo_training.models.driving",
            "source_integrity": verify_vendored_source(paths["upstream"]),
        }
        write_result(result, paths["result"], paths["schema"])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    try:
        assert_inputs(paths)
        run_quantization(args, paths, result)
        write_result(result, paths["result"], paths["schema"])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("Result: %s" % paths["result"])
        return 0
    except Exception as exc:
        result["status"] = "error"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            write_result(result, paths["result"], paths["schema"])
        except Exception:
            pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
