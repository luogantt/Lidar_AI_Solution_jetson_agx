#!/usr/bin/env python3
"""Reconstruct a complete SimLingo model from split exported artifacts."""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import quantize_and_benchmark as workflow


def _normalize_boundary_dtype(value: str) -> str:
    aliases = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "torch.float16": "fp16",
    }
    try:
        return aliases[value.lower()]
    except (AttributeError, KeyError):
        raise ValueError("Unsupported boundary dtype: %s" % value)


def _artifact_from_manifest(
    artifact_dir: Path,
    artifacts: Mapping[str, Any],
    key: str,
    fallback_name: str,
    strict_manifest: bool = False,
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


def _load_artifact_layout(
    artifact_dir: Path,
    boundary_dtype: str,
    quantization: str,
) -> Tuple[Path, Path, Dict[str, Any]]:
    layout = workflow.artifact_layout(boundary_dtype)
    manifest_path = artifact_dir / "artifact-manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise RuntimeError("Artifact manifest must contain a JSON object")
        manifest = loaded

    dtype_policy = manifest.get("dtype_policy", {})
    precision_profile = manifest.get("precision_profile", {})
    declared_dtype = None
    if isinstance(dtype_policy, Mapping):
        declared_dtype = dtype_policy.get("boundary_dtype")
    if declared_dtype is None and isinstance(precision_profile, Mapping):
        declared_dtype = precision_profile.get("boundary_dtype")

    schema_version = manifest.get("schema_version")
    legacy_bf16 = schema_version == "1.0" and declared_dtype is None
    strict_manifest = bool(manifest) and schema_version != "1.0"
    if legacy_bf16 and boundary_dtype != "bf16":
        raise RuntimeError(
            "Schema 1.0 artifact does not declare a boundary dtype and is therefore BF16; "
            "requested %s" % boundary_dtype
        )
    if declared_dtype is not None:
        declared_short = _normalize_boundary_dtype(declared_dtype)
        if declared_short != boundary_dtype:
            raise RuntimeError(
                "Artifact boundary dtype is %s, requested %s"
                % (declared_short, boundary_dtype)
            )

    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("Artifact manifest 'artifacts' must be an object")
    non_language_key = "non_language_%s" % boundary_dtype
    dense_key = "llm_merged_%s" % boundary_dtype
    if quantization in workflow.BOUNDARY_DTYPE_CHOICES:
        llm_key = dense_key
        llm_fallback = layout["baseline_llm"]
    else:
        llm_fallback = layout["%s_llm" % quantization]
        llm_key = llm_fallback

    non_language_path = _artifact_from_manifest(
        artifact_dir,
        artifacts,
        non_language_key,
        layout["non_language"],
        strict_manifest=strict_manifest,
    )
    llm_dir = _artifact_from_manifest(
        artifact_dir,
        artifacts,
        llm_key,
        llm_fallback,
        strict_manifest=strict_manifest,
    )
    manifest_report = {
        "path": str(manifest_path) if manifest_path.is_file() else None,
        "schema_version": schema_version,
        "legacy_schema_1_0_bf16": legacy_bf16,
        "strict_manifest_paths": strict_manifest,
        "declared_boundary_dtype": declared_dtype,
    }
    return non_language_path, llm_dir, manifest_report


def forward_driving_without_language(model: Any, driving_input: Any) -> Tuple[Any, Any, list]:
    """Run one vision/LLM/driving pass while skipping language generation.

    Upstream ``DrivingModel.forward(..., predict_language=False)`` passes the
    ``(features, logits)`` tuple returned by ``forward_model`` to a tensor-only
    splitter.  This wrapper performs the intended tuple unpacking without
    modifying the pinned upstream source snapshot.
    """

    # Match the upstream inference path: token IDs and placeholder values must
    # both come from prompt_inference.  forward_model reads ``prompt`` for the
    # placeholder values, so use a shallow NamedTuple copy with that field
    # replaced instead of mutating the caller's input.
    inference_input = driving_input._replace(prompt=driving_input.prompt_inference)
    adaptor_dict = model.adaptors(inference_input, inference=True)
    features, _ = model.forward_model(inference_input, adaptor_dict)
    outputs_by_adaptor = model.adaptors.split_outputs_by_adaptor(adaptor_dict, features)
    predictions = model.adaptors.driving.get_predictions(outputs_by_adaptor["driving"])
    return predictions["speed_wps"], predictions.get("route"), []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a complete dense or quantized SimLingo model from exported artifacts."
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/simlingo-3070"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/simlingo"))
    parser.add_argument("--internvl-dir", type=Path, default=None)
    parser.add_argument("--upstream", type=Path, default=workflow.VENDORED_SOURCE)
    parser.add_argument(
        "--quantization",
        choices=("bf16", "fp16", "nf4", "int8"),
        default="nf4",
        help="LLM artifact to load; bf16/fp16 select the matching dense baseline.",
    )
    parser.add_argument(
        "--boundary-dtype",
        choices=workflow.BOUNDARY_DTYPE_CHOICES,
        default=None,
        help="Floating dtype around Qwen. Defaults to the dense baseline dtype, or bf16.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_artifact_model(
    artifact_dir: Path,
    model_dir: Path,
    upstream: Path = workflow.VENDORED_SOURCE,
    internvl_dir: Path = None,
    quantization: str = "nf4",
    device: str = "cuda:0",
    boundary_dtype: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the full vision-language-driving module without the 2.57 GB checkpoint.

    The pinned InternVL2-1B snapshot remains necessary to construct the upstream
    remote-code classes. Trained vision/driving weights and Qwen weights come
    from the dtype-specific paths declared by the artifact manifest. Legacy
    schema 1.0 artifacts without a dtype declaration retain their BF16 layout.
    """

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import hydra
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    artifact_dir = artifact_dir.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    upstream = upstream.expanduser().resolve()
    base_dir = (
        internvl_dir.expanduser().resolve()
        if internvl_dir is not None
        else model_dir / "pretrained" / "InternVL2-1B"
    )
    if boundary_dtype is None:
        boundary_dtype = (
            quantization if quantization in workflow.BOUNDARY_DTYPE_CHOICES else "bf16"
        )
    boundary_dtype = _normalize_boundary_dtype(boundary_dtype)
    if quantization in workflow.BOUNDARY_DTYPE_CHOICES and quantization != boundary_dtype:
        raise ValueError(
            "Dense %s baseline requires --boundary-dtype %s, got %s"
            % (quantization, quantization, boundary_dtype)
        )
    if quantization not in ("bf16", "fp16", "nf4", "int8"):
        raise ValueError("Unsupported quantization artifact: %s" % quantization)

    config_path = artifact_dir / "simlingo-config.yaml"
    processor_dir = artifact_dir / "processor"
    non_language_path, llm_dir, manifest_report = _load_artifact_layout(
        artifact_dir,
        boundary_dtype,
        quantization,
    )
    torch_boundary_dtype = workflow.boundary_torch_dtype(torch, boundary_dtype)
    expected_dtype = workflow.boundary_dtype_name(boundary_dtype)

    required = (
        config_path,
        non_language_path,
        processor_dir,
        base_dir / "model.safetensors",
        llm_dir,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model inputs:\n  " + "\n  ".join(missing))
    source_report = workflow.verify_vendored_source(upstream)
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("A CUDA device is required")

    sys.path.insert(0, str(upstream))
    cfg = OmegaConf.load(str(config_path))
    cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
    cfg.model.vision_model.variant = str(base_dir)
    cfg.model.language_model.variant = str(base_dir)
    cfg.model.language_model.lora = False
    processor = AutoTokenizer.from_pretrained(processor_dir, local_files_only=True)
    added_tokens = processor.add_special_tokens(
        {"additional_special_tokens": workflow.SPECIAL_TOKENS}
    )
    if added_tokens != 0:
        raise RuntimeError(
            "Exported processor is incomplete; expected all SimLingo tokens, added %d"
            % added_tokens
        )
    processor.padding_side = "left"

    started = time.perf_counter()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch_boundary_dtype)
        model = hydra.utils.instantiate(
            cfg.model,
            cfg_data_module=cfg.data_module,
            processor=processor,
            cache_dir=str(base_dir),
            _recursive_=False,
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    non_language = load_file(str(non_language_path), device="cpu")
    non_language_float_dtypes = sorted(
        {
            str(value.dtype).replace("torch.", "")
            for value in non_language.values()
            if value.is_floating_point()
        }
    )
    if non_language_float_dtypes != [expected_dtype]:
        raise AssertionError(
            "Non-language artifact must contain %s floating tensors, found %s"
            % (expected_dtype, non_language_float_dtypes)
        )
    incompatible = model.load_state_dict(non_language, strict=False)
    illegal_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(("language_model.", "adaptors.language."))
    ]
    if incompatible.unexpected_keys or illegal_missing:
        raise RuntimeError(
            "Non-language artifact mismatch; unexpected=%s illegal_missing=%s"
            % (incompatible.unexpected_keys, illegal_missing)
        )

    cuda_index = torch.device(device).index or 0
    quantized_llm = AutoModelForCausalLM.from_pretrained(
        llm_dir,
        local_files_only=True,
        torch_dtype=torch_boundary_dtype,
        device_map={"": cuda_index},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    quantized_llm.eval()
    # The upstream LanguageAdaptor expects this convenience alias.
    quantized_llm.embed_tokens = quantized_llm.get_input_embeddings()

    from simlingo_training.models.adaptors.adaptors import LanguageAdaptor

    old_llm = model.language_model.model
    old_language_adaptor = model.adaptors.language
    model.language_model.model = quantized_llm
    model.language_model.lora = False
    model.language_model.vocab_size = quantized_llm.config.vocab_size
    model.language_model.hidden_size = quantized_llm.config.hidden_size
    model.language_model.max_position_embeddings = quantized_llm.config.max_position_embeddings
    model.adaptors.language = LanguageAdaptor(model.language_model)
    del old_llm, old_language_adaptor, non_language
    gc.collect()

    # A loaded bitsandbytes model must never be moved or recast through a
    # whole-model ``model.to(...)`` call. Move only the dense boundary modules.
    model.vision_model.to(device=device, dtype=torch_boundary_dtype)
    model.adaptors.driving.to(device=device, dtype=torch_boundary_dtype)
    model.wp_encoder.to(device=device, dtype=torch_boundary_dtype)
    model.eval()
    model.requires_grad_(False)

    input_embeddings = quantized_llm.get_input_embeddings()
    output_embeddings = quantized_llm.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("Qwen artifact must expose input embeddings and lm_head")
    input_dtype = str(input_embeddings.weight.dtype).replace("torch.", "")
    output_dtype = str(output_embeddings.weight.dtype).replace("torch.", "")
    language_dtype = str(quantized_llm.dtype).replace("torch.", "")
    component_dtypes = {
        "vision_encoder": workflow.floating_parameter_dtypes(model.vision_model),
        "waypoint_encoder": workflow.floating_parameter_dtypes(model.wp_encoder),
        "driving_heads": workflow.floating_parameter_dtypes(model.adaptors.driving),
        "language_non_quantized_parameters": workflow.floating_parameter_dtypes(
            quantized_llm
        ),
    }
    for component, dtypes in component_dtypes.items():
        if dtypes != [expected_dtype]:
            raise AssertionError(
                "%s must use %s floating parameters, found %s"
                % (component, expected_dtype, dtypes)
            )
    if input_dtype != expected_dtype or output_dtype != expected_dtype:
        raise AssertionError(
            "Language boundary dtype mismatch: embedding=%s lm_head=%s expected=%s"
            % (input_dtype, output_dtype, expected_dtype)
        )
    if language_dtype != expected_dtype:
        raise AssertionError(
            "Qwen model dtype must be %s, found %s" % (expected_dtype, language_dtype)
        )

    bnb_layer_counts = {
        "language_model": workflow.count_bnb_layers(quantized_llm),
        "vision_encoder": workflow.count_bnb_layers(model.vision_model),
        "waypoint_encoder": workflow.count_bnb_layers(model.wp_encoder),
        "driving_heads": workflow.count_bnb_layers(model.adaptors.driving),
    }
    if any(bnb_layer_counts[name] != 0 for name in (
        "vision_encoder",
        "waypoint_encoder",
        "driving_heads",
    )):
        raise AssertionError("bitsandbytes layers escaped Qwen: %s" % bnb_layer_counts)
    expected_bnb_layers = (
        workflow.EXPECTED_QWEN_LINEAR_LAYERS if quantization in ("nf4", "int8") else 0
    )
    if bnb_layer_counts["language_model"] != expected_bnb_layers:
        raise AssertionError(
            "Expected %d Qwen bitsandbytes layers for %s, found %d"
            % (
                expected_bnb_layers,
                quantization,
                bnb_layer_counts["language_model"],
            )
        )
    compute_dtypes = workflow.bnb_compute_dtypes(quantized_llm)
    expected_compute_dtypes = [expected_dtype] if quantization == "nf4" else []
    if compute_dtypes != expected_compute_dtypes:
        raise AssertionError(
            "bitsandbytes compute dtype mismatch: expected %s, found %s"
            % (expected_compute_dtypes, compute_dtypes)
        )
    loaded_in_4bit = bool(getattr(quantized_llm, "is_loaded_in_4bit", False))
    loaded_in_8bit = bool(getattr(quantized_llm, "is_loaded_in_8bit", False))
    expected_loaded_flags = {
        "bf16": (False, False),
        "fp16": (False, False),
        "nf4": (True, False),
        "int8": (False, True),
    }[quantization]
    if (loaded_in_4bit, loaded_in_8bit) != expected_loaded_flags:
        raise AssertionError(
            "Unexpected quantized load flags for %s: 4bit=%s 8bit=%s"
            % (quantization, loaded_in_4bit, loaded_in_8bit)
        )

    report = {
        "quantization": quantization,
        "boundary_dtype": expected_dtype,
        "device": device,
        "gpu": torch.cuda.get_device_name(cuda_index),
        "source_integrity": source_report,
        "artifact_manifest": manifest_report,
        "llm_artifact": str(llm_dir),
        "non_language_artifact": str(non_language_path),
        "processor_artifact": str(processor_dir),
        "loaded_in_4bit": loaded_in_4bit,
        "loaded_in_8bit": loaded_in_8bit,
        "language_model_dtype": language_dtype,
        "language_embedding_dtype": input_dtype,
        "language_lm_head_dtype": output_dtype,
        "non_language_artifact_float_dtypes": non_language_float_dtypes,
        "component_float_dtypes": component_dtypes,
        "bnb_layer_counts": bnb_layer_counts,
        "bnb_compute_dtypes": compute_dtypes,
        "missing_language_keys": len(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "load_seconds": round(time.perf_counter() - started, 3),
        "initial_predict_language": bool(model.predict_language),
        "driving_only_forward": "forward_driving_without_language",
    }
    return model, report


def main() -> int:
    args = parse_args()
    model, report = load_artifact_model(
        artifact_dir=args.artifact_dir,
        model_dir=args.model_dir,
        upstream=args.upstream,
        internvl_dir=args.internvl_dir,
        quantization=args.quantization,
        device=args.device,
        boundary_dtype=args.boundary_dtype,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
