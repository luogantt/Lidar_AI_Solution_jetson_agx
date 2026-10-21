#!/usr/bin/env python3
"""Reconstruct a complete SimLingo model from split exported artifacts."""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import quantize_and_benchmark as workflow


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
        description="Load a complete BF16 or NF4 SimLingo model from exported artifacts."
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/simlingo-3070"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/simlingo"))
    parser.add_argument("--internvl-dir", type=Path, default=None)
    parser.add_argument("--upstream", type=Path, default=workflow.VENDORED_SOURCE)
    parser.add_argument("--quantization", choices=("bf16", "nf4", "int8"), default="nf4")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_artifact_model(
    artifact_dir: Path,
    model_dir: Path,
    upstream: Path = workflow.VENDORED_SOURCE,
    internvl_dir: Path = None,
    quantization: str = "nf4",
    device: str = "cuda:0",
) -> Tuple[Any, Dict[str, Any]]:
    """Load the full vision-language-driving module without the 2.57 GB checkpoint.

    The pinned InternVL2-1B snapshot remains necessary to construct the upstream
    remote-code classes.  Trained vision/driving weights come from
    non_language_bf16.safetensors and Qwen weights come from the selected LLM
    artifact.
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
    config_path = artifact_dir / "simlingo-config.yaml"
    non_language_path = artifact_dir / "non_language_bf16.safetensors"
    processor_dir = artifact_dir / "processor"
    llm_names = {
        "bf16": "llm_merged_bf16",
        "nf4": "llm_bnb_nf4",
        "int8": "llm_bnb_int8",
    }
    llm_dir = artifact_dir / llm_names[quantization]

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
        torch.set_default_dtype(torch.bfloat16)
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
        torch_dtype=torch.bfloat16,
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

    # bitsandbytes modules must not be moved or recast through model.to().
    model.vision_model.to(device=device, dtype=torch.bfloat16)
    model.adaptors.driving.to(device=device, dtype=torch.bfloat16)
    model.wp_encoder.to(device=device, dtype=torch.bfloat16)
    model.eval()
    model.requires_grad_(False)

    report = {
        "quantization": quantization,
        "device": device,
        "gpu": torch.cuda.get_device_name(cuda_index),
        "source_integrity": source_report,
        "llm_artifact": str(llm_dir),
        "non_language_artifact": str(non_language_path),
        "processor_artifact": str(processor_dir),
        "loaded_in_4bit": bool(getattr(quantized_llm, "is_loaded_in_4bit", False)),
        "loaded_in_8bit": bool(getattr(quantized_llm, "is_loaded_in_8bit", False)),
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
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
