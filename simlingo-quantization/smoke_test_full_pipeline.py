#!/usr/bin/env python3
"""Run a deterministic vision-to-driving SimLingo artifact smoke test."""

import argparse
import datetime as _datetime
import gc
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

import load_quantized_model as artifact_loader
import quantize_and_benchmark as workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a dense baseline and NF4 across the complete SimLingo pipeline."
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/simlingo-3070"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/simlingo"))
    parser.add_argument("--internvl-dir", type=Path, default=None)
    parser.add_argument("--upstream", type=Path, default=workflow.VENDORED_SOURCE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--boundary-dtype",
        choices=workflow.BOUNDARY_DTYPE_CHOICES,
        default="bf16",
        help="Floating dtype for camera input, vision, embeddings and driving heads.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/simlingo-full-pipeline-smoke.json"),
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


def load_conversation_module(snapshot: Path) -> Any:
    path = snapshot / "conversation.py"
    spec = importlib.util.spec_from_file_location("simlingo_local_conversation", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_prompt(tokenizer: Any, snapshot: Path, num_image_tokens: int) -> Tuple[str, Any, Any]:
    import torch

    question = (
        "Current speed: 0.0 m/s. Target waypoint: "
        "<TARGET_POINT><TARGET_POINT>. What should the ego do next?"
    )
    conversation = load_conversation_module(snapshot)
    template = conversation.get_conv_template("internlm2-chat")
    template.append_message(template.roles[0], "<image>\n" + question)
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt()
    system_prompt = (
        template.system_template.replace("{system_message}", template.system_message)
        + template.sep
    )
    prompt = prompt.replace(system_prompt, "")
    image_tokens = "<img>" + "<IMG_CONTEXT>" * num_image_tokens + "</img>"
    prompt = prompt.replace("<image>", image_tokens, 1)
    encoded = tokenizer(
        [prompt],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    ids = encoded["input_ids"]
    valid = ids != tokenizer.pad_token_id
    target_id = int(tokenizer.convert_tokens_to_ids("<TARGET_POINT>"))
    context_id = int(tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))
    if int((ids == context_id).sum()) != num_image_tokens:
        raise AssertionError("Image placeholder count does not match image tiles")
    if int((ids == target_id).sum()) != 2:
        raise AssertionError("Expected exactly two target-point placeholders")
    return prompt, ids, valid


def build_input(
    model: Any,
    snapshot: Path,
    upstream: Path,
    device: str,
    boundary_dtype: str,
) -> Any:
    import torch

    sys.path.insert(0, str(upstream))
    from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
    from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess

    height, width = 512, 1024
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    rgb = np.stack(
        (
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.bitwise_xor(np.broadcast_to(x, (height, width)), np.broadcast_to(y, (height, width))),
        ),
        axis=-1,
    )
    cropped_height = int(height - (height * 4.8) // 16)

    from PIL import Image

    image = Image.fromarray(rgb[:cropped_height], mode="RGB")
    tiles_pil = dynamic_preprocess(
        image,
        image_size=448,
        use_thumbnail=False,
        max_num=2,
    )
    if len(tiles_pil) != 2:
        raise AssertionError("Expected two image tiles, got %d" % len(tiles_pil))
    transform = build_transform(448)
    tiles = torch.stack([transform(tile) for tile in tiles_pil])
    runtime_dtype = workflow.boundary_torch_dtype(torch, boundary_dtype)
    camera_images = tiles.unsqueeze(0).unsqueeze(0).to(
        device=device, dtype=runtime_dtype
    )

    tokens_per_patch = int((448 // 14) ** 2 * (0.5**2))
    prompt, ids, valid = build_prompt(
        model.tokenizer,
        snapshot,
        num_image_tokens=len(tiles_pil) * tokens_per_patch,
    )
    ids = ids.to(device)
    valid = valid.to(device)
    target_id = int(model.tokenizer.convert_tokens_to_ids("<TARGET_POINT>"))
    target_points = np.asarray([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32)
    inference_language = LanguageLabel(
        phrase_ids=ids,
        phrase_valid=valid,
        phrase_mask=valid,
        placeholder_values=[{target_id: target_points}],
        language_string=[prompt],
        loss_masking=torch.zeros_like(valid),
    )
    # Keep the training prompt observably different.  The artifact inference
    # wrapper must exclusively consume prompt_inference, as the upstream
    # DrivingModel.forward(inference=True) path intends.
    training_target_points = np.asarray(
        [[-100.0, 50.0], [-200.0, 100.0]], dtype=np.float32
    )
    training_language = LanguageLabel(
        phrase_ids=ids,
        phrase_valid=valid,
        phrase_mask=valid,
        placeholder_values=[{target_id: training_target_points}],
        language_string=["training-prompt-sentinel"],
        loss_masking=torch.zeros_like(valid),
    )
    return DrivingInput(
        camera_images=camera_images,
        image_sizes=torch.tensor([[cropped_height, width]], dtype=torch.long),
        camera_intrinsics=torch.eye(3, device=device).unsqueeze(0),
        camera_extrinsics=torch.eye(4, device=device).unsqueeze(0),
        vehicle_speed=torch.tensor([[0.0]], device=device),
        target_point=torch.tensor([[10.0, 0.0]], device=device),
        prompt=training_language,
        prompt_inference=inference_language,
    )


def tensor_comparison(candidate: Any, reference: Any) -> Dict[str, float]:
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


def run_mode(
    quantization: str,
    label: str,
    args: argparse.Namespace,
    snapshot: Path,
) -> Tuple[Dict[str, Any], Any, Any]:
    import torch

    model, load_report = artifact_loader.load_artifact_model(
        artifact_dir=args.artifact_dir,
        model_dir=args.model_dir,
        upstream=args.upstream,
        internvl_dir=snapshot,
        quantization=quantization,
        device=args.device,
        boundary_dtype=args.boundary_dtype,
    )
    model.predict_language = False
    load_report["predict_language_for_smoke"] = False
    if model.adaptors.driving.order != ["route", "speed_wps"]:
        raise AssertionError("Unexpected driving adaptor order: %s" % model.adaptors.driving.order)
    if sum(model.adaptors.driving.sizes.values()) != 30:
        raise AssertionError("Expected 30 driving queries")
    driving_input = build_input(
        model,
        snapshot,
        args.upstream.resolve(),
        args.device,
        args.boundary_dtype,
    )
    if driving_input.prompt.language_string == driving_input.prompt_inference.language_string:
        raise AssertionError("Smoke test requires distinct training and inference prompts")

    hidden_dtypes = []
    logits_dtypes = []
    adaptor_input_dtypes = []
    adaptor_mask_dtypes = []
    waypoint_embedding_dtypes = []

    def capture_language_hidden(_module: Any, _inputs: Any, output: Any) -> None:
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states:
            hidden_dtypes.append(
                str(hidden_states[-1].dtype).replace("torch.", "")
            )
        logits = getattr(output, "logits", None)
        if logits is not None:
            logits_dtypes.append(str(logits.dtype).replace("torch.", ""))

    def capture_adaptor_inputs(_module: Any, _inputs: Any, output: Any) -> None:
        if isinstance(output, dict):
            inputs = output.get("inputs")
            inputs_mask = output.get("inputs_mask")
            if inputs is not None:
                adaptor_input_dtypes.append(str(inputs.dtype).replace("torch.", ""))
            if inputs_mask is not None:
                adaptor_mask_dtypes.append(str(inputs_mask.dtype).replace("torch.", ""))

    def capture_waypoint_embeddings(_module: Any, _inputs: Any, output: Any) -> None:
        if output is not None:
            waypoint_embedding_dtypes.append(str(output.dtype).replace("torch.", ""))

    hidden_hook = model.language_model.model.register_forward_hook(
        capture_language_hidden
    )
    adaptor_hook = model.adaptors.register_forward_hook(capture_adaptor_inputs)
    waypoint_hook = model.wp_encoder.register_forward_hook(
        capture_waypoint_embeddings
    )

    def invoke() -> Any:
        return artifact_loader.forward_driving_without_language(model, driving_input)

    try:
        with torch.inference_mode():
            speed_wps, route, language = invoke()
            if tuple(speed_wps.shape) != (1, 10, 2):
                raise AssertionError(
                    "Unexpected speed waypoint shape: %s" % (speed_wps.shape,)
                )
            if tuple(route.shape) != (1, 20, 2):
                raise AssertionError("Unexpected route shape: %s" % (route.shape,))
            if language != []:
                raise AssertionError("Language output must be disabled for this smoke test")
            finite = bool(torch.isfinite(speed_wps).all() and torch.isfinite(route).all())
            if not finite:
                raise RuntimeError("Non-finite driving output in %s" % label)
            for _ in range(args.warmup):
                invoke()
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            timings = []
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                speed_wps, route, language = invoke()
                end.record()
                end.synchronize()
                timings.append(float(start.elapsed_time(end)))
    finally:
        hidden_hook.remove()
        adaptor_hook.remove()
        waypoint_hook.remove()

    speed_cpu = speed_wps.detach().cpu()
    route_cpu = route.detach().cpu()
    expected_dtype = workflow.boundary_dtype_name(args.boundary_dtype)
    observed_hidden_dtypes = sorted(set(hidden_dtypes))
    observed_logits_dtypes = sorted(set(logits_dtypes))
    observed_adaptor_dtypes = sorted(set(adaptor_input_dtypes))
    observed_adaptor_mask_dtypes = sorted(set(adaptor_mask_dtypes))
    observed_waypoint_dtypes = sorted(set(waypoint_embedding_dtypes))
    dtype_audit = {
        "expected": expected_dtype,
        "camera_input": str(driving_input.camera_images.dtype).replace("torch.", ""),
        "token_ids": str(driving_input.prompt_inference.phrase_ids.dtype).replace("torch.", ""),
        "token_mask": str(driving_input.prompt_inference.phrase_valid.dtype).replace("torch.", ""),
        "adaptor_inputs": observed_adaptor_dtypes,
        "adaptor_masks": observed_adaptor_mask_dtypes,
        "waypoint_embeddings": observed_waypoint_dtypes,
        "language_hidden": observed_hidden_dtypes,
        "language_logits": observed_logits_dtypes,
        "route_output": str(route.dtype).replace("torch.", ""),
        "speed_waypoints_output": str(speed_wps.dtype).replace("torch.", ""),
    }
    dtype_audit["passed"] = bool(
        dtype_audit["camera_input"] == expected_dtype
        and dtype_audit["token_ids"] == "int64"
        and dtype_audit["token_mask"] == "bool"
        and observed_adaptor_dtypes == [expected_dtype]
        and observed_adaptor_mask_dtypes == ["bool"]
        and observed_waypoint_dtypes == [expected_dtype]
        and observed_hidden_dtypes == [expected_dtype]
        and observed_logits_dtypes == [expected_dtype]
        and dtype_audit["route_output"] == expected_dtype
        and dtype_audit["speed_waypoints_output"] == expected_dtype
    )
    report = {
        "label": label,
        "load": load_report,
        "finite": finite,
        "runtime_dtype_audit": dtype_audit,
        "speed_waypoints_shape": list(speed_cpu.shape),
        "route_shape": list(route_cpu.shape),
        "latency_ms": {
            "mean": sum(timings) / len(timings),
            "p50": percentile(timings, 0.50),
            "p90": percentile(timings, 0.90),
            "min": min(timings),
            "max": max(timings),
        },
        "cuda_allocated_mib": torch.cuda.memory_allocated(args.device) / 2**20,
        "cuda_peak_mib": torch.cuda.max_memory_allocated(args.device) / 2**20,
        "speed_waypoints": speed_cpu.float().tolist(),
        "route": route_cpu.float().tolist(),
    }
    del model, driving_input, speed_wps, route
    gc.collect()
    torch.cuda.empty_cache()
    return report, speed_cpu, route_cpu


def comparison_passes(comparison: Dict[str, Dict[str, float]]) -> Tuple[bool, Dict[str, bool]]:
    route = comparison["route"]
    speed = comparison["speed_waypoints"]
    checks = {
        "route_cosine": route["cosine"] >= 0.999,
        "route_mean_abs": route["mean_abs"] <= 0.05,
        "route_rmse": route["rmse"] <= 0.075,
        "speed_waypoints_cosine": speed["cosine"] >= 0.995,
        "speed_waypoints_mean_abs": speed["mean_abs"] <= 0.35,
        "speed_waypoints_rmse": speed["rmse"] <= 0.50,
    }
    return all(checks.values()), checks


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    args.artifact_dir = args.artifact_dir.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.upstream = args.upstream.expanduser().resolve()
    snapshot = (
        args.internvl_dir.expanduser().resolve()
        if args.internvl_dir is not None
        else args.model_dir / "pretrained" / "InternVL2-1B"
    )

    results: Dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "scope": "synthetic RGB -> InternViT -> Qwen prefill -> route and speed waypoint heads",
        "timing_scope": (
            "CUDA events around model forward; excludes model/artifact loading, CPU image "
            "generation/cropping/tiling/normalization, and host-to-device transfer"
        ),
        "input": {
            "source": "deterministic synthetic 1024x512 RGB gradient/checkerboard",
            "tiles": [1, 1, 2, 3, 448, 448],
            "target_points": [[10.0, 0.0], [20.0, 0.0]],
            "predict_language": False,
            "training_prompt_distinct_from_inference_prompt": True,
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "boundary_dtype": workflow.boundary_dtype_name(args.boundary_dtype),
        "acceptance": {
            "finite": True,
            "runtime_dtype_matches_boundary": True,
            "route": {"cosine_min": 0.999, "mean_abs_max": 0.05, "rmse_max": 0.075},
            "speed_waypoints": {
                "cosine_min": 0.995,
                "mean_abs_max": 0.35,
                "rmse_max": 0.50,
            },
            "performance": {
                "latency_ratio_max": 1.20,
                "peak_cuda_saving_min_percent": 20.0,
            },
            "note": "Synthetic-image smoke thresholds; not a driving-accuracy threshold.",
        },
        "models": {},
    }
    reference_label = args.boundary_dtype
    candidate_label = (
        "nf4" if args.boundary_dtype == "bf16" else "nf4_fp16_boundaries"
    )
    results["reference_model"] = reference_label
    dense_report, dense_speed, dense_route = run_mode(
        reference_label, reference_label, args, snapshot
    )
    dense_report["accepted"] = bool(
        dense_report["finite"] and dense_report["runtime_dtype_audit"]["passed"]
    )
    results["models"][reference_label] = dense_report
    nf4_report, nf4_speed, nf4_route = run_mode(
        "nf4", candidate_label, args, snapshot
    )
    comparison_key = "comparison_to_%s" % reference_label
    comparison = {
        "speed_waypoints": tensor_comparison(nf4_speed, dense_speed),
        "route": tensor_comparison(nf4_route, dense_route),
    }
    comparison_accepted, comparison_checks = comparison_passes(comparison)
    latency_ratio = (
        nf4_report["latency_ms"]["mean"] / dense_report["latency_ms"]["mean"]
    )
    peak_cuda_saving_percent = 100.0 * (
        1.0 - nf4_report["cuda_peak_mib"] / dense_report["cuda_peak_mib"]
    )
    performance_checks = {
        "latency_ratio": latency_ratio <= 1.20,
        "peak_cuda_saving": peak_cuda_saving_percent >= 20.0,
    }
    nf4_report[comparison_key] = comparison
    nf4_report["acceptance_checks"] = comparison_checks
    nf4_report["performance_comparison_to_%s" % reference_label] = {
        "latency_ratio": latency_ratio,
        "latency_change_percent": 100.0 * (latency_ratio - 1.0),
        "peak_cuda_saving_percent": peak_cuda_saving_percent,
        "checks": performance_checks,
    }
    nf4_report["accepted"] = bool(
        nf4_report["finite"]
        and nf4_report["runtime_dtype_audit"]["passed"]
        and comparison_accepted
        and all(performance_checks.values())
    )
    results["models"][candidate_label] = nf4_report
    results["passed"] = bool(dense_report["accepted"] and nf4_report["accepted"])

    result_path = args.result.expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if results["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
