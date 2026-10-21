#!/usr/bin/env python3
"""Download only the files needed for SimLingo inference.

The official Hugging Face repository also contains optimizer shards.  Those are
not needed for inference and are intentionally excluded here.
"""

import argparse
import datetime as _datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional


REPO_ID = "RenzKa/simlingo"
SIMLINGO_REVISION = "26c7c89e797d4e25bbf640013317af8da26a5454"
INTERNVL_REPO_ID = "OpenGVLab/InternVL2-1B"
INTERNVL_REVISION = "0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187"
DEFAULT_FILES = (
    ".hydra/config.yaml",
    "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",
)
PUBLISHED_CHECKPOINT_BYTES = 2_569_679_322
PUBLISHED_CHECKPOINT_SHA256 = "ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28"
INTERNVL_FILES = (
    "added_tokens.json",
    "config.json",
    "configuration_intern_vit.py",
    "configuration_internvl_chat.py",
    "conversation.py",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "modeling_intern_vit.py",
    "modeling_internvl_chat.py",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)
INTERNVL_MODEL_BYTES = 1_876_395_376
INTERNVL_MODEL_SHA256 = "9420916a7fab7d2009f7907cdffa341c9cb6be7c5e0cf4ee193de16fde647dea"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the official SimLingo inference config and checkpoint."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/simlingo"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Hash every file. The two large weight files are always hashed and verified.",
    )
    parser.add_argument("--strict-size", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def planned_download(args: argparse.Namespace) -> Dict[str, object]:
    output = args.output_dir.expanduser().resolve()
    return {
        "repo_id": REPO_ID,
        "revision": SIMLINGO_REVISION,
        "output_dir": str(output),
        "files": list(DEFAULT_FILES),
        "internvl_repo_id": INTERNVL_REPO_ID,
        "internvl_revision": INTERNVL_REVISION,
        "internvl_output_dir": str(output / "pretrained" / "InternVL2-1B"),
        "internvl_files": list(INTERNVL_FILES),
        "excluded": "optimizer shards and DeepSpeed training state",
    }


def main() -> int:
    args = parse_args()
    plan = planned_download(args)
    if args.dry_run:
        print(json.dumps({"mode": "dry-run", **plan}, indent=2, ensure_ascii=False))
        return 0

    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before downloading: %s" % exc)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_revision: Optional[str] = None
    if not args.local_files_only:
        info = HfApi().model_info(REPO_ID, revision=SIMLINGO_REVISION)
        resolved_revision = info.sha

    records: List[Dict[str, object]] = []
    for filename in DEFAULT_FILES:
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            revision=SIMLINGO_REVISION,
            local_dir=str(output_dir),
            force_download=args.force,
            local_files_only=args.local_files_only,
        )
        path = Path(downloaded).resolve()
        record: Dict[str, object] = {
            "filename": filename,
            "path": str(path),
            "bytes": path.stat().st_size,
        }
        if args.sha256 or filename.endswith("pytorch_model.pt"):
            record["sha256"] = sha256_file(path)
        records.append(record)

    checkpoint_record = next(
        item for item in records if item["filename"].endswith("pytorch_model.pt")
    )
    size_matches = checkpoint_record["bytes"] == PUBLISHED_CHECKPOINT_BYTES
    hash_matches = checkpoint_record["sha256"] == PUBLISHED_CHECKPOINT_SHA256
    if not hash_matches:
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch: got %s, expected %s"
            % (checkpoint_record["sha256"], PUBLISHED_CHECKPOINT_SHA256)
        )
    if args.strict_size and not size_matches:
        raise RuntimeError(
            "Checkpoint size mismatch: got %s bytes, expected %s bytes. "
            "The upstream file may have changed."
            % (checkpoint_record["bytes"], PUBLISHED_CHECKPOINT_BYTES)
        )

    internvl_dir = output_dir / "pretrained" / "InternVL2-1B"
    snapshot_download(
        repo_id=INTERNVL_REPO_ID,
        revision=INTERNVL_REVISION,
        local_dir=str(internvl_dir),
        allow_patterns=list(INTERNVL_FILES),
        force_download=args.force,
        local_files_only=args.local_files_only,
    )
    internvl_records: List[Dict[str, object]] = []
    for filename in INTERNVL_FILES:
        path = (internvl_dir / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError("InternVL allowlisted file was not downloaded: %s" % path)
        record = {"filename": filename, "path": str(path), "bytes": path.stat().st_size}
        if args.sha256 or filename == "model.safetensors":
            record["sha256"] = sha256_file(path)
        internvl_records.append(record)
    internvl_weights = next(item for item in internvl_records if item["filename"] == "model.safetensors")
    if internvl_weights["bytes"] != INTERNVL_MODEL_BYTES:
        raise RuntimeError(
            "InternVL model size mismatch: got %s, expected %s"
            % (internvl_weights["bytes"], INTERNVL_MODEL_BYTES)
        )
    if internvl_weights["sha256"] != INTERNVL_MODEL_SHA256:
        raise RuntimeError(
            "InternVL model SHA-256 mismatch: got %s, expected %s"
            % (internvl_weights["sha256"], INTERNVL_MODEL_SHA256)
        )

    manifest = {
        "schema_version": "1.0",
        "downloaded_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "repo_id": REPO_ID,
        "requested_revision": SIMLINGO_REVISION,
        "resolved_revision": resolved_revision,
        "checkpoint_size_matches_published_file": size_matches,
        "checkpoint_sha256_matches": hash_matches,
        "files": records,
        "internvl": {
            "repo_id": INTERNVL_REPO_ID,
            "requested_revision": INTERNVL_REVISION,
            "files": internvl_records,
        },
    }
    manifest_path = output_dir / "download-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("Manifest: %s" % manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
