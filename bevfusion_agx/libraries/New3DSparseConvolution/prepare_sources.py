#!/usr/bin/env python3
"""Copy the reviewed, vendored spconv/cumm trees into an isolated build root.

Uses only the Python standard library and never accesses the network. Source
identities include every regular file's relative path and SHA256. Python bytecode
caches are ignored; Git metadata and compiled/build artifacts are rejected.
Changed vendored files, manifest changes, modified copied inputs, and legacy
unmanaged build roots require a new --deps-root. Nothing is deleted or replaced.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys


EXPECTED = {
    "spconv": ("2.3.8", "263d6b47425ef843c82f997b12d8b714013d216c"),
    "cumm": ("0.7.11", "b8d83b6511b70746d04485d0d2ec00896b735430"),
}
STATE_FILE = "source-state.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ignored(path):
    return "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo")


def source_inventory(root):
    files = {}
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ignored(relative):
            continue
        if ".git" in relative.parts:
            raise ValueError("Git metadata is not vendored source: " + str(path))
        if path.is_symlink():
            raise ValueError("Vendored source must contain regular files, not symlinks: " + str(path))
        if path.is_dir():
            if relative.parts[0] in ("build", "dist", "venv", ".venv", "_deps") or path.name.endswith(".egg-info"):
                raise ValueError("Build output found inside vendored source: " + str(path))
            continue
        if not path.is_file():
            raise ValueError("Unsupported source entry: " + str(path))
        if path.suffix.lower() in (".so", ".dll", ".pyd", ".o", ".obj", ".a", ".lib") or ".so." in path.name:
            raise ValueError("Compiled artifact found inside vendored source: " + str(path))
        files[relative.as_posix()] = sha256_file(path)
        total_bytes += path.stat().st_size
    if not files:
        raise ValueError("Vendored source tree is empty: " + str(root))
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return files, hashlib.sha256(canonical).hexdigest(), total_bytes


def inspect_sources(source_root):
    manifest_path = source_root / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported third_party/source-manifest.json schema")
    entries = manifest.get("sources", [])
    indexed = {entry["name"]: entry for entry in entries}
    if len(entries) != len(EXPECTED) or set(indexed) != set(EXPECTED):
        raise ValueError("Source manifest must describe exactly spconv and cumm")
    signature = {"manifest_sha256": sha256_file(manifest_path), "sources": []}
    inventories = {}
    for name, (version, commit) in EXPECTED.items():
        entry = indexed[name]
        if entry.get("version") != version or entry.get("commit") != commit or entry.get("path") != name:
            raise ValueError("Unexpected pinned version, commit or path for " + name)
        root = source_root / name
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Missing regular vendored source directory: " + str(root))
        files, tree_hash, total_bytes = source_inventory(root)
        license_file = entry.get("license_file")
        if license_file not in files:
            raise ValueError("Missing declared upstream license for " + name)
        signature["sources"].append({"name": name, "version": version, "commit": commit,
                                     "tree_sha256": tree_hash, "files": len(files), "bytes": total_bytes,
                                     "license_file": license_file, "license_sha256": files[license_file]})
        inventories[name] = files
    return signature, inventories


def require_fresh_root(reason, deps_root):
    raise ValueError("{}\nUse a new NEWSCN_DEPS_ROOT (or --deps-root); existing files were not replaced: {}".format(reason, deps_root))


def prepare_sources(source_root, deps_root, check_only=False):
    source_root = source_root.resolve(strict=True)
    deps_root = deps_root.resolve()
    # Do not allow the writable worktree to be placed inside the reviewed source.
    if deps_root == source_root or source_root in deps_root.parents or deps_root in source_root.parents:
        raise ValueError("Source and dependency roots must be separate, non-nested directories")
    signature, inventories = inspect_sources(source_root)
    state_path = deps_root / STATE_FILE
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1 or state.get("signature") != signature:
            require_fresh_root("Vendored source / provenance changed, or source-state schema differs.", deps_root)
        # Generated additions are allowed in _deps; originally copied files must
        # remain identical. This catches edits to the wrong, unreviewed copy.
        for name, files in inventories.items():
            for relative, expected_hash in files.items():
                copied = deps_root / name / relative
                if copied.is_symlink() or not copied.is_file() or sha256_file(copied) != expected_hash:
                    require_fresh_root("A copied source file was changed or removed: " + str(copied), deps_root)
        print("Verified existing source copies: " + str(deps_root))
        return signature
    if check_only:
        raise ValueError("No validated source-state.json in {}. Run build_deps.sh with a fresh NEWSCN_DEPS_ROOT first.".format(deps_root))
    if deps_root.exists() and (not deps_root.is_dir() or any(deps_root.iterdir())):
        require_fresh_root("Unmanaged/legacy dependency directory has no validated source-state.json.", deps_root)
    deps_root.mkdir(parents=True, exist_ok=True)
    for name, files in inventories.items():
        destination = deps_root / name
        destination.mkdir()
        for relative in files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / name / relative, target)
            if sha256_file(target) != files[relative]:
                raise ValueError("Source changed while being copied: " + relative)
    state = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
             "source_root": str(source_root), "signature": signature,
             "ignored_entries": ["__pycache__", "*.pyc", "*.pyo"]}
    # Publish only after both copies have passed their content checks.
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Copied vendored sources into: " + str(deps_root))
    for entry in signature["sources"]:
        print("  {} {} commit={} tree_sha256={}".format(entry["name"], entry["version"], entry["commit"], entry["tree_sha256"]))
    return signature


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent / "third_party")
    parser.add_argument("--deps-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="verify an existing managed copy without creating files")
    args = parser.parse_args(argv)
    prepare_sources(args.source_root, args.deps_root, check_only=args.check_only)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
