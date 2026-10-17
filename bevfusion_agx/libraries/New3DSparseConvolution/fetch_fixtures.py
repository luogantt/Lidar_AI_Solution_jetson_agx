#!/usr/bin/env python3
"""Fetch pinned upstream BEVFusion sparse-backbone fixtures, with SHA-256 checks."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

REVISION = "3f823115cb16d9ddca9561394b9fa49242ab8315"
BASE = (f"https://raw.githubusercontent.com/riteshkhrn/Lidar_AI_Solution/{REVISION}/"
        "libraries/New3DSparseConvolution/workspace/bevfusion/")
FILES = {
    "bevfusion.scn.xyz.onnx": "bf18bffa3f692ae7393a92c572851206257184805e8cd266978faf708f12f808",
    "bevfusion.scn.zyx.onnx": "a005753127532a5e1547de21404b0789715840182917e88f5076c1f4e0ce8e87",
    "infer.xyz.coors": "0ceedeacaed925c450b2cd6c598333c23fb44379b64c452f5ae368852cb1476c",
    "infer.xyz.dense": "735465737038a49f82309681af0b92898a0941a4b163d59afb3786447914067f",
    "infer.xyz.voxels": "dba40cfd0894e0bf9dd9b1d7a02d0bd952f83864d5655eaeff948f75e611b28e",
    "infer.zyx.coors": "2163f6bc715a337276b28ab808e02869e1b0a0ed6235253c8f265da2ee28d732",
    "infer.zyx.dense": "6f3c38780acbfac1a3b691f4872710dbbf231abacf4aee2ea8a1adaeee50e213",
    "infer.zyx.voxels": "dba40cfd0894e0bf9dd9b1d7a02d0bd952f83864d5655eaeff948f75e611b28e",
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "fixtures/bevfusion")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        destination = args.output / name
        if destination.is_file() and digest(destination) == expected:
            print(f"verified {destination}")
            continue
        temporary = destination.with_name(name + ".part")
        with urllib.request.urlopen(BASE + name, timeout=120) as source, temporary.open("wb") as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                output.write(chunk)
        if digest(temporary) != expected:
            raise RuntimeError(f"SHA-256 mismatch: {temporary}; not using this download")
        temporary.replace(destination)
        print(f"downloaded and verified {destination}")


if __name__ == "__main__":
    main()
