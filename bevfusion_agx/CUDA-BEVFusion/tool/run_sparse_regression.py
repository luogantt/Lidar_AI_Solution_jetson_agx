#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run all synthetic sparse fixtures, including numerical and repeatability checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((args.fixtures / "manifest.json").read_text())["cases"]
    compare = Path(__file__).with_name("compare_sparse_output.py")
    failed = []
    for case in cases:
        name = case["name"]
        output = args.output / f"{name}.dense"
        command = [str(args.checker.resolve()), "--model", str(args.fixtures / case["model"]),
                   "--voxels", str(args.fixtures / case["voxels"]),
                   "--coors", str(args.fixtures / case["coors"]),
                   "--grid", ",".join(map(str, case["grid"])),
                   "--output", str(output), "--warmup", "1", "--runs", "3"]
        with (args.output / f"{name}.log").open("w") as log:
            inference = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if inference.returncode:
            print(f"FAIL {name}: inference exited {inference.returncode}; see {name}.log", flush=True)
            failed.append(name)
            continue
        comparison = subprocess.run([
            sys.executable, str(compare), str(output), str(args.fixtures / case["reference"]),
            "--repeat", str(output) + ".first", "--max-abs", "0.002", "--max-mae", "0.0002",
            "--max-rmse", "0.0005", "--max-relative-l2", "0.003", "--min-cosine", "0.99999",
            "--json-report", str(args.output / f"{name}.metrics.json")])
        if comparison.returncode:
            failed.append(name)
        print(f"{'FAIL' if comparison.returncode else 'PASS'} {name}", flush=True)
    summary = {"total": len(cases), "passed": len(cases) - len(failed), "failed": failed}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
