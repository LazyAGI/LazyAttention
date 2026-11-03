# SPDX-License-Identifier: Apache-2.0
"""Run independent MAGI A/B processes and generate a Markdown/JSON report."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog="Additional arguments are forwarded to benchmarks.backward, e.g. --seqlen 4096 --candidate paired.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    args, extra = p.parse_known_args()
    reserved = {"--output", "--repeat", "--baseline", "--phase"}
    if any(x.split("=")[0] in reserved for x in extra):
        p.error("output/repeat/baseline/phase are controlled by this runner")
    if args.repeats < 1:
        p.error("repeats must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("choose a new or empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for repeat in range(args.repeats):
        output = args.output_dir / f"repeat-{repeat}.json"
        files.append(str(output))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.backward",
                "--baseline",
                "magi",
                *extra,
                "--repeat",
                str(repeat),
                "--output",
                str(output),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.report",
            *files,
            "--output",
            str(args.output_dir / "report.md"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
