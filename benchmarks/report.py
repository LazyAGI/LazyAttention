# SPDX-License-Identifier: Apache-2.0
"""Generate Markdown and machine-readable reports from paired benchmark JSON."""

import argparse
import hashlib
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from benchmarks.analyze import estimate


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        key = json.dumps(
            [
                r["fixture_sha256"],
                r.get("backends", ["index", "paired"]),
                r["phase"],
                r.get("protocol", "shared-workspace-v1"),
                r["source_sha256"],
                r.get("versions"),
                r.get("environment"),
                r.get("magi"),
                r["telemetry"][0]["gpu"].split(",")[1].strip(),
            ],
            sort_keys=True,
        )
        groups[key].append(r)
    result = []
    for group in groups.values():
        first = group[0]
        if len({r["repeat"] for r in group}) != len(group):
            raise ValueError("duplicate repeat IDs for one comparison")
        if any(not r.get("correctness") for r in group):
            raise ValueError("missing correctness checks")
        logs = [statistics.mean(b["dlog"] for b in r["blocks"]) for r in group]
        if len(group) >= 3:
            from scipy.stats import t

            mean = statistics.mean(logs)
            half = (
                float(t.ppf(0.975, len(logs) - 1))
                * statistics.stdev(logs)
                / math.sqrt(len(logs))
            )
            stats = dict(
                time_change_pct=100 * math.expm1(mean),
                ci95_pct=[100 * math.expm1(mean - half), 100 * math.expm1(mean + half)],
            )
            unit = "independent process Student-t"
        else:
            singles = [estimate([b["dlog"] for b in r["blocks"]]) for r in group]
            stats = dict(
                time_change_pct=100 * math.expm1(statistics.mean(logs)),
                ci95_pct=[
                    min(s["ci95_pct"][0] for s in singles),
                    max(s["ci95_pct"][1] for s in singles),
                ],
            )
            unit = "within-process circular block bootstrap envelope (fewer than 3 processes)"
        medians = [
            1000
            * statistics.median(
                x["ms"]
                for r in group
                for b in r["blocks"]
                for x in b["values"]
                if x["label"] == role
            )
            for role in (0, 1)
        ]
        errors = [e for r in group for check in r["correctness"] for e in check]
        if any(
            not math.isfinite(e["max_abs"])
            or not math.isfinite(e["rel_l2"])
            or e["max_abs"] > 0.0625
            or e["rel_l2"] > 0.005
            for e in errors
        ):
            raise ValueError("failed or nonfinite correctness metrics")
        result.append(
            dict(
                backends=first.get("backends", ["index", "paired"]),
                phase=first["phase"],
                fixture_sha256=first["fixture_sha256"],
                source_sha256=first["source_sha256"],
                shape=first["shape"],
                mapping=(
                    "external"
                    if first["config"].get("input")
                    else first["config"]["mapping"]
                ),
                mask=(
                    "external"
                    if first["config"].get("input")
                    else first["config"]["mask"]
                ),
                repeats=len(group),
                blocks=sum(len(r["blocks"]) for r in group),
                median_us=medians,
                **stats,
                interval_method=unit,
                protocol=first.get("protocol", "shared-workspace-v1"),
                max_abs=max(e["max_abs"] for e in errors),
                max_rel_l2=max(e["rel_l2"] for e in errors),
                magi=first.get("magi"),
                versions=first.get("versions"),
                environment=first.get("environment"),
                gpu=first["telemetry"][0]["gpu"],
                scope=first["scope"],
                excludes=first["excludes"],
            )
        )
    return result


def markdown(rows):
    lines = [
        "# Attention backward benchmark",
        "",
        "B/A − 1 is the paired GPU time change; negative means B is faster. "
        "Median times are descriptive; confidence intervals use paired log-ratios.",
        "",
        "| A → B | Shape [B,S,Hq,D] | Mapping / mask | Processes / blocks | A µs | B µs | B/A − 1 | 95% CI |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lo, hi = r["ci95_pct"]
        lines.append(
            f"| {' → '.join(r['backends'])} | {r['shape']} | {r['mapping']} / {r['mask']} | "
            f"{r['repeats']} / {r['blocks']} | {r['median_us'][0]:.3f} | {r['median_us'][1]:.3f} | "
            f"{r['time_change_pct']:+.3f}% | [{lo:+.3f}%, {hi:+.3f}%] |"
        )
    lines += ["", "## Validation and measurement details", ""]
    for i, r in enumerate(rows, 1):
        lines += [
            f"### Comparison {i}",
            "",
            f"- Phase: `{r['phase']}`; protocol: `{r['protocol']}`.",
            f"- Interval: {r['interval_method']}.",
            f"- Gradient max absolute error: {r['max_abs']:.6g}; max relative L2: {r['max_rel_l2']:.6g}.",
            f"- Fixture SHA256: `{r['fixture_sha256']}`.",
            f"- Scope: {r['scope']}.",
            f"- Excluded: {', '.join(r['excludes'])}.",
            f"- GPU snapshot: `{r['gpu']}`.",
            f"- Software: `{json.dumps(r['versions'],sort_keys=True)}`.",
        ]
        manifest = hashlib.sha256(
            json.dumps(r["source_sha256"], sort_keys=True).encode()
        ).hexdigest()
        lines.append(
            f"- Benchmark source manifest SHA256: `{manifest}` (full mapping in JSON/raw files)."
        )
        if r["magi"]:
            m = r["magi"]
            lines += [
                f"- MAGI ranges: {m['ranges']} ({m['range_source']}); auto range merge: {m['auto_range_merge']}.",
                "- Output addresses cross between roles. Private scratch is owned by each library; whole-workspace equality is not claimed.",
                "- MAGI FP32 clears, atomic accumulation and BF16 conversion are included. Forward and range preparation are excluded.",
                "- Generic range encoding is exact, but may be fragmented. Use verified external compact ranges when evaluating range-planning quality.",
            ]
        lines.append("")
    lines += [
        "No cross-case aggregate is inferred. Synthetic fixtures do not establish training throughput or a universal speedup.",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    json_path = a.output.with_suffix(".json")
    if a.output.suffix.lower() != ".md" or a.output.exists() or json_path.exists():
        p.error("choose a new .md output path (and unused matching .json path)")
    rows = summarize([json.loads(path.read_text()) for path in a.files])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(markdown(rows))
    json_path.write_text(json.dumps(rows, indent=2))
    print(a.output)


if __name__ == "__main__":
    main()
