#!/usr/bin/env python3
"""Aggregate peak-memory snapshots into a comparison table.

All metadata is parsed from the snapshot filename, which uses the format:
    {optim}_{device}_amp-{amp}_{compile|eager}_bs{batch_size}_{peak}GB_mem.pkl

Usage:
    # Summarize all snapshots in a directory:
    python summarize_memory.py mem_snapshots/

    # Summarize specific files:
    python summarize_memory.py results/*.pkl

    # Override the platform label (default: device from filename):
    python summarize_memory.py --platform "xpu (B50)" mem_snapshots/

    # Write to a markdown file instead of stdout:
    python summarize_memory.py -o summary.md mem_snapshots/

    # Set a custom baseline optimizer (default: AdamW):
    python summarize_memory.py --baseline AdamW mem_snapshots/
"""

import argparse
import re
import sys
from pathlib import Path

# Matches: {optim}_{device}_amp-{amp}_{compile|eager}_bs{batch_size}_{peak}GB_mem.pkl
_FILENAME_RE = re.compile(
    r"^(?P<optim>.+?)_(?P<device>cuda|xpu)_amp-(?P<amp>\w+)"
    r"_(?P<mode>compile|eager)_bs(?P<batch_size>\d+)"
    r"_(?P<peak>[0-9.]+)GB_mem\.pkl$"
)


def find_pkl_files(paths):
    """Resolve a list of paths (files or directories) to .pkl snapshot files."""
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.glob("*_mem.pkl")))
        elif p.is_file() and p.suffix == ".pkl":
            files.append(p)
    return files


def parse_filename(path):
    """Extract metadata from a snapshot filename.  Returns dict or None."""
    m = _FILENAME_RE.match(Path(path).name)
    if not m:
        return None
    return dict(
        optimizer=m.group("optim"),
        device=m.group("device"),
        amp=m.group("amp"),
        compile=m.group("mode") == "compile",
        batch_size=int(m.group("batch_size")),
        peak_memory_gb=float(m.group("peak")),
        source=str(path),
    )


def load_records(pkl_files, platform_override=None):
    records = []
    for f in pkl_files:
        meta = parse_filename(f)
        if meta is None:
            print(f"Warning: cannot parse filename, skipping: {f}", file=sys.stderr)
            continue
        meta["platform"] = platform_override or meta["device"]
        records.append(meta)
    return records


def build_table(records, baseline_optim="AdamW"):
    """Group by platform, compute ratios, return list of row dicts."""
    # group by platform
    platforms = {}
    for r in records:
        platforms.setdefault(r["platform"], []).append(r)

    rows = []
    for platform, recs in platforms.items():
        # find baseline
        baseline = None
        for r in recs:
            if r["optimizer"] == baseline_optim:
                baseline = r["peak_memory_gb"]
                break
        if baseline is None:
            # fallback: use the highest peak as baseline
            baseline = max(r["peak_memory_gb"] for r in recs)

        for r in sorted(recs, key=lambda x: x["peak_memory_gb"], reverse=True):
            ratio = r["peak_memory_gb"] / baseline if baseline > 0 else 0
            rows.append(dict(
                platform=r["platform"],
                optimizer=r["optimizer"],
                peak_memory_gb=r["peak_memory_gb"],
                ratio=ratio,
            ))
    return rows


def format_markdown(rows):
    lines = [
        "| Platform | Optimizer | Peak Memory (GB) | Ratio vs. baseline |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['platform']} | {r['optimizer']} | {r['peak_memory_gb']:.2f} | {r['ratio']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize memory benchmark results")
    parser.add_argument("paths", nargs="+", help=".pkl snapshot files or directories")
    parser.add_argument("--platform", default=None, help="Override platform label")
    parser.add_argument("--baseline", default="AdamW", help="Baseline optimizer name")
    parser.add_argument("-o", "--output", default=None, help="Write to file instead of stdout")
    args = parser.parse_args()

    pkl_files = find_pkl_files(args.paths)
    if not pkl_files:
        print("No snapshot .pkl files found.", file=sys.stderr)
        sys.exit(1)

    records = load_records(pkl_files, args.platform)
    rows = build_table(records, args.baseline)
    table = format_markdown(rows)

    if args.output:
        Path(args.output).write_text(table)
        print(f"Written to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
