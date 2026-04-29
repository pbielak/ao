#!/usr/bin/env python3
"""Extract per-module, per-phase memory allocations from memory snapshots.

Reads .pkl snapshots produced by torch.{cuda,xpu}.memory._dump_snapshot() and
produces allocation breakdowns by model module and training phase.

Usage:
    # Single snapshot breakdown:
    python extract_allocations.py snapshot.pkl

    # Compare two snapshots (e.g., XPU vs CUDA):
    python extract_allocations.py --compare xpu_mem.pkl cuda_mem.pkl

    # Write output to a file:
    python extract_allocations.py -o breakdown.md snapshot.pkl

    # Filter to only forward-pass allocations:
    python extract_allocations.py --phase forward snapshot.pkl
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Module classification: map stack frames to ViT sub-modules
# ---------------------------------------------------------------------------

# Ordered list of (pattern, label) – first match wins (innermost frame first).
_MODULE_PATTERNS = [
    ("patch_embed", "PatchEmbed"),
    ("attention_backward", "Attention"),
    ("attention.cpp", "Attention"),
    ("attention.py", "Attention"),
    ("mlp.py", "Mlp"),
    ("vision_transformer.py:_pos_embed", "PosEmbed"),
    ("vision_transformer.py:forward_head", "Head"),
    ("vision_transformer.py:forward_features", "TransformerBlock"),
    ("vision_transformer.py:forward", "VisionTransformer"),
    ("adam", "Optimizer(Adam)"),
]


def _classify_module(frames):
    """Return the innermost recognized module name from a list of frames."""
    for fr in frames:
        fn = fr.get("filename", "")
        name = fr.get("name", "")
        key = f"{fn}:{name}"
        for pattern, label in _MODULE_PATTERNS:
            if ":" in pattern:
                file_part, name_part = pattern.split(":", 1)
                if file_part in fn and name_part in name:
                    return label
            else:
                if pattern in fn.lower() or pattern in name.lower():
                    return label
    return "other"


# ---------------------------------------------------------------------------
# Phase classification: map stack frames to training phase
# ---------------------------------------------------------------------------

def _classify_phase(frames):
    """Classify an allocation into a training phase.

    Strategy — multi-signal approach:
    1. Look for the benchmark/training script frame to get coarse context
       (model creation, forward, backward, optimizer step).
    2. Use C++ frame hints (autograd engine, backward kernels) for allocations
       that lack a Python benchmark frame (common during backward in compiled models).
    3. Use Python frame hints as final fallback.
    """
    # --- Signal collection ---
    has_backward_py = False
    has_autograd_cpp = False
    has_backward_cpp = False
    has_benchmarking = False
    has_model_to = False
    has_cross_entropy_or_model_fwd = False
    has_module_forward = False
    has_loss_backward = False
    has_grad_scaler_step = False

    for fr in frames:
        fn = fr.get("filename", "")
        name = fr.get("name", "")
        fn_lower = fn.lower()
        name_lower = name.lower()

        if fn.endswith(".py"):
            # Benchmark script line-level classification
            if "benchmark_low_bit_adam" in fn_lower or "training" in fn_lower:
                # Heuristic: detect well-known patterns from the benchmark script
                if "to" == name_lower or "_apply" in name_lower:
                    has_model_to = True
                if "cross_entropy" in name_lower or ("model" in name_lower and "forward" in name_lower):
                    has_cross_entropy_or_model_fwd = True

            if "backward" in name_lower:
                has_backward_py = True
            if "benchmarking" in fn_lower or "do_bench" in name_lower:
                has_benchmarking = True

            # nn.Module.forward dispatched via _call_impl/_wrapped_call_impl
            if name_lower == "forward" and "module.py" not in fn_lower:
                has_module_forward = True

            # Detect optimizer step patterns
            basename = fn.rsplit("/", 1)[-1]
            if basename in ("adam.py", "adamw.py", "sgd.py"):
                has_grad_scaler_step = True
            if "grad_scaler" in name_lower and "step" in name_lower:
                has_grad_scaler_step = True
        else:
            # C++ hints — distinguish real autograd engine from dispatch keys
            if "autograd" in fn_lower and "Register" not in fn:
                has_autograd_cpp = True
            if "backward" in fn_lower or "backward" in name_lower:
                has_backward_cpp = True

    # Also check for the benchmark script frame: use it to locate the training
    # loop call (forward, backward, optim step) based on neighboring code.
    # We look for frames from nn/modules/module.py (model.to), functional.py
    # (F.cross_entropy → forward), _tensor.py (backward), grad_scaler (step).
    for fr in frames:
        fn = fr.get("filename", "")
        name = fr.get("name", "")
        if "functional.py" in fn and "cross_entropy" in name:
            has_cross_entropy_or_model_fwd = True
        if "_tensor.py" in fn and "backward" in name:
            has_loss_backward = True
        if "grad_scaler" in fn.lower() and "step" in name.lower():
            has_grad_scaler_step = True
        if fn.endswith("module.py") and name in ("to", "_apply", "convert"):
            has_model_to = True

    # --- Decision ---
    if has_model_to and not has_cross_entropy_or_model_fwd and not has_loss_backward and not has_module_forward:
        return "creation"
    if has_benchmarking:
        return "compilation"
    if has_grad_scaler_step and not has_loss_backward and not has_backward_cpp:
        return "optimizer"
    if has_loss_backward or has_backward_py or has_backward_cpp or has_autograd_cpp:
        return "backward"
    if has_cross_entropy_or_model_fwd or has_module_forward:
        return "forward"
    return "other"


def _classify_op(frames):
    """Return a human-readable op name from the innermost meaningful frame."""
    skip_basenames = {"module.py", "_ops.py", "__init__.py", "_compile.py"}
    py_frames = [f for f in frames if f.get("filename", "").endswith(".py")]

    # Find innermost Python frame that isn't dispatch boilerplate
    innermost = None
    for fr in py_frames:
        basename = fr["filename"].rsplit("/", 1)[-1]
        if basename not in skip_basenames:
            innermost = fr
            break

    if innermost is None:
        # Fall back to C++ frames for ops like SDPA
        for fr in frames:
            name = fr.get("name", "")
            if "scaled_dot_product" in name:
                return "scaled_dot_product_attention"
            if "flash_attention" in name:
                return "flash_attention"
            if "attention_backward" in name.lower():
                return "attention_backward"
        return "unknown"

    basename = innermost["filename"].rsplit("/", 1)[-1]
    fn_name = innermost.get("name", "")

    if basename == "linear.py":
        return "linear"
    if basename == "conv.py":
        return "conv2d"
    if basename == "activation.py":
        return fn_name if fn_name != "forward" else "activation"
    if basename == "normalization.py" or basename == "norm.py":
        return "layer_norm"
    if basename == "dropout.py":
        return "dropout"
    if "attention" in basename:
        # Could be SDPA or the attention module itself; check C++ for SDPA
        for fr in frames:
            if "scaled_dot_product" in fr.get("name", ""):
                return "scaled_dot_product_attention"
        return "attention_other"
    if basename == "functional.py":
        return fn_name  # e.g. cross_entropy, gelu, etc.

    return f"{basename.replace('.py', '')}:{fn_name}"


# ---------------------------------------------------------------------------
# Snapshot loading & processing
# ---------------------------------------------------------------------------

def load_snapshot(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_allocations(snap):
    """Replay alloc/free events and compute per-(module, phase) peak active memory.

    Returns a dict of (module, phase) -> dict with:
      - peak_bytes: high-water mark of active memory attributed to (module, phase)
      - alloc_count: number of alloc events
    Also returns global_peak_bytes: the peak of total active memory across all
    modules/phases combined.
    """
    device_traces = snap.get("device_traces", [])
    if not device_traces:
        return {}, 0

    dt = device_traces[0]

    # First pass: classify each alloc and record addr -> (module, phase, size)
    live = {}            # addr -> (module, phase, size)
    bucket_active = defaultdict(int)   # (module, phase) -> current active bytes
    bucket_peak = defaultdict(int)     # (module, phase) -> peak active bytes
    bucket_count = defaultdict(int)    # (module, phase) -> alloc count
    total_active = 0
    global_peak = 0

    for entry in dt:
        action = entry.get("action")
        addr = entry.get("addr")
        size = entry.get("size", 0)

        if action == "alloc":
            frames = entry.get("frames", [])
            module = _classify_module(frames)
            phase = _classify_phase(frames)
            key = (module, phase)

            live[addr] = (module, phase, size)
            bucket_active[key] += size
            bucket_count[key] += 1
            total_active += size

            if bucket_active[key] > bucket_peak[key]:
                bucket_peak[key] = bucket_active[key]
            if total_active > global_peak:
                global_peak = total_active

        elif action in ("free_completed", "free_requested"):
            info = live.pop(addr, None)
            if info is not None:
                mod, ph, sz = info
                bucket_active[(mod, ph)] -= sz
                total_active -= sz

    # Build result table: (module, phase) -> [peak_bytes, alloc_count]
    table = {}
    for key in set(bucket_peak.keys()) | set(bucket_count.keys()):
        table[key] = [bucket_peak.get(key, 0), bucket_count.get(key, 0)]

    return table, global_peak


def extract_module_ops(snap, module_filter):
    """Drill down into a single module: break it down by op and phase.

    Returns a table of (op, phase) -> [peak_bytes, alloc_count].
    """
    device_traces = snap.get("device_traces", [])
    if not device_traces:
        return {}

    dt = device_traces[0]

    live = {}
    bucket_active = defaultdict(int)
    bucket_peak = defaultdict(int)
    bucket_count = defaultdict(int)

    for entry in dt:
        action = entry.get("action")
        addr = entry.get("addr")
        size = entry.get("size", 0)

        if action == "alloc":
            frames = entry.get("frames", [])
            module = _classify_module(frames)
            if module != module_filter:
                live[addr] = None  # track addr but skip accounting
                continue

            phase = _classify_phase(frames)
            op = _classify_op(frames)
            key = (op, phase)

            live[addr] = key
            bucket_active[key] += size
            bucket_count[key] += 1

            if bucket_active[key] > bucket_peak[key]:
                bucket_peak[key] = bucket_active[key]

        elif action in ("free_completed", "free_requested"):
            info = live.pop(addr, None)
            if info is not None and info is not None:
                bucket_active[info] -= size

    table = {}
    for key in set(bucket_peak.keys()) | set(bucket_count.keys()):
        table[key] = [bucket_peak.get(key, 0), bucket_count.get(key, 0)]
    return table


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_PHASE_ORDER = ["creation", "compilation", "forward", "backward", "optimizer", "other"]
_MODULE_ORDER = [
    "PatchEmbed", "PosEmbed", "Attention", "Mlp",
    "TransformerBlock", "Head", "VisionTransformer",
    "Optimizer(Adam)", "other",
]


def _sort_key(module_phase):
    module, phase = module_phase
    pi = _PHASE_ORDER.index(phase) if phase in _PHASE_ORDER else 99
    mi = _MODULE_ORDER.index(module) if module in _MODULE_ORDER else 98
    return (pi, mi)


def format_single(table, label="", markdown=False, global_peak=0, col0_header="Module"):
    """Format a single snapshot breakdown."""
    rows = []
    for key in sorted(table.keys(), key=_sort_key):
        peak_bytes, count = table[key]
        module, phase = key
        rows.append((module, phase, f"{peak_bytes / 1e6:.1f}", str(count)))

    if markdown:
        lines = []
        if label:
            lines.append(f"### {label}\n")
        lines.append(f"| {col0_header} | Phase | Peak Active (MB) | Count |")
        lines.append("|---|---|---|---|")
        for module, phase, peak, count in rows:
            lines.append(f"| {module} | {phase} | {peak} | {count} |")
        if global_peak:
            lines.append(f"| **Global peak** | | **{global_peak / 1e6:.1f}** | |")
        return "\n".join(lines) + "\n"

    # Aligned plain-text
    headers = (col0_header, "Phase", "Peak Active (MB)", "Count")
    all_rows = [headers] + rows
    if global_peak:
        summary = ("Global peak", "", f"{global_peak / 1e6:.1f}", "")
        all_rows.append(summary)
    widths = [max(len(r[i]) for r in all_rows) for i in range(4)]

    def fmt_row(r):
        return "  ".join(
            r[i].rjust(widths[i]) if i >= 2 else r[i].ljust(widths[i])
            for i in range(4)
        )

    lines = []
    if label:
        lines.append(label)
        lines.append("")
    lines.append(fmt_row(headers))
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append(fmt_row(r))
    if global_peak:
        lines.append("  ".join("-" * w for w in widths))
        lines.append(fmt_row(summary))
    return "\n".join(lines) + "\n"


def format_comparison(table_a, table_b, label_a="A", label_b="B", markdown=False, col0_header="Module"):
    """Format a side-by-side comparison of two snapshots."""
    all_keys = sorted(set(table_a.keys()) | set(table_b.keys()), key=_sort_key)
    rows = []
    for key in all_keys:
        module, phase = key
        peak_a = table_a.get(key, [0, 0])[0]
        peak_b = table_b.get(key, [0, 0])[0]
        delta = peak_b - peak_a
        rows.append((
            module, phase,
            f"{peak_a / 1e6:.1f}", f"{peak_b / 1e6:.1f}", f"{delta / 1e6:+.1f}",
        ))

    if markdown:
        lines = [
            f"| {col0_header} | Phase | {label_a} (MB) | {label_b} (MB) | Delta (MB) |",
            "|---|---|---|---|---|",
        ]
        for module, phase, va, vb, d in rows:
            lines.append(f"| {module} | {phase} | {va} | {vb} | {d} |")
        return "\n".join(lines) + "\n"

    # Aligned plain-text
    headers = (col0_header, "Phase", f"{label_a} (MB)", f"{label_b} (MB)", "Delta (MB)")
    all_rows = [headers] + rows
    widths = [max(len(r[i]) for r in all_rows) for i in range(5)]

    def fmt_row(r):
        return "  ".join(
            r[i].rjust(widths[i]) if i >= 2 else r[i].ljust(widths[i])
            for i in range(5)
        )

    lines = [fmt_row(headers)]
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines) + "\n"


def _label_from_path(path):
    """Derive a short label from a snapshot filename.

    E.g. 'AdamW8bitAo_xpu_amp-bf16_compile_bs8_8.87GB_mem.pkl' -> 'AdamW8bitAo_xpu'
    """
    stem = Path(path).stem.replace("_mem", "")
    parts = stem.split("_")
    # Return optim + device (first two underscore-separated tokens)
    if len(parts) >= 2 and parts[1] in ("cuda", "xpu"):
        return f"{parts[0]}_{parts[1]}"
    return stem


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-module memory allocation breakdown from snapshots"
    )
    parser.add_argument("snapshots", nargs="+", help=".pkl snapshot file(s)")
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare two snapshots side-by-side (requires exactly 2 files)",
    )
    parser.add_argument(
        "--module", default=None,
        help="Drill down into a specific module (e.g. Attention, Mlp, PatchEmbed)",
    )
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown table")
    parser.add_argument("-o", "--output", default=None, help="Write to file")
    args = parser.parse_args()

    if args.module:
        # Drilldown mode: show op-level breakdown for a single module
        if args.compare:
            if len(args.snapshots) != 2:
                print("--compare requires exactly 2 snapshot files.", file=sys.stderr)
                sys.exit(1)
            snap_a = load_snapshot(args.snapshots[0])
            snap_b = load_snapshot(args.snapshots[1])
            tab_a = extract_module_ops(snap_a, args.module)
            tab_b = extract_module_ops(snap_b, args.module)
            label_a = _label_from_path(args.snapshots[0])
            label_b = _label_from_path(args.snapshots[1])
            output = format_comparison(
                tab_a, tab_b, label_a, label_b, markdown=args.markdown,
                col0_header="Op",
            )
        else:
            parts = []
            for path in args.snapshots:
                snap = load_snapshot(path)
                tab = extract_module_ops(snap, args.module)
                label = f"{_label_from_path(path)} — {args.module}"
                parts.append(format_single(
                    tab, label, markdown=args.markdown, col0_header="Op",
                ))
            output = "\n".join(parts)

    elif args.compare:
        if len(args.snapshots) != 2:
            print("--compare requires exactly 2 snapshot files.", file=sys.stderr)
            sys.exit(1)
        snap_a = load_snapshot(args.snapshots[0])
        snap_b = load_snapshot(args.snapshots[1])
        tab_a, _ = extract_allocations(snap_a)
        tab_b, _ = extract_allocations(snap_b)
        label_a = _label_from_path(args.snapshots[0])
        label_b = _label_from_path(args.snapshots[1])
        output = format_comparison(tab_a, tab_b, label_a, label_b, markdown=args.markdown)
    else:
        parts = []
        for path in args.snapshots:
            snap = load_snapshot(path)
            tab, global_peak = extract_allocations(snap)
            label = _label_from_path(path)
            parts.append(format_single(tab, label, markdown=args.markdown, global_peak=global_peak))
        output = "\n".join(parts)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
