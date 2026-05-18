"""Compute aggregate rubric statistics from qualitative evaluation outputs.

Analyzes rubric JSONL files (produced by qualitative_eval.py or
qualitative_analysis.py) and reports structural, weight-distribution,
and content metrics that characterize how rubrics evolve.

Metrics computed per rubric, then averaged across all prompts:

  Structural:  criteria count, average criterion text length
  Weights:     max weight, top-1 share (%), weight std dev
  Content (%): specific value embedding, negation, examples, label-only
  Type (%):    correctness, constraint, completeness

Usage examples::

    # Single file
    python scripts/eval/rubric_aggregate_stats.py \\
        /checkpoint/.../qualitative_eval/rubric/step_1000.jsonl

    # All steps in a rubric directory (prints trajectory table)
    python scripts/eval/rubric_aggregate_stats.py \\
        /checkpoint/.../qualitative_eval/rubric/

    # Compare multiple sources side-by-side (use LABEL:PATH syntax)
    python scripts/eval/rubric_aggregate_stats.py \\
        "Qwen3-8B:/checkpoint/.../qwen3_8b_base/rubrics.jsonl" \\
        "GPT-4.1:/checkpoint/.../gpt41/rubrics.jsonl" \\
        "V3 Main:/checkpoint/.../v3_main_margin_format/qualitative_eval/rubric/"

    # Output as CSV
    python scripts/eval/rubric_aggregate_stats.py \\
        /checkpoint/.../qualitative_eval/rubric/ --csv

    # Use the consolidated paper_qualitative_eval folder
    python scripts/eval/rubric_aggregate_stats.py \\
        "Qwen3-8B:/checkpoint/.../paper_qualitative_eval/qwen3_8b_prompted/rubrics.jsonl" \\
        "GPT-4.1:/checkpoint/.../paper_qualitative_eval/gpt41_prompted/rubrics.jsonl" \\
        "V3 Main:/checkpoint/.../paper_qualitative_eval/v3_main_margin_format/rubric/" \\
        "V2 Co-evo:/checkpoint/.../paper_qualitative_eval/v2_coevolve/rubric/" \\
        "Static:/checkpoint/.../paper_qualitative_eval/v3_frozen_prompted/rubric/" \\
        "RLCER:/checkpoint/.../paper_qualitative_eval/rlcer_evolving_old/rubric/"
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Criteria extraction (mirrors qualitative_analysis.py)
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_criteria(text: str) -> list[dict]:
    """Leniently extract rubric criteria from model output.

    Returns list of dicts with keys: criterion (str), weight (float).
    """
    cleaned = strip_thinking(text)

    # Strict JSON parse
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start : i + 1])
                    return parsed.get("criteria", [])
                except json.JSONDecodeError:
                    break

    # Regex fallback
    criteria: list[dict] = []
    pattern = re.compile(
        r'"criterion"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"weight"\s*:\s*([\d.]+)',
    )
    for m in pattern.finditer(cleaned):
        criterion_text = m.group(1).replace('\\"', '"').replace("\\n", " ")
        weight = float(m.group(2))
        criteria.append({"criterion": criterion_text, "weight": weight})

    return criteria


# ---------------------------------------------------------------------------
# Per-rubric statistics
# ---------------------------------------------------------------------------

_RE_SPECIFIC = re.compile(
    r"\d+\.?\d*|= |equals |formula|equation|code|pattern|exact|verbatim",
    re.IGNORECASE,
)
_RE_NEGATION = re.compile(
    r"\bavoids?\b|\bdoes not\b|\bwithout\b|\bmust not\b|\bshould not\b|\bdo not\b"
    r"|\bexclud\w*\b|\bno\s+\w+\b|\bnever\b|\brefrain\b|\bprohibit\b",
    re.IGNORECASE,
)
_RE_EXAMPLE = re.compile(
    r"\be\.g\.\b|\bsuch as\b|\bfor example\b|\bfor instance\b|\bincluding\b",
    re.IGNORECASE,
)
_RE_CORRECTNESS = re.compile(
    r"\bcorrect\w*\b|\baccurat\w*\b|\berror-free\b",
    re.IGNORECASE,
)
_RE_CONSTRAINT = re.compile(
    r"\bavoid\w*\b|\bexclud\w*\b|\bkeyword\b|\bmust not\b|\bshould not\b"
    r"|\bdo not\b|\bprohibit\b|\brestrict\b|\bforbid\w*\b",
    re.IGNORECASE,
)
_RE_COMPLETENESS = re.compile(
    r"\binclude\w*\b|\bcover\w*\b|\baddress\w*\b|\bcomprehensive\b|\bthorough\b",
    re.IGNORECASE,
)

LABEL_ONLY_MAX_LEN = 40


def compute_rubric_stats(criteria: list[dict]) -> dict[str, float]:
    """Compute statistics for a single rubric's criteria list."""
    n = len(criteria)
    if n == 0:
        return {
            "criteria_count": 0,
            "avg_criterion_len": 0,
            "weight_max": 0,
            "top1_share": 0,
            "weight_std": 0,
            "specific_pct": 0,
            "negation_pct": 0,
            "example_pct": 0,
            "label_only_pct": 0,
            "correctness_pct": 0,
            "constraint_pct": 0,
            "completeness_pct": 0,
        }

    texts = [c.get("criterion", "") for c in criteria]
    weights = [c.get("weight", 0.0) for c in criteria]
    total_weight = sum(weights) or 1.0

    max_w = max(weights)
    top1 = max_w / total_weight * 100
    mean_w = total_weight / n
    std_w = (sum((w - mean_w) ** 2 for w in weights) / n) ** 0.5

    avg_len = sum(len(t) for t in texts) / n

    n_specific = sum(1 for t in texts if _RE_SPECIFIC.search(t))
    n_negation = sum(1 for t in texts if _RE_NEGATION.search(t))
    n_example = sum(1 for t in texts if _RE_EXAMPLE.search(t))
    n_label_only = sum(1 for t in texts if len(t) <= LABEL_ONLY_MAX_LEN)
    n_correctness = sum(1 for t in texts if _RE_CORRECTNESS.search(t))
    n_constraint = sum(1 for t in texts if _RE_CONSTRAINT.search(t))
    n_completeness = sum(1 for t in texts if _RE_COMPLETENESS.search(t))

    return {
        "criteria_count": n,
        "avg_criterion_len": avg_len,
        "weight_max": max_w,
        "top1_share": top1,
        "weight_std": std_w,
        "specific_pct": n_specific / n * 100,
        "negation_pct": n_negation / n * 100,
        "example_pct": n_example / n * 100,
        "label_only_pct": n_label_only / n * 100,
        "correctness_pct": n_correctness / n * 100,
        "constraint_pct": n_constraint / n * 100,
        "completeness_pct": n_completeness / n * 100,
    }


# ---------------------------------------------------------------------------
# Aggregate over a JSONL file
# ---------------------------------------------------------------------------

STAT_KEYS = [
    "criteria_count", "avg_criterion_len",
    "weight_max", "top1_share", "weight_std",
    "specific_pct", "negation_pct", "example_pct", "label_only_pct",
    "correctness_pct", "constraint_pct", "completeness_pct",
]

DISPLAY_HEADERS = {
    "criteria_count": "#Crit",
    "avg_criterion_len": "Len",
    "weight_max": "MaxW",
    "top1_share": "Top-1%",
    "weight_std": "WtStd",
    "specific_pct": "Spec%",
    "negation_pct": "Neg%",
    "example_pct": "Eg%",
    "label_only_pct": "Label%",
    "correctness_pct": "Corr%",
    "constraint_pct": "Cnst%",
    "completeness_pct": "Cmpl%",
}


def aggregate_stats_from_file(path: Path) -> dict[str, float]:
    """Load a JSONL rubric file and return averaged statistics."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {k: 0.0 for k in STAT_KEYS}

    all_stats: list[dict[str, float]] = []
    for rec in records:
        text = rec.get("generation", rec.get("rubric", ""))
        criteria = extract_criteria(text)
        all_stats.append(compute_rubric_stats(criteria))

    avg: dict[str, float] = {}
    for k in STAT_KEYS:
        vals = [s[k] for s in all_stats]
        avg[k] = sum(vals) / len(vals)
    avg["n_prompts"] = len(records)
    return avg


def resolve_inputs(path: Path) -> list[tuple[str, Path]]:
    """Given a path, resolve to a list of (label, file) pairs.

    If path is a file, returns a single entry.
    If path is a directory, returns one entry per step_*.jsonl (sorted by step).
    """
    if path.is_file():
        return [(path.stem, path)]

    step_files = sorted(
        path.glob("step_*.jsonl"),
        key=lambda f: int(f.stem.split("_")[1]),
    )
    if step_files:
        return [(f"step_{f.stem.split('_')[1]}", f) for f in step_files]

    jsonl_files = sorted(path.glob("*.jsonl"))
    if jsonl_files:
        return [(f.stem, f) for f in jsonl_files]

    return []


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_val(key: str, val: float) -> str:
    if key == "criteria_count":
        return f"{val:.1f}"
    if key == "avg_criterion_len":
        return f"{val:.0f}"
    if key in ("weight_max", "weight_std"):
        return f"{val:.2f}"
    return f"{val:.1f}"


def print_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    """Print a formatted table to stdout."""
    headers = ["Source"] + [DISPLAY_HEADERS[k] for k in STAT_KEYS]
    col_widths = [max(len(h), 8) for h in headers]

    # Compute column widths from data
    for label, stats in rows:
        col_widths[0] = max(col_widths[0], len(label))
        for i, k in enumerate(STAT_KEYS, 1):
            col_widths[i] = max(col_widths[i], len(format_val(k, stats[k])))

    def fmt_row(vals: list[str]) -> str:
        return " | ".join(v.rjust(w) for v, w in zip(vals, col_widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for label, stats in rows:
        vals = [label] + [format_val(k, stats[k]) for k in STAT_KEYS]
        print(fmt_row(vals))
    print()


def print_csv(rows: list[tuple[str, dict[str, float]]], out: io.TextIOBase = sys.stdout) -> None:
    """Write stats as CSV."""
    writer = csv.writer(out)
    writer.writerow(["source"] + STAT_KEYS)
    for label, stats in rows:
        writer.writerow([label] + [round(stats[k], 2) for k in STAT_KEYS])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute aggregate rubric statistics from qualitative eval JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sources", nargs="*",
        help="JSONL file(s) or directories containing step_*.jsonl files. "
             "Use LABEL:PATH syntax to set a display label (e.g. 'V3 Main:/path/to/rubric/'). "
             "If a directory is given, all step files are analyzed as a trajectory.",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Output as CSV instead of formatted table.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON.",
    )
    args = parser.parse_args()

    if not args.sources:
        parser.print_help()
        sys.exit(1)

    # Parse LABEL:PATH pairs
    source_specs: list[tuple[str, Path]] = []
    for src in args.sources:
        if ":" in src and not Path(src).exists():
            label, path_str = src.split(":", 1)
            source_specs.append((label, Path(path_str)))
        else:
            p = Path(src)
            source_specs.append((p.name, p))

    all_rows: list[tuple[str, dict[str, float]]] = []

    for base_label, path in source_specs:
        entries = resolve_inputs(path)
        if not entries:
            print(f"Warning: no JSONL files found in {path}", file=sys.stderr)
            continue

        if len(entries) == 1:
            _, fpath = entries[0]
            stats = aggregate_stats_from_file(fpath)
            all_rows.append((base_label, stats))
        else:
            for step_label, fpath in entries:
                stats = aggregate_stats_from_file(fpath)
                row_label = f"{base_label} {step_label}" if len(source_specs) > 1 else step_label
                all_rows.append((row_label, stats))

    if not all_rows:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    if args.json_output:
        out = []
        for label, stats in all_rows:
            out.append({"source": label, **{k: round(stats[k], 2) for k in STAT_KEYS}})
        json.dump(out, sys.stdout, indent=2)
        print()
    elif args.csv:
        print_csv(all_rows)
    else:
        print_table(all_rows)


if __name__ == "__main__":
    main()
