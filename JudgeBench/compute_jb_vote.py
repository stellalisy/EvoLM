#!/usr/bin/env python3
"""Compute JudgeBench vote accuracy from a result JSONL file.

Usage:
    python compute_jb_vote.py <result.jsonl>

Outputs per-category and overall vote accuracy matching the logic in
utils/metrics.py (reverse_order=True, i.e. two-game "vote" metric).

Categories:
    Knowledge  = mmlu-pro*
    Reasoning  = livebench-reasoning*
    Math       = livebench-math*
    Coding     = livecodebench*
    Vote       = overall (all 350 pairs)
    Avg        = macro-average of the four category accuracies
"""

import json
import sys
from typing import List, Dict, Any


def flip_judgment(decision: str) -> str:
    """Flip A<->B in a decision string (mirrors utils/metrics.py)."""
    if decision == "A>B":
        return "B>A"
    elif decision == "B>A":
        return "A>B"
    return decision


def vote_accuracy(pairs: List[Dict[str, Any]]) -> float:
    """Compute vote accuracy using the two-game voting logic from metrics.py.

    For each pair with two judgments (forward and reverse order):
      - judgment[0] is the forward game: judge sees (A, B)
      - judgment[1] is the reverse game: judge sees (B, A)

    The reverse game's decision is flipped (A>B <-> B>A) to normalize it
    to the same frame of reference as the forward game.

    Then a counter tallies:
      +1 if a decision matches the gold label
      -1 if it matches the flipped label
       0 if it's a tie (A=B) or None

    If counter > 0: correct.  If counter < 0: incorrect.  If 0: tie (wrong).
    Vote accuracy = n_correct / n_pairs.
    """
    if not pairs:
        return 0.0

    n_correct = 0
    for pair in pairs:
        label = pair["label"]
        j1, j2 = pair["judgments"]

        decision1 = j1["decision"] if j1 is not None else None
        decision2 = flip_judgment(j2["decision"] if j2 is not None else None)

        counter = 0
        for decision in [decision1, decision2]:
            if decision == label:
                counter += 1
            elif decision == flip_judgment(label):
                counter -= 1

        if counter > 0:
            n_correct += 1

    return 100.0 * n_correct / len(pairs)


CATEGORIES = [
    ("Knowledge", "mmlu-pro"),
    ("Reasoning", "livebench-reasoning"),
    ("Math", "livebench-math"),
    ("Coding", "livecodebench"),
]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result.jsonl>", file=sys.stderr)
        sys.exit(1)

    fpath = sys.argv[1]
    pairs = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    # Verify all pairs have 2 judgments (reverse_order=True)
    n_single = sum(1 for p in pairs if len(p.get("judgments", [])) < 2)
    if n_single:
        print(f"WARNING: {n_single}/{len(pairs)} pairs have < 2 judgments "
              f"(single-game mode?)", file=sys.stderr)

    cat_scores = {}
    for cat_name, prefix in CATEGORIES:
        cat_pairs = [p for p in pairs if p["source"].startswith(prefix)]
        if cat_pairs:
            acc = vote_accuracy(cat_pairs)
            cat_scores[cat_name] = acc
            print(f"{cat_name}: {acc:.2f}% ({len(cat_pairs)} pairs)")
        else:
            print(f"{cat_name}: N/A (0 pairs)")

    overall = vote_accuracy(pairs)
    print(f"Vote: {overall:.2f}% ({len(pairs)} pairs)")

    if cat_scores:
        avg = sum(cat_scores.values()) / len(cat_scores)
        print(f"Avg: {avg:.2f}%")


if __name__ == "__main__":
    main()
