"""
data_loader.py — Load and join Task 3 data files.

Task 3.1 JSON structure (one entry per sequence):
{
    "timeline_id":  "0cac13e357",
    "change_type":  "Switch" | "Escalation",
    "sequence_id":  "S_sequence1" | "E_sequence1",
    "postindices":  [2, 3],
    "postids":      ["13a844f48c", "751ec3360a"],
    "summary":      "Gold summary text..."
}

Post-level JSON structure (Task 1/2 output):
{
    "post_id":      "13a844f48c",
    "timeline_id":  "0cac13e357",
    "text":         "Post text...",
    "A": {"present": 1, "score": 0.8},
    "B": {"present": 0, "score": 0.1},
    "C": {"present": 1, "score": 0.6},
    "D": {"present": 0, "score": 0.2},
    "state":        "adaptive" | "maladaptive" | "neutral"
}
"""

import json
import os
from typing import Dict, List


# ── Load Task 3.1 summary file ────────────────────────────────

def load_sequences_from_summary_file(filepath: str) -> List[Dict]:
    """
    Load the Task 3.1 training JSON.
    Returns a list of sequence dicts, each with an added 'direction' field.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        entries = list(raw.values())
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"Unexpected JSON format in {filepath}")

    sequences = []
    for entry in entries:
        seq = {
            "timeline_id":  entry.get("timeline_id", "unknown"),
            "change_type":  entry.get("change_type", "Switch"),
            "sequence_id":  entry.get("sequence_id", ""),
            "postindices":  entry.get("postindices", []),
            "postids":      entry.get("postids", []),
            "gold_summary": entry.get("summary", ""),
            "posts":        [],
        }
        seq["direction"] = _infer_direction(seq["gold_summary"], seq["change_type"])
        sequences.append(seq)

    return sequences


def _infer_direction(summary_text: str, change_type: str) -> str:
    """
    Heuristically infer deterioration vs improvement from gold summary keywords.
    """
    text = summary_text.lower()
    deterioration_kw = [
        "deteriorat", "worsen", "escalat", "decline", "increas", "overwhelm",
        "collapse", "crisis", "distress", "maladaptive dominan",
    ]
    improvement_kw = [
        "improv", "recover", "stabiliz", "adaptive dominan", "resilien",
        "better", "reduc", "resolv", "switch to adaptive", "positive shift",
    ]

    det_score = sum(1 for kw in deterioration_kw if kw in text)
    imp_score = sum(1 for kw in improvement_kw if kw in text)

    if det_score > imp_score:
        return "deterioration"
    elif imp_score > det_score:
        return "improvement"
    return "mixed"


# ── Load post-level annotations (Task 1 / 2 output) ──────────

def load_posts(filepath: str) -> Dict[str, Dict]:
    """
    Load post-level annotation file.
    Returns a dict keyed by post_id.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Post file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return {p["post_id"]: p for p in raw if "post_id" in p}
    elif isinstance(raw, dict):
        return raw
    else:
        raise ValueError(f"Unexpected format in {filepath}")


# ── Join sequences with post annotations ─────────────────────

def attach_posts_to_sequences(sequences: List[Dict], posts: Dict[str, Dict]) -> List[Dict]:
    """
    For each sequence, look up its postids in the posts dict
    and attach the full post objects.
    """
    missing = 0
    for seq in sequences:
        attached = []
        for pid in seq["postids"]:
            if pid in posts:
                attached.append(posts[pid])
            else:
                missing += 1
        seq["posts"] = attached

    if missing:
        print(f"[data_loader] Warning: {missing} post IDs not found in post file.")

    return sequences


# ── Load test sequences (no gold summaries) ───────────────────

def load_test_sequences(test_json_path: str, timelines_dir: str) -> List[Dict]:
    """
    Load test sequences and attach post text from raw timeline files.
    Timeline files must have a 'posts' list where each post has:
      'post_id', 'post' (text), 'post_index', and optionally 'date'.
    """
    with open(test_json_path, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    # Build post lookup from all timeline JSON files
    post_lookup: Dict[str, Dict] = {}
    for fname in os.listdir(timelines_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(timelines_dir, fname), "r", encoding="utf-8") as f:
            timeline = json.load(f)
        for post in timeline.get("posts", []):
            post_lookup[post["post_id"]] = {
                "post_id":    post["post_id"],
                "text":       post["post"],   # raw field is "post", normalized to "text"
                "post_index": post["post_index"],
                "date":       post.get("date", ""),
            }

    missing = 0
    result = []
    for entry in sequences:
        seq = {
            "timeline_id":  entry["timeline_id"],
            "sequence_id":  entry["sequence_id"],
            "postindices":  entry["postindices"],
            "postids":      entry["postids"],
            "change_type":  "Unknown",
            "gold_summary": "",
            "direction":    "unknown",
            "posts":        [],
        }
        for pid in entry["postids"]:
            if pid in post_lookup:
                seq["posts"].append(post_lookup[pid])
            else:
                missing += 1
        result.append(seq)

    if missing:
        print(f"[data_loader] Warning: {missing} post IDs not found in timeline files.")

    return result


# ── Group sequences by timeline ───────────────────────────────

def group_by_timeline(sequences: List[Dict]) -> Dict[str, List[Dict]]:
    """Group sequences by timeline_id for Task 3.2 analysis."""
    grouped: Dict[str, List[Dict]] = {}
    for seq in sequences:
        tid = seq["timeline_id"]
        grouped.setdefault(tid, []).append(seq)
    return grouped


# ── CLI test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_train.json"
    seqs = load_sequences_from_summary_file(path)
    by_tl = group_by_timeline(seqs)

    print(f"\nLoaded {len(seqs)} sequences across {len(by_tl)} timelines.\n")
    print(f"{'Timeline':<15} {'# Seqs':<8} {'Change Types':<25} {'Directions'}")
    print("-" * 65)
    for tid, slist in sorted(by_tl.items()):
        ctypes = ", ".join(set(s["change_type"] for s in slist))
        dirs   = ", ".join(set(s["direction"]   for s in slist))
        print(f"{tid:<15} {len(slist):<8} {ctypes:<25} {dirs}")

    if len(sys.argv) > 2:
        posts = load_posts(sys.argv[2])
        seqs  = attach_posts_to_sequences(seqs, posts)
        n_with_posts = sum(1 for s in seqs if s["posts"])
        print(f"\nAttached posts: {n_with_posts}/{len(seqs)} sequences have post text.")
