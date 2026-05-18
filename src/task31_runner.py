"""
task31_runner.py — Task 3.1: Generate structured narrative summaries.

For each sequence (Switch or Escalation event), produces a 3-part summary:
  1. CENTRAL THEME
  2. WITHIN-STATE DYNAMICS
  3. BETWEEN-STATE DYNAMICS
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from prompt_builder import (
    TASK31_SYSTEM,
    build_task31_prompt,
    build_task31_prompt_with_example,
)
from llm_generator import generate_batch


# ── Main runner ───────────────────────────────────────────────

def run_task31(
    sequences: List[Dict],
    backend: str = "mock",
    model: str = "llama3:8b",
    output_dir: str = "outputs",
    few_shot: bool = False,
    max_new_tokens: int = 700,
    temperature: float = 0.3,
) -> List[Dict]:
    """
    Generate Task 3.1 summaries for all sequences.

    Args:
        sequences:      List of sequence dicts from data_loader.
        backend:        LLM backend — 'mock', 'ollama', or 'hf'.
        model:          Model name (used by ollama/hf backends).
        output_dir:     Directory to write results and submission files.
        few_shot:       If True, prepend the previous gold summary as a reference example.
        max_new_tokens: Token budget for generation.
        temperature:    Sampling temperature.

    Returns:
        List of result dicts with generated summary + metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Building prompts for {len(sequences)} sequences...")
    prompts = []
    for i, seq in enumerate(sequences):
        if few_shot and i > 0 and sequences[i - 1].get("gold_summary"):
            user_prompt = build_task31_prompt_with_example(seq, sequences[i - 1])
        else:
            user_prompt = build_task31_prompt(seq, include_posts=bool(seq.get("posts")))
        prompts.append({"system": TASK31_SYSTEM, "user": user_prompt})

    print(f"  Running generation (backend={backend}, model={model})...")
    generated_texts = generate_batch(
        prompts=prompts,
        backend=backend,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        verbose=True,
    )

    results = []
    for seq, gen_text in zip(sequences, generated_texts):
        result = {
            "timeline_id":       seq["timeline_id"],
            "sequence_id":       seq["sequence_id"],
            "change_type":       seq["change_type"],
            "direction":         seq.get("direction", "unknown"),
            "postids":           seq["postids"],
            "gold_summary":      seq.get("gold_summary", ""),
            "generated_summary": gen_text,
            "parsed_sections":   _parse_sections(gen_text),
        }
        results.append(result)

    # Save full results
    out_path = os.path.join(output_dir, "task31_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(results)} summaries → {out_path}")

    # Save Codabench submission format
    _save_submission_format(results, output_dir)

    return results


# ── Section parsing ───────────────────────────────────────────

def _parse_sections(text: str) -> Dict[str, str]:
    """
    Extract the three named sections from a generated summary.
    Falls back gracefully when headers are missing or malformed.
    """
    sections = {"central_theme": "", "within_state": "", "between_state": ""}

    patterns = [
        (r"1\.\s*CENTRAL THEME[:\n]+(.*?)(?=2\.\s*WITHIN|$)",      "central_theme"),
        (r"2\.\s*WITHIN.STATE DYNAMICS[:\n]+(.*?)(?=3\.\s*BETWEEN|$)", "within_state"),
        (r"3\.\s*BETWEEN.STATE DYNAMICS[:\n]+(.*?)$",               "between_state"),
    ]
    for pattern, key in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            sections[key] = m.group(1).strip()

    # Fallback: dump everything into central_theme if no headers matched
    if not any(sections.values()):
        sections["central_theme"] = text.strip()

    return sections


# ── Submission format ─────────────────────────────────────────

def _save_submission_format(results: List[Dict], output_dir: str):
    """Save Codabench submission JSON: keyed by '{timeline_id}_{sequence_id}'."""
    submission = {}
    for r in results:
        key = f"{r['timeline_id']}_{r['sequence_id']}"
        submission[key] = {
            "timeline_id": r["timeline_id"],
            "sequence_id": r["sequence_id"],
            "change_type": r["change_type"],
            "summary":     r["generated_summary"],
        }
    path = os.path.join(output_dir, "task31_submission.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
    print(f"  Saved Codabench submission → {path}")


# ── Evaluation ────────────────────────────────────────────────

def evaluate_summaries(results: List[Dict], output_dir: str = "outputs") -> Dict:
    """
    Evaluate generated summaries against gold using ROUGE and BERTScore (if installed).
    Skips gracefully when gold summaries are absent or libraries are missing.
    """
    pairs = [(r["gold_summary"], r["generated_summary"])
             for r in results if r.get("gold_summary")]

    if not pairs:
        print("  No gold summaries found — skipping evaluation.")
        return {}

    gold_texts, gen_texts = zip(*pairs)
    metrics: Dict = {}

    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rL = [], [], []
        for gold, gen in zip(gold_texts, gen_texts):
            s = scorer.score(gold, gen)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rL.append(s["rougeL"].fmeasure)
        metrics["rouge1"] = round(sum(r1) / len(r1), 4)
        metrics["rouge2"] = round(sum(r2) / len(r2), 4)
        metrics["rougeL"] = round(sum(rL) / len(rL), 4)
        print(f"\n  ROUGE-1: {metrics['rouge1']}  "
              f"ROUGE-2: {metrics['rouge2']}  "
              f"ROUGE-L: {metrics['rougeL']}")
    except ImportError:
        print("  rouge_score not installed — skipping ROUGE. "
              "Install with: pip install rouge_score")

    # BERTScore (optional, slow on CPU)
    try:
        from bert_score import score as bert_score
        _, _, F1 = bert_score(list(gen_texts), list(gold_texts), lang="en", verbose=False)
        metrics["bertscore_f1"] = round(F1.mean().item(), 4)
        print(f"  BERTScore F1: {metrics['bertscore_f1']}")
    except ImportError:
        pass

    eval_path = os.path.join(output_dir, "task31_eval.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Evaluation saved → {eval_path}")

    return metrics
