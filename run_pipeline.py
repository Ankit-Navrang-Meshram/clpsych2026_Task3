"""
run_pipeline.py — End-to-end pipeline for CLPsych 2026 Task 3.

Usage examples:

  # Quick smoke test with mock backend (no model needed)
  python run_pipeline.py --data data/train.json --backend mock

  # Run Task 3.1 only with Ollama
  python run_pipeline.py --data data/train.json --backend ollama --model llama3:8b --task 31

  # Run both tasks with post annotations
  python run_pipeline.py --data data/train.json --posts data/posts.json --backend ollama

  # Test mode (no gold summaries)
  python run_pipeline.py --test data/test.json --timelines data/timelines/ --backend ollama

  # Few-shot mode
  python run_pipeline.py --data data/train.json --backend ollama --few-shot

  # Evaluate Task 3.1 after generation (requires rouge_score)
  python run_pipeline.py --data data/train.json --backend mock --evaluate
"""

import argparse
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import (
    load_sequences_from_summary_file,
    load_posts,
    attach_posts_to_sequences,
    load_test_sequences,
)
from task31_runner import run_task31, evaluate_summaries
from task32_runner import run_task32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CLPsych 2026 Task 3 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data sources (mutually exclusive: train vs test mode)
    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data", metavar="FILE",
        help="Training JSON with gold summaries (Task 3.1 format)",
    )
    data_group.add_argument(
        "--test", metavar="FILE",
        help="Test JSON without gold summaries",
    )

    p.add_argument(
        "--timelines", metavar="DIR",
        help="Directory of raw timeline JSONs (required with --test)",
    )
    p.add_argument(
        "--posts", metavar="FILE",
        help="Post-level annotation JSON (Task 1/2 output, optional)",
    )

    # Task selection
    p.add_argument(
        "--task", choices=["31", "32", "both"], default="both",
        help="Which task(s) to run (default: both)",
    )

    # Generation settings
    p.add_argument("--backend", choices=["mock", "ollama", "hf"], default="mock",
                   help="LLM backend (default: mock)")
    p.add_argument("--model", default="llama3:8b",
                   help="Model name for ollama/hf backends (default: llama3:8b)")
    p.add_argument("--max-tokens", type=int, default=700,
                   help="Max new tokens per generation (default: 700)")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Sampling temperature (default: 0.3)")

    # Options
    p.add_argument("--few-shot", action="store_true",
                   help="Use few-shot prompting for Task 3.1")
    p.add_argument("--evaluate", action="store_true",
                   help="Run ROUGE/BERTScore evaluation after Task 3.1 (requires gold data)")
    p.add_argument("--output-dir", default="outputs",
                   help="Directory for output files (default: outputs/)")

    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "═" * 60)
    print("  CLPsych 2026 — Task 3 Pipeline")
    print("═" * 60)

    # ── Load sequences ────────────────────────────────────────
    if args.data:
        print(f"\n[1/4] Loading training data: {args.data}")
        sequences = load_sequences_from_summary_file(args.data)

        if args.posts:
            print(f"      Attaching posts from: {args.posts}")
            posts = load_posts(args.posts)
            sequences = attach_posts_to_sequences(sequences, posts)

    else:  # --test mode
        if not args.timelines:
            print("ERROR: --timelines DIR is required when using --test")
            sys.exit(1)
        print(f"\n[1/4] Loading test data: {args.test}")
        print(f"      Timelines directory: {args.timelines}")
        sequences = load_test_sequences(args.test, args.timelines)

    n_with_posts = sum(1 for s in sequences if s.get("posts"))
    print(f"      Loaded {len(sequences)} sequences ({n_with_posts} with post text)")

    # ── Task 3.1 ──────────────────────────────────────────────
    task31_results = []
    if args.task in ("31", "both"):
        print(f"\n[2/4] Task 3.1 — Narrative summary generation")
        task31_results = run_task31(
            sequences=sequences,
            backend=args.backend,
            model=args.model,
            output_dir=args.output_dir,
            few_shot=args.few_shot,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        if args.evaluate:
            print(f"\n[3/4] Evaluating Task 3.1 summaries...")
            evaluate_summaries(task31_results, output_dir=args.output_dir)
        else:
            print("\n[3/4] Evaluation skipped (use --evaluate to run ROUGE/BERTScore)")

    else:
        print("\n[2/4] Task 3.1 skipped")
        print("[3/4] Evaluation skipped")

    # ── Task 3.2 ──────────────────────────────────────────────
    if args.task in ("32", "both"):
        print(f"\n[4/4] Task 3.2 — Dynamic signature extraction")

        # Prefer sequences enriched with generated summaries when available
        source = task31_results if task31_results else sequences
        # For Task 3.2, use gold_summary when available, else generated_summary
        for seq in source:
            if not seq.get("gold_summary") and seq.get("generated_summary"):
                seq["gold_summary"] = seq["generated_summary"]

        run_task32(
            sequences=source,
            backend=args.backend,
            model=args.model,
            output_dir=args.output_dir,
            max_new_tokens=min(args.max_tokens * 2, 1500),
            temperature=max(args.temperature - 0.1, 0.1),
        )
    else:
        print("\n[4/4] Task 3.2 skipped")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  Done. Outputs written to: {os.path.abspath(args.output_dir)}/")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
