"""
task32_runner.py — Task 3.2: Extract 6 recurrent dynamic signatures.

Aggregates all timeline summaries and identifies:
  - 3 signatures of DETERIORATION (patterns preceding / accompanying worsening)
  - 3 signatures of IMPROVEMENT   (patterns preceding / accompanying recovery)

in terms of ABCD dynamics and self-state interactions.
"""

import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader import group_by_timeline
from prompt_builder import TASK32_SYSTEM, build_task32_prompt
from llm_generator import generate


# ── Main runner ───────────────────────────────────────────────

def run_task32(
    sequences: List[Dict],
    backend: str = "mock",
    model: str = "llama3:8b",
    output_dir: str = "outputs",
    max_new_tokens: int = 1500,
    temperature: float = 0.2,
) -> Dict:
    """
    Run Task 3.2 signature extraction across all sequences.

    Args:
        sequences:      List of sequence dicts (ideally with generated/gold summaries).
        backend:        LLM backend — 'mock', 'ollama', or 'hf'.
        model:          Model name (used by ollama/hf backends).
        output_dir:     Directory to write outputs.
        max_new_tokens: Token budget for generation.
        temperature:    Sampling temperature.

    Returns:
        Dict with 'raw_output', 'signatures', 'n_sequences', 'n_timelines'.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group by direction
    by_direction: Dict[str, List[Dict]] = defaultdict(list)
    for seq in sequences:
        by_direction[seq.get("direction", "mixed")].append(seq)

    print("  Sequences by direction:")
    for d, seqs in by_direction.items():
        print(f"    {d}: {len(seqs)}")

    if len(sequences) < 5:
        print(f"  WARNING: Only {len(sequences)} sequences — patterns may not be meaningful.")

    user_prompt = build_task32_prompt(dict(by_direction))

    print(f"\n  Running signature extraction (backend={backend}, model={model})...")
    raw_output = generate(
        system_prompt=TASK32_SYSTEM,
        user_prompt=user_prompt,
        backend=backend,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    parsed = _parse_signatures(raw_output)

    result = {
        "raw_output":  raw_output,
        "signatures":  parsed,
        "n_sequences": len(sequences),
        "n_timelines": len(group_by_timeline(sequences)),
    }

    # Save raw text
    raw_path = os.path.join(output_dir, "task32_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_output)
    print(f"  Saved raw output → {raw_path}")

    # Save structured JSON
    json_path = os.path.join(output_dir, "task32_signatures.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved structured signatures → {json_path}")

    # Save human-readable submission text
    _save_submission_text(raw_output, parsed, output_dir)

    _print_signature_summary(parsed)

    return result


# ── Parsing ───────────────────────────────────────────────────

def _parse_signatures(text: str) -> Dict:
    """
    Parse LLM output into {"deterioration": [...], "improvement": [...]}.
    Falls back gracefully if section headers are missing.
    """
    result: Dict = {"deterioration": [], "improvement": []}

    det_block = _extract_block(text, r"DETERIORATION")
    imp_block = _extract_block(text, r"IMPROVEMENT")

    result["deterioration"] = _extract_signatures_from_block(det_block, expected=3)
    result["improvement"]   = _extract_signatures_from_block(imp_block, expected=3)

    if not result["deterioration"] and not result["improvement"]:
        result["raw_unparsed"] = text

    return result


def _extract_block(text: str, header_re: str) -> str:
    """Extract the section of text that follows a given header pattern."""
    pattern = rf"{header_re}.*?(?=\n\n[A-Z]{{4,}}|\Z)"
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(0).strip() if m else text


def _extract_signatures_from_block(block: str, expected: int = 3) -> List[Dict]:
    """
    Split a block into individual signature entries and parse each one.
    Expects numbered items or 'Signature N' headers.
    """
    sigs: List[Dict] = []

    chunks = re.split(
        r"\n(?=(?:Signature\s+\d|SIGNATURE\s+\d|\d+\.\s+[A-Z]))",
        block,
        flags=re.IGNORECASE,
    )

    for chunk in chunks:
        if len(chunk.strip()) < 50:
            continue
        sig = _parse_one_signature(chunk)
        if sig:
            sigs.append(sig)
        if len(sigs) >= expected:
            break

    if not sigs and block.strip():
        sigs = [{"name": "Signature", "raw": block.strip()}]

    return sigs


def _parse_one_signature(text: str) -> Optional[Dict]:
    """Parse structured fields from a single signature block."""
    sig: Dict = {}

    field_patterns = {
        "name":                  r"(?:NAME|Signature\s+\d+)[:\s]+([^\n]+)",
        "abcd_pattern":          r"(?:ABCD\s+PATTERN)[:\s]+(.*?)(?=(?:SELF-STATE|EXAMPLES|CLINICAL)|$)",
        "self_state_pattern":    r"(?:SELF.STATE\s+PATTERN)[:\s]+(.*?)(?=(?:EXAMPLES|CLINICAL)|$)",
        "examples":              r"(?:EXAMPLES)[:\s]+(.*?)(?=(?:CLINICAL)|$)",
        "clinical_significance": r"(?:CLINICAL\s+SIGNIFICANCE)[:\s]+(.*?)$",
    }

    for key, pattern in field_patterns.items():
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            sig[key] = m.group(1).strip()

    if not sig:
        return None

    if "name" not in sig:
        first_line = text.strip().splitlines()[0]
        sig["name"] = re.sub(r"^\d+\.\s*", "", first_line).strip()

    return sig


# ── Submission output ─────────────────────────────────────────

def _save_submission_text(raw: str, parsed: Dict, output_dir: str):
    """Save a clean, human-readable submission text file."""
    lines = [
        "CLPsych 2026 Shared Task — Task 3.2 Submission",
        "6 Recurrent Dynamic Signatures",
        "=" * 60,
        "",
        "DETERIORATION SIGNATURES",
        "-" * 40,
    ]

    for i, sig in enumerate(parsed.get("deterioration", []), 1):
        lines.append(f"\nDeterioration Signature {i}: {sig.get('name', 'Unnamed')}")
        for field in ["abcd_pattern", "self_state_pattern", "examples", "clinical_significance"]:
            if field in sig:
                label = field.replace("_", " ").title()
                lines.append(f"  {label}: {sig[field]}")

    lines += ["", "IMPROVEMENT SIGNATURES", "-" * 40]

    for i, sig in enumerate(parsed.get("improvement", []), 1):
        lines.append(f"\nImprovement Signature {i}: {sig.get('name', 'Unnamed')}")
        for field in ["abcd_pattern", "self_state_pattern", "examples", "clinical_significance"]:
            if field in sig:
                label = field.replace("_", " ").title()
                lines.append(f"  {label}: {sig[field]}")

    path = os.path.join(output_dir, "task32_submission.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved submission text → {path}")


def _print_signature_summary(parsed: Dict):
    print("\n  ── Signature Summary ──────────────────────────────")
    for direction in ("deterioration", "improvement"):
        sigs = parsed.get(direction, [])
        print(f"  {direction.upper()} ({len(sigs)} found):")
        for i, sig in enumerate(sigs, 1):
            print(f"    {i}. {sig.get('name', 'Unnamed')}")
    print()
