"""
prompt_builder.py — Build structured prompts for Task 3.1 and 3.2.

ABCD framework:
  A = Affect & emotional states
  B = Behavioral patterns & actions
  C = Cognitive patterns & beliefs
  D = Drivers / context / stressors

Self-states:
  Adaptive    = healthy coping, hope, engagement
  Maladaptive = distress, avoidance, crisis, hopelessness
"""

from typing import Dict, List


# ── Task 3.1 system prompt ────────────────────────────────────

TASK31_SYSTEM = """You are a clinical psychologist specializing in suicidality and narrative analysis.
Your task is to write a structured narrative summary of a sequence of social media posts surrounding a psychological change event (Switch or Escalation) in a person's timeline.

Use the ABCD framework throughout:
  A = Affect (emotions, mood)
  B = Behavior (actions, coping)
  C = Cognition (thoughts, beliefs, self-perception)
  D = Drivers (context, stressors, triggers)

Your summary must cover THREE parts:
1. CENTRAL THEME: The dominant ABCD theme and how it evolves across the sequence.
2. WITHIN-STATE DYNAMICS: How ABCD sub-elements interact inside the adaptive or maladaptive self-state.
3. BETWEEN-STATE DYNAMICS: How adaptive and maladaptive states relate and shift in dominance across the sequence.

Be specific, clinically grounded, and concise. Write in third person ("The individual...").
Do not include bullet points. Write in flowing prose."""


# ── Task 3.1 prompt builders ──────────────────────────────────

def build_task31_prompt(sequence: Dict, include_posts: bool = True) -> str:
    """
    Build a Task 3.1 user prompt for a single sequence.
    Works with or without attached post texts.
    """
    tid       = sequence["timeline_id"]
    ctype     = sequence["change_type"]
    sid       = sequence["sequence_id"]
    posts     = sequence.get("posts", [])
    direction = sequence.get("direction", "unknown")

    lines = [
        f"TIMELINE ID: {tid}",
        f"SEQUENCE ID: {sid}",
        f"CHANGE EVENT TYPE: {ctype}",
        f"INFERRED DIRECTION: {direction}",
        "",
    ]

    if include_posts and posts:
        lines.append("POST SEQUENCE (chronological):")
        for i, p in enumerate(posts, 1):
            text  = p.get("text", "[no text]")
            state = p.get("state", "unknown")
            abcd  = _format_abcd(p)
            lines.append(f"\n  Post {i} [state={state}] {abcd}")
            lines.append(f"  \"{text[:500]}\"")   # cap per post to control context length
    else:
        lines.append("(No post text available — generate summary based on sequence metadata.)")

    lines += [
        "",
        "Write a structured narrative summary with the THREE required sections:",
        "1. CENTRAL THEME",
        "2. WITHIN-STATE DYNAMICS",
        "3. BETWEEN-STATE DYNAMICS",
    ]

    return "\n".join(lines)


def build_task31_prompt_with_example(sequence: Dict, example: Dict) -> str:
    """
    Few-shot variant: prepend one gold example before the target sequence prompt.
    """
    example_block = (
        "EXAMPLE (gold reference):\n"
        f"[Timeline: {example['timeline_id']} | Sequence: {example['sequence_id']}]\n"
        f"{example.get('gold_summary', '')}\n"
        "---\n"
        "Now write the summary for the following sequence in the same style:\n\n"
    )
    return example_block + build_task31_prompt(sequence)


def _format_abcd(post: Dict) -> str:
    """Format ABCD presence flags as a compact inline tag, e.g. [A=✓(0.80) B=✗(0.10)]."""
    parts = []
    for dim in ["A", "B", "C", "D"]:
        val = post.get(dim, {})
        if isinstance(val, dict):
            present = val.get("present", 0)
            score   = val.get("score", 0.0)
            parts.append(f"{dim}={'✓' if present else '✗'}({score:.2f})")
        elif isinstance(val, (int, float)):
            parts.append(f"{dim}={'✓' if val else '✗'}")
    return "[" + " ".join(parts) + "]" if parts else ""


# ── Task 3.2 system prompt ────────────────────────────────────

TASK32_SYSTEM = """You are a clinical researcher analyzing patterns of psychological deterioration and improvement in suicidal ideation timelines.

You will be given a collection of narrative summaries from multiple individuals' social media timelines.
Each summary describes the ABCD dynamics (Affect, Behavior, Cognition, Drivers) and self-state interactions surrounding a Switch or Escalation event.

Your task is to identify 6 RECURRENT DYNAMIC SIGNATURES across the full collection:
  - 3 signatures of DETERIORATION (patterns that consistently precede or accompany worsening)
  - 3 signatures of IMPROVEMENT (patterns that consistently precede or accompany recovery)

For each signature:
  1. Give it a short descriptive name (e.g., "Cognitive rigidity under escalating drivers")
  2. Describe the ABCD dynamic pattern (which dimensions dominate, how they interact)
  3. Describe the self-state interaction pattern (how adaptive/maladaptive states shift)
  4. Cite 2–3 specific examples from the summaries provided (use timeline_id)
  5. Explain the clinical significance (1–2 sentences)

Write in formal academic prose. Be specific about ABCD interactions — avoid vague generalities."""


# ── Task 3.2 prompt builder ───────────────────────────────────

def build_task32_prompt(summaries_by_direction: Dict[str, List[Dict]]) -> str:
    """
    Build a Task 3.2 prompt from summaries grouped by direction.

    Args:
        summaries_by_direction: {"deterioration": [...], "improvement": [...], "mixed": [...]}
    """
    total = sum(len(v) for v in summaries_by_direction.values())
    lines = [
        "TASK: Identify 6 recurrent dynamic signatures across the following timeline summaries.",
        "Provide 3 DETERIORATION signatures and 3 IMPROVEMENT signatures.",
        "",
        f"Total summaries: {total}",
        "",
    ]

    for direction, seqs in summaries_by_direction.items():
        if not seqs:
            continue
        lines.append(f"── {direction.upper()} TIMELINES ({len(seqs)} sequences) ──")
        for seq in seqs:
            tid     = seq.get("timeline_id", "?")
            sid     = seq.get("sequence_id", "?")
            ctype   = seq.get("change_type", "?")
            summary = seq.get("gold_summary", "[no summary]")
            lines.append(f"\n[{tid} | {sid} | {ctype}]")
            lines.append(summary[:800])   # cap to manage context length
        lines.append("")

    lines += [
        "─" * 60,
        "",
        "Now identify and describe the 6 recurrent dynamic signatures.",
        "Format each signature clearly with:",
        "  NAME, ABCD PATTERN, SELF-STATE PATTERN, EXAMPLES, CLINICAL SIGNIFICANCE.",
    ]

    return "\n".join(lines)
