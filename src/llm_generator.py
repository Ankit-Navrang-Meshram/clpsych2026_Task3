"""
llm_generator.py — LLM inference with multiple backends.

Supported backends:
  mock    — deterministic placeholder, no model required (for testing)
  ollama  — local Ollama REST API  (requires: ollama serve)
  hf      — HuggingFace transformers (GPU recommended)
"""

import requests
from typing import Dict, List


# ── Public API ────────────────────────────────────────────────

def generate(
    system_prompt: str,
    user_prompt: str,
    backend: str = "mock",
    model: str = "llama3:8b",
    max_new_tokens: int = 600,
    temperature: float = 0.3,
    **kwargs,
) -> str:
    """
    Generate a response given system + user prompts.
    Returns the raw text response string.
    """
    if backend == "mock":
        return _mock_generate(system_prompt, user_prompt)
    elif backend == "ollama":
        return _ollama_generate(system_prompt, user_prompt, model, max_new_tokens, temperature)
    elif backend == "hf":
        return _hf_generate(system_prompt, user_prompt, model, max_new_tokens, temperature)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: mock, ollama, hf")


def generate_batch(
    prompts: List[Dict],
    backend: str = "mock",
    model: str = "llama3:8b",
    max_new_tokens: int = 600,
    temperature: float = 0.3,
    verbose: bool = True,
) -> List[str]:
    """
    Generate responses for a list of {"system": ..., "user": ...} dicts.
    Shows a progress counter when verbose=True.
    """
    results = []
    n = len(prompts)
    for i, p in enumerate(prompts, 1):
        if verbose:
            print(f"  Generating {i}/{n}...", end="\r", flush=True)
        try:
            out = generate(
                system_prompt=p["system"],
                user_prompt=p["user"],
                backend=backend,
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as e:
            print(f"\n  [WARNING] Generation failed for item {i}: {e}")
            out = f"[GENERATION ERROR: {e}]"
        results.append(out)

    if verbose:
        print(f"  Generated {n}/{n} responses.    ")
    return results


# ── Mock backend ──────────────────────────────────────────────

def _mock_generate(system_prompt: str, user_prompt: str) -> str:
    """
    Returns a structured placeholder matching the expected output format.
    Useful for end-to-end pipeline testing without a real model.
    """
    tid, ctype = "unknown", "Switch"
    for line in user_prompt.splitlines():
        if line.startswith("TIMELINE ID:"):
            tid = line.split(":", 1)[1].strip()
        if line.startswith("CHANGE EVENT TYPE:"):
            ctype = line.split(":", 1)[1].strip()

    return (
        f"1. CENTRAL THEME\n"
        f"The individual (timeline {tid}) demonstrates a central preoccupation with affect "
        f"dysregulation and cognitive rigidity, with the {ctype.lower()} event marking a "
        f"notable shift in the balance between adaptive and maladaptive self-states. "
        f"Affect (A) remains the dominant ABCD dimension throughout, modulated by behavioral "
        f"withdrawal (B) and negative cognitive schemas (C).\n\n"
        f"2. WITHIN-STATE DYNAMICS\n"
        f"Within the maladaptive state, heightened negative affect (A) reinforces ruminative "
        f"cognitions (C), which in turn suppress adaptive behaviors (B). Drivers (D) such as "
        f"perceived social isolation and unresolved interpersonal conflict sustain the "
        f"maladaptive cycle, preventing natural recovery. Within adaptive state periods, "
        f"behavioral engagement (B) temporarily buffers affective distress (A).\n\n"
        f"3. BETWEEN-STATE DYNAMICS\n"
        f"The adaptive and maladaptive states alternate with the maladaptive state gaining "
        f"progressive dominance as the sequence unfolds. The {ctype.lower()} event represents "
        f"a tipping point where driver (D) intensity exceeds the individual's cognitive coping "
        f"capacity (C), causing a rapid collapse of adaptive functioning."
    )


# ── Ollama backend ────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"


def _ollama_generate(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": temperature,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Start it first:\n"
            "  ollama serve   (in a separate terminal)"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


# ── HuggingFace backend ───────────────────────────────────────

_hf_pipeline = None


def _hf_generate(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    global _hf_pipeline
    if _hf_pipeline is None:
        print(f"\n[HF] Loading model '{model}' (this may take a minute)...")
        from transformers import pipeline
        _hf_pipeline = pipeline(
            "text-generation",
            model=model,
            device_map="auto",
            torch_dtype="auto",
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    out = _hf_pipeline(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,
    )
    return out[0]["generated_text"].strip()


# ── CLI test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "mock"
    model   = sys.argv[2] if len(sys.argv) > 2 else "llama3:8b"
    print(f"Testing backend={backend}, model={model}")
    result = generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say hello in one sentence.",
        backend=backend,
        model=model,
    )
    print("Response:", result)
