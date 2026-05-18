# CLPsych 2026 — Task 3: Psychological Change Narrative Analysis

A pipeline for generating structured narrative summaries (Task 3.1) and extracting recurrent dynamic signatures (Task 3.2) from social media timelines, using the ABCD framework.

---

## Project Structure

```
clpsych_task3/
├── run_pipeline.py        # Main CLI entry point
├── requirements.txt
├── src/
│   ├── data_loader.py     # Load & join sequence / post data
│   ├── llm_generator.py   # LLM backends: mock | ollama | hf
│   ├── prompt_builder.py  # Prompt construction for Task 3.1 & 3.2
│   ├── task31_runner.py   # Task 3.1: narrative summary generation
│   └── task32_runner.py   # Task 3.2: dynamic signature extraction
├── data/                  # Place your data files here (gitignored)
└── outputs/               # Generated results written here (gitignored)
```

---

## ABCD Framework

| Label | Dimension | Description |
|-------|-----------|-------------|
| **A** | Affect | Emotions, mood, emotional reactivity |
| **B** | Behavior | Actions, coping strategies, engagement |
| **C** | Cognition | Thoughts, beliefs, self-perception |
| **D** | Drivers | Context, stressors, triggers |

**Self-states:** Adaptive (healthy coping) ↔ Maladaptive (distress, avoidance, crisis)

---

## Setup

```bash
git clone https://github.com/<your-username>/clpsych_task3.git
cd clpsych_task3
pip install -r requirements.txt
```

### Optional: Ollama (local LLM)
```bash
# Install from https://ollama.com/download, then:
ollama serve          # start the server
ollama pull llama3:8b # download a model
```

---

## Usage

### Quick smoke test (no model required)
```bash
python run_pipeline.py --data data/train.json --backend mock
```

### Task 3.1 only — with Ollama
```bash
python run_pipeline.py \
  --data data/train.json \
  --backend ollama \
  --model llama3:8b \
  --task 31
```

### Both tasks — with post annotations
```bash
python run_pipeline.py \
  --data data/train.json \
  --posts data/posts.json \
  --backend ollama \
  --task both
```

### Test mode (no gold summaries)
```bash
python run_pipeline.py \
  --test data/test.json \
  --timelines data/timelines/ \
  --backend ollama
```

### Few-shot prompting + evaluation
```bash
python run_pipeline.py \
  --data data/train.json \
  --backend ollama \
  --few-shot \
  --evaluate
```

### All CLI options
```
--data FILE          Training JSON with gold summaries
--test FILE          Test JSON (no gold summaries)
--timelines DIR      Raw timeline JSONs directory (required with --test)
--posts FILE         Post-level annotation JSON (optional)
--task {31,32,both}  Which task(s) to run (default: both)
--backend            mock | ollama | hf  (default: mock)
--model              Model name for ollama/hf (default: llama3:8b)
--max-tokens INT     Max new tokens per generation (default: 700)
--temperature FLOAT  Sampling temperature (default: 0.3)
--few-shot           Few-shot prompting for Task 3.1
--evaluate           Run ROUGE/BERTScore evaluation (requires rouge_score)
--output-dir DIR     Output directory (default: outputs/)
```

---

## Data Format

### Training / Task 3.1 JSON (`--data`)
```json
[
  {
    "timeline_id": "0cac13e357",
    "change_type": "Switch",
    "sequence_id": "S_sequence1",
    "postindices": [2, 3],
    "postids":     ["13a844f48c", "751ec3360a"],
    "summary":     "Gold summary text..."
  }
]
```

### Post annotation JSON (`--posts`)
```json
[
  {
    "post_id":     "13a844f48c",
    "timeline_id": "0cac13e357",
    "text":        "Post text...",
    "A": {"present": 1, "score": 0.8},
    "B": {"present": 0, "score": 0.1},
    "C": {"present": 1, "score": 0.6},
    "D": {"present": 0, "score": 0.2},
    "state":       "adaptive"
  }
]
```

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/task31_results.json` | Full results with generated + gold summaries |
| `outputs/task31_submission.json` | Codabench submission format |
| `outputs/task31_eval.json` | ROUGE / BERTScore metrics (if `--evaluate`) |
| `outputs/task32_raw.txt` | Raw LLM output for Task 3.2 |
| `outputs/task32_signatures.json` | Parsed dynamic signatures |
| `outputs/task32_submission.txt` | Human-readable submission text |

---

## Backends

| Backend | When to use |
|---------|-------------|
| `mock` | Testing the pipeline without a model |
| `ollama` | Local inference — fastest setup, good quality |
| `hf` | HuggingFace transformers — flexible, GPU recommended |

---

## Evaluation

Install optional dependencies to enable `--evaluate`:

```bash
pip install rouge_score      # ROUGE-1, ROUGE-2, ROUGE-L
pip install bert_score       # BERTScore F1 (slow on CPU)
```

---

## License

This code is released for research use only. Shared task data is subject to CLPsych 2026 data use agreements.
