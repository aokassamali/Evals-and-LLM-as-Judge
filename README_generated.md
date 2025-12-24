# Evals-and-LLM-as-Judge (Ollama + API) — SciFact Eval Harness

A **reproducible evaluation harness** for comparing **local Ollama models** and **hosted API models** on **SciFact-style claim verification** (SUPPORTS / REFUTES / NEI).

It gives you:
- **Run artifacts**: one JSONL per model × dataset in `runs/` (immutable “source of truth”)
- **Metrics**: label + evidence scoring, aggregated leaderboards in `results/`
- **Reliability**: inter-model agreement + κ statistics
- **Disagreement triage**: where models flip labels + evidence behavior patterns
- Optional hooks for **LLM-as-a-judge** (when enabled)

> This README is generated from the repo’s current `results/` artifacts (snapshot date: 2025-12-24).

---

## Repo layout

```
.
├── configs/        # model + dataset specs (what to run)
├── data/           # datasets / cached files (keep large files out of git)
├── metrics/        # metric logic (label + evidence scoring)
├── prompts/        # prompt templates
├── results/        # metrics outputs + reliability + disagreement reports
├── runs/           # raw per-example model outputs (.jsonl) — source of truth
├── scripts/        # entrypoints (eval + analytics)
├── run_all_evals.bat
├── run_all_analytics.bat
├── LICENSE
└── requirements / requirements.txt
```

### Key scripts (`scripts/`)
- `eval_one_model.py` — run one model on one dataset → writes `runs/*.jsonl`
- `eval_many_models.py` — run a matrix (from `configs/`)
- `compute_single_model_metrics.py` — score one run file → writes `results/metrics_json/*.json`
- `compute_multiple_model_metrics.py` — score all run files → writes `results/metrics_table.json`
- `compute_reliability.py` — agreement + Cohen’s κ (pairwise) + Fleiss’ κ (overall)
- `analyze_disagreements.py` — label flips + evidence behavior summaries
- `ollama_client.py` — local inference wrapper
- `openai_backend.py` — API inference wrapper (when enabled)
- `judge_runner.py` — optional judge pass (when enabled)
- `make_scifact_oracle_jsonl.py` — generate an “oracle” run for sanity checks

---

## Quickstart (Windows)

### 1) Setup environment
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If your dependency file is named `requirements` (no `.txt`), use:
```bat
pip install -r requirements
```

### 2) (Optional) Install & run Ollama
Install Ollama, then pull at least one model:
```bat
ollama pull llama3.1:8b
ollama pull qwen3:8b
ollama pull deepseek-r1:8b
ollama pull gemma3:4b
```

Confirm:
```bat
ollama list
```

### 3) Run the full pipeline
**Run evals (generate `runs/*.jsonl`):**
```bat
run_all_evals.bat
```

**Run analytics (generate `results/*`):**
```bat
run_all_analytics.bat
```

> The batch scripts are the “golden path” on Windows. If you rename folders (e.g., `results/`), update the `.bat` files and any hard-coded paths inside `scripts/`.

---

## What a “run” is (and why it matters)

A **run file** is a JSONL where **each line is one dataset example** with:
- model metadata (`model`, provider/host, decoding params)
- prompt + response
- extracted prediction (`pred_label`, optionally evidence sentences / ids)
- example id (`id`) so everything can be joined later

Runs are stored under:
- `runs/scifact_dev200_<model>_t0.0.jsonl` (example naming)

**Important:** `runs/` is the source of truth.  
You can re-run analytics any time without re-running inference.

---

## Metrics computed

### Label metrics (3-way classification)
- Accuracy
- Macro-F1 (equal weight for SUPPORTS / REFUTES / NEI)

### Evidence metrics (sentence-level)
- Precision / Recall / F1 over selected evidence sentences

### Evidence behavior diagnostics
Useful for spotting “gaming” patterns:
- `avg_predicted_sentences` — average # sentences cited
- `pct_pred_shotgun` — cites too many sentences (low precision)
- `pct_cite_all_unnecessary` — “cite-all” even when unnecessary
- `pct_empty_when_pred_non_nei` — predicts SUPPORTS/REFUTES but gives no evidence
- `pct_nonempty_when_pred_nei` — gives evidence when predicting NEI
- `correct_label_wrong_evidence` — right label, wrong justification

---

## Current results snapshot (SciFact dev200)

This table comes from `results/metrics_table.json`.

| Model | Label Acc | Macro-F1 | Evidence F1 | Evidence P/R | Avg #sent | Shotgun | Cite-all (unnec.) | Correct label / wrong ev | Errors | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.2_t0.0 | 0.780 | 0.780 | 0.741 | 0.733/0.876 | 1.27 | 3.5% | 0.0% | 3 | 0 | 200 |
| qwen3_8b_t0.0 | 0.690 | 0.656 | 0.664 | 0.649/0.866 | 1.62 | 10.0% | 1.0% | 2 | 0 | 200 |
| llama3.1_8b_t0.0 | 0.450 | 0.409 | 0.331 | 0.330/0.778 | 2.83 | 22.0% | 7.5% | 15 | 0 | 200 |
| deepseek-r1_8b_t0.0 | 0.480 | 0.389 | 0.597 | 0.568/0.779 | 1.65 | 15.2% | 1.0% | 1 | 2 | 198 |
| gemma3_4b_t0.0 | 0.470 | 0.388 | 0.424 | 0.397/0.871 | 2.65 | 23.5% | 2.0% | 5 | 0 | 200 |

**How to read this table**
- If **Label Acc/Macro-F1** is low: the model is misunderstanding the claim–document relationship.
- If **Evidence Recall** is high but **Precision** is low: the model is “shotgunning” citations.
- High **Correct label / wrong ev**: the model guessed the right label but couldn’t justify it correctly.

### Per-label breakdown (top model)
**Per-label P/R/F1 for `gpt-5.2_t0.0`**

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| SUPPORTS | 0.887 | 0.635 | 0.740 | 74 |
| REFUTES | 0.867 | 0.750 | 0.804 | 52 |
| NEI | 0.686 | 0.946 | 0.795 | 74 |

See the full per-label breakdowns in: `results/label_breakdown.json`.

---

## Reliability (do models agree?)

Reliability is computed on the **shared example ids** across runs.

- Runs included: **5**
- Shared IDs across all runs: **198**
- Fleiss’ κ (all runs): **0.161**
- Disagreement items (all runs): **171**

### Highest agreement pairs (by Cohen’s κ)
- **gpt-5.2_t0.0 ↔ qwen3_8b_t0.0**: agreement 68.0%, κ=0.474 (n=200)
- **deepseek-r1_8b_t0.0 ↔ qwen3_8b_t0.0**: agreement 67.7%, κ=0.359 (n=198)
- **gemma3_4b_t0.0 ↔ llama3.1_8b_t0.0**: agreement 67.0%, κ=0.326 (n=200)

### Lowest agreement pairs (by Cohen’s κ)
- **deepseek-r1_8b_t0.0 ↔ gemma3_4b_t0.0**: agreement 27.3%, κ=0.047 (n=198)
- **deepseek-r1_8b_t0.0 ↔ llama3.1_8b_t0.0**: agreement 26.3%, κ=0.091 (n=198)
- **gemma3_4b_t0.0 ↔ gpt-5.2_t0.0**: agreement 37.5%, κ=0.101 (n=200)

See full output in: `results/reliability_dev200.json`.

---

## Disagreement analysis (what’s flipping, and why?)

`analyze_disagreements.py` generates a report that summarizes:
- label flips (e.g., NEI ↔ SUPPORTS/REFUTES, SUPPORTS ↔ REFUTES)
- evidence behavior buckets (`normal`, `empty`, `shotgun`, `cite_all`)
- lightweight “claim flags” among disagreements (numeric / causal / hedged, etc.)

Output:
- `results/disagreement_summary.md`

### Report excerpt
```text
# Disagreement summary (ref = scifact_dev200_gpt-5.2_t0.0)

Shared IDs across gold + 5 runs: **198**

## scifact_dev200_deepseek-r1_8b_t0.0 vs scifact_dev200_gpt-5.2_t0.0

- n: **198**
- label flips: `{'NEI<->(S/R)': 77, 'same': 119, 'SUPPORTS<->REFUTES': 2}`
- evidence behavior: `{'cite_all': 2, 'normal': 85, 'empty': 95, 'shotgun': 16}`
- claim flags among disagreements: `{'numeric': 13, 'causal': 22, 'hedged': 1}`
- correct label but wrong evidence (vs gold): **1**

### Examples: SUPPORTS<->REFUTES

- `scifact_dev_1290_4687948` gold=SUPPORTS  ref=SUPPORTS  scifact_dev200_deepseek-r1_8b_t0.0=REFUTES  other_ev=1 (normal)
  - There is an inverse relationship between hip fractures and statin use.
- `scifact_dev_249_1568684` gold=REFUTES  ref=REFUTES  scifact_dev200_deepseek-r1_8b_t0.0=SUPPORTS  other_ev=2 (normal)
  - Chenodeosycholic acid treatment reduces whole-body energy expenditure.

### Examples: NEI<->(S/R)

- `scifact_dev_475_18678095` gold=SUPPORTS  ref=NEI  scifact_dev200_deepseek-r1_8b_t0.0=SUPPORTS  other_ev=3 (normal)
  - Glycolysis is one of the primary glycometabolic pathways in cells.
- `scifact_dev_100_4381486` gold=SUPPORTS  ref=SUPPORTS  scifact_dev200_deepseek-r1_8b_t0.0=NEI  other_ev=8 (cite_all)
  - All hematopoietic stem cells segregate their chromosomes randomly.
- `scifact_dev_513_13230773` gold=REFUTES  ref=REFUTES  scifact_dev200_deepseek-r1_8b_t0.0=NEI  other_ev=0 (empty)
  - High cardiopulmonary fitness causes increased mortality rate.
- `scifact_dev_274_11614737` gold=REFUTES  ref=REFUTES  scifact_dev200_deepseek-r1_8b_t0.0=NEI  other_ev=5 (normal)
  - Combination nicotine replacement therapies with varenicline or bupropion lead to significantly higher long-term abstinence rates at 52 weeks than varenicline monotherapy.
- `scifact_dev_501_17930286` gold=SUPPORTS  ref=SUPPORTS  scifact_dev200_deepseek-r1_8b_t0.0=NEI  other_ev=8 (shotgun)
```

---

## Adding new models

1) Add/edit the model spec in `configs/` (see existing files for the schema).
2) Ensure the backend supports it:
   - Ollama: `scripts/ollama_client.py`
   - API: `scripts/openai_backend.py` (when enabled)

**Benchmarking best practices**
- Keep `temperature=0.0` for comparability.
- Keep the same prompt template and dataset slice.
- Treat different tags as different models (e.g., `llama3.1:8b` vs `llama3.1:8b_q4_0`).

---

## Troubleshooting

### Batch scripts fail after renaming folders
If you renamed directories (common ones: `runs/`, `results/`, `scripts/`), update:
- `run_all_evals.bat`
- `run_all_analytics.bat`
- any hard-coded paths inside `scripts/*.py`

### Ollama connection issues
If calls to `http://localhost:11434` fail:
- confirm Ollama is running
- confirm the model exists: `ollama list`
- try interactive run: `ollama run llama3.1:8b`

---

## Reproducibility checklist

- Log model tag + decoding params for every record ✅
- Keep `runs/*.jsonl` immutable once created ✅
- Re-run analytics without regenerating runs ✅
- Version prompts (hash or template id) ✅
- Store summary artifacts under `results/` ✅

---

## License
See `LICENSE`.
