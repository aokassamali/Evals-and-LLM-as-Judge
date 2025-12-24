# LLM-as-Judge Reliability Study

How reliable are LLMs when used as evaluators? This project measures inter-model agreement and failure modes across local and frontier models on a scientific claim verification task.

## Key Findings

- **Inter-model agreement is poor.** Fleiss' kappa across all 5 models is 0.16 (slight agreement). Even the best-performing pair (GPT-5.2 ↔ Qwen3) only reaches κ = 0.47.
- **Model quality matters more than size.** Qwen3 8B (69%) substantially outperforms other similar-sized local models (45-48%).-
- **Evidence quality correlates with model capability.** GPT-5.2 shows calibrated citation, Qwen3 shows moderate shotgunning (10%), and smaller models cite excessively (20-25%).
- **Same-sized models are not interchangeable.** Among 8B models, accuracy ranges from 45% (Llama3.1) to 69% (Qwen3) - a 24-point spread that's comparable to the 9-point gap between Qwen3 and frontier GPT-5.2. Model selection matters more than parameter count.

## Results

### Model Performance (n=200 claims)

| Model | Accuracy | Macro F1 | Evidence P | Evidence R | Evidence F1 |
|-------|----------|----------|------------|------------|-------------|
| GPT-5.2 | **0.780** | **0.780** | **0.733** | **0.876** | **0.741** |
| Qwen3 8B | 0.690 | 0.656 | 0.649 | 0.866 | 0.664 |
| DeepSeek-R1 8B | 0.480 | 0.389 | 0.568 | 0.779 | 0.597 |
| Gemma3 4B | 0.470 | 0.388 | 0.397 | 0.871 | 0.424 |
| Llama3.1 8B | 0.450 | 0.409 | 0.330 | 0.778 | 0.331 |

### Inter-Model Agreement (Cohen's Kappa)

|  | DeepSeek | Gemma3 | GPT-5.2 | Llama3.1 | Qwen3 |
|--|----------|--------|---------|----------|-------|
| DeepSeek | — | 0.05 | 0.25 | 0.09 | 0.36 |
| Gemma3 | | — | 0.10 | 0.33 | 0.21 |
| GPT-5.2 | | | — | 0.11 | **0.47** |
| Llama3.1 | | | | — | 0.22 |

Fleiss' kappa (all 5 models): **0.161**

### Failure Patterns

| Model | % Empty Evidence (non-NEI) | % Cite-All (unnecessary) | % Shotgun | Correct Label, Wrong Evidence |
|-------|---------------------------|--------------------------|-----------|------------------------------|
| GPT-5.2 | 0.0% | 0.0% | 3.5% | 3 |
| Qwen3 8B | 0.0% | 1.0% | 10.0% | 2 |
| DeepSeek-R1 8B | 0.0% | 1.0% | 15.2% | 1 |
| Gemma3 4B | 0.0% | 2.0% | 23.5% | 5 |
| Llama3.1 8B | 0.0% | 7.5% | 22.0% | 15 |

## Methodology

### Task: Scientific Claim Verification

Given a claim and evidence sentences from a scientific abstract, classify as:
- **SUPPORTS**: Evidence directly supports the claim
- **REFUTES**: Evidence contradicts the claim  
- **NEI**: Not enough information to determine

Models must also cite which sentence IDs support their judgment.

### Dataset

[SciFact](https://github.com/allenai/scifact) dev set, 200 randomly sampled claims with oracle evidence (gold document provided). This isolates the reasoning task from retrieval.

### Models Evaluated

**Local (via Ollama):**
- Llama 3.1 8B
- Qwen3 8B
- DeepSeek-R1 8B
- Gemma3 4B

**Frontier (via OpenAI API):**
- GPT-5.2

All runs at temperature=0.0 for reproducibility.

### Metrics

**Label metrics:** Accuracy, macro F1, per-class precision/recall/F1

**Evidence metrics:** Sentence-level precision, recall, F1 against gold evidence

**Reliability metrics:** 
- Pairwise percent agreement and Cohen's kappa
- Fleiss' kappa across all models
- Disagreement analysis by claim type (numeric, causal, hedged)

**Failure diagnostics:**
- `% empty_when_not_NEI`: Model predicts SUPPORTS/REFUTES but cites no evidence
- `% cite_all_unnecessary`: Model cites every sentence when fewer would suffice
- `% shotgun`: Over-citation with low precision (>1.5× gold sentences, <70% precision)
- `correct_label_wrong_evidence`: Right answer, wrong reasoning

## Project Structure

```
├── data/
│   └── scifact_oracle_dev_200.jsonl    # Evaluation dataset
├── prompts/
│   └── scifact_judge.txt               # Prompt template
├── scripts/
│   ├── make_scifact_oracle_jsonl.py    # Dataset preparation
│   ├── eval_one_model.py               # Single model evaluation
│   ├── eval_many_models.py             # Batch evaluation runner
│   ├── compute_single_model_metrics.py # Per-model metrics
│   ├── compute_multiple_model_metrics.py # Aggregate metrics table
│   ├── compute_reliability.py          # Inter-rater agreement
│   └── analyze_disagreements.py        # Failure mode analysis
├── runs/                               # Raw model outputs (JSONL)
├── results/
│   ├── metrics_table.md                # Summary performance table
│   ├── reliability_dev200.json         # Agreement statistics
│   └── disagreement_summary.md         # Qualitative error analysis
├── run_all_evals.bat                   # Run all model evaluations
└── run_all_analytics.bat               # Compute all metrics
```

## Reproducing Results

### Prerequisites

```bash
pip install requests pydantic scikit-learn openai
```

For local models, install [Ollama](https://ollama.ai) and pull:
```bash
ollama pull llama3.1:8b
ollama pull qwen3:8b
ollama pull deepseek-r1:8b
ollama pull gemma3:4b
```

### 1. Prepare Dataset

```bash
python scripts/make_scifact_oracle_jsonl.py --split dev --n 200 --seed 0
```

### 2. Run Evaluations

```bash
# Local models
python scripts/eval_many_models.py \
    --models llama3.1:8b qwen3:8b deepseek-r1:8b gemma3:4b \
    --input data/scifact_oracle_dev_200.jsonl \
    --out_dir runs \
    --temperature 0.0

# Frontier model (requires OPENAI_API_KEY)
python scripts/eval_one_model.py \
    --backend openai \
    --model gpt-5.2 \
    --input data/scifact_oracle_dev_200.jsonl \
    --output runs/scifact_dev200_gpt-5.2_t0.0.jsonl
```

### 3. Compute Metrics

```bash
# Or run everything:
./run_all_analytics.bat
```

Individual scripts:
```bash
# Performance metrics
python scripts/compute_multiple_model_metrics.py \
    --gold data/scifact_oracle_dev_200.jsonl \
    --runs_dir runs

# Inter-model reliability
python scripts/compute_reliability.py \
    --runs_dir runs \
    --out results/reliability_dev200.json

# Disagreement analysis
python scripts/analyze_disagreements.py \
    --gold data/scifact_oracle_dev_200.jsonl \
    --runs_dir runs \
    --ref scifact_dev200_gpt-5.2_t0.0
```

## Implications for LLM-as-Judge

1. **Don't assume agreement.** If you're using LLM judges for evaluation, different models will give systematically different scores. Inter-rater reliability should be measured and reported.

2. **Local models have systematic biases.** The SUPPORTS over-prediction suggests local models may inflate positive evaluations in other judge tasks too.

3. **Evidence grounding matters.** Models that get the right label often cite wrong evidence—they may be pattern-matching rather than reasoning.

4. **Frontier models aren't perfect either.** GPT-5.2 still disagrees with ground truth 22% of the time on this structured task.

## License

MIT

## Citation

If you use this work, please cite:

```
@misc{kassamali2024llmjudge,
  author = {Kassamali, Asad},
  title = {LLM-as-Judge Reliability Study},
  year = {2024},
  url = {https://github.com/aokassamali/Evals-and-LLM-as-Judge}
}
```
