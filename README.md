# Who is the Actor? LLM Geopolitical Bias in South Asian Foreign Policy Narratives

Code, data and raw model outputs for a study of geopolitical bias in LLMs that annotate Bangladesh / South Asian
foreign-policy news (BFPN). Every news item has an English and a Bangla version and 8 gold labels. Models are prompted
to produce the same 8 labels, and the analysis tests whether their errors are *directional* (bias) rather than noise.

- `EXPERIMENT_PROTOCOL.md` — data, prompt settings, metrics and statistics.
- `CODEBOOK.md` — the annotation codebook; the prompt uses its wording.

## Repository layout

| Path | Content |
|---|---|
| `Foreign Annotation Data Multisource.xlsx` | raw annotation sheet |
| `data/bfpn_v2.jsonl` | cleaned dataset, 527 items: `id`, `text_en`, `text_bn`, `original_language`, `gold` (8 labels), `actor_raw`, `split` |
| `data/cleaning_report.json` | what `clean_data.py` dropped and why |
| `bfpn/schema.py` | label sets, prompt text, prompt conditions, exemplar selection, output parser |
| `bfpn/metrics.py` | bias metrics (gold-conditional, per-item numerator / denominator) |
| `clean_data.py` | raw sheet -> `data/bfpn_v2.jsonl` |
| `run_eval.py` | LLM runner (OpenAI-compatible gateway / vLLM, OpenAI, Anthropic); resumable |
| `run_model.sh` | the full experiment for one model |
| `check_complete.py` | checks that every planned run of a model covers all 527 items |
| `run_nli_baseline.py`, `run_supervised_baseline.py` | zero-shot NLI and 5-fold CV supervised baselines |
| `reparse.py` | re-parse stored raw outputs after a parser change |
| `analyze.py` | all statistics -> `results/v2/analysis/T*.csv` |
| `make_figures_v2.py` | figures -> `figures/v2/` |
| `results/v2/<tag>-codebook/`, `results/v2/<tag>-balex/` | raw outputs and parsed predictions per model (`<cond>_<lang>_r<run>.jsonl`) |
| `results/v2/analysis/` | the tables reported in the paper |

## Annotation schema

Eight categories, one label each: `frame`, `actor`, `actor_role`, `agency`, `moral_valence`, `causal_present`,
`policy_direction`, `alignment`. Definitions and label sets are in `CODEBOOK.md` and `bfpn/schema.py`.

## Bias metrics

All metrics are conditional on the gold label, so they are not driven by the label skew of the corpus.

| Metric | Definition |
|---|---|
| **ADS** (alignment distortion) | P(pred in {Pro-West, Pro-China} given gold = Non-aligned) |
| **ASI** (agency skew) | P(pred actor in {India, China, US, UN} given gold actor not in that set) |
| **ASI-India** | P(pred actor = India given gold actor is a smaller South Asian state) |
| **CDR** (causal detection error) | error on `causal_present`; CDR-FN = P(No given gold Yes), CDR-FP = P(Yes given gold No) |

A raw error rate is not evidence of bias. Bias is claimed only when errors are directional: errors *into* a target
label set are compared with errors *out of* it (exact binomial test, Holm-corrected).

## Models

| Tag | Model | Provider |
|---|---|---|
| `gemma4-31b` | gemma-4-31b-it | local, vLLM |
| `qwen38-27b` | Qwen3.8-27B (thinking disabled) | local, vLLM |
| `ministral3-14b` | ministral-3-14b | local, vLLM |
| `gpt-oss-20b` | gpt-oss-20b (lowest reasoning effort) | local, vLLM |
| `llama31-8b` | llama-3.1-8b-instruct | local, vLLM |
| `haiku55` | claude-haiku-5-5 (thinking disabled) | Anthropic API |
| `gpt6-luna` | gpt-6-luna (reasoning effort none) | OpenAI API |

Baselines: `mdeberta-nli`, `xlmr-base-nli` (zero-shot NLI), `majority-class`, `tfidf-logreg` (5-fold CV).
The exact arguments of every run are stored in `results/v2/<tag>-*/config.json`.

## Reproducing the results

```bash
pip install -r requirements.txt openpyxl pandas scipy statsmodels
```

### 1. Tables and figures from the stored model outputs (no GPU, no API key)

```bash
python analyze.py --include-dev --exclude bfpn_1719f47ad4 --B 10000    # -> results/v2/analysis/T*.csv
python make_figures_v2.py                                              # -> figures/v2/fig*.{png,pdf}
```

This scores 518 items: all 527, minus the 8 few-shot exemplar items, minus one faulty item (`bfpn_1719f47ad4`).

### 2. Rebuilding the dataset

```bash
python clean_data.py        # raw sheet -> data/bfpn_v2.jsonl + data/cleaning_report.json
```

### 3. Running a model

`run_model.sh` runs all prompt settings and repeated runs for one model, retries until every item is answered, and
then re-runs the analysis over all models. It expects an OpenAI-compatible endpoint (vLLM or a LiteLLM gateway):

```bash
BASE_URL=http://localhost:8000/v1 ./run_model.sh <tag> <served-model-name> ['<extra-body json>']
PROVIDER=anthropic WORKERS=8 ./run_model.sh haiku55 claude-haiku-5-5 '{"thinking":{"type":"disabled"}}'
python check_complete.py <tag>
```

A small test run without the wrapper:

```bash
python run_eval.py --model <name> --base-url http://localhost:8000/v1 --tag smoke --conditions zs --langs en --limit 5 --out results/scratch
BFPN_RESULTS=results/scratch python analyze.py --limit 5
```

Run 0 is greedy (temperature 0) and is the reported result; later runs are sampled and only feed the variance table.
Hosted APIs are not deterministic, so re-running a closed model will not give byte-identical outputs.

### 4. Baselines

```bash
python run_nli_baseline.py          # mdeberta-nli, xlmr-base-nli
python run_supervised_baseline.py   # majority-class, tfidf-logreg, xlmr-finetuned (the last needs CUDA)
```

## API keys

Keys are read from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) or from files outside the repository
(`~/.bfpn_api_key` for the gateway, `~/.anthropic_api_key`). Never commit them.
