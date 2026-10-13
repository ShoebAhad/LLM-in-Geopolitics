# Experimental Protocol (paper-ready)

Every model is run with one command and receives exactly the same items, prompts and runs:

    PY=~/.venvs/bfpn/bin/python ./run_model.sh <tag> <gateway-model-name> ['<extra-body json>']

## 1. Data

| | Items |
|---|---|
| Annotated news items (each has all 8 labels; English and Bangla version of the same text) | 527 |
| Held out as few-shot exemplars (never scored) | 8 |
| Removed (one faulty item) | 1 |
| **Scored for every model and every prompt setting** | **518** |

Gold labels are a two-annotator consensus following the annotation codebook (`CODEBOOK.md`).
The prompt uses the codebook wording verbatim, so models and annotators receive the same definitions.

Categories (8): frame, actor, actor role, agency, moral valence, causal link present, policy direction, alignment.

## 2. Prompt settings (2 x 2)

| Setting | Exemplars | Chain-of-thought | Runs per model |
|---|---|---|---|
| Zero-shot (primary) | no | no | 5 |
| Few-shot | 5 | no | 1 + 1 with a second exemplar set |
| Zero-shot + CoT | no | yes | 1 |
| Few-shot + CoT | 5 | yes | 3 |

- Run 0 is deterministic (temperature 0, seed 0) and is the reported result.
- Additional runs use temperature 0.7 with seeds 1, 2, ... and are used only to report run-to-run variation (mean, SD, min, max).
- The second exemplar set covers all alignment labels ("balanced") and tests sensitivity to the choice of exemplars.
- Output is a JSON object with one label per category; labels outside the schema are counted as errors.
- Output budget: 4,000 tokens (16,000 with CoT). Context window: 32,768 tokens for every model.

Total per model: 11 runs x 527 items x 2 text versions = 11,594 model calls.

## 3. Models

All models are open-weight, served locally with vLLM on one RTX 5090 (32 GB), one model at a time.

| Tag | Weights | Reasoning mode |
|---|---|---|
| qwen38-27b | sakamakismile/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4 (4-bit community build) | thinking disabled |
| gpt-oss-20b | openai/gpt-oss-20b | built-in reasoning at the lowest effort (cannot be disabled) |
| (next models) | | |

### Non-LLM baselines

Two zero-shot NLI classifiers from the earlier version of the paper, re-run locally on the same 527 items (`run_nli_baseline.py`):
`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (tag `mdeberta-nli`) and `symanto/xlm-roberta-base-snli-mnli-anli-xnli` (tag `xlmr-base-nli`).
One NLI classification per category with a fixed hypothesis template; the label with the highest entailment score is taken.
They have a single setting (zero-shot, deterministic) and read only the first 512 tokens of each text.

Supervised baselines (`run_supervised_baseline.py`), 5-fold cross-validation by item so that every item is predicted by a model
that never saw it: `majority-class` (most frequent training label), `tfidf-logreg` (character n-gram TF-IDF + logistic regression,
one classifier per category) and `xlmr-finetuned` (xlm-roberta-base with one output head per category, first 512 tokens).
These use the gold labels for training, so they are a reference for how learnable the labels are, not a zero-shot competitor.

## 4. Metrics

All bias metrics are conditional on the gold label, so they are not affected by the label skew of the corpus.

| Metric | Definition | Scored items |
|---|---|---|
| Accuracy | mean accuracy over the 8 categories | 518 |
| ADS (alignment distortion) | P(pred in {Pro-West, Pro-China} given gold = Non-aligned) | 309 |
| ASI (agency skew) | P(pred actor in {India, China, US, UN} given gold actor not in that set) | about 290 |
| ASI-India | P(pred actor = India given gold actor is a smaller South Asian state) | about 253 |
| CDR (causal detection error) | error on "causal link present"; CDR-FN = P(No given gold Yes), CDR-FP = P(Yes given gold No) | 518 (308 / 210) |

A raw error rate is not evidence of bias. Bias is claimed only when errors are **directional**:
for each metric we count errors *into* the target label set and errors *out of* it and test the split against 50/50
(exact binomial test, Holm-corrected).

## 5. Statistics

- 95% confidence intervals: item-level bootstrap, 10,000 resamples.
- Between-model and between-setting differences: paired bootstrap on the same items, Holm-corrected.
- Prompt effects (exemplars, CoT, interaction): logistic GEE clustered on item.
- Run-to-run variation: mean, SD, min and max over the sampled runs.

## 6. Reported results

1. Main table: accuracy, ADS, ASI, ASI-India, CDR-FN, CDR-FP per model, zero-shot and few-shot, with 95% CIs.
2. Direction-of-error table: into vs out of each target set, with corrected p-values.
3. Prompt-setting table: effect of exemplars and of chain-of-thought.
4. Robustness: run-to-run variation and the second exemplar set.
5. Shared errors across models (`T13_shared_errors.csv`): share of items every model gets wrong, compared with the share expected
   if the models erred independently (product of their error rates), and the share where they give the same wrong label.
6. Direction of shared errors (`T14_shared_error_direction.csv`): for each label pair, items where every model turns gold A into B
   versus gold B into A (exact binomial test, Holm-corrected). An item is counted once.

## 7. Limitations to state

- Small classes (Pro-China 31, Mixed 43, Myanmar 7, UN 5) cannot support class-level claims.
- The alignment result depends on the prompt setting and on the exemplars; it is reported per setting, not as a fixed model trait.
- The Qwen model is a quantized community build, not the official release.
- Reasoning cannot be switched off for gpt-oss-20b (run at the lowest effort), so reasoning mode differs between models.
- One region (South Asia) and one codebook.
