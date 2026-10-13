"""
Non-LLM baselines from the v1 paper (zero-shot NLI classifiers), re-run locally on the SAME 527 items as the LLMs.
Same hypothesis templates as the v1 NLI baseline; the actor label set is the v2 one (10 actors).
One NLI classification per category; the label with the highest entailment score wins.
Output has the same format as run_eval.py (results/v2/<tag>/zs_<lang>_r0.jsonl), so analyze.py reads it unchanged.

  ~/.venvs/bfpn-nli/bin/python run_nli_baseline.py            # both models, EN + BN
  ~/.venvs/bfpn-nli/bin/python run_nli_baseline.py --models mdeberta

The premise is cut to the model's 512-token window (the LLMs read the full text): this is a property of the baseline.
"""
import argparse, json, os, time
from transformers import pipeline
from bfpn.schema import CATEGORIES, VALID_LABELS

MODELS = {   # tag -> Hugging Face id (the two baselines reported in v1)
    "mdeberta-nli":  "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "xlmr-base-nli": "symanto/xlm-roberta-base-snli-mnli-anli-xnli",
}
HYPOTHESIS_TEMPLATE = {
    "frame":            "The dominant narrative frame of this text is {}.",
    "actor":            "The central narrative actor in this text is {}.",
    "actor_role":       "The main actor in this text is acting as a {}.",
    "agency":           "The main actor has {} narrative agency in this text.",
    "moral_valence":    "The main actor is portrayed in a {} light in this text.",
    "causal_present":   "Explicit cause-effect language is {} in this text.",
    "policy_direction": "The policy direction described in this text is {}.",
    "alignment":        "The geopolitical alignment of this text is {}.",
}
LABEL_DISPLAY = {
    "frame": {"Humanitarian": "humanitarian", "Security": "security", "Economic": "economic", "Strategic": "strategic", "Normative": "normative"},
    "actor": {"Bangladesh": "Bangladesh", "India": "India", "Pakistan": "Pakistan", "Sri Lanka": "Sri Lanka", "Bhutan": "Bhutan",
              "Myanmar": "Myanmar", "China": "China", "US": "the United States", "UN": "the United Nations", "Other": "another actor"},
    "actor_role": {"Initiator": "initiator", "Respondent": "respondent", "Mediator": "mediator", "Victim": "victim"},
    "agency": {"High": "high", "Medium": "medium", "Low": "low"},
    "moral_valence": {"Positive": "positive", "Neutral": "neutral", "Negative": "negative"},
    "causal_present": {"Yes": "present", "No": "absent"},
    "policy_direction": {"Cooperation": "cooperation", "Pressure": "pressure", "Intervention": "intervention", "Neutrality": "neutrality"},
    "alignment": {"Pro-West": "pro-Western", "Pro-China": "pro-China", "Non-aligned": "non-aligned", "Mixed": "mixed"},
}
for c in CATEGORIES: assert set(LABEL_DISPLAY[c]) == set(VALID_LABELS[c]), c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--langs", nargs="+", default=["en", "bn"]); ap.add_argument("--out", default="results/v2")
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    items = [json.loads(l) for l in open("data/bfpn_v2.jsonl")]
    for tag in a.models:
        clf = pipeline("zero-shot-classification", model=MODELS[tag], device=a.device)
        outdir = os.path.join(a.out, tag); os.makedirs(outdir, exist_ok=True)
        json.dump({"model": MODELS[tag], "tag": tag, "method": "zero-shot NLI, one classification per category", "max_tokens_input": 512},
                  open(os.path.join(outdir, "config.json"), "w"), indent=1)
        for lang in a.langs:
            path = os.path.join(outdir, f"zs_{lang}_r0.jsonl")
            done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
            todo = [it for it in items if it["id"] not in done]; t0 = time.time()
            preds = {it["id"]: {} for it in todo}
            for c in CATEGORIES:
                back = {v: k for k, v in LABEL_DISPLAY[c].items()}
                res = clf([it["text_" + lang] for it in todo], candidate_labels=list(back), hypothesis_template=HYPOTHESIS_TEMPLATE[c],
                          multi_label=False, batch_size=a.batch, truncation="only_first") if todo else []
                for it, r in zip(todo, res): preds[it["id"]][c] = back[r["labels"][0]]
            with open(path, "a") as f:
                for it in todo:
                    p = preds[it["id"]]
                    f.write(json.dumps({"id": it["id"], "model": tag, "condition": "zs", "lang": lang, "run": 0, "schema": "nli-templates",
                                        "pred": p, "parsed": True, "finish": "stop", "chars": len(it["text_" + lang]),
                                        "chars_full": len(it["text_" + lang]), "text_fraction_used": 1.0, "raw": json.dumps(p)}, ensure_ascii=False) + "\n")
            print(f"done {path} ({len(todo)} new, {time.time() - t0:.0f}s)", flush=True)

if __name__ == "__main__": main()
