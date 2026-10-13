"""
Supervised non-LLM baselines, 5-fold cross-validation on the SAME 527 items as the LLMs (folds split by item, so the
English and Bangla version of an item are always in the same fold). Every item is predicted by a model that never saw it.
  majority-class : most frequent training-fold label per category
  tfidf-logreg   : character n-gram TF-IDF + logistic regression, one classifier per category
  xlmr-finetuned : xlm-roberta-base fine-tuned with one output head per category (first 512 tokens)
Output has the same format as run_eval.py (results/v2/<tag>/sup_<lang>_r0.jsonl, condition "sup"), so analyze.py reads it.

  ~/.venvs/bfpn-nli/bin/python run_supervised_baseline.py [--models majority-class tfidf-logreg xlmr-finetuned]
"""
import argparse, json, os, time
import numpy as np
from bfpn.schema import CATEGORIES, VALID_LABELS

SEED, FOLDS = 42, 5

def folds(n):
    idx = np.random.default_rng(SEED).permutation(n)
    return [np.sort(idx[k::FOLDS]) for k in range(FOLDS)]

def run_majority(items, F):
    out = {l: [None] * len(items) for l in ("en", "bn")}
    for te in F:
        tr = np.setdiff1d(np.arange(len(items)), te)
        maj = {c: max(VALID_LABELS[c], key=lambda v: sum(items[i]["gold"][c] == v for i in tr)) for c in CATEGORIES}
        for i in te:
            for l in out: out[l][i] = dict(maj)
    return out

def run_tfidf(items, F):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    out = {l: [dict() for _ in items] for l in ("en", "bn")}
    for l in out:
        X_txt = [it["text_" + l] for it in items]
        for te in F:
            tr = np.setdiff1d(np.arange(len(items)), te)
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=200000)
            Xtr, Xte = vec.fit_transform([X_txt[i] for i in tr]), vec.transform([X_txt[i] for i in te])
            for c in CATEGORIES:
                y = [items[i]["gold"][c] for i in tr]
                p = LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(Xtr, y).predict(Xte)
                for i, v in zip(te, p): out[l][i][c] = str(v)
    return out

def run_xlmr(items, F, epochs=8, lr=2e-5, batch=16, max_len=512):
    import torch, torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base"); dev = "cuda"
    L2I = {c: {v: k for k, v in enumerate(VALID_LABELS[c])} for c in CATEGORIES}
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.enc = AutoModel.from_pretrained("xlm-roberta-base"); s.drop = nn.Dropout(0.1)
            s.heads = nn.ModuleList([nn.Linear(s.enc.config.hidden_size, len(VALID_LABELS[c])) for c in CATEGORIES])
        def forward(s, **x):
            h = s.drop(s.enc(**x).last_hidden_state[:, 0]); return [hd(h) for hd in s.heads]
    def enc(texts): return tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
    out = {l: [dict() for _ in items] for l in ("en", "bn")}
    for k, te in enumerate(F):
        torch.manual_seed(SEED + k); t0 = time.time()
        tr = np.setdiff1d(np.arange(len(items)), te)
        train = [(items[i]["text_" + l], [L2I[c][items[i]["gold"][c]] for c in CATEGORIES]) for i in tr for l in ("en", "bn")]
        net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * ((len(train) + batch - 1) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy="linear")
        rng = np.random.default_rng(SEED + k); net.train()
        for ep in range(epochs):
            order = rng.permutation(len(train))
            for b in range(0, len(order), batch):
                bt = [train[j] for j in order[b:b + batch]]; y = torch.tensor([t[1] for t in bt], device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = net(**enc([t[0] for t in bt]))
                loss = sum(nn.functional.cross_entropy(lg.float(), y[:, j]) for j, lg in enumerate(logits))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            for l in ("en", "bn"):
                for b in range(0, len(te), 32):
                    ids = te[b:b + 32]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = net(**enc([items[i]["text_" + l] for i in ids]))
                    for j, c in enumerate(CATEGORIES):
                        for i, a in zip(ids, logits[j].float().argmax(1).tolist()): out[l][i][c] = VALID_LABELS[c][a]
        print(f"  xlmr fold {k + 1}/{FOLDS} done ({time.time() - t0:.0f}s, last loss {loss.item():.2f})", flush=True)
        del net, opt; torch.cuda.empty_cache()
    return out

RUNNERS = {"majority-class": run_majority, "tfidf-logreg": run_tfidf, "xlmr-finetuned": run_xlmr}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=list(RUNNERS), choices=list(RUNNERS))
    ap.add_argument("--out", default="results/v2"); a = ap.parse_args()
    items = [json.loads(l) for l in open("data/bfpn_v2.jsonl")]; F = folds(len(items))
    for tag in a.models:
        t0 = time.time(); preds = RUNNERS[tag](items, F)
        outdir = os.path.join(a.out, tag); os.makedirs(outdir, exist_ok=True)
        json.dump({"tag": tag, "method": "5-fold cross-validation by item, out-of-fold predictions", "seed": SEED}, open(os.path.join(outdir, "config.json"), "w"), indent=1)
        for l in ("en", "bn"):
            with open(os.path.join(outdir, f"sup_{l}_r0.jsonl"), "w") as f:
                for it, p in zip(items, preds[l]):
                    f.write(json.dumps({"id": it["id"], "model": tag, "condition": "sup", "lang": l, "run": 0, "schema": "supervised-cv",
                                        "pred": p, "parsed": True, "finish": "stop", "chars": len(it["text_" + l]), "chars_full": len(it["text_" + l]),
                                        "text_fraction_used": 1.0, "raw": json.dumps(p)}, ensure_ascii=False) + "\n")
        print(f"done {tag} ({time.time() - t0:.0f}s)", flush=True)

if __name__ == "__main__": main()
