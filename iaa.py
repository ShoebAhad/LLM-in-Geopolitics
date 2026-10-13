"""
Inter-annotator agreement on an independently re-annotated subsample (the gold labels in bfpn_v2 are adjudicated, so
agreement has to be measured on labels given WITHOUT seeing gold or the other annotator).
  python iaa.py sample [--n 120] [--min-per-label 8]     -> data/iaa_sample.xlsx   (texts only, no labels; hand to annotators)
  python iaa.py score A.xlsx [B.xlsx]                    -> results/v2/analysis/T15_iaa.csv + a LaTeX table on stdout
With one file the comparison is annotator vs adjudicated gold; with two it is A vs B (plus each vs gold).
Reports raw agreement, Cohen's kappa (item bootstrap CI) and Gwet's AC1 (kappa is depressed by the skewed labels).
"""
import argparse, json, random, sys
import numpy as np, pandas as pd
from bfpn.schema import CATEGORIES, VALID_LABELS, select_exemplars

DATA, SHEET, EXCLUDE = "data/bfpn_v2.jsonl", "data/iaa_sample.xlsx", {"bfpn_1719f47ad4"}

def scored_items():
    items = [json.loads(l) for l in open(DATA)]
    dev = [i for i in items if i["split"] == "dev"]
    ex = {e["id"] for m in ("default", "balanced") for e in select_exemplars(dev, mode=m)}
    return [i for i in items if i["id"] not in ex | EXCLUDE]

def sample(n, k, seed):
    items = scored_items(); rng = random.Random(seed); rng.shuffle(items)
    chosen = {}
    for c in CATEGORIES:
        for lab in VALID_LABELS[c]:
            pool = [i for i in items if i["gold"][c] == lab]
            have = sum(i["gold"][c] == lab for i in chosen.values())
            for i in pool:
                if have >= min(k, len(pool)): break
                if i["id"] not in chosen: chosen[i["id"]] = i; have += 1
    for i in items:
        if len(chosen) >= n: break
        chosen.setdefault(i["id"], i)
    rows = sorted(chosen.values(), key=lambda i: i["id"])
    df = pd.DataFrame([{"id": i["id"], "text": i["text_" + i["original_language"]], **{c: "" for c in CATEGORIES}} for i in rows])
    with pd.ExcelWriter(SHEET) as w:
        df.to_excel(w, sheet_name="annotate", index=False)
        pd.DataFrame([{"category": c, "allowed_labels": " | ".join(VALID_LABELS[c])} for c in CATEGORIES]).to_excel(w, sheet_name="labels", index=False)
    print(f"{len(df)} items -> {SHEET} (texts in the original language, gold withheld)")
    for c in CATEGORIES: print(f"  {c:17s}", dict(pd.Series([i['gold'][c] for i in rows]).value_counts()))

def read(path):
    df = pd.read_excel(path, sheet_name="annotate").set_index("id")
    out = {}
    for c in CATEGORIES:
        canon = {v.lower(): v for v in VALID_LABELS[c]}
        s = df[c].map(lambda v: canon.get(str(v).strip().lower()))
        if s.isna().any(): sys.exit(f"{path}: {int(s.isna().sum())} missing/out-of-schema labels in '{c}': {sorted(set(df[c][s.isna()].astype(str)))}")
        out[c] = s
    return pd.DataFrame(out)

def kappa(a, b, w=None):
    w = np.ones(len(a)) if w is None else w; labs = sorted(set(a) | set(b))
    po = (w * (a == b)).sum() / w.sum()
    pa = np.array([(w * (a == l)).sum() for l in labs]) / w.sum(); pb = np.array([(w * (b == l)).sum() for l in labs]) / w.sum()
    pe = (pa * pb).sum()
    return (po - pe) / (1 - pe) if pe < 1 else np.nan

def ac1(a, b, q):
    po = (a == b).mean(); pi = np.array([((a == l).mean() + (b == l).mean()) / 2 for l in set(a) | set(b)])
    pe = (pi * (1 - pi)).sum() / (q - 1)
    return (po - pe) / (1 - pe)

def score(paths, B, seed):
    gold = pd.DataFrame({i["id"]: i["gold"] for i in scored_items()}).T
    ann = [read(p) for p in paths]
    ids = sorted(set.intersection(*[set(a.index) for a in ann]) & set(gold.index)); n = len(ids)
    pairs = [("A vs B", ann[0], ann[1]), ("A vs gold", ann[0], gold), ("B vs gold", ann[1], gold)] if len(ann) == 2 else [("annotator vs gold", ann[0], gold)]
    W = np.random.default_rng(seed).multinomial(n, np.ones(n) / n, size=B).astype(float)
    rows = []
    for name, x, y in pairs:
        for c in CATEGORIES:
            a, b = x.loc[ids, c].values.astype(object), y.loc[ids, c].values.astype(object)
            bs = np.array([kappa(a, b, w) for w in W]); lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            rows.append(dict(pair=name, category=c, n=n, agreement=(a == b).mean(), kappa=kappa(a, b), lo=lo, hi=hi, ac1=ac1(a, b, len(VALID_LABELS[c]))))
    T = pd.DataFrame(rows); T.to_csv("results/v2/analysis/T15_iaa.csv", index=False)
    print(T.round(3).to_string(index=False), "\n")
    main = T[T.pair == pairs[0][0]]
    print("\\begin{tabular}{lrrrr}\n\\toprule\nCategory & Labels & Agree. & $\\kappa$ [95\\% CI] & AC1 \\\\\n\\midrule")
    for r in main.itertuples():
        print(f"{r.category.replace('_', ' ').capitalize()} & {len(VALID_LABELS[r.category])} & {100 * r.agreement:.1f} & {r.kappa:.2f} [{r.lo:.2f}, {r.hi:.2f}] & {r.ac1:.2f} \\\\")
    print(f"\\midrule\nMean & & {100 * main.agreement.mean():.1f} & {main.kappa.mean():.2f} & {main.ac1.mean():.2f} \\\\\n\\bottomrule\n\\end{{tabular}}\n% {pairs[0][0]}, n = {n} items")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--n", type=int, default=120); s.add_argument("--min-per-label", type=int, default=8); s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score"); s.add_argument("files", nargs="+"); s.add_argument("--B", type=int, default=10000); s.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    sample(a.n, a.min_per_label, a.seed) if a.cmd == "sample" else score(a.files[:2], a.B, a.seed)
