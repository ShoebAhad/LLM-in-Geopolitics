"""
Statistics for the revised paper. Reads data/bfpn_v2.jsonl + results/v2/<tag>/*.jsonl, writes CSVs to results/v2/analysis/.
All uncertainty is ITEM-CLUSTER bootstrap (an item's EN+BN rows move together; models share the same resamples so
model differences are paired).  Run 0 (greedy) is the primary prediction; runs >=1 feed the variance table.
  python analyze.py [--B 5000] [--conds zs fs_cot]
"""
import argparse, itertools, json, os, glob
import numpy as np, pandas as pd
from scipy.stats import binomtest
from bfpn.schema import CATEGORIES, VALID_LABELS, extract_json, validate
from bfpn.metrics import METRICS, ind_ACC, rate, boot_weights, boot_rate, ci

FLOWS = {
    "major_power_attribution": ("actor", {"India", "China", "US", "UN"}, "toward India/China/US/UN"),
    "india_attribution": ("actor", {"India"}, "toward India"),
    "polarization": ("alignment", {"Pro-West", "Pro-China"}, "toward Pro-West/Pro-China"),
    "pro_west": ("alignment", {"Pro-West"}, "toward Pro-West"),
    "non_aligned": ("alignment", {"Non-aligned"}, "toward Non-aligned"),
    "strategic_default": ("frame", {"Strategic"}, "toward Strategic"),
    "causal_yes": ("causal_present", {"Yes"}, "toward Yes (FP) vs No (FN)"),
}
R = os.environ.get("BFPN_RESULTS", "results/v2"); OUT = f"{R}/analysis"

def load(items, conds=None):
    ids = [i["id"] for i in items]; pos = {k: n for n, k in enumerate(ids)}
    gold = {c: np.array([i["gold"][c] for i in items], dtype=object) for c in CATEGORIES}
    preds, seen, st = {}, {}, {}
    for f in glob.glob(f"{R}/*/*_r*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if (conds and r["condition"] not in conds) or r["id"] not in pos: continue
            key = (r["model"], r["condition"], r["lang"], r["run"])
            d = preds.setdefault(key, {c: np.array([None] * len(ids), dtype=object) for c in CATEGORIES})
            seen.setdefault(key, set()).add(r["id"])
            pred = validate(extract_json(r["raw"])) if r.get("raw") else r["pred"]
            for c in CATEGORIES: d[c][pos[r["id"]]] = pred.get(c)
            s = st.setdefault(key, dict(rows=0, unparsed=0, truncated=0, input_shortened=0))
            s["rows"] += 1; s["unparsed"] += (extract_json(r["raw"]) is None) if r.get("raw") else (not r.get("parsed", True))
            s["truncated"] += r.get("finish") == "length"; s["input_shortened"] += r.get("text_fraction_used", 1.0) < 1.0
    incomplete = [k for k in preds if len(seen[k]) < len(ids)]
    for k in incomplete: print("skipping incomplete config", k, f"{len(seen[k])}/{len(ids)}")
    integ = []
    for k in sorted(preds):
        miss = len(ids) - len(seen[k])
        integ.append(dict(model=k[0], cond=k[1], lang=k[2], run=k[3], expected=len(ids), rows=st[k]["rows"], unique_ids=len(seen[k]),
                          duplicates=st[k]["rows"] - len(seen[k]), missing=miss, unparsed=int(st[k]["unparsed"]),
                          truncated=int(st[k]["truncated"]), input_shortened=int(st[k]["input_shortened"]),
                          **{f"invalid_{c}": int(sum(x is None for x in preds[k][c])) - miss for c in CATEGORIES},
                          used_in_analysis=k not in incomplete))
    return gold, {k: v for k, v in preds.items() if k not in incomplete}, integ

def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1, run)
    return adj

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--conds", nargs="*"); ap.add_argument("--include-dev", action="store_true", help="use all items except the 5 few-shot exemplars"); ap.add_argument("--exclude", nargs="*", default=[], help="item ids to drop (e.g. bfpn_1719f47ad4)"); ap.add_argument("--limit", type=int, default=0, help="first N test items (pilot)"); ap.add_argument("--meta", default="models_meta.json")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    items = [json.loads(l) for l in open("data/bfpn_v2.jsonl")]
    from bfpn.schema import select_exemplars
    dev_pool = [i for i in items if i["split"] == "dev"]
    ex_ids = {e["id"] for m_ in ("default", "balanced") for e in select_exemplars(dev_pool, mode=m_)}
    items = [i for i in items if (i["id"] not in ex_ids) if (a.include_dev or i["split"] == "test") if i["id"] not in set(a.exclude)]
    if a.limit: items = items[:a.limit]
    gold, P, integ = load(items, a.conds); n = len(items)
    for c in CATEGORIES: assert set(gold[c]) <= set(VALID_LABELS[c]), f"gold label outside schema in {c}: {set(gold[c]) - set(VALID_LABELS[c])}"
    T0 = pd.DataFrame(integ); T0.to_csv(f"{OUT}/T0_data_integrity.csv", index=False)
    W = boot_weights(n, a.B, seed=42)
    models = sorted({k[0] for k in P}); conds = sorted({k[1] for k in P})
    prim = {k: v for k, v in P.items() if k[3] == 0}
    print(f"{n} test items | {len(models)} models | conds {conds} | B={a.B}")

    rows = []
    for (m, c, l, r), p in sorted(prim.items()):
        for name, (fn, cat) in {**METRICS, "ACC_all8": (None, None)}.items():
            if name == "ACC_all8":
                num = np.mean([ind_ACC(gold[x], p[x])[0] for x in CATEGORIES], axis=0); den = np.ones(n)
            else:
                num, den = fn(gold[cat], p[cat])
            bs = boot_rate(num, den, W); lo, hi = ci(bs)
            rows.append(dict(model=m, cond=c, lang=l, metric=name, value=rate(num, den), lo=lo, hi=hi,
                             num=int(num.sum()), den=int(den.sum())))
    T1 = pd.DataFrame(rows); T1.to_csv(f"{OUT}/T1_main.csv", index=False)

    rows = []
    for (m, c, l, r), p in sorted(prim.items()):
        for name, (cat, S, lab) in FLOWS.items():
            g_, p_ = gold[cat], p[cat]; v = np.array([x is not None for x in p_])
            gi = np.array([x in S for x in g_]); pi = np.array([x in S for x in p_])
            into = int((~gi & pi & v).sum()); out = int((gi & ~pi & v).sum())
            pv = binomtest(into, into + out, 0.5).pvalue if into + out else 1.0
            rows.append(dict(model=m, cond=c, lang=l, flow=name, label=lab, into=into, out_of=out,
                             net=into - out, share_into=into / (into + out) if into + out else np.nan, p_binom=pv))
    A = pd.DataFrame(rows)
    if len(A): A["p_holm"] = holm(A.p_binom.values)
    A.to_csv(f"{OUT}/T2_directional_asymmetry.csv", index=False)

    rows, prs = [], []
    for (m, c, l, r), p in sorted(prim.items()):
        for cat in CATEGORIES:
            g_, p_ = gold[cat], p[cat]; v = np.array([x is not None for x in p_])
            err = int(((g_ != p_) & v).sum()); tin = tout = tpair = 0
            for L in VALID_LABELS[cat]:
                gi, pi = g_ == L, p_ == L
                into, out = int((~gi & pi).sum()), int((gi & ~pi & v).sum()); tin += into; tout += out
                d = (W @ (pi.astype(np.float32) - gi.astype(np.float32))) / n * 100; lo, hi = ci(d)
                rows.append(dict(model=m, cond=c, lang=l, category=cat, label=L, gold_n=int(gi.sum()), pred_n=int(pi.sum()),
                                 correct=int((gi & pi).sum()), into=into, out_of=out, invalid=int((gi & ~v).sum()), net=into - out,
                                 share_into=into / (into + out) if into + out else np.nan,
                                 recall=(gi & pi).sum() / gi.sum() if gi.sum() else np.nan,
                                 precision=(gi & pi).sum() / pi.sum() if pi.sum() else np.nan,
                                 shift_pp=(int(pi.sum()) - int(gi.sum())) / n * 100, lo=lo, hi=hi,
                                 p_binom=binomtest(into, into + out, 0.5).pvalue if into + out else 1.0))
            for x, y in itertools.combinations(VALID_LABELS[cat], 2):
                xy, yx = int(((g_ == x) & (p_ == y)).sum()), int(((g_ == y) & (p_ == x)).sum()); tpair += xy + yx
                nx, ny = int(((g_ == x) & v).sum()), int(((g_ == y) & v).sum())
                prs.append(dict(model=m, cond=c, lang=l, category=cat, a=x, b=y, gold_a=nx, gold_b=ny, a_to_b=xy, b_to_a=yx,
                                net_to_b=xy - yx, share_to_b=xy / (xy + yx) if xy + yx else np.nan,
                                rate_a_to_b=xy / nx if nx else np.nan, rate_b_to_a=yx / ny if ny else np.nan,
                                p_binom=binomtest(xy, xy + yx, 0.5).pvalue if xy + yx else 1.0))
            ok = int((g_ == p_).sum())
            assert tin == tout == tpair == err and ok + err + int((~v).sum()) == n, f"flow counts do not reconcile: {(m, c, l, cat)}"
    for T, name in ((pd.DataFrame(rows), "T2b_label_flows"), (pd.DataFrame(prs), "T2c_pair_flows")):
        if len(T): T["p_holm"] = holm(T.p_binom.values)
        T.to_csv(f"{OUT}/{name}.csv", index=False)

    rows = []
    for (m, c, l, r), p in sorted(P.items()):
        for name, (fn, cat) in METRICS.items():
            rows.append(dict(model=m, cond=c, lang=l, run=r, metric=name, value=rate(*fn(gold[cat], p[cat]))))
    V = pd.DataFrame(rows)
    if V.run.nunique() > 1:
        V.groupby(["model", "cond", "lang", "metric"]).value.agg(["mean", "std", "min", "max", "count"]).reset_index() \
         .to_csv(f"{OUT}/T3_run_variance.csv", index=False)

    rows, trans = [], []
    for m in models:
        for c in conds:
            pe, pb = prim.get((m, c, "en", 0)), prim.get((m, c, "bn", 0))
            if pe is None or pb is None: continue
            for cat in CATEGORIES:
                both = np.array([x is not None and y is not None for x, y in zip(pe[cat], pb[cat])])
                flip = ((pe[cat] != pb[cat]) & both).astype(float)
                ce, cb = (pe[cat] == gold[cat]), (pb[cat] == gold[cat])
                b = int((ce & ~cb & both).sum()); cc = int((~ce & cb & both).sum())
                pv = binomtest(b, b + cc, 0.5).pvalue if b + cc else 1.0
                bs = boot_rate(flip, both.astype(float), W); lo, hi = ci(bs)
                dacc = boot_rate(ce.astype(float), np.ones(n), W) - boot_rate(cb.astype(float), np.ones(n), W)
                rows.append(dict(model=m, cond=c, category=cat, flip_rate=rate(flip, both), flip_lo=lo, flip_hi=hi,
                                 acc_en=ce.mean(), acc_bn=cb.mean(), acc_diff_en_minus_bn=ce.mean() - cb.mean(),
                                 diff_lo=ci(dacc)[0], diff_hi=ci(dacc)[1], en_only_correct=b, bn_only_correct=cc,
                                 p_mcnemar_exact=pv))
                if cat in ("actor", "alignment", "frame"):
                    for (x, y), k in pd.Series(list(zip(pe[cat][both], pb[cat][both]))).value_counts().items() if both.any() else []:
                        if x != y: trans.append(dict(model=m, cond=c, category=cat, en_pred=x, bn_pred=y, count=int(k)))
    L = pd.DataFrame(rows)
    if len(L): L["p_holm"] = holm(L.p_mcnemar_exact.values)
    L.to_csv(f"{OUT}/T4_language_invariance.csv", index=False)
    pd.DataFrame(trans).to_csv(f"{OUT}/T4b_flip_transitions.csv", index=False)

    rows = []
    for m in models:
        for c in conds:
            for cat in CATEGORIES:
                fl = []
                for l in ("en", "bn"):
                    rs = sorted(r for (mm, cc, ll, r) in P if mm == m and cc == c and ll == l and r >= 1)
                    if len(rs) >= 2:
                        x, y = P[(m, c, l, rs[0])][cat], P[(m, c, l, rs[1])][cat]
                        ok = np.array([u is not None and v is not None for u, v in zip(x, y)])
                        fl.append(float(((x != y) & ok).sum() / max(1, ok.sum())))
                pe, pb = prim.get((m, c, "en", 0)), prim.get((m, c, "bn", 0))
                if pe is None or pb is None or not fl: continue
                ok = np.array([u is not None and v is not None for u, v in zip(pe[cat], pb[cat])])
                lang_flip = float(((pe[cat] != pb[cat]) & ok).sum() / max(1, ok.sum()))
                rows.append(dict(model=m, cond=c, category=cat, en_bn_flip=lang_flip, within_lang_run_flip=float(np.mean(fl)),
                                 excess_flip=lang_flip - float(np.mean(fl))))
    pd.DataFrame(rows).to_csv(f"{OUT}/T4e_noise_floor.csv", index=False)

    rows = []
    for m in models:
        for c in conds:
            pe, pb = prim.get((m, c, "en", 0)), prim.get((m, c, "bn", 0))
            if pe is None or pb is None: continue
            for name, (fn, cat) in METRICS.items():
                ne, de = fn(gold[cat], pe[cat]); nb, db = fn(gold[cat], pb[cat])
                d = boot_rate(ne, de, W) - boot_rate(nb, db, W); lo, hi = ci(d)
                pv = min(1, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))
                rows.append(dict(model=m, cond=c, metric=name, en=rate(ne, de), bn=rate(nb, db),
                                 diff=rate(ne, de) - rate(nb, db), lo=lo, hi=hi, p_boot=pv, den_en=int(de.sum()), den_bn=int(db.sum())))
    D = pd.DataFrame(rows)
    if len(D): D["p_holm"] = holm(D.p_boot.values)
    D.to_csv(f"{OUT}/T4c_language_bias_diff.csv", index=False)

    try:
        import statsmodels.api as sm, statsmodels.formula.api as smf
        recs = []
        for (m, c, l, r), p in prim.items():
            for cat in CATEGORIES:
                for i in range(n): recs.append((i, m, c, l, cat, int(p[cat][i] == gold[cat][i])))
        G = pd.DataFrame(recs, columns=["item", "model", "cond", "lang", "cat", "correct"])
        res = []
        for cat in CATEGORIES + ["ALL"]:
            sub = G if cat == "ALL" else G[G.cat == cat]
            f = "correct ~ C(lang) + C(model) + C(cond)" + (" + C(cat)" if cat == "ALL" else "")
            mod = smf.gee(f, "item", data=sub, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            k = "C(lang)[T.en]"
            res.append(dict(category=cat, en_vs_bn_logodds=mod.params[k], se=mod.bse[k], p=mod.pvalues[k],
                            odds_ratio=float(np.exp(mod.params[k]))))
        pd.DataFrame(res).to_csv(f"{OUT}/T4d_gee_language.csv", index=False)
    except Exception as e:
        print("GEE skipped:", e)

    rows = []
    for c, l in itertools.product(conds, ["en", "bn"]):
        for m1, m2 in itertools.combinations(models, 2):
            p1, p2 = prim.get((m1, c, l, 0)), prim.get((m2, c, l, 0))
            if p1 is None or p2 is None: continue
            for name, (fn, cat) in METRICS.items():
                n1, d1 = fn(gold[cat], p1[cat]); n2, d2 = fn(gold[cat], p2[cat])
                dd = boot_rate(n1, d1, W) - boot_rate(n2, d2, W); lo, hi = ci(dd)
                pv = min(1, 2 * min(np.nanmean(dd <= 0), np.nanmean(dd >= 0)))
                rows.append(dict(cond=c, lang=l, model_a=m1, model_b=m2, metric=name, diff=rate(n1, d1) - rate(n2, d2),
                                 lo=lo, hi=hi, p_boot=pv))
    M = pd.DataFrame(rows)
    if len(M): M["p_holm"] = holm(M.p_boot.values)
    M.to_csv(f"{OUT}/T5_model_pairs.csv", index=False)

    rows = []
    for (m, c, l, r), p in prim.items():
        for a_ in sorted(set(gold["actor"])):
            mask = gold["actor"] == a_
            rows.append(dict(model=m, cond=c, lang=l, gold_actor=a_, n=int(mask.sum()),
                             actor_acc=float((p["actor"][mask] == a_).mean()),
                             pred_India=float((p["actor"][mask] == "India").mean()),
                             pred_major=float(np.mean([x in {"India", "China", "US", "UN"} for x in p["actor"][mask]]))))
    pd.DataFrame(rows).to_csv(f"{OUT}/T6a_by_gold_actor.csv", index=False)
    conf = {}
    for (m, c, l, r), p in prim.items():
        for cat in ("actor", "alignment", "frame"):
            for g_, p_ in zip(gold[cat], p[cat]):
                if g_ != p_: conf[(cat, g_, p_ or "INVALID", l)] = conf.get((cat, g_, p_ or "INVALID", l), 0) + 1
    pd.DataFrame([dict(category=k[0], gold=k[1], pred=k[2], lang=k[3], count=v) for k, v in conf.items()]) \
      .sort_values("count", ascending=False).to_csv(f"{OUT}/T6b_confusions.csv", index=False)
    diff_rows = []
    for cat in CATEGORIES:
        wrong = np.mean([(p[cat] != gold[cat]) for p in prim.values()], axis=0)
        for i, it in enumerate(items):
            diff_rows.append(dict(id=it["id"], category=cat, gold=gold[cat][i], frac_configs_wrong=wrong[i],
                                  unanimous_wrong=bool(wrong[i] == 1.0), original_language=it["original_language"]))
    DF = pd.DataFrame(diff_rows); DF.to_csv(f"{OUT}/T6c_item_difficulty.csv", index=False)

    rows = []
    targets = [("actor", {"India"}), ("actor", {"India", "China", "US", "UN"}), ("alignment", {"Pro-West"}),
               ("alignment", {"Pro-China"}), ("alignment", {"Non-aligned"}), ("frame", {"Strategic"}),
               ("causal_present", {"Yes"})]
    for (m, c, l, r), p in sorted(prim.items()):
        for cat, S in targets:
            gi = np.array([x in S for x in gold[cat]], float); pi = np.array([x in S for x in p[cat]], float)
            d = (W @ pi - W @ gi) / n * 100; lo, hi = ci(d)
            rows.append(dict(model=m, cond=c, lang=l, category=cat, label="+".join(sorted(S)),
                             gold_share=gi.mean() * 100, pred_share=pi.mean() * 100, shift_pp=(pi.mean() - gi.mean()) * 100, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(f"{OUT}/T7_share_distortion.csv", index=False)

    rows = []
    for (m, c, l, r), p in sorted(prim.items()):
        for grp, mask in [("gold_actor=India", gold["actor"] == "India"), ("gold_actor=Bangladesh", gold["actor"] == "Bangladesh"),
                          ("gold_actor=Pakistan", gold["actor"] == "Pakistan"),
                          ("gold_align=Pro-West", gold["alignment"] == "Pro-West"), ("gold_align=Non-aligned", gold["alignment"] == "Non-aligned")]:
            nu, de = METRICS["CDR_FN"][0](gold["causal_present"], p["causal_present"])
            nu, de = nu & mask, de & mask
            rows.append(dict(model=m, cond=c, lang=l, group=grp, cdr_fn=rate(nu, de), num=int(nu.sum()), den=int(de.sum())))
    pd.DataFrame(rows).to_csv(f"{OUT}/T8_cdrfn_by_group.csv", index=False)

    if os.path.exists(a.meta):
        meta = json.load(open(a.meta)); rows = []
        for (m, c, l, r), p in sorted(prim.items()):
            if m not in meta: continue
            for name in ("ASI", "ADS", "CDR_FN"):
                fn, cat = METRICS[name]; nu, de = fn(gold[cat], p[cat])
                rows.append(dict(model=m, vendor_country=meta[m]["country"], params_b=meta[m].get("params_b"),
                                 open_weight=meta[m].get("open"), cond=c, lang=l, metric=name, value=rate(nu, de), den=int(de.sum())))
        pd.DataFrame(rows).to_csv(f"{OUT}/T9_vendor_exploratory.csv", index=False)

    try:
        import statsmodels.api as sm, statsmodels.formula.api as smf
        FS = {"zs": (0, 0), "zs_cot": (0, 1), "fs": (1, 0), "fs_cot": (1, 1)}
        res = []
        for m in models:
            recs = []
            for (mm, c, l, r), p in prim.items():
                if mm != m or c not in FS: continue
                for cat in CATEGORIES:
                    cc = (p[cat] == gold[cat]).astype(int)
                    for i in range(n): recs.append((i, FS[c][0], FS[c][1], l, cat, int(cc[i])))
            if len({(x[1], x[2]) for x in recs}) < 4: continue
            G2 = pd.DataFrame(recs, columns=["item", "fewshot", "cot", "lang", "cat", "correct"])
            for cat in CATEGORIES + ["ALL"]:
                sub = G2 if cat == "ALL" else G2[G2.cat == cat]
                f = "correct ~ fewshot * cot + C(lang)" + (" + C(cat)" if cat == "ALL" else "")
                mod = smf.gee(f, "item", data=sub, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
                for term in ("fewshot", "cot", "fewshot:cot"):
                    res.append(dict(model=m, category=cat, term=term, logodds=mod.params[term], se=mod.bse[term], p=mod.pvalues[term]))
        pd.DataFrame(res).to_csv(f"{OUT}/T10_prompt_ablation_gee.csv", index=False)
    except Exception as e:
        print("T10 skipped:", e)

    orig = np.array([i["original_language"] for i in items])
    rows = []
    for m in models:
        for c in conds:
            pe, pb = prim.get((m, c, "en", 0)), prim.get((m, c, "bn", 0))
            if pe is None or pb is None: continue
            for og in ("en", "bn"):
                mask = orig == og
                if mask.sum() < 5: continue
                for cat in CATEGORIES + ["ALL8"]:
                    if cat == "ALL8":
                        ce = np.mean([(pe[x] == gold[x]) for x in CATEGORIES], axis=0); cb = np.mean([(pb[x] == gold[x]) for x in CATEGORIES], axis=0)
                    else:
                        ce, cb = (pe[cat] == gold[cat]).astype(float), (pb[cat] == gold[cat]).astype(float)
                    native, trans = (ce, cb) if og == "en" else (cb, ce)
                    Wm = W * mask.astype(np.float32)
                    d = (Wm @ (native - trans)) / np.maximum(Wm.sum(1), 1); lo, hi = ci(d)
                    rows.append(dict(model=m, cond=c, original_language=og, n=int(mask.sum()), category=cat,
                                     acc_native=float(native[mask].mean()), acc_translated=float(trans[mask].mean()),
                                     native_minus_translated=float((native - trans)[mask].mean()), lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(f"{OUT}/T4f_native_vs_translated.csv", index=False)

    from statsmodels.stats.proportion import proportion_confint
    rows = []
    for (m, c, l, r), p in sorted(prim.items()):
        for act in sorted(set(gold["actor"])):
            mask = (gold["alignment"] == "Non-aligned") & (gold["actor"] == act)
            if mask.sum() < 8: continue
            pw = int((p["alignment"][mask] == "Pro-West").sum()); pc = int((p["alignment"][mask] == "Pro-China").sum()); k = int(mask.sum())
            lo, hi = proportion_confint(pw + pc, k, method="wilson")
            rows.append(dict(model=m, cond=c, lang=l, gold_actor=act, n_gold_nonaligned=k, to_pro_west=pw, to_pro_china=pc,
                             polarized_rate=(pw + pc) / k, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(f"{OUT}/T11_alignment_drift_by_actor.csv", index=False)

    rows = []
    for m in models:
        for l in ("en", "bn"):
            for c1, c2 in itertools.combinations(sorted({k[1] for k in prim if k[0] == m}), 2):
                p1, p2 = prim.get((m, c1, l, 0)), prim.get((m, c2, l, 0))
                if p1 is None or p2 is None: continue
                for name, (fn, cat) in list(METRICS.items()) + [("ACC_all8", (None, None))]:
                    if name == "ACC_all8":
                        n1 = np.mean([(p1[x] == gold[x]) for x in CATEGORIES], axis=0); n2 = np.mean([(p2[x] == gold[x]) for x in CATEGORIES], axis=0)
                        d = (W @ (n2 - n1)) / n; v1, v2 = n1.mean(), n2.mean()
                    else:
                        a1, b1 = fn(gold[cat], p1[cat]); a2, b2 = fn(gold[cat], p2[cat])
                        d = boot_rate(a2, b2, W) - boot_rate(a1, b1, W); v1, v2 = rate(a1, b1), rate(a2, b2)
                    lo, hi = ci(d)
                    rows.append(dict(model=m, lang=l, cond_a=c1, cond_b=c2, metric=name, a=v1, b=v2, diff_b_minus_a=v2 - v1, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(f"{OUT}/T12_condition_pairs.csv", index=False)

    mm = [m for m in models if m.endswith("-codebook")]; rows, crow = [], []
    if len(mm) >= 2:
        for c in conds:
            ls = [l for l in ("en", "bn") if all((m, c, l, 0) in prim for m in mm)]
            if not ls: continue
            for cat in CATEGORIES:
                g = gold[cat]; P_ = {(m, l): prim[(m, c, l, 0)][cat] for m in mm for l in ls}
                err = {m: sum((P_[(m, l)] != g).astype(float) for l in ls) for m in mm}
                allw = sum(np.all([P_[(m, l)] != g for m in mm], axis=0).astype(float) for l in ls)
                same = sum((np.all([P_[(m, l)] != g for m in mm], axis=0) & np.all([P_[(m, l)] == P_[(mm[0], l)] for m in mm], axis=0)
                            & np.array([x is not None for x in P_[(mm[0], l)]])).astype(float) for l in ls)
                agree = sum(np.all([P_[(m, l)] == P_[(mm[0], l)] for m in mm], axis=0).astype(float) for l in ls)
                den = np.full(n, float(len(ls)))
                b_all = boot_rate(allw, den, W); b_exp = np.prod([boot_rate(err[m], den, W) for m in mm], axis=0)
                exp = float(np.prod([rate(err[m], den) for m in mm])); lo, hi = ci(b_all - b_exp); slo, shi = ci(boot_rate(same, den, W))
                rows.append(dict(cond=c, category=cat, n_models=len(mm), **{f"error_{m[:-9]}": rate(err[m], den) for m in mm},
                                 all_wrong=rate(allw, den), expected_all_wrong_if_independent=exp, ratio_observed_to_expected=rate(allw, den) / exp if exp else np.nan,
                                 excess=rate(allw, den) - exp, excess_lo=lo, excess_hi=hi,
                                 same_wrong_label=rate(same, den), same_lo=slo, same_hi=shi,
                                 share_of_shared_errors_with_same_label=same.sum() / allw.sum() if allw.sum() else np.nan,
                                 all_models_same_label=rate(agree, den)))
                cnt, cnt_v = {}, {}
                for l in ls:
                    for i in range(n):
                        pv = P_[(mm[0], l)][i]
                        if pv is None or pv == g[i] or any(P_[(m, l)][i] != pv for m in mm): continue
                        cnt.setdefault((g[i], pv), set()).add(i); cnt_v[(g[i], pv)] = cnt_v.get((g[i], pv), 0) + 1
                for (a_, b_), ids_ in cnt.items():
                    rev = len(cnt.get((b_, a_), ())); k = len(ids_)
                    if k < rev or (k == rev and a_ > b_): continue
                    crow.append(dict(cond=c, category=cat, gold=a_, shared_pred=b_, items=k, item_versions=cnt_v[(a_, b_)], reverse_items=rev,
                                     share_this_direction=k / (k + rev), gold_n=int((g == a_).sum()), rate_of_gold=k / max(1, int((g == a_).sum())),
                                     p_binom=binomtest(k, k + rev, 0.5).pvalue))
        pd.DataFrame(rows).to_csv(f"{OUT}/T13_shared_errors.csv", index=False)
        C = pd.DataFrame(crow)
        if len(C): C["p_holm"] = holm(C.p_binom.values); C = C.sort_values(["cond", "items"], ascending=[True, False])
        C.to_csv(f"{OUT}/T14_shared_error_direction.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    s = T1[(T1.lang.isin(["en", "bn"]))].pivot_table(index=["model", "cond", "lang"], columns="metric", values="value")
    print(s.round(3))
    if len(T0):
        T0["invalid_labels"] = T0.filter(like="invalid_").sum(axis=1)
        chk = ["missing", "duplicates", "unparsed", "truncated", "input_shortened", "invalid_labels"]; bad = T0[(T0[chk] > 0).any(axis=1)]
        print(f"\ndata integrity: {len(T0) - len(bad)}/{len(T0)} result files clean" + ("" if not len(bad) else " (details in T0_data_integrity.csv)\n" + bad[["model", "cond", "lang", "run"] + chk].to_string(index=False)))
    print("\nCSV tables written to", OUT)

if __name__ == "__main__": main()
