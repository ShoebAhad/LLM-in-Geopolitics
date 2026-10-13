"""
Clean 'Foreign Annotation Data Multisource.xlsx' (Sheet1) into bfpn_v2.

Output: data/bfpn_v2.jsonl   one row per UNIQUE item (id, texts in en/bn, gold, split)
        data/cleaning_report.json

Decisions (agreed with authors): drop Actor_Role == Unclear; Hindi skipped (EN+BN only);
actor set is South-Asia focused; one id per unique text, both languages share the id.
"""
import hashlib, json, re, sys
from collections import Counter
import pandas as pd

SRC = "Foreign Annotation Data Multisource.xlsx"
OUT = "data/bfpn_v2.jsonl"
REPORT = "data/cleaning_report.json"
SEED = 42
DEV_N = 40            # held-out items used only for few-shot exemplars / prompt checks

BN = re.compile(r"[ঀ-৿]")
def bn_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum(1 for c in letters if BN.match(c)) / max(1, len(letters))

def norm_text(t): return re.sub(r"\s+", " ", str(t)).strip()
def item_id(t):   return "bfpn_" + hashlib.md5(norm_text(t).encode()).hexdigest()[:10]

# Actor policy (authors, 5 Oct): the label set is the explicit list below (country names; "US" and "UN" as written in the
# codebook). EVERY other country or organisation (EU, ASEAN, BRICS, QUAD, IOM, Japan, Nepal, Gulf states, ...) is "Other".
# actor_raw keeps the original actor, spelled as a proper country/organisation name.
ACTOR_LIST = ["Bangladesh", "India", "Pakistan", "Sri Lanka", "Bhutan", "Myanmar", "China", "US", "UN"]
ACTOR_MAP = {"usa": "US", "united states": "US", "srilanka": "Sri Lanka", "malayasia": "Malaysia", "turkiye": "Turkey",
             "quad": "QUAD", "european": "EU", "european union": "EU"}
FRAME_MAP = {"legal": "Normative"}
ALIGN_MAP = {"non-aligned": "Non-aligned", "pro_china": "Pro-China", "pro-china": "Pro-China",
             "pro-west": "Pro-West", "mixed": "Mixed"}
POLICY_MAP = {"co-operation": "Cooperation"}

def main():
    raw = pd.read_excel(SRC, sheet_name="Sheet1").iloc[:, :14]
    rep = {"rows_in_sheet": len(raw)}
    df = raw[raw["Actor"].notna()].copy()               # annotated rows only
    rep["annotated_rows"] = len(df)
    for c in ["Frame", "Actor", "Actor_Role", "Agency", "Moral_valence", "Causal_Claim",
              "policy_direction", "Alignment"]:
        df[c] = df[c].astype(str).str.strip()

    unclear = df["Actor_Role"].str.lower() == "unclear"
    rep["dropped_unclear_role"] = int(unclear.sum())
    df = df[~unclear]

    df["Frame"] = df["Frame"].map(lambda v: FRAME_MAP.get(v.lower(), v))
    df["Alignment"] = df["Alignment"].map(lambda v: ALIGN_MAP.get(v.lower(), v))
    df["policy_direction"] = df["policy_direction"].map(lambda v: POLICY_MAP.get(v.lower(), v))
    df["actor_raw"] = df["Actor"].map(lambda v: ACTOR_MAP.get(v.lower(), v))
    df["Actor_Role"] = df["Actor_Role"].str.capitalize()

    # which column is English / Bangla?  'Texts' = original, translation col = the other language
    rows, conflicts, bad = {}, [], []
    for _, r in df.iterrows():
        orig, trans = norm_text(r["Texts"]), norm_text(r["Translation (EN->BN/BN->EN)"])
        if not orig or not trans or trans.lower() == "nan":
            bad.append("missing text/translation"); continue
        o_bn = bn_ratio(orig) > 0.3
        t_bn = bn_ratio(trans) > 0.3
        if o_bn == t_bn:
            bad.append("translation in same script as original"); continue
        en, bn = (trans, orig) if o_bn else (orig, trans)
        gold = {"frame": r["Frame"], "actor": r["actor_raw"], "actor_role": r["Actor_Role"],
                "agency": r["Agency"], "moral_valence": r["Moral_valence"],
                "causal_present": r["Causal_Claim"], "policy_direction": r["policy_direction"],
                "alignment": r["Alignment"]}
        iid = item_id(orig)
        if iid in rows:
            if rows[iid]["gold_raw"] != gold: conflicts.append(iid)
            continue
        rows[iid] = {"id": iid, "original_language": "bn" if o_bn else "en",
                     "text_en": en, "text_bn": bn, "gold_raw": gold}
    rep["dropped_bad_text_rows"] = dict(Counter(bad))
    rep["duplicate_text_label_conflicts_dropped"] = len(set(conflicts))
    for c in set(conflicts): rows.pop(c, None)

    # explicit actor list; everything else -> Other (raw kept)
    cnt = Counter(v["gold_raw"]["actor"] for v in rows.values())
    for v in rows.values():
        g = dict(v["gold_raw"]); v["actor_raw"] = g["actor"]
        if g["actor"] not in ACTOR_LIST: g["actor"] = "Other"
        v["gold"] = g; del v["gold_raw"]
    rep["actor_set"] = ACTOR_LIST + ["Other"]
    rep["actor_raw_counts"] = dict(cnt.most_common())
    rep["mapped_to_Other"] = {a: n for a, n in cnt.most_common() if a not in ACTOR_LIST}

    # stratified dev split by alignment x original_language (seeded, deterministic)
    items = sorted(rows.values(), key=lambda v: v["id"])
    import random; rnd = random.Random(SEED)
    groups = {}
    for v in items: groups.setdefault((v["gold"]["alignment"], v["original_language"]), []).append(v)
    dev = set()
    for k, g in groups.items():
        rnd.shuffle(g)
        dev.update(x["id"] for x in g[:max(1, round(DEV_N * len(g) / len(items)))])
    for v in items: v["split"] = "dev" if v["id"] in dev else "test"

    with open(OUT, "w") as f:
        for v in items: f.write(json.dumps(v, ensure_ascii=False) + "\n")
    rep["n_items"] = len(items)
    rep["n_dev"] = sum(v["split"] == "dev" for v in items)
    rep["n_test"] = rep["n_items"] - rep["n_dev"]
    rep["n_evaluations_per_model_condition_test"] = rep["n_test"] * 2
    rep["original_language"] = dict(Counter(v["original_language"] for v in items))
    for c in ["frame", "actor", "actor_role", "agency", "moral_valence", "causal_present",
              "policy_direction", "alignment"]:
        rep[f"gold_{c}"] = dict(Counter(v["gold"][c] for v in items).most_common())
    rep["chars_en"] = pd.Series([len(v["text_en"]) for v in items]).describe().round(0).to_dict()
    rep["chars_bn"] = pd.Series([len(v["text_bn"]) for v in items]).describe().round(0).to_dict()
    json.dump(rep, open(REPORT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))

if __name__ == "__main__": main()
