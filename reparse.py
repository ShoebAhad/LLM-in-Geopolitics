"""Re-derive 'pred'/'parsed' from the stored raw model output (after a parser fix). Usage: python reparse.py <results dir>"""
import glob, json, sys
from bfpn.schema import extract_json, validate
n = changed = 0
for f in glob.glob(f"{sys.argv[1]}/*/*_r*.jsonl"):
    rows = [json.loads(l) for l in open(f)]
    for r in rows:
        j = extract_json(r["raw"]); new = validate(j); n += 1
        if new != r["pred"] or (j is not None) != r["parsed"]: changed += 1
        r["pred"], r["parsed"] = new, j is not None
    with open(f, "w") as o:
        for r in rows: o.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"re-parsed {n} records, {changed} changed")
