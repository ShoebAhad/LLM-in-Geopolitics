"""Exit 0 if every planned (tag, condition, language, run) file has all items; else print what is missing and exit 1.
Optional argument: model tag prefix (default qwen38-27b), e.g. `check_complete.py gpt-oss-20b`."""
import json, os, sys
T = sys.argv[1] if len(sys.argv) > 1 else "qwen38-27b"
R = "results/v2"; MAIN = f"{T}-codebook"; BAL = f"{T}-balex"
plan = []   # (tag, cond, run, expected unique ids)
for c in ("zs", "fs", "zs_cot", "fs_cot"): plan.append((MAIN, c, 0, 527))     # 486 test + 41 dev
for c in ("zs", "fs_cot"):
    for r in (1, 2): plan.append((MAIN, c, r, 527))
for r in (3, 4): plan.append((MAIN, "zs", r, 527))
plan.append((BAL, "fs", 0, 527))
missing, gaps = [], []   # gaps = rows that exist but are unusable or weakened (no answer, cut off, input shortened)
for tag, c, r, n in plan:
    for l in ("en", "bn"):
        f = f"{R}/{tag}/{c}_{l}_r{r}.jsonl"
        rows = [json.loads(x) for x in open(f)] if os.path.exists(f) else []
        have = len({x["id"] for x in rows})
        if have < n: missing.append(f"{tag}/{c}_{l}_r{r}: {have}/{n}")
        bad = dict(unparsed=sum(not x["parsed"] for x in rows), truncated=sum(x["finish"] == "length" for x in rows),
                   input_shortened=sum(x.get("text_fraction_used", 1) < 1 for x in rows))
        if any(bad.values()): gaps.append(f"{tag}/{c}_{l}_r{r}: {bad}")
print("COMPLETE" if not missing else "MISSING:\n  " + "\n  ".join(missing))
print("NO GAPS" if not gaps else "GAPS (rows present but weakened):\n  " + "\n  ".join(gaps))
sys.exit(1 if missing else 0)
