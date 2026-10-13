"""
Unified evaluation runner (replaces run_claude_eval / run_opus_eval / run_few_shot / run_baseline_eval).

Open-weight (vLLM or ollama, OpenAI-compatible):
  python run_eval.py --model Qwen/Qwen3-8B --base-url http://localhost:8001/v1 --tag qwen3-8b \
      --conditions zs zs_cot fs fs_cot --langs en bn --runs 0 1 2 3
Closed: --provider openai | anthropic  (keys from OPENAI_API_KEY / ANTHROPIC_API_KEY)

Run 0 = greedy (temperature 0, seed 0) = primary result. Runs >=1 = sampled (temperature --temp, seed=run)
to quantify run-to-run variance. Output is resumable: results/v2/<tag>/<cond>_<lang>_r<run>.jsonl
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from bfpn.schema import build_prompt, select_exemplars, extract_json, validate, CONDITIONS

def load_items(path="data/bfpn_v2.jsonl", max_chars=60000):
    items = [json.loads(l) for l in open(path)]
    n_trunc = 0
    for it in items:
        m = max(len(it["text_en"]), len(it["text_bn"]))
        if m > max_chars:
            r = max_chars / m; n_trunc += 1
            for k in ("text_en", "text_bn"): it[k] = it[k][:int(len(it[k]) * r)]
    return items, n_trunc

import threading, collections as _c, re as _re, datetime as _dt
class Throttle:
    """Shared limiter for a rate-limited gateway (limit: 500k tokens/min per key, possibly shared and counted differently from
    the server's usage numbers). Sliding-window budget + GLOBAL cooldown on any 429 + AIMD: the budget is cut by 40% on each 429 and
    recovers 5% per quiet 30 s, so retries never storm the gateway."""
    def __init__(self, limit, window=60.0):
        self.base = float(limit); self.limit = self.base if self.base >= 5_000_000 else min(self.base, 300000.0); self.window = window
        self.lock = threading.Lock(); self.log = _c.deque(); self.inflight = 0
        self.cooldown_until = 0.0; self.last_hit = 0.0; self.last_up = time.time(); self.hits = 0
        self.max_p = 64000; self.inflight_p = 0; self.ratio = 0.6
    def acquire(self, est, est_p=0):
        while True:
            with self.lock:
                now = time.time()
                if now >= self.cooldown_until:
                    while self.log and now - self.log[0][0] > self.window: self.log.popleft()
                    if now - self.last_hit > 30 and now - self.last_up > 30 and self.limit < self.base:
                        self.limit = min(self.base, self.limit * 1.05); self.last_up = now
                    used = sum(t for _, t in self.log) + self.inflight
                    ok_p = self.inflight_p + est_p <= self.max_p or self.inflight_p == 0
                    if ok_p and (used + est <= self.limit or (not self.log and self.inflight == 0)):
                        self.inflight += est; self.inflight_p += est_p; return
            time.sleep(0.5)
    def release(self, est, actual, est_p=0, plen=0, prompt_tokens=0):
        with self.lock:
            self.inflight_p -= est_p
            if plen and prompt_tokens: self.ratio = 0.8 * self.ratio + 0.2 * (prompt_tokens / plen)
            self.inflight -= est; self.log.append((time.time(), actual if actual else est))
    def hit(self, wait):
        with self.lock:
            now = time.time(); self.hits += 1; self.last_hit = now
            self.cooldown_until = max(self.cooldown_until, now + wait + 10)
            self.limit = max(40000.0, self.limit * 0.6)
            print(f"  [429 #{self.hits}] global cooldown {wait + 10:.0f}s, budget -> {self.limit/1000:.0f}k tok/min", flush=True)
THROTTLE = None
USAGE = {"input_tokens": 0, "output_tokens": 0}; USAGE_LOCK = threading.Lock()

def get_key(a):
    k = os.environ.get("OPENAI_API_KEY")
    p = os.path.expanduser(a.key_file)
    if not k and os.path.exists(p): k = open(p).read().strip()
    return k or "EMPTY"

def make_caller(a):
    if a.provider == "anthropic":
        import anthropic
        p = os.path.expanduser("~/.anthropic_api_key")
        key = os.environ.get("ANTHROPIC_API_KEY") or (open(p).read().strip() if os.path.exists(p) else None)
        cl = anthropic.Anthropic(api_key=key, max_retries=6, timeout=600)
        extra = json.loads(a.extra_body) if a.extra_body else {}
        def call(prompt, temp, seed, max_tokens):
            kw = {} if a.no_temp else {"temperature": temp}
            r = cl.messages.create(model=a.model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}], **kw, **extra)
            with USAGE_LOCK: USAGE["input_tokens"] += r.usage.input_tokens; USAGE["output_tokens"] += r.usage.output_tokens
            return "".join(b.text for b in r.content if b.type == "text"), ("length" if r.stop_reason == "max_tokens" else r.stop_reason)
    else:
        import openai
        cl = openai.OpenAI(base_url=a.base_url, api_key=get_key(a), timeout=600)
        extra = json.loads(a.extra_body) if a.extra_body else None
        def call(prompt, temp, seed, max_tokens):
            kw = {"extra_body": extra} if extra else {}
            if not a.no_temp: kw["temperature"] = temp
            if not a.no_seed: kw["seed"] = seed
            tok_key = "max_completion_tokens" if a.provider == "openai" else "max_tokens"
            kw[tok_key] = max_tokens
            est = len(prompt) // 2 + 600; est_p = int(len(prompt) * THROTTLE.ratio) + 1
            THROTTLE.acquire(est, est_p); actual = pt = 0
            try:
                r = cl.chat.completions.create(model=a.model, messages=[{"role": "user", "content": prompt}], **kw)
                actual = getattr(r.usage, "total_tokens", 0) or 0; pt = getattr(r.usage, "prompt_tokens", 0) or 0
                if a.provider == "openai":
                    with USAGE_LOCK: USAGE["input_tokens"] += pt; USAGE["output_tokens"] += getattr(r.usage, "completion_tokens", 0) or 0
            finally:
                THROTTLE.release(est, actual, est_p, len(prompt), pt)
            return (r.choices[0].message.content or ""), r.choices[0].finish_reason
    return call

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--provider", default="openai-compat", choices=["openai-compat", "openai", "anthropic"])
    ap.add_argument("--base-url", default=os.environ.get("BFPN_BASE_URL", "http://localhost:8001/v1"))
    ap.add_argument("--conditions", nargs="+", default=["zs", "fs_cot"], choices=list(CONDITIONS))
    ap.add_argument("--langs", nargs="+", default=["en", "bn"])
    ap.add_argument("--runs", nargs="+", type=int, default=[0])
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--split", default="test", choices=["test", "dev", "all"])
    ap.add_argument("--limit", type=int, default=0, help="debug: first N items")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-inflight-tokens", type=int, default=64000, help="max PROMPT tokens in flight at once (server KV-cache protection)")
    ap.add_argument("--tpm", type=int, default=400000, help="client-side tokens-per-minute budget (gateway limit is 500k)")
    ap.add_argument("--max-chars", type=int, default=60000, help="pre-truncation cap (default above the longest document, i.e. none); context overflow is handled by automatic shrinking")
    ap.add_argument("--extra-body", default="", help='JSON, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    ap.add_argument("--no-temp", action="store_true"); ap.add_argument("--no-seed", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=600, help="completion budget, non-CoT conditions")
    ap.add_argument("--max-tokens-cot", type=int, default=3000, help="completion budget, CoT conditions")
    ap.add_argument("--exemplars", default="default", choices=["default", "balanced"], help="few-shot exemplar set (balanced = covers all alignment labels)")
    ap.add_argument("--schema", default="codebook", choices=["codebook", "paper"], help="prompt wording: codebook-faithful (default) or the v1 paper wording")
    ap.add_argument("--key-file", default="~/.bfpn_api_key", help="used if OPENAI_API_KEY is not set; keep OUTSIDE the repo")
    ap.add_argument("--out", default="results/v2")
    ap.add_argument("--keep-caps", action="store_true", help="do not raise --max-tokens/--max-tokens-cot to the generous floors below")
    a = ap.parse_args()
    if not a.keep_caps: a.max_tokens, a.max_tokens_cot = max(a.max_tokens, 4000), max(a.max_tokens_cot, 16000)
    if a.tpm == 400000: a.tpm = 470000

    items, n_trunc = load_items(max_chars=a.max_chars)
    dev = [i for i in items if i["split"] == "dev"]
    exemplars = select_exemplars(dev, mode=a.exemplars)
    test = [i for i in items if a.split == "all" or i["split"] == a.split]
    if a.limit: test = test[:a.limit]
    print(f"{len(test)} items ({n_trunc} truncated at {a.max_chars} chars) | exemplars: {[e['id'] for e in exemplars]}")
    global THROTTLE; THROTTLE = Throttle(a.tpm); THROTTLE.max_p = a.max_inflight_tokens
    call = make_caller(a)
    outdir = os.path.join(a.out, a.tag); os.makedirs(outdir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(outdir, "config.json"), "w"), indent=1)

    for cond in a.conditions:
        for lang in a.langs:
            for run in a.runs:
                path = os.path.join(outdir, f"{cond}_{lang}_r{run}.jsonl")
                done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
                todo = [it for it in test if it["id"] not in done]
                if not todo: continue
                temp = 0.0 if run == 0 else a.temp
                max_tokens = a.max_tokens_cot if CONDITIONS[cond][1] else a.max_tokens
                def work(it):
                    text, used, mt, err, empty = it["text_" + lang], 1.0, max_tokens, "", 0
                    for attempt in range(40):
                        prompt = build_prompt(text, cond, exemplars, lang, a.schema)
                        try:
                            raw, fin = call(prompt, temp, run + 1000 * empty, mt)
                            if fin == "length" and not raw.strip() and empty < 3: empty += 1; continue
                            break
                        except Exception as e:
                            msg = str(e).lower(); err = f"{type(e).__name__}: {str(e)[:300]}"
                            if "context" in msg or "maximum" in msg or "too long" in msg:
                                mw = _re.search(r"context length is (\d+)", msg); mi = _re.search(r"(\d+) input tokens", msg)
                                win, inp = (int(mw.group(1)) if mw else 32768), (int(mi.group(1)) if mi else None)
                                if inp and "at least" not in msg and win - inp - 64 >= 1024:
                                    mt = min(mt, win - inp - 64)
                                elif mt > 1024:
                                    mt = max(1024, int(mt * 0.8))
                                else:
                                    used *= 0.9; text = it["text_" + lang][:int(len(it["text_" + lang]) * used)]
                                raw, fin = "", "error:context"; continue
                            if "429" in msg or "rate limit" in msg:
                                m = _re.search(r"resets at: ([0-9: -]+) UTC", str(e))
                                wait = 20.0
                                if m:
                                    try: wait = max(3.0, (_dt.datetime.strptime(m.group(1).strip(), "%Y-%m-%d %H:%M:%S") - _dt.datetime.utcnow()).total_seconds() + 2)
                                    except Exception: pass
                                THROTTLE.hit(min(wait, 90)); raw, fin = "", "error:RateLimit"; continue
                            raw, fin = "", f"error:{type(e).__name__}"; time.sleep(2 ** min(attempt, 5) * 2)
                    if fin.startswith("error"):
                        print(f"  WARN failed {it['id']} {cond}/{lang}/r{run} ({fin}): {err}", flush=True); return None
                    pred = validate(extract_json(raw))
                    return {"id": it["id"], "model": a.tag, "condition": cond, "lang": lang, "run": run,
                            "schema": a.schema, "pred": pred, "parsed": extract_json(raw) is not None, "finish": fin,
                            "chars": len(text), "max_tokens_used": mt, "empty_retries": empty, "chars_full": len(it["text_" + lang]), "text_fraction_used": round(used, 3), "raw": raw}
                t0 = time.time(); failed = 0
                with ThreadPoolExecutor(a.workers) as ex, open(path, "a") as f:
                    from concurrent.futures import as_completed
                    futs = [ex.submit(work, it) for it in todo]
                    for k, r in enumerate((ft.result() for ft in as_completed(futs)), 1):
                        if r is None: failed += 1; continue
                        f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                        if k % 10 == 0 or k == len(todo):
                            el = time.time() - t0; rt = k / el
                            print(f"  [{a.tag}] {cond}/{lang}/r{run}: {k}/{len(todo)} ({100*k/len(todo):.0f}%) "
                                  f"{rt*60:.1f} items/min | elapsed {el/60:.1f}m | ETA {(len(todo)-k)/rt/60:.1f}m", flush=True)
                rs = [json.loads(l) for l in open(path)]
                bad = sum(not r["parsed"] for r in rs); cut = sum(r["finish"] == "length" for r in rs)
                print(f"done {path} ({len(todo) - failed} new, {failed} FAILED-not-saved, {time.time()-t0:.0f}s) | unparsed {bad}/{len(rs)} | truncated {cut}/{len(rs)}")
                if USAGE["input_tokens"]:
                    with open(os.path.join(outdir, "usage.jsonl"), "a") as uf: uf.write(json.dumps({"file": os.path.basename(path), "calls_saved": len(todo) - failed, **USAGE}) + "\n")
                    print(f"  usage {os.path.basename(path)}: {USAGE['input_tokens']:,} input, {USAGE['output_tokens']:,} output tokens", flush=True)
                    USAGE["input_tokens"] = USAGE["output_tokens"] = 0
                if cut > 0.02 * len(rs): print("  WARNING: >2% truncated -> raise --max-tokens / lower reasoning effort", flush=True)

if __name__ == "__main__": main()
