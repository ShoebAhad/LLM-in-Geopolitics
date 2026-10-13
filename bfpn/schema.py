"""Label sets, prompt construction (with ablation switches), output parsing. v2 schema."""
import json, re

VALID_LABELS = {
    "frame":            ["Humanitarian", "Security", "Economic", "Strategic", "Normative"],
    "actor":            ["Bangladesh", "India", "Pakistan", "Sri Lanka", "Bhutan", "Myanmar",
                         "China", "US", "UN", "Other"],
    "actor_role":       ["Initiator", "Respondent", "Mediator", "Victim"],
    "agency":           ["High", "Medium", "Low"],
    "moral_valence":    ["Positive", "Neutral", "Negative"],
    "causal_present":   ["Yes", "No"],
    "policy_direction": ["Cooperation", "Pressure", "Intervention", "Neutrality"],
    "alignment":        ["Pro-West", "Pro-China", "Non-aligned", "Mixed"],
}
CATEGORIES = list(VALID_LABELS)

# metric label sets
MAJOR = {"India", "China", "US", "UN"}                       # "major power" set M
SMALL_SA = {"Bangladesh", "Pakistan", "Sri Lanka", "Bhutan", "Myanmar"}
POLARIZED = {"Pro-West", "Pro-China"}

SCHEMA_PAPER = """=== ANNOTATION SCHEMA & RULES ===

FRAME - dominant lens only (ignore secondary signals):
  Humanitarian : refugees, suffering, moral obligation, aid
  Security     : threats, military, borders, weapons, conflict
  Economic     : trade, investment, sanctions, loans, tariffs
  Strategic    : power, alliances, geopolitical influence, diplomacy
  Normative    : law, sovereignty, human rights, democratic norms
  Decision: Economic + Strategic both present -> Strategic
            Security + Humanitarian both present -> choose the ACTION frame

ACTOR - narrative center, ONE only:
  Bangladesh | India | Pakistan | Sri Lanka | Bhutan | Myanmar | China | US | UN | Other
  (Other = any actor not listed, e.g. Gulf states, ASEAN members, blocs)
  Priority order: Who initiates? -> Who is most discussed? -> Who is evaluated?
  Tie-break: Bangladesh

ACTOR_ROLE - narrative positioning:
  Initiator  : starts/launches the action
  Respondent : reacts to someone else's action
  Mediator   : facilitates between other parties
  Victim     : suffers from external forces

AGENCY - control over the action:
  High   : unilateral decisions ("decided","imposed","launched","demanded")
  Medium : negotiates, urges, requests ("proposed","called for","urged")
  Low    : affected, awaiting, pressured ("dependent on","awaiting","impacted")

MORAL_VALENCE - how the PRIMARY actor is portrayed:
  Positive : responsible, praised, legitimate
  Neutral  : descriptive/factual, no clear judgment
  Negative : criticized, blamed, portrayed as harmful

CAUSAL_PRESENT - explicit cause-effect language only:
  Yes : "because","led to","resulted in","due to"; Bangla: "কারণে","ফলে","যার ফলে"
  No  : no explicit causal connector (implied causality does NOT count)

POLICY_DIRECTION - implied action:
  Cooperation  : dialogue, aid, agreements, joint action
  Pressure     : sanctions, criticism, demands, protests
  Intervention : direct military/administrative involvement
  Neutrality   : balance, avoidance, non-commitment

ALIGNMENT - geopolitical stance implied by the narrative:
  Pro-West   : US/EU/NATO aligned framing
  Pro-China  : BRI/Beijing aligned framing
  Non-aligned: balancing multiple powers, strategic neutrality
  Mixed      : multiple alignments equally present (use SPARINGLY)
  CRITICAL: Western audience != Pro-West | criticising West != Pro-China
            Non-aligned != passive or inactive
"""


# Codebook-faithful variant: wording follows the authors' annotation codebook (Sec. 4-6). Differences from SCHEMA_PAPER:
#  * no "Economic+Strategic -> Strategic" frame tie-break (codebook: dominant frame only, ignore secondary signals)
#  * ALIGNMENT is "direct or indirect beneficiary" / which bloc the involved country mostly allies with,
#    not "stance of the framing"; no "use Mixed sparingly"
#  * CAUSAL: explicit cause-effect stated in the text (not "connector words only")
#  * actor hierarchy has the 4th step (mentioned first) and special cases A-E
SCHEMA_CODEBOOK = """=== ANNOTATION CODEBOOK ===
Annotate how the narrative is constructed, NOT whether it is factually true. Annotate the whole document.
Follow this order: main story -> frame -> actor -> role -> agency -> moral valence -> causal claim -> policy direction -> alignment.
Assign exactly ONE label per category.

FRAME - the dominant lens through which the foreign policy action is described (choose the dominant frame only; ignore secondary signals):
  Humanitarian : refugees, suffering, moral obligation
  Security     : threats, conflict, borders
  Economic     : trade, investment, sanctions
  Strategic    : power, alliances, influence (cues: partnership, influence, cooperation, alliance)
  Normative    : law, legitimacy, global norms

ACTOR - the main decision-maker or narrative focus; exactly ONE. Always answer with the country name:
  Bangladesh | India | Pakistan | Sri Lanka | Bhutan | Myanmar | China | US | UN | Other
  Other = ANY country not listed above (e.g. Japan, Nepal, Saudi Arabia) and ANY organization or bloc other than the UN
  (e.g. EU, ASEAN, BRICS, QUAD, IOM). Do not invent new labels.
  Hierarchy: (1) actor initiating the action, (2) actor most discussed, (3) actor being evaluated, (4) actor mentioned first.
  If still unclear: Bangladesh when Bangladesh is involved; otherwise the actor that acts as the major (leading, dominant) party,
  e.g. India or Pakistan when that country is the one driving the narrative.
  Cases: symmetric interaction ("Bangladesh and India agreed") -> Bangladesh; external pressure ("US pressured Bangladesh") -> the pressuring actor (US);
         mediation ("UN facilitated talks") -> UN; victim narrative ("Bangladesh suffered") -> Bangladesh;
         background actor ("Bangladesh acted with support from China") -> Bangladesh.
  The actor is the narrative center, not every actor mentioned.

ACTOR_ROLE - how the actor is positioned in the narrative (narrative role, not legal role):
  Initiator : starts the action | Respondent : reacts | Mediator : facilitates | Victim : suffers

AGENCY - how much control the actor has:
  High   : decided, launched, demanded, imposed
  Medium : negotiated, requested, urged, proposed
  Low    : affected, pressured, influenced

MORAL_VALENCE - how the actor is portrayed (media framing, not truth):
  Positive : responsible, legitimate, praised, successful
  Neutral  : descriptive
  Negative : criticized, failed

CAUSAL_PRESENT - is an explicit cause-effect relation stated in the document?
  Yes : explicit causality present (e.g. "because of", "led to", "resulted in", "due to"; Bangla: "কারণে", "ফলে")
  No  : no causal relation stated

POLICY_DIRECTION - the action implied:
  Cooperation : dialogue, aid | Pressure : sanctions, criticism | Intervention : direct involvement | Neutrality : balance, avoidance

ALIGNMENT - the geopolitical alignment implied by the narrative. Consider the direct or indirect beneficiary: for example, if the action
  involves a country that mostly allies with the western bloc from a geopolitical lens, the narrative is Pro-West.
  Pro-West   : US/EU aligned, or the direct/indirect beneficiary is the western bloc
  Pro-China  : China aligned, or the direct/indirect beneficiary is China
  Non-aligned: strategic neutrality (an active stance; NOT passive neutrality)
  Mixed      : multiple alignments
"""

SCHEMAS = {"paper": SCHEMA_PAPER, "codebook": SCHEMA_CODEBOOK}

JSON_TEMPLATE = ('{"frame": "...", "actor": "...", "actor_role": "...", "agency": "...", '
                 '"moral_valence": "...", "causal_present": "...", "policy_direction": "...", '
                 '"alignment": "..."}')

# Conditions are a 2x2 of independent switches: few-shot exemplars x chain-of-thought.
CONDITIONS = {"zs": (False, False), "zs_cot": (False, True),
              "fs": (True, False),  "fs_cot": (True, True)}

def select_exemplars(dev_items, k=5, mode="default"):
    """default: one short dev item per frame (alignment mix is whatever that gives: 4 Non-aligned + 1 Pro-West).
    balanced: covers the alignment labels (Pro-West, Pro-China, Mixed, Non-aligned x2), varied frames/actors, shortest first."""
    if mode == "balanced":
        chosen, used_f, used_a = [], set(), set()
        for lab in ["Pro-West", "Pro-China", "Mixed", "Non-aligned", "Non-aligned"]:
            c = [d for d in dev_items if d["gold"]["alignment"] == lab and d not in chosen and len(d["text_en"]) < 3000]
            c.sort(key=lambda d: (d["gold"]["frame"] in used_f, d["gold"]["actor"] in used_a, len(d["text_en"]), d["id"]))
            if c: chosen.append(c[0]); used_f.add(c[0]["gold"]["frame"]); used_a.add(c[0]["gold"]["actor"])
        return chosen[:k]
    """Deterministic: one short dev item per frame, preferring unseen (actor, alignment) pairs."""
    chosen, seen = [], set()
    for frame in VALID_LABELS["frame"]:
        cands = [d for d in dev_items if d["gold"]["frame"] == frame and len(d["text_en"]) < 1800]
        cands.sort(key=lambda d: ((d["gold"]["actor"], d["gold"]["alignment"]) in seen,
                                  len(d["text_en"]), d["id"]))
        if cands:
            chosen.append(cands[0]); seen.add((cands[0]["gold"]["actor"], cands[0]["gold"]["alignment"]))
    return chosen[:k]

def build_prompt(text, condition, exemplars=None, lang="en", schema="codebook"):
    fewshot, cot = CONDITIONS[condition]
    parts = ["You are an expert in international relations and geopolitical narrative analysis.",
             SCHEMAS[schema]]
    if fewshot:
        parts.append("=== EXAMPLES ===")
        for i, ex in enumerate(exemplars, 1):
            parts.append(f"Example {i}:\nDocument: {ex['text_' + lang]}\nOutput:\n"
                         + json.dumps(ex["gold"], ensure_ascii=False, indent=1))
    parts.append("=== TASK ===\nAnalyze the document below.\n- Focus on NARRATIVE FRAMING, not factual truth\n"
                 "- Choose ONLY ONE label per category, using exactly the label spellings above")
    if cot:
        parts.append("- First reason step by step following the codebook order (main story, frame, actor, role, agency, valence, causal claim, policy, alignment), "
                     "then end with a line 'FINAL_JSON:' followed by the JSON object")
    else:
        parts.append("- Output VALID JSON only, no explanation, no markdown fences")
    parts.append(f"\nDocument:\n{text}\n\nOutput format: {JSON_TEMPLATE}\n"
                 + ("Reasoning then FINAL_JSON:" if cot else "Output (JSON only):"))
    return "\n\n".join(parts)

def extract_json(raw):
    """Return the LAST JSON object in the text that looks like an annotation (>=4 schema keys).
    Scans every '{' with raw_decode, so repeated objects, fences and brace-containing reasoning are all safe."""
    if not raw: return None
    dec, best, i = json.JSONDecoder(), None, raw.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(raw, i)
            if isinstance(obj, dict):
                d = {str(k).lower(): v for k, v in obj.items()}
                if sum(k in VALID_LABELS for k in d) >= 4: best = d
            i = raw.find("{", end)
        except json.JSONDecodeError:
            i = raw.find("{", i + 1)
    return best

def validate(pred):
    """Out-of-schema labels -> None (kept separate so invalid rate is measurable)."""
    out = {}
    for c in CATEGORIES:
        v = (pred or {}).get(c)
        v = v.strip() if isinstance(v, str) else v
        if c == "actor" and isinstance(v, str) and v and v not in VALID_LABELS[c]:
            # the codebook defines Other as "any country or organization not listed": a single unlisted actor (Japan, EU, ...) IS Other,
            # exactly as the gold labels were cleaned. Applied to every model. Multi-actor answers stay invalid.
            if v in ("United States", "USA", "U.S.", "United States of America"): v = "US"
            elif not any(sep in v for sep in (",", "/", ";", " and ", "|")) and len(v) <= 40: v = "Other"
        out[c] = v if v in VALID_LABELS[c] else None
    return out
