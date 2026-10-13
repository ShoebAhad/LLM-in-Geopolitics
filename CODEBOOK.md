# Annotation Codebook (v2.1, single source of truth for the prompts)

Bilingual (Bangla / English), theory-grounded annotation of how foreign-policy events are narrated: **narrative framing, actor agency, causal reasoning, geopolitical alignment**.
`bfpn/schema.py::SCHEMA_CODEBOOK` is generated from this document by hand; if one changes, change the other.

## 1. Unit and principle
- **Unit:** one full document (document level, not sentence level).
- **Principle:** annotate how the narrative is constructed, **not** whether it is true.
- **Do not annotate:** every entity, every sentence, or syntactic features. Only concepts that map to foreign-policy theory.
- Exactly **one label per category**.

## 2. Workflow (follow in order)
1. Read the full document. 2. Ask "what is the main story?" 3. Frame. 4. Primary actor. 5. Actor role. 6. Agency. 7. Moral valence. 8. Causal claim. 9. Policy direction. 10. Alignment.

## 3. Schema

| Category | Question | Labels |
|---|---|---|
| `frame` | Dominant lens through which the foreign-policy action is described | Humanitarian · Security · Economic · Strategic · Normative |
| `actor` | Main decision-maker / narrative focus | Bangladesh · India · Pakistan · Sri Lanka · Bhutan · Myanmar · China · US · UN · **Other** |
| `actor_role` | How the actor is positioned (narrative role, not legal role) | Initiator · Respondent · Mediator · Victim |
| `agency` | How much control the actor has | High · Medium · Low |
| `moral_valence` | How the actor is portrayed (media framing, not truth) | Positive · Neutral · Negative |
| `causal_present` | Is an explicit cause-effect relation stated? | Yes · No |
| `policy_direction` | Action implied | Cooperation · Pressure · Intervention · Neutrality |
| `alignment` | Geopolitical alignment implied (direct or indirect beneficiary) | Pro-West · Pro-China · Non-aligned · Mixed |

### Frame
Humanitarian: refugees, suffering, moral obligation. Security: threats, conflict, borders. Economic: trade, investment, sanctions. Strategic: power, alliances, influence (cues: partnership, influence, cooperation, alliance). Normative: law, legitimacy, global norms.
**Rule:** choose the dominant frame only; ignore secondary signals. (There is no "Economic + Strategic → Strategic" tie-break.)

### Actor
- **Label set (updated 5 Oct):** Bangladesh, India, Pakistan, Sri Lanka, Bhutan, Myanmar, China, US, UN, Other. Labels are **country names** (`US` and `UN` as written).
- **Other** = any country not in the list (e.g. Japan, Nepal, Saudi Arabia) **and any organization or bloc other than the UN** (EU, ASEAN, BRICS, QUAD, IOM, ...).
- Hierarchy: (1) actor initiating the action, (2) actor most discussed, (3) actor being evaluated, (4) actor mentioned first.
- Tie-break (confirmed by authors, 5 Oct): **Bangladesh** when Bangladesh is involved; otherwise the actor that **acts as the major (leading, dominant) party** — e.g. India or Pakistan when that country is the one driving the narrative.
- Cases: A symmetric interaction ("Bangladesh and India agreed") → Bangladesh. B external pressure ("US pressured Bangladesh") → US. C mediation ("UN facilitated talks") → UN. D victim narrative ("Bangladesh suffered") → Bangladesh. E background actor ("Bangladesh acted with support from China") → Bangladesh.

### Actor role / agency / valence
Role: Initiator starts the action; Respondent reacts; Mediator facilitates; Victim suffers.
Agency: High (decided, launched, demanded, imposed); Medium (negotiated, requested, urged, proposed); Low (affected, pressured, influenced).
Valence: Positive (responsible, legitimate, praised, successful); Neutral (descriptive); Negative (criticized, failed).

### Causal claim
Yes: explicit causality present ("because of", "led to", "resulted in", "due to"; Bangla "কারণে", "ফলে"). No: no causal relation stated.

### Policy direction
Cooperation (dialogue, aid); Pressure (sanctions, criticism); Intervention (direct involvement); Neutrality (balance, avoidance).

### Alignment
Consider the **direct or indirect beneficiary**. Example: if the action involves a country that mostly allies with the western bloc, the narrative is Pro-West.
Pro-West: US/EU aligned or the beneficiary is the western bloc. Pro-China: China aligned or the beneficiary is China. Non-aligned: strategic neutrality (an active stance, not passive neutrality). Mixed: multiple alignments.

## 4. Data decisions (applied by `clean_data.py`)
- Rows with `actor_role = Unclear` are dropped.
- `Legal` frame → Normative. Spelling normalised: `USA`→US, `Srilanka`→Sri Lanka, `Malayasia`→Malaysia, `Pro_China`→Pro-China, `Non-Aligned`→Non-aligned, `Co-Operation`→Cooperation.
- Actors outside the label set (incl. EU, ASEAN, BRICS, QUAD, IOM and all other countries) → `Other`; the original is kept in `actor_raw`.
- Same text with conflicting labels is dropped (7 items) until the annotators resolve it.

## 5. Decisions and open items
Resolved (5 Oct): actor tie-break without Bangladesh = the actor acting as the major party (India / Pakistan); the 7 duplicate-text label conflicts stay **dropped**.
Still open: (1) confirm `Other` (41 items, 7.8%) stays one class; (2) independent labels from both annotators for the κ table.
