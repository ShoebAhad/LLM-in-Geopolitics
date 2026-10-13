"""
Paper figures for the v2 experiments (seven LLMs). Reads data/bfpn_v2.jsonl, results/v2/<tag>-codebook/*.jsonl and
results/v2/analysis/T13/T14, writes 300-dpi PNG + vector PDF to figures/v2/.
    ~/.venvs/bfpn/bin/python make_figures_v2.py
Every number is from run 0 with the English and Bangla versions of each item pooled; intervals are 95% item-cluster
bootstrap CIs (the same 518 scored items as analyze.py: all items minus the few-shot exemplars and bfpn_1719f47ad4).
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from analyze import load
from bfpn.schema import CATEGORIES, select_exemplars
from bfpn.metrics import METRICS, ind_ACC, boot_weights, ci

OUT = "figures/v2"; A = "results/v2/analysis"; os.makedirs(OUT, exist_ok=True)
MODELS = [("gemma4-31b", "Gemma 4 31B"), ("qwen38-27b", "Qwen3.8 27B"), ("ministral3-14b", "Ministral 3 14B"),
          ("gpt-oss-20b", "gpt-oss 20B"), ("llama31-8b", "Llama 3.1 8B"), ("haiku55", "Claude Haiku 5.5"), ("gpt6-luna", "GPT-6 Luna")]
N_OPEN = 5   # the first five are open-weight, the last two are closed
CONDS = [("zs", "Zero-shot"), ("zs_cot", "Zero-shot\n+ CoT"), ("fs", "Few-shot"), ("fs_cot", "Few-shot\n+ CoT")]
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.weight": "bold", "font.size": 12, "axes.labelweight": "bold",
                     "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.labelsize": 12, "xtick.labelsize": 11.5,
                     "ytick.labelsize": 12, "legend.fontsize": 11.5, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})

# ── data ──────────────────────────────────────────────────────────────────────
items = [json.loads(l) for l in open("data/bfpn_v2.jsonl")]
dev = [i for i in items if i["split"] == "dev"]
ex_ids = {e["id"] for m in ("default", "balanced") for e in select_exemplars(dev, mode=m)}
items = [i for i in items if i["id"] not in ex_ids and i["id"] != "bfpn_1719f47ad4"]
gold, P, _ = load(items); n = len(items); W = boot_weights(n, 5000, seed=42)
def pred(tag, cond, lang): return P[(f"{tag}-codebook", cond, lang, 0)]

def pooled(tag, cond, fn):
    """fn(gold_dict, pred_dict) -> per-item (num, den); summed over the two language versions, bootstrapped over items."""
    num = np.zeros(n); den = np.zeros(n)
    for lang in ("en", "bn"):
        a, b = fn(gold, pred(tag, cond, lang)); num += a; den += b
    bs = (W @ num.astype(np.float32)) / (W @ den.astype(np.float32)); lo, hi = ci(bs)
    return num.sum() / den.sum(), lo, hi
def metric(name): return lambda g, p: METRICS[name][0](g[METRICS[name][1]], p[METRICS[name][1]])
def acc(g, p): return np.mean([ind_ACC(g[c], p[c])[0] for c in CATEGORIES], axis=0), np.ones(n)
def share_shift(cat, label):   # predicted share minus gold share of one label
    return lambda g, p: ((p[cat] == label).astype(float) - (g[cat] == label).astype(float), np.ones(n))

# ── shared drawing helpers ────────────────────────────────────────────────────
def model_axis(ax, labels=True):
    y = np.arange(len(MODELS))[::-1]
    ax.set_yticks(y); ax.set_yticklabels([m[1] for m in MODELS])
    ax.set_ylim(-0.6, len(MODELS) - 0.4); ax.tick_params(axis="y", length=0)
    ax.axhline(len(MODELS) - N_OPEN - 0.5, color=INK2, lw=1.0, ls=(0, (4, 3)))   # open-weight above, closed below
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.9); ax.set_axisbelow(True)
    return y
def group_tags(ax):
    t = ax.get_yaxis_transform()
    ax.text(1.02, len(MODELS) - N_OPEN / 2 - 0.5, "Open-weight", rotation=270, va="center", ha="left", transform=t, color=INK2, fontsize=11)
    ax.text(1.02, (len(MODELS) - N_OPEN) / 2 - 0.5, "Closed", rotation=270, va="center", ha="left", transform=t, color=INK2, fontsize=11)
def zs_fs_dots(ax, fn, scale=1.0):
    """Zero-shot (blue circle) and few-shot (orange square) with 95% CI whiskers, one row per model."""
    y = model_axis(ax, labels=False)
    for (tag, _), yy in zip(MODELS, y):
        for cond, col, mk, dy in (("zs", BLUE, "o", 0.16), ("fs", ORANGE, "s", -0.16)):
            v, lo, hi = (scale * x for x in pooled(tag, cond, fn))
            ax.plot([lo, hi], [yy + dy] * 2, color=col, lw=2.2, solid_capstyle="round", zorder=2)
            ax.plot(v, yy + dy, mk, color=col, ms=8.5, mec="white", mew=1.4, zorder=3)
ZS_FS_LEGEND = [Line2D([], [], color=BLUE, marker="o", ms=8.5, mec="white", lw=2.2, label="Zero-shot"),
                Line2D([], [], color=ORANGE, marker="s", ms=8.5, mec="white", lw=2.2, label="Few-shot")]
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("wrote", f"{OUT}/{name}.png / .pdf")
def pct(ax): ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

# ── Fig 1: accuracy ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.4, 4.3))
zs_fs_dots(ax, acc, scale=100); ax.set_yticklabels([m[1] for m in MODELS]); group_tags(ax); pct(ax)
ax.set_xlim(56, 76); ax.set_xticks(range(56, 77, 4)); ax.set_ylim(-0.6, len(MODELS) + 0.45); ax.set_xlabel("Accuracy, mean of the eight categories")
t1 = pd.read_csv(f"{A}/T1_main.csv"); t1 = t1[(t1.metric == "ACC_all8") & (t1.cond == "sup")]
for tag, lab, ha, dx in (("majority-class", "Majority class", "left", 0.25), ("tfidf-logreg", "TF-IDF + logistic\nregression (supervised)", "right", -0.25)):
    v = 100 * t1[t1.model == tag].value.mean()   # non-LLM reference: mean of the English and Bangla versions, 5-fold cross-validation
    ax.axvline(v, color=INK2, lw=1.6, ls=(0, (1, 2)), zorder=1)
    ax.text(v + dx, len(MODELS) - 0.48, lab, ha=ha, va="bottom", fontsize=10.5, color=INK2, linespacing=1.1)
ax.legend(handles=ZS_FS_LEGEND, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.6)
save(fig, "fig1_accuracy")

# ── Fig 2: where gold Non-aligned items go ────────────────────────────────────
ALIGN = [("Non-aligned", GREY, INK, "Non-aligned (correct)"), ("Mixed", AQUA, INK, "Mixed"),
         ("Pro-West", BLUE, "white", "Pro-West"), ("Pro-China", ORANGE, INK, "Pro-China")]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), sharey=True, gridspec_kw={"wspace": 0.15})
for ax, (cond, title) in zip(axes, (("zs", "Zero-shot"), ("fs", "Few-shot"))):
    y = model_axis(ax, labels=(ax is axes[0])); ax.grid(False)
    for (tag, _), yy in zip(MODELS, y):
        left = 0.0
        for lab, col, tcol, _ in ALIGN:
            v = 100 * pooled(tag, cond, lambda g, p, lab=lab: ((p["alignment"] == lab) & (g["alignment"] == "Non-aligned"),
                                                                 (g["alignment"] == "Non-aligned") & np.array([x is not None for x in p["alignment"]])))[0]
            ax.barh(yy, v, left=left, height=0.66, color=col, edgecolor="white", linewidth=2.0)
            if v >= 6: ax.text(left + v / 2, yy, f"{v:.0f}", ha="center", va="center", color=tcol, fontsize=11.5)
            left += v
    ax.set_xlim(0, 100); pct(ax); ax.set_title(title); ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
fig.supxlabel("Share of gold Non-aligned items, by predicted label", y=-0.03, fontsize=12, fontweight="bold")
group_tags(axes[1])
fig.legend(handles=[Patch(facecolor=c, edgecolor="white", label=l) for _, c, _, l in ALIGN], loc="lower center",
           bbox_to_anchor=(0.5, 0.96), ncol=4, frameon=False, handlelength=1.3, columnspacing=1.6)
save(fig, "fig2_alignment_destinations")

# ── Fig 3: shared errors ──────────────────────────────────────────────────────
CAT = {"frame": "Frame", "actor": "Actor", "actor_role": "Actor role", "agency": "Agency", "moral_valence": "Moral valence",
       "causal_present": "Causal link", "policy_direction": "Policy direction", "alignment": "Alignment"}
t13 = pd.read_csv(f"{A}/T13_shared_errors.csv"); t13 = t13[t13.cond == "zs"].sort_values("all_wrong")
assert int(t13.n_models.iloc[0]) == len(MODELS), "T13 was not computed over the seven models: re-run analyze.py"
t14 = pd.read_csv(f"{A}/T14_shared_error_direction.csv"); t14 = t14[t14.cond == "zs"]
DIRS = [("actor_role", "Respondent", "Initiator"), ("policy_direction", "Neutrality", "Cooperation"), ("frame", "Economic", "Strategic"),
        ("moral_valence", "Neutral", "Positive"), ("agency", "Medium", "High"), ("causal_present", "Yes", "No")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"wspace": 0.62})
yy = np.arange(len(t13)); obs = 100 * t13.all_wrong.values; exp = 100 * t13.expected_all_wrong_if_independent.values
a1.barh(yy, obs, height=0.62, color=BLUE)
a1.plot(exp, yy, "|", color=INK, ms=17, mew=2.6, zorder=3)
for y_, o in zip(yy, obs): a1.text(o + 0.5, y_, f"{o:.0f}%", va="center", ha="left", fontsize=11.5)
a1.set_yticks(yy); a1.set_yticklabels([CAT[c] for c in t13.category]); a1.set_xlim(0, 27); pct(a1)
a1.set_xlabel("Items that all seven models label wrongly"); a1.set_title("Errors shared by all seven models")
a1.legend(handles=[Patch(facecolor=BLUE, label="Observed"), Line2D([], [], color=INK, marker="|", ms=14, mew=2.6, lw=0, label="Expected if errors\nwere independent")],
          loc="lower right", frameon=False, handlelength=1.2, borderaxespad=0.1, labelspacing=0.7)
rows = []
for cat, g_, p_ in DIRS:
    r = t14[(t14.category == cat) & (t14.gold == g_) & (t14.shared_pred == p_)]
    if len(r): rows.append((f"{g_} → {p_}", CAT[cat], 100 * r.rate_of_gold.iloc[0], 100 * r.share_this_direction.iloc[0]))
rows = sorted(rows, key=lambda r: r[2]); yy = np.arange(len(rows))
a2.barh(yy, [r[2] for r in rows], height=0.62, color=BLUE)
for y_, r in zip(yy, rows): a2.text(r[2] + 1.2, y_, f"{r[2]:.0f}%", va="center", ha="left", fontsize=11.5)
a2.set_yticks(yy); a2.set_yticklabels([f"{r[0]}\n({r[1]})" for r in rows], linespacing=1.15); a2.set_xlim(0, 75); pct(a2)
a2.set_xlabel("Gold items given the same wrong label by all seven"); a2.set_title("Direction of the shared errors")
for ax in (a1, a2):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.grid(axis="x", color=GRID, lw=0.9); ax.set_axisbelow(True)
save(fig, "fig3_shared_errors")

# ── Fig 4: attribution and alignment bias on one scale ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True, gridspec_kw={"wspace": 0.16})
for ax, (name, title) in zip(axes, (("ASI_India", "Agency given to India\n(gold actor: a smaller state)"),
                                    ("ASI", "Agency given to a major power\n(gold actor: not a major power)"),
                                    ("ADS", "Non-aligned read as\nPro-West or Pro-China"))):
    zs_fs_dots(ax, metric(name), scale=100); ax.set_xlim(0, 60); pct(ax); ax.set_title(title); ax.set_xlabel("Share of gold items")
axes[0].set_yticklabels([m[1] for m in MODELS]); group_tags(axes[2])
fig.legend(handles=ZS_FS_LEGEND, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False, handlelength=1.6)
save(fig, "fig4_bias_metrics")

# ── Fig 5: prompt ablation ────────────────────────────────────────────────────
ramp = LinearSegmentedColormap.from_list("blue", ["#eef5fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), gridspec_kw={"wspace": 0.52})
for ax, (fn, title, vmin, vmax) in zip(axes, ((metric("ADS"), "Non-aligned read as Pro-West or Pro-China", 0, 55), (acc, "Accuracy, mean of the eight categories", 60, 73))):
    M = np.array([[100 * pooled(tag, c, fn)[0] for c, _ in CONDS] for tag, _ in MODELS])
    ax.imshow(M, cmap=ramp, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center", fontsize=12.5, color="white" if (M[i, j] - vmin) / (vmax - vmin) > 0.55 else INK)
    ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([c[1] for c in CONDS]); ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([m[1] for m in MODELS])
    ax.set_xticks(np.arange(-0.5, len(CONDS)), minor=True); ax.set_yticks(np.arange(-0.5, len(MODELS)), minor=True)
    ax.grid(which="minor", color="white", lw=2.5); ax.tick_params(which="both", length=0); ax.set_title(title, pad=10)
    ax.axhline(N_OPEN - 0.5, color=INK, lw=1.6)
    for s in ax.spines.values(): s.set_visible(False)
save(fig, "fig5_prompt_ablation")

# ── Fig 6: polarisation by gold actor (zero-shot) ─────────────────────────────
def pol_for(actor):
    def fn(g, p):
        d = (g["alignment"] == "Non-aligned") & (g["actor"] == actor) & np.array([x is not None for x in p["alignment"]])
        return np.array([x in ("Pro-West", "Pro-China") for x in p["alignment"]]) & d, d
    return fn
fig, ax = plt.subplots(figsize=(7.8, 4.5)); y = model_axis(ax)
ACT = [("India", BLUE, "o", 0.22), ("Pakistan", ORANGE, "s", 0.0), ("Bangladesh", AQUA, "^", -0.22)]
for (tag, _), yy in zip(MODELS, y):
    for actor, col, mk, dy in ACT:
        v, lo, hi = (100 * x for x in pooled(tag, "zs", pol_for(actor)))
        ax.plot([lo, hi], [yy + dy] * 2, color=col, lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot(v, yy + dy, mk, color=col, ms=8.5, mec="white", mew=1.4, zorder=3)
ax.set_xlim(-1.5, 92); pct(ax); group_tags(ax)
ax.set_xlabel("Gold Non-aligned items read as Pro-West or Pro-China (zero-shot)")
ax.legend(handles=[Line2D([], [], color=c, marker=m, ms=8.5, mec="white", lw=2.2, label=f"Actor: {a}") for a, c, m, _ in ACT],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.6, columnspacing=1.4)
save(fig, "fig6_polarisation_by_actor")

# ── Fig 7: frame and causal diagnostics ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True, gridspec_kw={"wspace": 0.16})
zs_fs_dots(axes[0], share_shift("frame", "Strategic"), scale=100); axes[0].axvline(0, color=INK, lw=1.2); axes[0].set_xlim(-8, 28)
axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
axes[0].set_title("Strategic frame:\npredicted share minus gold share"); axes[0].set_xlabel("Percentage points")
for ax, name, title in ((axes[1], "CDR_FN", "Causal links missed\n(gold: link present)"), (axes[2], "CDR_FP", "Causal links invented\n(gold: no link)")):
    zs_fs_dots(ax, metric(name), scale=100); ax.set_xlim(0, 85); pct(ax); ax.set_title(title); ax.set_xlabel("Share of gold items")
axes[0].set_yticklabels([m[1] for m in MODELS]); group_tags(axes[2])
fig.legend(handles=ZS_FS_LEGEND, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False, handlelength=1.6)
save(fig, "fig7_frame_causal")
