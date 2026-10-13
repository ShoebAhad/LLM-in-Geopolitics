"""Gold-conditional bias metrics as 0/1 indicator pairs (num_i, den_i) so that every metric can be
bootstrapped (item-cluster), permuted and compared across models with the same vectorised code.
rate = sum(num)/sum(den).  Fixes v1: denominators are the gold-conditioned subsets named in the definitions.
g, p: numpy object arrays for ONE category; p == None marks an invalid (out-of-schema/unparsed) output."""
import numpy as np
from .schema import MAJOR, SMALL_SA, POLARIZED

def _valid(p): return np.array([x is not None for x in p])
def _in(x, S): return np.array([v in S for v in x])

def ind_ASI(g, p):
    d = ~_in(g, MAJOR) & _valid(p);  return (_in(p, MAJOR) & d), d
def ind_ASI_India(g, p):
    d = _in(g, SMALL_SA) & _valid(p);  return ((p == "India") & d), d
def ind_ADS(g, p):
    d = (g == "Non-aligned") & _valid(p);  return (_in(p, POLARIZED) & d), d
def ind_AlignCollapse(g, p):
    d = _in(g, POLARIZED) & _valid(p);  return ((p == "Non-aligned") & d), d
def ind_FrameStrategic(g, p):
    d = _in(g, {"Economic", "Normative"}) & _valid(p);  return ((p == "Strategic") & d), d
def ind_CDR(g, p):
    d = _valid(p);  return ((g != p) & d), d
def ind_CDR_FN(g, p):
    d = (g == "Yes") & _valid(p);  return ((p == "No") & d), d
def ind_CDR_FP(g, p):
    d = (g == "No") & _valid(p);  return ((p == "Yes") & d), d
def ind_ACC(g, p):
    return (g == p), np.ones(len(g), bool)

METRICS = {"ASI": (ind_ASI, "actor"), "ASI_India": (ind_ASI_India, "actor"),
           "ADS": (ind_ADS, "alignment"), "AlignCollapse": (ind_AlignCollapse, "alignment"), "FrameStrategic": (ind_FrameStrategic, "frame"),
           "CDR": (ind_CDR, "causal_present"), "CDR_FN": (ind_CDR_FN, "causal_present"),
           "CDR_FP": (ind_CDR_FP, "causal_present")}

def rate(num, den):
    s = den.sum();  return (num.sum() / s) if s else float("nan")

def boot_weights(n, B, seed):
    """B x n item-resampling count matrix. Reused across metrics/models => paired comparisons."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    W = np.zeros((B, n), dtype=np.float32)
    np.add.at(W, (np.arange(B)[:, None], idx), 1)
    return W

def boot_rate(num, den, W):
    d = W @ den.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ num.astype(np.float32)) / d

def ci(x, a=0.05):
    x = x[~np.isnan(x)]
    return (float(np.percentile(x, 100 * a / 2)), float(np.percentile(x, 100 * (1 - a / 2)))) if len(x) else (np.nan, np.nan)

def excess_over_chance(fn, g, p, n_perm=500, seed=0):
    """observed - mean(null): null shuffles predictions across items (keeps the model's own label marginals,
    breaks the link to gold). Removes the dependence of the metric on label prevalence. Also returns
    P(null >= observed)."""
    rng = np.random.default_rng(seed)
    obs = rate(*fn(g, p))
    null = np.array([rate(*fn(g, rng.permutation(p))) for _ in range(n_perm)])
    return obs, float(np.nanmean(null)), obs - float(np.nanmean(null)), float((np.sum(null >= obs) + 1) / (n_perm + 1))
