"""
P1-B: Fair (symmetric) threshold audit.

Both ACE-Net and the baseline are evaluated with val-optimal per-class
thresholds. Run after p1b_extract_ace_probs.py has produced the ACE-Net
probabilities. This is the main threshold table; p1a is the conservative bound.

Usage:
  python scripts/p1b_threshold_audit_fair.py

Output:
  results/p1b_threshold_audit_fair.json
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SEEDS: List[int]   = [42, 123, 456, 789, 1024]
ACE_AUDIT_DIR: str = "results/ace_net_goemotions_L8_K32_audit"
BL_AUDIT_DIR: str  = "results/deberta_goemotions_audit"
OUTPUT_JSON: str   = "results/p1b_threshold_audit_fair.json"
N_THRESHOLDS: int  = 200
TAIL3: List[str]   = ["pride", "nervousness", "relief"]

# Reference values (ddof=1) at a fixed 0.5 threshold (sanity checks).
MACRO_BASELINE_FIXED = 0.4559
MACRO_ACE_FIXED      = 0.4965


def opt_thresholds_from_val(
    val_probs: np.ndarray, val_labels: np.ndarray
) -> np.ndarray:
    """Find the per-class optimal threshold on the val set. Returns a (28,) array."""
    thresholds = np.linspace(0.05, 0.95, N_THRESHOLDS)
    best_thr = np.full(28, 0.5)
    for c in range(28):
        best_f1 = -1.0
        for t in thresholds:
            preds = (val_probs[:, c] >= t).astype(int)
            f1 = float(f1_score(val_labels[:, c], preds, zero_division=0))
            if f1 > best_f1:
                best_f1, best_thr[c] = f1, t
    return best_thr


def eval_with_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    thr: np.ndarray,
    label_names: List[str],
    tail3: List[str],
) -> Dict:
    """Evaluate predictions with a per-class threshold array."""
    preds = (probs >= thr[None, :]).astype(int)
    macro    = float(f1_score(labels, preds, average="macro", zero_division=0))
    micro    = float(f1_score(labels, preds, average="micro", zero_division=0))
    weighted = float(f1_score(labels, preds, average="weighted", zero_division=0))
    tail3_f1 = {
        cls: float(f1_score(labels[:, label_names.index(cls)], preds[:, label_names.index(cls)], zero_division=0))
        for cls in tail3 if cls in label_names
    }
    return {"macro_f1": macro, "micro_f1": micro, "weighted_f1": weighted, "tail3_f1": tail3_f1}


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(
        ((n1 - 1) * np.std(a, ddof=1) ** 2 + (n2 - 1) * np.std(b, ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, ci: float = 0.95):
    rng = np.random.default_rng(0)
    diffs = [float(np.mean(rng.choice(a, len(a), replace=True)) - np.mean(rng.choice(b, len(b), replace=True)))
             for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2
    return float(np.percentile(diffs, alpha * 100)), float(np.percentile(diffs, (1 - alpha) * 100))


def process_model(
    audit_dir: str,
    label_names: List[str],
    tail3: List[str],
    model_name: str,
) -> Tuple[List[float], List[Dict]]:
    """Load probs per seed, apply the val-optimal threshold, return per-seed results."""
    macro_list = []
    full_list  = []
    for seed in SEEDS:
        sd = os.path.join(audit_dir, f"seed_{seed}")
        assert os.path.exists(sd), f"[{model_name}] seed dir not found: {sd}"
        val_probs   = np.load(os.path.join(sd, "val_probs.npy"))
        val_labels  = np.load(os.path.join(sd, "val_labels.npy"))
        test_probs  = np.load(os.path.join(sd, "test_probs.npy"))
        test_labels = np.load(os.path.join(sd, "test_labels.npy"))

        opt_thr = opt_thresholds_from_val(val_probs, val_labels)
        res     = eval_with_threshold(test_probs, test_labels, opt_thr, label_names, tail3)
        macro_list.append(res["macro_f1"])
        full_list.append(res)
        print(f"  [{model_name}] seed={seed}: macro_f1(opt)={res['macro_f1']:.4f}")
    return macro_list, full_list


def main() -> None:
    print("=" * 70)
    print("P1-B: Fair (Symmetric) Threshold Audit")
    print("  Both ACE-Net and Baseline: val -> opt per-class threshold -> test")
    print("=" * 70)

    sys.path.insert(0, ".")
    from src.data.registry import get_loader
    loader = get_loader("goemotions", cache_dir=".cache/datasets")
    label_names: List[str] = loader.label_names
    assert len(label_names) == 28

    print("\n[ACE-Net] val -> opt threshold -> test")
    ace_macro, ace_full = process_model(ACE_AUDIT_DIR, label_names, TAIL3, "ACE-Net")

    print("\n[Baseline] val -> opt threshold -> test")
    bl_macro, bl_full = process_model(BL_AUDIT_DIR, label_names, TAIL3, "Baseline")

    ace_arr = np.array(ace_macro)
    bl_arr  = np.array(bl_macro)

    ace_mean, ace_std = float(np.mean(ace_arr)), float(np.std(ace_arr, ddof=1))
    bl_mean,  bl_std  = float(np.mean(bl_arr)),  float(np.std(bl_arr, ddof=1))

    print(f"\n  ACE-Net(opt): {ace_mean:.4f} +/- {ace_std:.4f}")
    print(f"  Baseline(opt): {bl_mean:.4f} +/- {bl_std:.4f}")

    delta   = ace_mean - bl_mean
    t, p    = stats.ttest_ind(ace_arr, bl_arr, equal_var=False)
    d       = cohen_d(ace_arr, bl_arr)
    ci_lo, ci_hi = bootstrap_ci(ace_arr, bl_arr)

    print("\n[Stats] ACE-Net(opt) vs Baseline(opt)")
    print(f"  Delta: {delta:+.4f}  Welch t={t:.4f}  p={p:.4f}")
    print(f"  Cohen's d={d:.4f}")
    print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    print("\n[Tail-3] ACE-Net(opt) vs Baseline(opt)")
    tail3_out = {}
    for cls in TAIL3:
        ace_t  = [r["tail3_f1"].get(cls, float("nan")) for r in ace_full]
        bl_t   = [r["tail3_f1"].get(cls, float("nan")) for r in bl_full]
        print(f"  {cls}: ACE={np.mean(ace_t):.4f}+/-{np.std(ace_t, ddof=1):.4f}  "
              f"BL={np.mean(bl_t):.4f}+/-{np.std(bl_t, ddof=1):.4f}")
        tail3_out[cls] = {
            "ace_opt_mean": float(np.mean(ace_t)), "ace_opt_std": float(np.std(ace_t, ddof=1)),
            "bl_opt_mean":  float(np.mean(bl_t)),  "bl_opt_std":  float(np.std(bl_t, ddof=1)),
            "delta": float(np.mean(ace_t) - np.mean(bl_t)),
        }

    result = {
        "experiment": "P1-B Fair (Symmetric) Threshold Audit",
        "note": "Both models use a val-optimal per-class threshold. Main threshold table.",
        "ace_net_opt_threshold": {
            "macro_f1_values": [float(v) for v in ace_arr],
            "macro_f1_mean": ace_mean,
            "macro_f1_std":  ace_std,
        },
        "baseline_opt_threshold": {
            "macro_f1_values": [float(v) for v in bl_arr],
            "macro_f1_mean": bl_mean,
            "macro_f1_std":  bl_std,
        },
        "statistics": {
            "delta_ace_minus_baseline": float(delta),
            "welch_t": float(t),
            "welch_p": float(p),
            "significant_at_05": bool(p < 0.05),
            "cohens_d": float(d),
            "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        },
        "reference_fixed_05": {
            "baseline": MACRO_BASELINE_FIXED,
            "ace_net":  MACRO_ACE_FIXED,
        },
        "tail3": tail3_out,
        "seeds": SEEDS,
        "n_thresholds": N_THRESHOLDS,
    }
    os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved: {OUTPUT_JSON}")
    print("=" * 70)
    print("P1-B COMPLETE")
    print(f"  ACE-Net(opt):  {ace_mean:.4f} +/- {ace_std:.4f}")
    print(f"  Baseline(opt): {bl_mean:.4f} +/- {bl_std:.4f}")
    print(f"  Delta={delta:+.4f}  p={p:.4f}  d={d:.4f}")
    outcome = "ACE-Net > Baseline even with symmetric threshold opt" if delta > 0 else "Baseline(opt) >= ACE-Net(opt)"
    print(f"  {outcome}")
    print("=" * 70)


if __name__ == "__main__":
    main()
