"""
P1-A: Conservative threshold audit.

ACE-Net is evaluated at a fixed 0.5 threshold (read from the saved
test_metrics.json). The baseline is given a free per-class threshold
optimization (tuned on its own val split, applied to test). If ACE-Net still
dominates under this asymmetric handicap, it is a strong conservative bound.
The symmetric (fair) comparison is in p1b_threshold_audit_fair.py.

Usage:
  python scripts/p1a_threshold_audit_conservative.py

Output:
  results/p1a_threshold_audit_conservative.json
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SEEDS: List[int] = [42, 123, 456, 789, 1024]
ACE_DIR: str     = "results/ace_net_goemotions_L8_K32"
BASELINE_DIR: str = "results/deberta_goemotions_audit"
OUTPUT_JSON: str  = "results/p1a_threshold_audit_conservative.json"
N_THRESHOLDS: int = 200
TAIL3: List[str]  = ["pride", "nervousness", "relief"]

# Reference values (ddof=1) at a fixed 0.5 threshold.
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
    thr,
    label_names: List[str],
    tail3: List[str],
) -> Dict:
    """Evaluate predictions with a scalar or per-class threshold."""
    if isinstance(thr, np.ndarray) and thr.ndim == 1:
        preds = (probs >= thr[None, :]).astype(int)
    else:
        preds = (probs >= thr).astype(int)

    macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    micro = float(f1_score(labels, preds, average="micro", zero_division=0))
    weighted = float(f1_score(labels, preds, average="weighted", zero_division=0))

    tail3_f1 = {}
    for cls in tail3:
        if cls in label_names:
            c = label_names.index(cls)
            tail3_f1[cls] = float(
                f1_score(labels[:, c], preds[:, c], zero_division=0)
            )

    return {
        "macro_f1": macro,
        "micro_f1": micro,
        "weighted_f1": weighted,
        "tail3_f1": tail3_f1,
    }


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled std."""
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.std(a, ddof=1) ** 2 + (n2 - 1) * np.std(b, ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    return float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else 0.0


def bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, ci: float = 0.95
) -> Tuple[float, float]:
    """Bootstrap 95% CI for the mean difference (a - b)."""
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(float(np.mean(sa) - np.mean(sb)))
    alpha = (1.0 - ci) / 2
    return float(np.percentile(diffs, alpha * 100)), float(
        np.percentile(diffs, (1 - alpha) * 100)
    )


def main() -> None:
    print("=" * 70)
    print("P1-A: Conservative Threshold Audit")
    print("  ACE-Net: fixed 0.5 threshold (from test_metrics.json)")
    print("  Baseline: val -> per-class opt threshold -> test")
    print("=" * 70)

    sys.path.insert(0, ".")
    from src.data.registry import get_loader

    loader = get_loader("goemotions", cache_dir=".cache/datasets")
    label_names: List[str] = loader.label_names
    assert len(label_names) == 28, f"Expected 28 labels, got {len(label_names)}"

    # ACE-Net per-seed macro_f1 at the fixed 0.5 threshold.
    print("\n[ACE-Net] Loading per-seed test_metrics.json (fixed threshold 0.5)")
    ace_macro: List[float] = []
    for seed in SEEDS:
        path = os.path.join(ACE_DIR, f"seed_{seed}", "test_metrics.json")
        assert os.path.exists(path), f"Not found: {path}"
        with open(path) as f:
            m = json.load(f)
        macro = float(m["macro_f1"])
        ace_macro.append(macro)
        print(f"  seed={seed}: macro_f1={macro:.4f}")

    ace_arr = np.array(ace_macro)
    ace_mean = float(np.mean(ace_arr))
    ace_std  = float(np.std(ace_arr, ddof=1))
    print(f"  ACE-Net mean={ace_mean:.4f} +/- {ace_std:.4f}")

    # Baseline val -> optimal threshold -> test.
    print("\n[Baseline] val -> per-class optimal threshold -> test evaluation")
    baseline_macro: List[float] = []
    baseline_opt_thr: List[np.ndarray] = []
    baseline_full: List[Dict] = []

    for seed in SEEDS:
        sd = os.path.join(BASELINE_DIR, f"seed_{seed}")
        val_probs  = np.load(os.path.join(sd, "val_probs.npy"))
        val_labels = np.load(os.path.join(sd, "val_labels.npy"))
        test_probs  = np.load(os.path.join(sd, "test_probs.npy"))
        test_labels = np.load(os.path.join(sd, "test_labels.npy"))

        assert val_probs.shape[1]  == 28, f"val_probs shape error: {val_probs.shape}"
        assert test_probs.shape[1] == 28, f"test_probs shape error: {test_probs.shape}"

        opt_thr = opt_thresholds_from_val(val_probs, val_labels)
        baseline_opt_thr.append(opt_thr)

        res = eval_with_threshold(test_probs, test_labels, opt_thr, label_names, TAIL3)
        baseline_macro.append(res["macro_f1"])
        baseline_full.append(res)

        print(
            f"  seed={seed}: macro_f1(opt)={res['macro_f1']:.4f}  "
            f"thr_mean={opt_thr.mean():.3f}  thr_std={opt_thr.std():.3f}"
        )

    bl_arr = np.array(baseline_macro)
    bl_mean = float(np.mean(bl_arr))
    bl_std  = float(np.std(bl_arr, ddof=1))
    print(f"  Baseline(opt) mean={bl_mean:.4f} +/- {bl_std:.4f}")

    print("\n[Stats] ACE-Net(0.5) vs Baseline(opt)")
    t_stat, p_val = stats.ttest_ind(ace_arr, bl_arr, equal_var=False)
    d = cohen_d(ace_arr, bl_arr)
    ci_lo, ci_hi = bootstrap_ci(ace_arr, bl_arr)
    delta = ace_mean - bl_mean

    print(f"  Delta (ACE - BL_opt): {delta:+.4f}")
    print(f"  Welch t={t_stat:.4f}, p={p_val:.4f}")
    print(f"  Cohen's d={d:.4f}")
    print(f"  95% Bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    print(
        f"\n  Reference: baseline(0.5)={MACRO_BASELINE_FIXED:.4f}, "
        f"ACE-Net(0.5)={MACRO_ACE_FIXED:.4f}, "
        f"baseline(opt)={bl_mean:.4f}"
    )
    print(f"  Delta(opt vs fixed): {bl_mean - MACRO_BASELINE_FIXED:+.4f}")

    print("\n[Tail-3] Comparison (ACE-Net fixed vs Baseline opt)")
    tail3_summary = {}
    for cls in TAIL3:
        bl_tail = [r["tail3_f1"].get(cls, float("nan")) for r in baseline_full]
        bl_tail_mean = float(np.mean(bl_tail))
        bl_tail_std  = float(np.std(bl_tail, ddof=1))
        print(f"  {cls}: baseline_opt={bl_tail_mean:.4f} +/- {bl_tail_std:.4f}")
        tail3_summary[cls] = {
            "baseline_opt_mean": bl_tail_mean,
            "baseline_opt_std": bl_tail_std,
            "values": [float(v) for v in bl_tail],
        }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    result = {
        "experiment": "P1-A Conservative Threshold Audit",
        "note": (
            "ACE-Net uses a fixed 0.5 threshold. "
            "The baseline uses a val-optimal per-class threshold. "
            "The symmetric comparison is in p1b."
        ),
        "ace_net": {
            "threshold": "fixed_0.5",
            "macro_f1_values": [float(v) for v in ace_arr],
            "macro_f1_mean": ace_mean,
            "macro_f1_std": ace_std,
        },
        "baseline_opt_threshold": {
            "threshold": "val_optimal_per_class",
            "macro_f1_values": [float(v) for v in bl_arr],
            "macro_f1_mean": bl_mean,
            "macro_f1_std": bl_std,
        },
        "statistics": {
            "delta_ace_minus_bl_opt": float(delta),
            "welch_t": float(t_stat),
            "welch_p": float(p_val),
            "significant_at_05": bool(p_val < 0.05),
            "cohens_d": float(d),
            "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        },
        "reference": {
            "baseline_fixed_05": MACRO_BASELINE_FIXED,
            "ace_net_fixed_05": MACRO_ACE_FIXED,
            "delta_opt_gain_for_baseline": float(bl_mean - MACRO_BASELINE_FIXED),
        },
        "tail3": tail3_summary,
        "seeds": SEEDS,
        "n_thresholds": N_THRESHOLDS,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved: {OUTPUT_JSON}")
    print("=" * 70)
    print("P1-A COMPLETE")
    print(f"  ACE-Net(0.5): {ace_mean:.4f} +/- {ace_std:.4f}")
    print(f"  Baseline(opt): {bl_mean:.4f} +/- {bl_std:.4f}")
    print(
        f"  Delta={delta:+.4f}  p={p_val:.4f}  "
        f"{'ACE-Net wins even with baseline threshold opt' if delta > 0 else 'Baseline(opt) > ACE-Net(fixed)'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
