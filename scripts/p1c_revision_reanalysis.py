#!/usr/bin/env python3
"""
P1-C: Revision re-analysis (CPU only, no model, no GPU).

Consumes the saved sigmoid probabilities for the DeBERTa baseline and ACE-Net
(L8/K32) on GoEmotions, produced by p1b_extract_ace_probs.py, and computes:

  T1  threshold analysis
      (a) fixed-0.5 macro-F1 (sanity vs paper: BL 0.4559 / ACE 0.4965)
      (b) symmetric fair audit: per-class threshold tuned on each method's
          own val split, applied to test -> macro-F1 gap
      (c) threshold-independent: macro Average Precision (area under PR)
  T2  per-class breakdown at 0.5: support (test + train), precision, recall,
      F1, predicted-positive count, and AP, for BL and ACE; tail-3 flagged
  T3  label-frequency control: per-class dF1 and dAP vs train frequency
  T7  simple signals: per-example baseline confidence (max prob) and margin
      (top1-top2) correlated with the per-example improvement, compared
      against the label-based dL signal

All figures are mean +/- std over the five seeds. Nothing here trains or calls
an LLM. Outputs land in results/revision/.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.registry import get_loader

from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

HERE      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ACE_DIR   = os.path.join(HERE, "results", "ace_net_goemotions_L8_K32_audit")
BL_DIR    = os.path.join(HERE, "results", "deberta_goemotions_audit")
OUT_DIR   = os.path.join(HERE, "results", "revision")
SEEDS     = [42, 123, 456, 789, 1024]
EPS       = 1e-12

os.makedirs(OUT_DIR, exist_ok=True)


def load_probs(base, seed, split):
    d = os.path.join(base, f"seed_{seed}")
    p = os.path.join(d, f"{split}_probs.npy")
    l = os.path.join(d, f"{split}_labels.npy")
    if not (os.path.exists(p) and os.path.exists(l)):
        return None, None
    return np.load(p).astype(np.float64), np.load(l).astype(np.int64)


def macro_f1_at(probs, labels, thr):
    """thr: scalar or per-class vector."""
    pred = (probs >= thr).astype(np.int64)
    return f1_score(labels, pred, average="macro", zero_division=0)


def tune_per_class_threshold(val_probs, val_labels, grid):
    """Per-class threshold maximizing that class's F1 on val."""
    C = val_labels.shape[1]
    thr = np.full(C, 0.5)
    for c in range(C):
        best_f1, best_t = -1.0, 0.5
        yc = val_labels[:, c]
        pc = val_probs[:, c]
        for t in grid:
            f1 = f1_score(yc, (pc >= t).astype(np.int64), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[c] = best_t
    return thr


def per_example_f1(probs, labels, thr):
    pred = (probs >= thr).astype(np.int64)
    tp = (pred & labels).sum(1)
    fp = (pred & (1 - labels)).sum(1)
    fn = ((1 - pred) & labels).sum(1)
    denom = (2 * tp + fp + fn)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1


def per_example_bce(probs, labels):
    p = np.clip(probs, EPS, 1 - EPS)
    return -(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean(1)


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() < EPS or y.std() < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    loader = get_loader("goemotions")
    names  = list(loader.label_names)
    C      = loader.n_labels
    train_records, _, _ = loader.load_splits()
    train_labels = np.array([r["label"] for r in train_records], dtype=np.int64)
    train_support = train_labels.sum(0)
    tail3_idx = list(np.argsort(train_support)[:3])          # 3 rarest by train frequency
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)

    report = {"seeds": SEEDS, "label_names": names,
              "train_support": {names[i]: int(train_support[i]) for i in range(C)},
              "tail3": [names[i] for i in tail3_idx]}

    acc = {k: [] for k in
           ["bl_f1_0.5", "ace_f1_0.5",
            "bl_f1_opt", "ace_f1_opt",
            "bl_mAP", "ace_mAP"]}
    perclass = {k: np.zeros((len(SEEDS), C)) for k in
                ["bl_P", "bl_R", "bl_F1", "bl_posrate", "bl_AP",
                 "ace_P", "ace_R", "ace_F1", "ace_posrate", "ace_AP"]}
    test_support_runs = []
    forge = {k: [] for k in ["r_conf_df1", "r_margin_df1", "r_dL_df1",
                             "r_conf_df1_fc", "r_margin_df1_fc", "r_dL_df1_fc"]}

    missing = []
    for si, seed in enumerate(SEEDS):
        bp, bl_lab = load_probs(BL_DIR, seed, "test")
        ap, ace_lab = load_probs(ACE_DIR, seed, "test")
        bvp, bvl = load_probs(BL_DIR, seed, "val")
        avp, avl = load_probs(ACE_DIR, seed, "val")
        if bp is None or ap is None:
            missing.append(seed); continue
        labels = bl_lab
        assert np.array_equal(bl_lab, ace_lab), f"label mismatch seed {seed}"
        test_support_runs.append(labels.sum(0))

        # T1a fixed 0.5
        acc["bl_f1_0.5"].append(macro_f1_at(bp, labels, 0.5))
        acc["ace_f1_0.5"].append(macro_f1_at(ap, labels, 0.5))
        # T1b symmetric fair: tune each on its own val
        bthr = tune_per_class_threshold(bvp, bvl, grid)
        athr = tune_per_class_threshold(avp, avl, grid)
        acc["bl_f1_opt"].append(macro_f1_at(bp, labels, bthr))
        acc["ace_f1_opt"].append(macro_f1_at(ap, labels, athr))
        # T1c threshold-independent macro AP
        acc["bl_mAP"].append(average_precision_score(labels, bp, average="macro"))
        acc["ace_mAP"].append(average_precision_score(labels, ap, average="macro"))

        # T2 per-class at 0.5
        bpred = (bp >= 0.5).astype(np.int64)
        apred = (ap >= 0.5).astype(np.int64)
        for c in range(C):
            perclass["bl_P"][si, c]  = precision_score(labels[:, c], bpred[:, c], zero_division=0)
            perclass["bl_R"][si, c]  = recall_score(labels[:, c], bpred[:, c], zero_division=0)
            perclass["bl_F1"][si, c] = f1_score(labels[:, c], bpred[:, c], zero_division=0)
            perclass["bl_posrate"][si, c] = bpred[:, c].sum()
            perclass["bl_AP"][si, c] = average_precision_score(labels[:, c], bp[:, c]) if labels[:, c].sum() > 0 else np.nan
            perclass["ace_P"][si, c]  = precision_score(labels[:, c], apred[:, c], zero_division=0)
            perclass["ace_R"][si, c]  = recall_score(labels[:, c], apred[:, c], zero_division=0)
            perclass["ace_F1"][si, c] = f1_score(labels[:, c], apred[:, c], zero_division=0)
            perclass["ace_posrate"][si, c] = apred[:, c].sum()
            perclass["ace_AP"][si, c] = average_precision_score(labels[:, c], ap[:, c]) if labels[:, c].sum() > 0 else np.nan

        # T7 simple signals vs the label-based dL signal (per example, at 0.5)
        conf   = bp.max(1)
        sorted_p = np.sort(bp, 1)
        margin = sorted_p[:, -1] - sorted_p[:, -2]           # top1 - top2
        df1    = per_example_f1(ap, labels, 0.5) - per_example_f1(bp, labels, 0.5)
        dL     = per_example_bce(bp, labels) - per_example_bce(ap, labels)  # positive = ACE lowers loss
        forge["r_conf_df1"].append(pearson(conf, df1))
        forge["r_margin_df1"].append(pearson(margin, df1))
        forge["r_dL_df1"].append(pearson(dL, df1))
        # frequency-controlled subset: examples carrying at least one rarer label
        has_rare = labels[:, np.argsort(train_support)[:8]].sum(1) > 0
        if has_rare.sum() > 10:
            forge["r_conf_df1_fc"].append(pearson(conf[has_rare], df1[has_rare]))
            forge["r_margin_df1_fc"].append(pearson(margin[has_rare], df1[has_rare]))
            forge["r_dL_df1_fc"].append(pearson(dL[has_rare], df1[has_rare]))

    def ms(x):
        x = [v for v in x if v == v]
        return [float(np.mean(x)), float(np.std(x, ddof=1) if len(x) > 1 else 0.0), len(x)]

    report["missing_seeds"] = missing
    report["T1_threshold"] = {
        "fixed_0.5": {"baseline": ms(acc["bl_f1_0.5"]), "ace": ms(acc["ace_f1_0.5"]),
                      "gap_mean": float(np.mean(acc["ace_f1_0.5"]) - np.mean(acc["bl_f1_0.5"]))},
        "symmetric_val_optimal": {"baseline": ms(acc["bl_f1_opt"]), "ace": ms(acc["ace_f1_opt"]),
                                  "gap_mean": float(np.mean(acc["ace_f1_opt"]) - np.mean(acc["bl_f1_opt"]))},
        "threshold_independent_macroAP": {"baseline": ms(acc["bl_mAP"]), "ace": ms(acc["ace_mAP"]),
                                          "gap_mean": float(np.mean(acc["ace_mAP"]) - np.mean(acc["bl_mAP"]))},
    }

    test_support = np.mean(np.array(test_support_runs), 0) if test_support_runs else np.zeros(C)
    pc_out = {}
    for c in range(C):
        pc_out[names[c]] = {
            "train_support": int(train_support[c]),
            "test_support": float(test_support[c]),
            "is_tail3": c in tail3_idx,
            "baseline": {"P": ms(perclass["bl_P"][:, c]), "R": ms(perclass["bl_R"][:, c]),
                         "F1": ms(perclass["bl_F1"][:, c]), "pred_pos": ms(perclass["bl_posrate"][:, c]),
                         "AP": ms(perclass["bl_AP"][:, c])},
            "ace": {"P": ms(perclass["ace_P"][:, c]), "R": ms(perclass["ace_R"][:, c]),
                    "F1": ms(perclass["ace_F1"][:, c]), "pred_pos": ms(perclass["ace_posrate"][:, c]),
                    "AP": ms(perclass["ace_AP"][:, c])},
        }
    report["T2_per_class"] = pc_out

    # T3 label-frequency control: correlation of per-class gain with train freq
    dF1 = (perclass["ace_F1"] - perclass["bl_F1"]).mean(0)
    dAP = (np.nan_to_num(perclass["ace_AP"]) - np.nan_to_num(perclass["bl_AP"])).mean(0)
    report["T3_freq_control"] = {
        "pearson_dF1_vs_trainfreq": pearson(train_support, dF1),
        "pearson_dAP_vs_trainfreq": pearson(train_support, dAP),
        "mean_dF1_tail3": float(np.mean([dF1[i] for i in tail3_idx])),
        "mean_dF1_nontail": float(np.mean([dF1[i] for i in range(C) if i not in tail3_idx])),
        "mean_dAP_tail3": float(np.mean([dAP[i] for i in tail3_idx])),
        "mean_dAP_nontail": float(np.mean([dAP[i] for i in range(C) if i not in tail3_idx])),
    }

    report["T7_simple_signals_vs_forge"] = {
        "full_test": {"conf_vs_dF1": ms(forge["r_conf_df1"]),
                      "margin_vs_dF1": ms(forge["r_margin_df1"]),
                      "dL_vs_dF1": ms(forge["r_dL_df1"])},
        "rare_subset": {"conf_vs_dF1": ms(forge["r_conf_df1_fc"]),
                        "margin_vs_dF1": ms(forge["r_margin_df1_fc"]),
                        "dL_vs_dF1": ms(forge["r_dL_df1_fc"])},
    }

    with open(os.path.join(OUT_DIR, "p1c_reanalysis.json"), "w") as f:
        json.dump(report, f, indent=2)

    T1 = report["T1_threshold"]
    def fmt(x): return f"{x[0]:.4f}+/-{x[1]:.4f} (n={x[2]})"
    lines = []
    lines.append("# P1-C Revision Re-analysis (CPU, from saved probabilities)\n")
    lines.append(f"Seeds used: {[s for s in SEEDS if s not in missing]}  missing={missing}\n")
    lines.append("## T1 Threshold analysis\n")
    lines.append("| Setting | Baseline | ACE-Net | Gap |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Fixed 0.5 (sanity) | {fmt(T1['fixed_0.5']['baseline'])} | {fmt(T1['fixed_0.5']['ace'])} | {T1['fixed_0.5']['gap_mean']:+.4f} |")
    lines.append(f"| Symmetric val-optimal | {fmt(T1['symmetric_val_optimal']['baseline'])} | {fmt(T1['symmetric_val_optimal']['ace'])} | {T1['symmetric_val_optimal']['gap_mean']:+.4f} |")
    lines.append(f"| Macro AP (threshold-free) | {fmt(T1['threshold_independent_macroAP']['baseline'])} | {fmt(T1['threshold_independent_macroAP']['ace'])} | {T1['threshold_independent_macroAP']['gap_mean']:+.4f} |")
    lines.append("\n## T3 Frequency control\n")
    t3 = report["T3_freq_control"]
    lines.append(f"- Pearson(dF1, train freq) = {t3['pearson_dF1_vs_trainfreq']:.3f}  (negative => gain concentrates on rare classes)")
    lines.append(f"- Pearson(dAP, train freq) = {t3['pearson_dAP_vs_trainfreq']:.3f}  (threshold-free)")
    lines.append(f"- mean dF1 tail-3 = {t3['mean_dF1_tail3']:+.4f} ; non-tail = {t3['mean_dF1_nontail']:+.4f}")
    lines.append(f"- mean dAP tail-3 = {t3['mean_dAP_tail3']:+.4f} ; non-tail = {t3['mean_dAP_nontail']:+.4f}")
    lines.append("\n## T7 Simple signals vs label-based dL (Pearson r with per-example dF1)\n")
    f7 = report["T7_simple_signals_vs_forge"]
    lines.append("| Signal | Full test | Rare subset |")
    lines.append("|---|---|---|")
    lines.append(f"| baseline confidence | {fmt(f7['full_test']['conf_vs_dF1'])} | {fmt(f7['rare_subset']['conf_vs_dF1'])} |")
    lines.append(f"| baseline margin | {fmt(f7['full_test']['margin_vs_dF1'])} | {fmt(f7['rare_subset']['margin_vs_dF1'])} |")
    lines.append(f"| dL (label-based) | {fmt(f7['full_test']['dL_vs_dF1'])} | {fmt(f7['rare_subset']['dL_vs_dF1'])} |")
    lines.append("\n## T2 Per-class (tail-3 detail)\n")
    lines.append("| Class | train n | BL F1 | ACE F1 | BL AP | ACE AP | BL predpos | ACE predpos |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in range(C):
        if c in tail3_idx:
            d = pc_out[names[c]]
            lines.append(f"| {names[c]} | {d['train_support']} | {d['baseline']['F1'][0]:.3f} | {d['ace']['F1'][0]:.3f} | {d['baseline']['AP'][0]:.3f} | {d['ace']['AP'][0]:.3f} | {d['baseline']['pred_pos'][0]:.1f} | {d['ace']['pred_pos'][0]:.1f} |")
    with open(os.path.join(OUT_DIR, "p1c_reanalysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved: {OUT_DIR}/p1c_reanalysis.json and .md")


if __name__ == "__main__":
    main()
