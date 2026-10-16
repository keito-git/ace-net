#!/usr/bin/env python3
"""
P1-D: ACE-Net context-vector analysis (CPU only).

Provides a direct analysis of the input-dependent context vectors and a
mean-vector substitution test for whether the full mechanism behaves as a
constant learned prior.

For each seed it:
  1. runs ACE-Net over the GoEmotions test set and captures the per-input
     context vector c_i = pi_i^T D via model._live_c;
  2. measures how input-dependent those vectors are (cosine of each c_i to the
     mean cbar, and relative dispersion ||c_i - cbar|| / ||cbar||);
  3. substitutes the single mean vector cbar for every input via
     model._swap_context and re-runs inference (mean-vector substitution);
  4. compares macro-F1 (normal) vs macro-F1 (mean-substituted).

If the mean-substituted score matches the normal score and the vectors cluster
tightly around cbar, the adaptive allocation is inert and the mechanism acts as
a constant learned prior (the paper's diagnostic claim).

CPU only. No training, no LLM. Reuses the loader and checkpoints from
p1b_extract_ace_probs.py. Output: results/revision/p1d_context_vectors.{json,md}
"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, before torch import
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.registry import get_loader
from src.models.ace_net_classifier import ACENetClassifier
from sklearn.metrics import f1_score

HERE        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR    = os.path.join(HERE, "results", "ace_net_goemotions_L8_K32")
OUT_DIR     = os.path.join(HERE, "results", "revision")
BACKBONE    = "microsoft/deberta-v3-base"
MODEL_CACHE = ".cache/models"
SEEDS       = [42, 123, 456, 789, 1024]
MAX_LENGTH  = 128
BATCH_SIZE  = 8
INJECT_LAYER, N_PROTOTYPES, TAU_MIN, TAU_SCALE = 8, 32, 0.3, 1.0
os.makedirs(OUT_DIR, exist_ok=True)


def collate(batch):
    return {"input_ids": torch.stack([b[0] for b in batch]),
            "attention_mask": torch.stack([b[1] for b in batch]),
            "labels": torch.stack([b[2] for b in batch])}


@torch.no_grad()
def infer_capture_c(model, loader):
    """Normal mode: return probs (N,C) and stacked context vectors C (N,d)."""
    model.eval(); model._swap_context = None
    probs, cs = [], []
    for batch in loader:
        logits = model(batch["input_ids"], batch["attention_mask"])
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        cs.append(model._live_c.cpu().numpy())  # (B,d) set in the hook
    return np.concatenate(probs), np.concatenate(cs)


@torch.no_grad()
def infer_swap(model, loader, cbar):
    """Mean-substitution: inject the single vector cbar for every input."""
    model.eval()
    cbar_t = torch.tensor(cbar, dtype=torch.float32)
    probs = []
    for batch in loader:
        B = batch["input_ids"].shape[0]
        model._swap_context = cbar_t.unsqueeze(0).expand(B, -1).clone()
        logits = model(batch["input_ids"], batch["attention_mask"])
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    model._swap_context = None
    return np.concatenate(probs)


def main():
    loader = get_loader("goemotions")
    _, _, test_records = loader.load_splits()
    texts  = [r["text"] for r in test_records]
    labels = np.array([r["label"] for r in test_records], dtype=np.int64)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BACKBONE, cache_dir=MODEL_CACHE, use_fast=True)
    enc = tok(texts, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    labels_t = torch.tensor(labels, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], labels_t),
                             batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    agg = {k: [] for k in ["f1_normal", "f1_meansub", "cos_to_mean_mean",
                           "cos_to_mean_p05", "rel_dispersion", "pred_agreement"]}
    per_seed = {}
    for seed in SEEDS:
        ckpt_path = os.path.join(CKPT_DIR, f"seed_{seed}", "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"seed {seed}: checkpoint missing, skip"); continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        proto = sd["prototype_dict"].float().numpy()
        model = ACENetClassifier(backbone=BACKBONE, n_labels=loader.n_labels, dict_init=proto,
                                 inject_layer=INJECT_LAYER, n_prototypes=N_PROTOTYPES,
                                 tau_min=TAU_MIN, tau_scale=TAU_SCALE, use_ib_gate=False,
                                 is_multilabel=True, cache_dir=MODEL_CACHE)
        model.load_state_dict(sd, strict=True); model.eval()

        t0 = time.time()
        probs_n, C = infer_capture_c(model, test_loader)   # C: (N,d)
        cbar = C.mean(0)                                    # (d,)
        probs_s = infer_swap(model, test_loader, cbar)

        f1_n = f1_score(labels, (probs_n >= 0.5).astype(int), average="macro", zero_division=0)
        f1_s = f1_score(labels, (probs_s >= 0.5).astype(int), average="macro", zero_division=0)

        cbar_norm = np.linalg.norm(cbar) + 1e-12
        cos = (C @ cbar) / (np.linalg.norm(C, axis=1) * cbar_norm + 1e-12)
        rel_disp = np.linalg.norm(C - cbar, axis=1).mean() / cbar_norm
        pred_agree = float((((probs_n >= 0.5) == (probs_s >= 0.5)).mean()))

        agg["f1_normal"].append(f1_n); agg["f1_meansub"].append(f1_s)
        agg["cos_to_mean_mean"].append(float(cos.mean()))
        agg["cos_to_mean_p05"].append(float(np.percentile(cos, 5)))
        agg["rel_dispersion"].append(float(rel_disp))
        agg["pred_agreement"].append(pred_agree)
        per_seed[seed] = {"f1_normal": f1_n, "f1_meansub": f1_s,
                          "cos_to_mean_mean": float(cos.mean()),
                          "cos_to_mean_p05": float(np.percentile(cos, 5)),
                          "rel_dispersion": float(rel_disp),
                          "pred_agreement": pred_agree,
                          "cbar_norm": float(cbar_norm)}
        print(f"seed {seed}: f1_normal={f1_n:.4f} f1_meansub={f1_s:.4f} "
              f"cos_to_mean={cos.mean():.3f} rel_disp={rel_disp:.3f} "
              f"agree={pred_agree:.3f}  [{time.time()-t0:.0f}s]")

    def ms(x):
        return [float(np.mean(x)), float(np.std(x, ddof=1) if len(x) > 1 else 0.0), len(x)]
    report = {"seeds": SEEDS, "per_seed": per_seed,
              "summary": {k: ms(v) for k, v in agg.items()},
              "delta_f1_meansub_minus_normal": float(np.mean(agg["f1_meansub"]) - np.mean(agg["f1_normal"]))}
    with open(os.path.join(OUT_DIR, "p1d_context_vectors.json"), "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    def fmt(x): return f"{x[0]:.4f}+/-{x[1]:.4f}"
    md = ["# P1-D Context-vector analysis (mean-vector substitution)\n",
          f"Seeds: {list(per_seed.keys())}\n",
          "| Quantity | Value |", "|---|---|",
          f"| macro-F1 normal ACE-Net | {fmt(s['f1_normal'])} |",
          f"| macro-F1 mean-vector substitution | {fmt(s['f1_meansub'])} |",
          f"| delta (meansub - normal) | {report['delta_f1_meansub_minus_normal']:+.4f} |",
          f"| cosine(c_i, cbar) mean | {fmt(s['cos_to_mean_mean'])} |",
          f"| cosine(c_i, cbar) 5th pct | {fmt(s['cos_to_mean_p05'])} |",
          f"| relative dispersion ||c_i-cbar||/||cbar|| | {fmt(s['rel_dispersion'])} |",
          f"| prediction agreement (0.5) normal vs meansub | {fmt(s['pred_agreement'])} |",
          "\nInterpretation: cosine near 1 and low dispersion => the per-input "
          "context vectors cluster tightly around their mean; a near-zero delta "
          "=> replacing every input's vector with the single mean leaves macro-F1 "
          "essentially unchanged, i.e. the adaptive allocation is inert and the "
          "mechanism acts as a constant learned prior (the paper's diagnostic claim)."]
    with open(os.path.join(OUT_DIR, "p1d_context_vectors.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nSaved: {OUT_DIR}/p1d_context_vectors.json and .md")


if __name__ == "__main__":
    main()
