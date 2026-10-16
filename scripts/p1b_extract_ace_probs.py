"""
P1-B: Extract ACE-Net sigmoid probabilities for the fair threshold audit.

Loads the best_model.pt checkpoint for each seed, runs CPU-only inference on
the GoEmotions val and test sets, and saves the raw sigmoid probabilities.
Run p1b_threshold_audit_fair.py afterwards for the symmetric comparison.

Runs on CPU only: CUDA_VISIBLE_DEVICES is set to "" before importing torch.

Usage:
  python scripts/p1b_extract_ace_probs.py

Output (per seed):
  results/ace_net_goemotions_L8_K32_audit/seed_{seed}/
    val_probs.npy, test_probs.npy, val_labels.npy, test_labels.npy, env_log.json
"""

import json
import os
import sys
import time
from typing import List

import numpy as np

# CPU only: set CUDA_VISIBLE_DEVICES before any torch import.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.registry import get_loader
from src.models.ace_net_classifier import ACENetClassifier

SEEDS: List[int]  = [42, 123, 456, 789, 1024]
CKPT_DIR: str     = "results/ace_net_goemotions_L8_K32"
OUTPUT_DIR: str   = "results/ace_net_goemotions_L8_K32_audit"
BACKBONE: str     = "microsoft/deberta-v3-base"
MAX_LENGTH: int   = 128
BATCH_SIZE: int   = 8
INJECT_LAYER: int = 8
N_PROTOTYPES: int = 32
TAU_MIN: float    = 0.3
TAU_SCALE: float  = 1.0
DATASET: str      = "goemotions"
CACHE_DIR: str    = ".cache/datasets"
MODEL_CACHE: str  = ".cache/models"


def build_dataloader(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> DataLoader:
    """Minimal DataLoader from pre-tokenized tensors."""
    from torch.utils.data import TensorDataset
    ds = TensorDataset(input_ids, attention_mask, labels)

    class _CollateFn:
        def __call__(self, batch):
            ids   = torch.stack([b[0] for b in batch])
            masks = torch.stack([b[1] for b in batch])
            lbls  = torch.stack([b[2] for b in batch])
            return {"input_ids": ids, "attention_mask": masks, "labels": lbls}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_CollateFn())


@torch.no_grad()
def run_inference(model: ACENetClassifier, loader: DataLoader) -> np.ndarray:
    """Return sigmoid probs (N, C) from forward passes."""
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        logits = model(input_ids, attention_mask)
        probs  = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def build_env_log(seed: int) -> dict:
    """Minimal env manifest for reproducibility."""
    import platform, socket, hashlib
    ckpt_path = os.path.join(CKPT_DIR, f"seed_{seed}", "best_model.pt")
    try:
        with open(ckpt_path, "rb") as f:
            data_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        data_hash = "N/A"
    return {
        "script": __file__,
        "seed": seed,
        "ckpt_path": ckpt_path,
        "ckpt_sha256_prefix": data_hash,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "cpu",
        "inject_layer": INJECT_LAYER,
        "n_prototypes": N_PROTOTYPES,
        "tau_min": TAU_MIN,
        "tau_scale": TAU_SCALE,
    }


def main() -> None:
    print("=" * 70)
    print("P1-B: Extract ACE-Net probabilities (CPU only)")
    print(f"  CKPT_DIR: {CKPT_DIR}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print("=" * 70)

    print("\nLoading GoEmotions dataset...")
    data_loader = get_loader(DATASET, cache_dir=CACHE_DIR)
    assert data_loader.n_labels == 28, f"Expected 28 labels, got {data_loader.n_labels}"
    assert data_loader.is_multilabel, "GoEmotions must be multi-label"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BACKBONE, cache_dir=MODEL_CACHE, use_fast=True
    )

    def tokenize_texts(texts, max_length=MAX_LENGTH):
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    print("  Loading splits via load_splits()...")
    _, val_records, test_records = data_loader.load_splits()
    val_texts_list  = [r["text"] for r in val_records]
    val_labels_np   = np.array([r["label"] for r in val_records], dtype=np.float32)
    test_texts_list = [r["text"] for r in test_records]
    test_labels_np  = np.array([r["label"] for r in test_records], dtype=np.float32)

    print("  Tokenizing val split...")
    val_ids, val_masks = tokenize_texts(val_texts_list)
    val_labels_arr = torch.tensor(val_labels_np, dtype=torch.float32)

    print("  Tokenizing test split...")
    test_ids, test_masks = tokenize_texts(test_texts_list)
    test_labels_arr = torch.tensor(test_labels_np, dtype=torch.float32)

    print(f"  val:  {val_ids.shape[0]} samples  labels={val_labels_np.shape}")
    print(f"  test: {test_ids.shape[0]} samples  labels={test_labels_np.shape}")

    val_loader  = build_dataloader(val_ids,  val_masks,  val_labels_arr,  BATCH_SIZE)
    test_loader = build_dataloader(test_ids, test_masks, test_labels_arr, BATCH_SIZE)

    for seed in SEEDS:
        print(f"\n{'-'*50}")
        print(f"Seed {seed}")
        print(f"{'-'*50}")

        ckpt_path = os.path.join(CKPT_DIR, f"seed_{seed}", "best_model.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        t0 = time.time()

        # weights_only=False: the checkpoint stores non-tensor values (epoch, val_macro_f1).
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd   = ckpt["state_dict"]

        proto_np = sd["prototype_dict"].float().numpy()
        assert proto_np.shape == (N_PROTOTYPES, 768), (
            f"Unexpected prototype_dict shape: {proto_np.shape}"
        )

        model = ACENetClassifier(
            backbone=BACKBONE,
            n_labels=data_loader.n_labels,
            dict_init=proto_np,
            inject_layer=INJECT_LAYER,
            n_prototypes=N_PROTOTYPES,
            tau_min=TAU_MIN,
            tau_scale=TAU_SCALE,
            use_ib_gate=False,
            is_multilabel=True,
            cache_dir=MODEL_CACHE,
        )
        model.load_state_dict(sd, strict=True)
        model.eval()
        print(f"  Loaded: epoch={ckpt['epoch']}, val_macro_f1={ckpt['val_macro_f1']:.4f}")

        assert not model.prototype_dict.requires_grad, "prototype_dict must be frozen"

        print("  Running val inference...")
        val_probs = run_inference(model, val_loader)
        print(f"  val_probs: {val_probs.shape}, range=[{val_probs.min():.4f}, {val_probs.max():.4f}]")

        print("  Running test inference...")
        test_probs = run_inference(model, test_loader)
        print(f"  test_probs: {test_probs.shape}, range=[{test_probs.min():.4f}, {test_probs.max():.4f}]")

        # Sanity: macro_f1 at 0.5 should match the saved test_metrics.json.
        from sklearn.metrics import f1_score as sk_f1
        preds = (test_probs >= 0.5).astype(int)
        macro_check = float(sk_f1(test_labels_arr.numpy().astype(int), preds, average="macro", zero_division=0))
        print(f"  Sanity macro_f1(0.5)={macro_check:.4f}")

        existing_path = os.path.join(CKPT_DIR, f"seed_{seed}", "test_metrics.json")
        if os.path.exists(existing_path):
            with open(existing_path) as f:
                existing = json.load(f)
            expected = float(existing["macro_f1"])
            diff = abs(macro_check - expected)
            print(f"  Existing macro_f1={expected:.4f}  diff={diff:.5f}", end="")
            if diff < 1e-3:
                print("  [OK]")
            else:
                print(f"  [WARN] diff={diff:.5f} > 1e-3")

        out_seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
        os.makedirs(out_seed_dir, exist_ok=True)

        np.save(os.path.join(out_seed_dir, "val_probs.npy"), val_probs.astype(np.float32))
        np.save(os.path.join(out_seed_dir, "test_probs.npy"), test_probs.astype(np.float32))
        np.save(os.path.join(out_seed_dir, "val_labels.npy"), val_labels_arr.numpy().astype(np.int8))
        np.save(os.path.join(out_seed_dir, "test_labels.npy"), test_labels_arr.numpy().astype(np.int8))

        env_log = build_env_log(seed)
        env_log["sanity_macro_f1_at_05"] = float(macro_check)
        with open(os.path.join(out_seed_dir, "env_log.json"), "w") as f:
            json.dump(env_log, f, indent=2)

        elapsed = time.time() - t0
        print(f"  Saved to {out_seed_dir}  [{elapsed:.0f}s]")

        del model

    print("\n" + "=" * 70)
    print("P1-B COMPLETE")
    print(f"  Probs saved in: {OUTPUT_DIR}")
    print("  Next: run p1b_threshold_audit_fair.py for the symmetric comparison")
    print("=" * 70)


if __name__ == "__main__":
    main()
