"""
ACE-Net fine-tuning loop for emotion classification.

Implements a custom training loop for ACENetClassifier using the same
hyperparameters as BaselineConfig for a fair comparison. The custom loop is
required for the diversity regularization loss and attention-mask pass-through.
Loss is BCEWithLogitsLoss for multi-label and CrossEntropyLoss for multi-class.
Total loss = task_loss + c3_lambda * c3_loss + diversity_weight * div_loss
           + spread_weight * spread_loss.

CUDA_VISIBLE_DEVICES must be set in os.environ before torch is imported; this
is handled by the training entry point.
"""
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from src.baselines.deberta_baseline import BaselineConfig
from src.data.base_loader import BaseEmotionLoader
from src.eval.metrics import compute_all_metrics, aggregate_seeds, save_metrics_json
from src.eval.subgroup_eval import evaluate_subgroups, format_subgroup_report
from src.models.ace_net_classifier import ACENetClassifier
from src.models.dictionary_builder import load_or_build_dictionary
from src.utils.env_log import log_environment
from src.utils.seed import set_seed


@dataclass
class ACENetConfig:
    """
    ACE-Net specific hyperparameters, extending BaselineConfig (shared HPs are
    not duplicated here).
    """
    # Prototype dictionary
    context_file: str = "data/raw/corpus_contexts.json"
    dict_cache_path: str = "data/processed/dict_K32.npy"
    n_prototypes: int = 32
    force_rebuild_dict: bool = False

    # Injection
    inject_layer: int = 8

    # Temperature
    tau_min: float = 0.3
    tau_scale: float = 1.0

    # IB gate (off by default)
    use_ib_gate: bool = False

    # Diversity regularization weight
    diversity_weight: float = 0.2

    # C3 swap-contrastive auxiliary loss (off by default)
    c3_loss: bool = False
    c3_lambda: float = 0.1
    c3_margin: float = 0.05
    c3_t_resample: int = 2
    c3_monitor: bool = False

    # Intra-batch c* spread loss (off by default). Minimizing the mean
    # off-diagonal cosine pushes the per-sample context vectors apart.
    spread_weight: float = 0.0

    # Ablation flags (default off = full ACE-Net)
    fixed_tau: bool = False
    no_gumbel: bool = False
    input_concat: bool = False
    free_vector: bool = False

    # Output
    output_dir: str = "results/ace_net_goemotions"


class _EmotionDataset(Dataset):
    """Thin Dataset wrapper holding tokenized tensors and labels."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        texts: List[str],
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.texts = texts

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "idx": idx,
        }


def _get_linear_schedule_with_warmup(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Linear warmup then linear decay."""
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def _tokenize_records(
    records: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    is_multilabel: bool,
    n_labels: int,
) -> _EmotionDataset:
    """Tokenize records and return an _EmotionDataset."""
    texts = [r["text"] for r in records]
    raw_labels = [r["label"] for r in records]

    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    if is_multilabel:
        label_tensor = torch.tensor(np.stack(raw_labels), dtype=torch.float32)
    else:
        label_tensor = torch.tensor(raw_labels, dtype=torch.long)

    return _EmotionDataset(
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"],
        labels=label_tensor,
        texts=texts,
    )


@torch.no_grad()
def _evaluate(
    model: ACENetClassifier,
    dataloader: DataLoader,
    is_multilabel: bool,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a dataloader and return (y_pred, y_true). For multi-label,
    y_pred is a thresholded binary matrix; for multi-class, an argmax array.
    """
    model.eval()
    device = next(model.parameters()).device
    all_logits, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        logits = model(input_ids, attention_mask)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    all_logits_cat = torch.cat(all_logits, dim=0).numpy()
    all_labels_cat = torch.cat(all_labels, dim=0).numpy()

    if is_multilabel:
        probs = 1.0 / (1.0 + np.exp(-all_logits_cat))
        y_pred = (probs >= threshold).astype(int)
    else:
        y_pred = np.argmax(all_logits_cat, axis=-1)

    return y_pred, all_labels_cat


@torch.no_grad()
def _compute_c_diversity(model, dataloader, device):
    """
    Compute context-vector diversity metrics on a dataloader: pairwise cosine
    similarity, effective number of prototypes, and allocation entropy. Returns
    an empty dict when no allocation is available (free_vector mode).
    """
    import numpy as _np
    model.eval()
    all_pi = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        _ = model(input_ids, attention_mask)
        diag = model.get_diagnostics()
        if 'pi' in diag and diag['pi'] is not None:
            all_pi.append(diag['pi'].float().cpu())

    if not all_pi:
        return {}

    import torch as _torch
    all_pi_np = _torch.cat(all_pi, dim=0).numpy()
    D_np = model.prototype_dict.float().cpu().numpy()
    c_star = all_pi_np @ D_np

    norms = _np.linalg.norm(c_star, axis=1, keepdims=True).clip(1e-9)
    c_norm = c_star / norms
    gram = c_norm @ c_norm.T
    N = gram.shape[0]
    triu_i, triu_j = _np.triu_indices(N, k=1)
    cos_vals = gram[triu_i, triu_j]

    pi_mean = all_pi_np.mean(axis=0)
    pi_safe = _np.clip(pi_mean, 1e-9, None)
    H_batch = float(-_np.sum(pi_safe * _np.log(pi_safe)))
    N_eff = float(_np.exp(H_batch))

    pi_samp = _np.clip(all_pi_np, 1e-9, None)
    H_sample = -_np.sum(pi_samp * _np.log(pi_samp), axis=1)

    return {
        'c_pairwise_cos_mean': float(_np.mean(cos_vals)),
        'c_pairwise_cos_std': float(_np.std(cos_vals)),
        'N_eff': N_eff,
        'pi_batch_entropy': H_batch,
        'pi_sample_entropy_mean': float(_np.mean(H_sample)),
        'pi_top3_usage': sorted(pi_mean.tolist(), reverse=True)[:3],
        'n_test_samples_div': int(N),
    }


def run_ace_net_baseline(
    loader: BaseEmotionLoader,
    cfg: BaselineConfig,
    ace_cfg: ACENetConfig,
    seed: int,
) -> Dict[str, Any]:
    """
    Train and evaluate ACENetClassifier on a single seed.

    The hyperparameters in cfg must match those used for the DeBERTa baseline
    for a valid fair comparison, and os.environ["CUDA_VISIBLE_DEVICES"] must be
    set by the caller before this function runs. Returns a dict with the test
    metrics, subgroup metrics, and training log.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = os.path.join(ace_cfg.output_dir, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    log_environment(
        seed=seed,
        dataset_name=loader.__class__.__name__,
        output_dir=run_dir,
        extra={
            "method": "ACENet-C3" if ace_cfg.c3_loss else "ACENet",
            "inject_layer": ace_cfg.inject_layer,
            "n_prototypes": ace_cfg.n_prototypes,
            "tau_min": ace_cfg.tau_min,
            "use_ib_gate": ace_cfg.use_ib_gate,
            "diversity_weight": ace_cfg.diversity_weight,
            "c3_loss": ace_cfg.c3_loss,
            "c3_lambda": ace_cfg.c3_lambda,
            "c3_margin": ace_cfg.c3_margin,
            "c3_t_resample": ace_cfg.c3_t_resample,
            "c3_monitor": ace_cfg.c3_monitor,
            "tau_scale": ace_cfg.tau_scale,
            "spread_weight": ace_cfg.spread_weight,
            "fixed_tau": ace_cfg.fixed_tau,
            "no_gumbel": ace_cfg.no_gumbel,
            "input_concat": ace_cfg.input_concat,
            "free_vector": ace_cfg.free_vector,
        },
    )

    c3_label = (f" | C3=ON lambda={ace_cfg.c3_lambda} m={ace_cfg.c3_margin}"
                if ace_cfg.c3_loss else " | C3=OFF")
    print(f"\n[ACE-Net] seed={seed} | device={device}{c3_label}")
    print(f"  inject_layer={ace_cfg.inject_layer} | K={ace_cfg.n_prototypes} | "
          f"tau_min={ace_cfg.tau_min} | tau_scale={ace_cfg.tau_scale} | gate={'ON' if ace_cfg.use_ib_gate else 'OFF'}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_records, val_records, test_records = loader.load_splits()
    train_ds = _tokenize_records(train_records, tokenizer, cfg.max_length,
                                  loader.is_multilabel, loader.n_labels)
    val_ds   = _tokenize_records(val_records,   tokenizer, cfg.max_length,
                                  loader.is_multilabel, loader.n_labels)
    test_ds  = _tokenize_records(test_records,  tokenizer, cfg.max_length,
                                  loader.is_multilabel, loader.n_labels)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.dataloader_num_workers, pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.dataloader_num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.dataloader_num_workers, pin_memory=True,
    )

    dict_init = load_or_build_dictionary(
        context_file=ace_cfg.context_file,
        cache_path=ace_cfg.dict_cache_path,
        n_clusters=ace_cfg.n_prototypes,
        backbone=cfg.model_name,
        force_rebuild=ace_cfg.force_rebuild_dict,
        cache_dir=".cache/models",
    )
    print(f"  Dictionary: {dict_init.shape} (from {ace_cfg.context_file})")

    model = ACENetClassifier(
        backbone=cfg.model_name,
        n_labels=loader.n_labels,
        dict_init=dict_init,
        inject_layer=ace_cfg.inject_layer,
        n_prototypes=ace_cfg.n_prototypes,
        tau_min=ace_cfg.tau_min,
        tau_scale=ace_cfg.tau_scale,
        use_ib_gate=ace_cfg.use_ib_gate,
        is_multilabel=loader.is_multilabel,
        cache_dir=".cache/models",
        fixed_tau=ace_cfg.fixed_tau,
        no_gumbel=ace_cfg.no_gumbel,
        input_concat=ace_cfg.input_concat,
        free_vector=ace_cfg.free_vector,
    ).to(device)

    model.assert_no_label_gradient_in_dict()
    print("  [OK] prototype_dict.requires_grad=False (label-free invariant)")

    if loader.is_multilabel:
        task_criterion = nn.BCEWithLogitsLoss()
    else:
        task_criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = _get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = cfg.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(run_dir, "best_model.pt")
    no_improve_count = 0
    train_log: List[Dict[str, float]] = []

    print(f"  Training: {cfg.num_epochs} epochs, {len(train_loader)} steps/epoch")

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_task_loss = 0.0
        epoch_div_loss = 0.0
        epoch_c3_loss = 0.0
        epoch_spread_loss = 0.0
        tau_list, H_list = [], []
        c3_c_sel_list: List[float] = []
        c3_d_match_list: List[float] = []
        c3_d_swap_list: List[float] = []
        c3_entropy_list: List[float] = []
        c3_maxprob_list: List[float] = []
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward-1: main pass (gradient attached).
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_1 = model(input_ids, attention_mask)

                if loader.is_multilabel:
                    task_loss = task_criterion(logits_1, labels.float())
                else:
                    task_loss = task_criterion(logits_1, labels.long())

                # Must be computed from forward-1's _live_pa before any forward-2.
                div_loss = model.compute_diversity_loss()

                # c* spread loss (anti-collapse): off-diagonal mean cosine of the
                # per-sample context vectors recomputed with gradient from _live_pa.
                spread_loss_val = torch.tensor(0.0, device=device)
                if ace_cfg.spread_weight > 0.0 and input_ids.size(0) >= 2:
                    live_pa = model._live_pa
                    c_live = live_pa @ model.prototype_dict.float()
                    c_live_norm = F.normalize(c_live, dim=-1)
                    cos_mat = c_live_norm @ c_live_norm.T
                    B = input_ids.size(0)
                    eye_mask = torch.eye(B, dtype=torch.bool, device=device)
                    spread_loss_val = cos_mat.masked_fill(eye_mask, 0.0).sum() / (B * (B - 1))

            # C3 swap-contrastive auxiliary loss (requires fp32, i.e. use_amp False).
            c3_loss_val = torch.tensor(0.0, device=device)
            if ace_cfg.c3_loss and input_ids.size(0) >= 2:
                c_batch = model.get_live_c()
                sigma_1 = torch.sigmoid(logits_1)

                # Forward-2: re-sample (new Gumbel noise + new dropout mask).
                with torch.no_grad():
                    logits_2 = model(input_ids, attention_mask)
                    sigma_2 = torch.sigmoid(logits_2)

                # Forward-swap: inject a permuted context from other examples.
                # Use a derangement (no fixed points) so d_swap != d_match.
                perm = torch.randperm(input_ids.size(0), device=device)
                arange = torch.arange(input_ids.size(0), device=device)
                fixed = (perm == arange).nonzero(as_tuple=True)[0]
                for idx in fixed:
                    swap_with = (idx + 1) % input_ids.size(0)
                    perm[idx], perm[swap_with] = perm[swap_with].clone(), perm[idx].clone()
                c_swap = c_batch[perm]
                model.set_swap_context(c_swap)
                with torch.no_grad():
                    logits_swap = model(input_ids, attention_mask)
                    sigma_swap = torch.sigmoid(logits_swap)
                model.clear_swap_context()

                # L2 distances; clamp(min=1e-8) avoids sqrt(0) gradient instability.
                d_match = (sigma_1 - sigma_2.detach()).pow(2).sum(dim=-1).clamp(min=1e-8).sqrt()
                d_swap  = (sigma_1 - sigma_swap.detach()).pow(2).sum(dim=-1).clamp(min=1e-8).sqrt()

                c3_loss_val = torch.clamp(
                    d_match - d_swap + ace_cfg.c3_margin, min=0.0
                ).mean()

                with torch.no_grad():
                    c_sel_b = (d_swap.detach() - d_match.detach()).mean().item()
                    d_m_b   = d_match.detach().mean().item()
                    d_s_b   = d_swap.detach().mean().item()
                    p = sigma_1.detach()
                    H_bin = -(p * p.clamp(1e-9).log() + (1 - p) * (1 - p).clamp(1e-9).log())
                    ent_b     = H_bin.mean().item()
                    maxprob_b = sigma_1.detach().max(dim=-1)[0].mean().item()
                c3_c_sel_list.append(c_sel_b)
                c3_d_match_list.append(d_m_b)
                c3_d_swap_list.append(d_s_b)
                c3_entropy_list.append(ent_b)
                c3_maxprob_list.append(maxprob_b)

            total_loss = (task_loss
                          + ace_cfg.c3_lambda * c3_loss_val
                          + ace_cfg.diversity_weight * div_loss
                          + ace_cfg.spread_weight * spread_loss_val)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_task_loss   += task_loss.item()
            epoch_div_loss    += div_loss.item()
            epoch_c3_loss     += c3_loss_val.item()
            epoch_spread_loss += spread_loss_val.item()

            diag = model.get_diagnostics()
            if diag:
                tau_list.extend(diag["tau"].cpu().view(-1).tolist())
                H_list.extend(diag["H_norm"].cpu().view(-1).tolist())

        val_pred, val_true = _evaluate(model, val_loader, loader.is_multilabel)
        val_macro_f1 = float(f1_score(
            val_true, val_pred, average="macro", zero_division=0
        ))

        tau_arr = np.array(tau_list)
        H_arr = np.array(H_list)
        N_eff = float(np.exp(H_arr * math.log(ace_cfg.n_prototypes)).mean()) if len(H_arr) else 0.0
        tau_corr_H = float(np.corrcoef(tau_arr, H_arr)[0, 1]) if len(tau_arr) > 10 else 0.0

        c3_epoch_diag: Dict[str, float] = {}
        if ace_cfg.c3_loss and c3_c_sel_list:
            c_sel_mean = float(np.mean(c3_c_sel_list))
            c_sel_std  = float(np.std(c3_c_sel_list))
            c3_epoch_diag = {
                "C_sel_mean": c_sel_mean,
                "C_sel_std": c_sel_std,
                "d_match_mean": float(np.mean(c3_d_match_list)),
                "d_swap_mean":  float(np.mean(c3_d_swap_list)),
                "sigma_entropy_mean": float(np.mean(c3_entropy_list)),
                "sigma_max_prob_mean": float(np.mean(c3_maxprob_list)),
                "L_C3_value": epoch_c3_loss / len(train_loader),
                "L_task_value": epoch_task_loss / len(train_loader),
                "r_dmatch_freq": float("nan"),
                "r_monitor_epoch": float("nan"),
                "code_usage_entropy": float("nan"),
            }
            if c_sel_mean < 0.01:
                print(f"  [C3-ALERT] Epoch {epoch+1}: C_sel={c_sel_mean:.4f} < 0.01 (collapse risk)")

        epoch_info = {
            "epoch": epoch + 1,
            "task_loss": epoch_task_loss / len(train_loader),
            "div_loss": epoch_div_loss / len(train_loader),
            "c3_loss": epoch_c3_loss / len(train_loader),
            "spread_loss": epoch_spread_loss / len(train_loader),
            "val_macro_f1": val_macro_f1,
            "tau_mean": float(tau_arr.mean()) if len(tau_arr) else 0.0,
            "tau_std": float(tau_arr.std()) if len(tau_arr) else 0.0,
            "N_eff": N_eff,
            "corr_tau_H": tau_corr_H,
            "time_sec": time.time() - t0,
            "c3_diagnostics": c3_epoch_diag,
        }
        train_log.append(epoch_info)

        c3_print = ""
        if ace_cfg.c3_loss and c3_epoch_diag:
            c3_print = (f" | C3={epoch_info['c3_loss']:.4f}"
                        f" C_sel={c3_epoch_diag['C_sel_mean']:.4f}")
        spread_print = ""
        if ace_cfg.spread_weight > 0.0:
            spread_print = f" | spread={epoch_info['spread_loss']:.4f}"
        print(
            f"  Epoch {epoch+1}/{cfg.num_epochs} | "
            f"task_loss={epoch_info['task_loss']:.4f} | "
            f"div_loss={epoch_info['div_loss']:.4f}{c3_print}{spread_print} | "
            f"val_mF1={val_macro_f1:.4f} | "
            f"N_eff={N_eff:.1f} | corr(tau,H)={tau_corr_H:.2f} | "
            f"{epoch_info['time_sec']:.0f}s"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            no_improve_count = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": best_epoch,
                "val_macro_f1": best_val_macro_f1,
                "seed": seed,
            }, best_ckpt_path)
        else:
            no_improve_count += 1
            if no_improve_count >= cfg.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={cfg.early_stopping_patience})")
                break

    print(f"  Best epoch: {best_epoch} | val_mF1={best_val_macro_f1:.4f}")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  Loaded best checkpoint (epoch={ckpt['epoch']})")

    test_pred, test_true = _evaluate(model, test_loader, loader.is_multilabel)
    test_texts = test_ds.texts

    task_type_str = "multilabel" if loader.is_multilabel else "classification"
    test_metrics = compute_all_metrics(
        test_true, test_pred, loader.label_names, task_type=task_type_str
    )
    test_metrics["seed"] = seed
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_val_macro_f1"] = best_val_macro_f1

    print("  Computing c* diversity on the test set...")
    c_div = _compute_c_diversity(model, test_loader, device)
    test_metrics.update(c_div)
    if c_div:
        print(
            f"  c*_cos={c_div['c_pairwise_cos_mean']:.4f}"
            f" N_eff={c_div['N_eff']:.2f}"
            f" pi_H={c_div['pi_batch_entropy']:.4f}"
        )

    subgroup_results = evaluate_subgroups(
        texts=test_texts,
        y_true=test_true,
        y_pred=test_pred,
        label_names=loader.label_names,
        task_type=task_type_str,
    )

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        test_metrics["peak_gpu_mem_gb"] = peak_mem_gb
        print(f"  Peak GPU memory: {peak_mem_gb:.2f} GB")

    save_metrics_json(test_metrics, os.path.join(run_dir, "test_metrics.json"))
    save_metrics_json(subgroup_results, os.path.join(run_dir, "subgroup_metrics.json"))
    save_metrics_json({"train_log": train_log}, os.path.join(run_dir, "train_log.json"))

    print(
        f"  [seed={seed}] macro_f1={test_metrics['macro_f1']:.4f} | "
        f"weighted_f1={test_metrics.get('weighted_f1', float('nan')):.4f}"
    )
    print(format_subgroup_report(subgroup_results))

    return {
        "test_metrics": test_metrics,
        "subgroup_metrics": subgroup_results,
        "train_log": train_log,
    }
