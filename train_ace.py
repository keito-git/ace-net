"""
ACE-Net training entry point (fine-tuning for emotion classification).

Usage:
  # GoEmotions 5-seed run
  python train_ace.py --dataset goemotions --cuda_device 0

  # Single-seed run
  python train_ace.py --dataset goemotions --seeds 42 --cuda_device 0

  # Override the injection layer for an ablation
  python train_ace.py --dataset goemotions --inject_layer 5 --cuda_device 0

CUDA_VISIBLE_DEVICES must be set before any torch import; this script sets it
from --cuda_device before importing torch.
"""
import argparse
import json
import os
import sys
from typing import List

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


CANONICAL_SEEDS: List[int] = [42, 123, 456, 789, 1024]

# Default diversity weight (matches the --diversity_weight argparse default).
DEFAULT_DIVERSITY_WEIGHT: float = 0.2


def build_ablation_suffix(
    fixed_tau: bool = False,
    no_gumbel: bool = False,
    input_concat: bool = False,
    free_vector: bool = False,
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
) -> str:
    """
    Encode ablation conditions into an output-dir suffix so ablation runs never
    collide with the primary-configuration output dir. The default configuration
    (all flags False, diversity_weight == 0.2) returns "".
    """
    suffix = ""
    if fixed_tau:
        suffix += "_fixedtau"
    if no_gumbel:
        suffix += "_nogumbel"
    if input_concat:
        suffix += "_inputconcat"
    if free_vector:
        suffix += "_freevec"
    if diversity_weight != DEFAULT_DIVERSITY_WEIGHT:
        suffix += f"_divw{diversity_weight}"
    return suffix


def check_output_dir_guard(output_dir: str, overwrite_ok: bool) -> None:
    """
    Refuse to run if output_dir already contains aggregated_metrics.json (a
    completed experiment) unless --overwrite_ok was passed.
    """
    marker = os.path.join(output_dir, "aggregated_metrics.json")
    if os.path.isdir(output_dir) and os.path.isfile(marker) and not overwrite_ok:
        print(
            f"[GUARD] Output dir already holds completed results: {output_dir}\n"
            f"        ({marker} exists)\n"
            f"        Refusing to overwrite. Pass --overwrite_ok to re-run into\n"
            f"        this directory."
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACE-Net fine-tuning experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset key from registry (e.g. goemotions)")
    parser.add_argument("--list_datasets", action="store_true",
                        help="List available datasets and exit")

    # Paths
    parser.add_argument("--context_file", type=str,
                        default="data/raw/corpus_contexts.json",
                        help="Path to the distilled context sentences JSON")
    parser.add_argument("--dict_cache", type=str,
                        default=None,
                        help="Path to cached dict .npy (auto-generated if None)")
    parser.add_argument("--cache_dir", type=str, default=".cache/datasets")
    parser.add_argument("--output_dir", type=str, default="results")

    # Dataset-specific paths
    parser.add_argument("--local_data_dir", type=str, default=None)
    parser.add_argument("--local_csv_path", type=str, default=None)

    # Shared hyperparameters (must match the DeBERTa baseline for fair comparison)
    parser.add_argument("--seeds", nargs="+", type=int, default=CANONICAL_SEEDS,
                        help="Seed list. Default is the canonical 5-seed protocol.")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--backbone", type=str, default="microsoft/deberta-v3-base",
                        help="HuggingFace backbone id. Use microsoft/deberta-v3-large "
                             "with --inject_layer 16 for the large configuration.")

    # ACE-Net specific
    parser.add_argument("--inject_layer", type=int, default=8,
                        help="0-indexed layer index (8 = Layer 9 in 1-indexed)")
    parser.add_argument("--n_prototypes", type=int, default=32,
                        help="K: number of prototypes")
    parser.add_argument("--tau_min", type=float, default=0.3,
                        help="Minimum temperature tau_min")
    parser.add_argument("--use_ib_gate", action="store_true",
                        help="Enable the IB gate (default OFF)")
    parser.add_argument("--diversity_weight", type=float, default=0.2,
                        help="Weight for the diversity regularization term")
    parser.add_argument("--force_rebuild_dict", action="store_true",
                        help="Rebuild the dictionary even if a cache exists")

    # Core-component ablations
    parser.add_argument("--fixed_tau", action="store_true",
                        help="Disable entropy-weighted tau(x); use constant tau = tau_min")
    parser.add_argument("--no_gumbel", action="store_true",
                        help="Deterministic softmax in training (no Gumbel noise)")
    parser.add_argument("--input_concat", action="store_true",
                        help="Late fusion: add c to the final [CLS] before the classifier")
    parser.add_argument("--free_vector", action="store_true",
                        help="Replace the dictionary and attention with a single learnable vector")

    # Overwrite-safety guard
    parser.add_argument("--overwrite_ok", action="store_true",
                        help="Allow writing into an output dir that already contains "
                             "aggregated_metrics.json")

    # C3 swap-contrastive auxiliary loss (--no_fp16 recommended when enabled)
    parser.add_argument("--c3_loss", action="store_true",
                        help="Enable the C3 swap-contrastive auxiliary loss (default OFF)")
    parser.add_argument("--c3_lambda", type=float, default=0.1,
                        help="C3 loss coefficient")
    parser.add_argument("--c3_margin", type=float, default=0.05,
                        help="Swap-contrastive hinge margin")
    parser.add_argument("--c3_t_resample", type=int, default=2,
                        help="Number of re-sample forward passes")
    parser.add_argument("--c3_monitor", action="store_true",
                        help="Enable per-epoch C3 collapse diagnostics")
    parser.add_argument("--c3_full_grad", action="store_true",
                        help="Use a symmetric gradient over both branches (3x cost, ablation only)")
    parser.add_argument("--spread_weight", type=float, default=0.0,
                        help="Weight of the intra-batch c* spread loss (anti-collapse). "
                             "0.0 disables it; try 0.05-0.2 to de-collapse the dictionary.")
    parser.add_argument("--tau_scale", type=float, default=1.0,
                        help="Scale on softplus(tau_head) output. 1.0 = original behavior; "
                             "0.2 is a recommended anti-collapse value. Must be in (0, 1.0].")

    # Infrastructure
    parser.add_argument("--cuda_device", type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES value (set before torch import)")

    return parser.parse_args()


def compute_s1_s4_strata(
    texts: List[str],
    no_explicit_emotion_flags: List[bool],
    short_text_flags: List[bool],
) -> dict:
    """
    Map stratify.py flags to four strata by explicitness and length:
      S1 explicit + long, S2 explicit + short,
      S3 implicit + long, S4 implicit + short.
    Returns a dict of stratum_name -> list of sample indices.
    """
    strata = {"S1_explicit_long": [], "S2_explicit_short": [],
               "S3_implicit_long": [], "S4_implicit_short": []}
    for i, (implicit, short) in enumerate(zip(no_explicit_emotion_flags, short_text_flags)):
        explicit = not implicit
        if explicit and not short:
            strata["S1_explicit_long"].append(i)
        elif explicit and short:
            strata["S2_explicit_short"].append(i)
        elif implicit and not short:
            strata["S3_implicit_long"].append(i)
        else:
            strata["S4_implicit_short"].append(i)
    return strata


def main() -> None:
    args = parse_args()

    if args.list_datasets:
        from src.data.registry import list_datasets
        print(list_datasets())
        sys.exit(0)

    # Set CUDA_VISIBLE_DEVICES before any torch import.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    import torch
    from src.data.registry import get_loader, get_registry
    from src.baselines.deberta_baseline import BaselineConfig
    from src.baselines.ace_net_baseline import ACENetConfig, run_ace_net_baseline
    from src.eval.metrics import aggregate_seeds, welch_ttest, save_metrics_json, format_seed_table
    from src.utils.stratify import compute_strata_flags

    # Dataset
    loader_kwargs = {}
    if args.local_data_dir:
        loader_kwargs["local_data_dir"] = args.local_data_dir
        loader_kwargs["data_dir"] = args.local_data_dir
    if args.local_csv_path:
        loader_kwargs["local_csv_path"] = args.local_csv_path

    loader = get_loader(args.dataset, cache_dir=args.cache_dir, **loader_kwargs)
    registry = get_registry()
    dataset_info = registry[args.dataset]

    # Dict cache path. The backbone variant tag keeps base/large dicts separate.
    dict_cache = args.dict_cache
    backbone_tag = "_large" if "large" in args.backbone else ""

    if dict_cache is None:
        dict_cache = os.path.join(
            "data", "processed",
            f"dict_K{args.n_prototypes}_L{args.inject_layer}{backbone_tag}.npy"
        )

    # Configs (hyperparameters shared with the DeBERTa baseline).
    cfg = BaselineConfig(
        model_name=args.backbone,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        fp16=not args.no_fp16,
        output_dir=os.path.join(args.output_dir, f"deberta_{args.dataset}{backbone_tag}"),
    )

    # Build an output-dir name that encodes all non-default conditions.
    c3_suffix = ""
    if args.c3_loss:
        c3_suffix = f"_c3_lam{args.c3_lambda}_m{args.c3_margin}"
    anticollapse_suffix = ""
    default_dict = os.path.join("data", "processed",
                                f"dict_K{args.n_prototypes}_L{args.inject_layer}{backbone_tag}.npy")
    if dict_cache != default_dict:
        dict_tag = "_pca" if "pca" in dict_cache else "_custdict"
        anticollapse_suffix = dict_tag + anticollapse_suffix
    if args.tau_scale != 1.0:
        anticollapse_suffix += f"_ts{args.tau_scale}"
    if args.spread_weight > 0.0:
        anticollapse_suffix += f"_sw{args.spread_weight}"

    ablation_suffix = build_ablation_suffix(
        fixed_tau=args.fixed_tau,
        no_gumbel=args.no_gumbel,
        input_concat=args.input_concat,
        free_vector=args.free_vector,
        diversity_weight=args.diversity_weight,
    )

    ace_cfg = ACENetConfig(
        context_file=args.context_file,
        dict_cache_path=dict_cache,
        n_prototypes=args.n_prototypes,
        force_rebuild_dict=args.force_rebuild_dict,
        inject_layer=args.inject_layer,
        tau_min=args.tau_min,
        tau_scale=args.tau_scale,
        use_ib_gate=args.use_ib_gate,
        diversity_weight=args.diversity_weight,
        c3_loss=args.c3_loss,
        c3_lambda=args.c3_lambda,
        c3_margin=args.c3_margin,
        c3_t_resample=args.c3_t_resample,
        c3_monitor=args.c3_monitor,
        spread_weight=args.spread_weight,
        fixed_tau=args.fixed_tau,
        no_gumbel=args.no_gumbel,
        input_concat=args.input_concat,
        free_vector=args.free_vector,
        output_dir=os.path.join(
            args.output_dir,
            f"ace_net_{args.dataset}{backbone_tag}_L{args.inject_layer}_K{args.n_prototypes}"
            + ("_gate" if args.use_ib_gate else "")
            + c3_suffix
            + anticollapse_suffix
            + ablation_suffix,
        ),
    )

    check_output_dir_guard(ace_cfg.output_dir, args.overwrite_ok)

    # --no_fp16 is recommended when --c3_loss is enabled (fp16 can produce NaNs).
    if args.c3_loss and not args.no_fp16:
        print("[WARNING] --c3_loss is more stable with --no_fp16. Forcing fp32.")
        cfg = cfg.__class__(
            **{k: (False if k == "fp16" else v) for k, v in cfg.__dict__.items()}
        )

    c3_str = (f"ON  lambda={ace_cfg.c3_lambda} m={ace_cfg.c3_margin} "
              f"T={ace_cfg.c3_t_resample} monitor={'ON' if ace_cfg.c3_monitor else 'OFF'}"
              if ace_cfg.c3_loss else "OFF")
    print(f"\n{'='*70}")
    print("ACE-Net Experiment")
    print(f"Dataset : {dataset_info.display_name}")
    print(f"Task    : {dataset_info.task_type.value}")
    print(f"Labels  : {dataset_info.n_labels}")
    print(f"Seeds   : {args.seeds}")
    print(f"Device  : CUDA_VISIBLE_DEVICES={args.cuda_device}")
    print(f"fp16    : {cfg.fp16 and torch.cuda.is_available()}")
    print(f"ACE-Net : inject_layer={ace_cfg.inject_layer} K={ace_cfg.n_prototypes} "
          f"tau_min={ace_cfg.tau_min} gate={'ON' if ace_cfg.use_ib_gate else 'OFF'} "
          f"div_weight={ace_cfg.diversity_weight}")
    print(f"Ablation: fixed_tau={ace_cfg.fixed_tau} no_gumbel={ace_cfg.no_gumbel} "
          f"input_concat={ace_cfg.input_concat} free_vector={ace_cfg.free_vector}")
    print(f"C3-Loss : {c3_str}")
    print(f"Dict    : {ace_cfg.context_file}")
    print(f"Output  : {ace_cfg.output_dir}")
    print(f"{'='*70}\n")

    from src.data.task_type import TaskType
    if dataset_info.task_type == TaskType.REGRESSION:
        print("WARNING: Regression task. ACE-Net regression head not implemented.")
        sys.exit(0)

    # 5-seed loop
    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed} / {len(args.seeds)}")
        print(f"{'='*60}")
        result = run_ace_net_baseline(loader, cfg, ace_cfg, seed=seed)
        all_results.append(result["test_metrics"])

    # Aggregate across seeds (aggregate_seeds requires >= 2 seeds).
    if len(all_results) >= 2:
        aggregated = aggregate_seeds(all_results)
    else:
        print(f"[INFO] Single-seed run ({len(all_results)} seed): "
              "skipping aggregate_seeds (requires >= 2). "
              "Per-seed test_metrics.json is the result.")
        aggregated = None

    final = {
        "method": "ACENet",
        "dataset": args.dataset,
        "inject_layer": ace_cfg.inject_layer,
        "n_prototypes": ace_cfg.n_prototypes,
        "tau_min": ace_cfg.tau_min,
        "use_ib_gate": ace_cfg.use_ib_gate,
        "diversity_weight": ace_cfg.diversity_weight,
        "fixed_tau": ace_cfg.fixed_tau,
        "no_gumbel": ace_cfg.no_gumbel,
        "input_concat": ace_cfg.input_concat,
        "free_vector": ace_cfg.free_vector,
        "seeds": args.seeds,
        "per_seed": all_results,
        "aggregated": aggregated,
    }

    out_path = os.path.join(ace_cfg.output_dir, "aggregated_metrics.json")
    os.makedirs(ace_cfg.output_dir, exist_ok=True)
    save_metrics_json(final, out_path)

    print(f"\n{'='*70}")
    if aggregated is not None:
        print("ACE-Net Multi-seed Summary")
        print(f"{'='*70}")
        print(format_seed_table(aggregated))
    else:
        print("ACE-Net Single-seed Result")
        print(f"{'='*70}")
        for key in ["macro_f1", "weighted_f1", "micro_f1"]:
            if key in all_results[0]:
                print(f"  {key}: {all_results[0][key]:.4f}")

    if aggregated is not None:
        for key in ["macro_f1", "weighted_f1", "micro_f1"]:
            if key in aggregated:
                vals = aggregated[key]["values"]
                print(f"\n{key} per seed: {[f'{v:.4f}' for v in vals]}")
                print(f"  mean={aggregated[key]['mean']:.4f} std={aggregated[key]['std']:.4f}")

    print(f"\nAll results saved: {ace_cfg.output_dir}")
    print(f"Aggregated:        {out_path}")


if __name__ == "__main__":
    main()
