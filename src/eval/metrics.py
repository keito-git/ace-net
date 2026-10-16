"""
Evaluation metrics: classification, multi-label, and regression tasks.

Provides F1 (compute_f1), regression metrics (compute_regression_metrics),
a unified dispatcher (compute_all_metrics), multi-seed aggregation
(aggregate_seeds), Welch's t-test (welch_ttest), JSON persistence
(save_metrics_json), and an ASCII summary table (format_seed_table).

compute_all_metrics() always returns the full metric set; callers should not
report a cherry-picked subset.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr
from scipy import stats as scipy_stats
from sklearn.metrics import f1_score, mean_absolute_error


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    zero_division: int = 0,
) -> float:
    """Compute F1 score (macro, weighted, micro, or binary)."""
    return float(f1_score(y_true, y_pred, average=average, zero_division=zero_division))


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dim_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute Pearson r, MAE, RMSE for regression tasks. For multi-dimensional
    targets (e.g. VAD), reports per-dimension values plus the average.
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_dims = y_true.shape[1]
    metrics: Dict[str, float] = {}
    rs, maes, rmses = [], [], []

    for d in range(n_dims):
        yt = y_true[:, d] if y_true.ndim > 1 else y_true.flatten()
        yp = y_pred[:, d] if y_pred.ndim > 1 else y_pred.flatten()
        dim_label = dim_names[d] if dim_names and d < len(dim_names) else str(d)

        if len(yt) < 3:
            r = float("nan")
        else:
            r, _ = pearsonr(yt, yp)
            r = float(r)
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

        rs.append(r)
        maes.append(mae)
        rmses.append(rmse)

        if n_dims > 1:
            metrics[f"pearson_r_{dim_label}"] = r
            metrics[f"mae_{dim_label}"] = mae
            metrics[f"rmse_{dim_label}"] = rmse

    metrics["pearson_r"] = float(np.nanmean(rs))
    metrics["mae"] = float(np.mean(maes))
    metrics["rmse"] = float(np.mean(rmses))
    return metrics


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    task_type: str = "classification",
    zero_division: int = 0,
) -> Dict[str, float]:
    """
    Compute the complete evaluation metric set, dispatched by task type.

    task_type is one of 'classification', 'multilabel', 'regression',
    'cause_extraction', 'sarcasm'. Per-class F1 is always included.
    """
    if task_type == "regression":
        return compute_regression_metrics(y_true, y_pred, label_names)

    macro_f1 = compute_f1(y_true, y_pred, "macro", zero_division)
    weighted_f1 = compute_f1(y_true, y_pred, "weighted", zero_division)
    micro_f1 = compute_f1(y_true, y_pred, "micro", zero_division)

    metrics: Dict[str, float] = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
    }

    if task_type in ("sarcasm", "cause_extraction"):
        binary_f1 = compute_f1(y_true, y_pred, "binary", zero_division)
        metrics["f1_binary"] = binary_f1

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=zero_division)
    for i, score in enumerate(per_class_f1):
        key_name = label_names[i] if (label_names and i < len(label_names)) else str(i)
        metrics[f"f1_class_{key_name}"] = float(score)

    return metrics


def aggregate_seeds(
    seed_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate per-seed metric dicts into mean and std (ddof=1). Requires at
    least 2 seed runs; the protocol uses 5.
    """
    if len(seed_metrics) < 2:
        raise ValueError(
            f"aggregate_seeds requires at least 2 seed runs, got {len(seed_metrics)}. "
            "Protocol requires 5."
        )
    if len(seed_metrics) < 5:
        import warnings
        warnings.warn(
            f"Only {len(seed_metrics)} seed runs provided; 5 required.", stacklevel=2
        )

    all_keys = sorted(set(seed_metrics[0].keys()))
    result: Dict[str, Dict[str, Any]] = {}
    for key in all_keys:
        if key in ("seed",):
            continue
        scalar_vals: list = []
        non_scalar_vals: list = []
        for m in seed_metrics:
            if key not in m:
                continue
            v = m[key]
            if isinstance(v, str):
                continue
            try:
                scalar_vals.append(float(v))
            except (TypeError, ValueError):
                non_scalar_vals.append(v)
        if scalar_vals:
            result[key] = {
                "mean": float(np.mean(scalar_vals)),
                "std": float(np.std(scalar_vals, ddof=1) if len(scalar_vals) > 1 else 0.0),
                "n": len(scalar_vals),
                "values": scalar_vals,
            }
        elif non_scalar_vals:
            result[key] = {
                "per_seed": non_scalar_vals,
                "n": len(non_scalar_vals),
            }
    return result


def welch_ttest(
    values_a: List[float],
    values_b: List[float],
    metric_name: str = "macro_f1",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Welch's t-test (two-sided, unequal variance) comparing two methods'
    per-seed results. Requires at least 2 seeds per condition.
    """
    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "metric": metric_name,
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "alpha": alpha,
            "n_a": len(values_a),
            "n_b": len(values_b),
            "mean_a": float(np.mean(values_a)) if values_a else float("nan"),
            "mean_b": float(np.mean(values_b)) if values_b else float("nan"),
            "delta": float("nan"),
            "interpretation": (
                f"INSUFFICIENT DATA (n_a={len(values_a)}, n_b={len(values_b)}). "
                "Minimum 2 seeds per condition required."
            ),
        }

    t_stat, p_value = scipy_stats.ttest_ind(values_a, values_b, equal_var=False)
    significant = bool(float(p_value) < alpha)
    mean_a, mean_b = float(np.mean(values_a)), float(np.mean(values_b))
    return {
        "metric": metric_name,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": significant,
        "alpha": alpha,
        "n_a": len(values_a),
        "n_b": len(values_b),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": mean_a - mean_b,
        "interpretation": (
            f"Significant (p={p_value:.4f} < {alpha}): "
            f"A ({'>' if mean_a > mean_b else '<'} by {abs(mean_a - mean_b):.4f}) B"
            if significant
            else f"Not significant (p={p_value:.4f} >= {alpha})"
        ),
    }


def save_metrics_json(metrics: Dict[str, Any], output_path: str) -> None:
    """Save a metrics dict to JSON (creates parent dirs if needed)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] Saved: {path}")


def format_seed_table(
    aggregated: Dict[str, Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> str:
    """Format aggregated seed results as an ASCII table."""
    if metrics is None:
        metrics = ["macro_f1", "weighted_f1", "micro_f1", "pearson_r", "mae", "rmse"]
    lines = [
        "+-------------------------+----------------------------+",
        "| Metric                  | Mean +/- Std  (n)          |",
        "+-------------------------+----------------------------+",
    ]
    for key in metrics:
        if key not in aggregated:
            continue
        agg = aggregated[key]
        mean, std, n = agg["mean"], agg["std"], agg["n"]
        val_str = f"{mean:.4f} +/- {std:.4f}  (n={n})"
        lines.append(f"| {key:<23} | {val_str:<26} |")
    lines.append("+-------------------------+----------------------------+")
    return "\n".join(lines)
