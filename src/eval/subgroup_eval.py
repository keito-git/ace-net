"""
Subgroup evaluation: per context-deficiency stratum performance metrics.

Supports classification/multi-label (Macro-F1, Weighted-F1), regression
(Pearson r, MAE, RMSE), and binary sarcasm/cause tasks (binary F1).
"""
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from src.utils.stratify import compute_strata_flags


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Pearson r, MAE, RMSE for continuous targets."""
    if len(y_true) < 3:
        return {"pearson_r": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"pearson_r": float(r), "mae": mae, "rmse": rmse}


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    zero_division: int = 0,
) -> Dict[str, float]:
    """Macro-F1 and Weighted-F1 for classification / multi-label."""
    from src.eval.metrics import compute_all_metrics
    return compute_all_metrics(y_true, y_pred, label_names, zero_division)


def evaluate_subgroups(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    task_type: str = "classification",
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate performance for each context-deficiency subgroup.

    Returns a dict mapping stratum_name -> metric_dict (each includes
    'n_samples'); an 'overall' entry is always included as a reference.
    """
    strata_flags = compute_strata_flags(texts)

    def _compute(yt: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
        if task_type == "regression":
            return _regression_metrics(yt, yp)
        return _classification_metrics(yt, yp, label_names)

    results: Dict[str, Dict[str, Any]] = {}

    overall_metrics = _compute(y_true, y_pred)
    overall_metrics["n_samples"] = int(len(texts))
    results["overall"] = overall_metrics

    for stratum_name, flags in strata_flags.items():
        indices = [i for i, flag in enumerate(flags) if flag]
        n = len(indices)
        if n == 0:
            results[stratum_name] = {"n_samples": 0, "note": "no samples in stratum"}
            continue
        sub_true = y_true[indices]
        sub_pred = y_pred[indices]
        sub_metrics = _compute(sub_true, sub_pred)
        sub_metrics["n_samples"] = n
        results[stratum_name] = sub_metrics

    return results


def format_subgroup_report(
    subgroup_results: Dict[str, Dict[str, Any]],
    primary_metric: str = "macro_f1",
) -> str:
    """Return a human-readable subgroup report string."""
    lines = [
        "=" * 70,
        "Subgroup Evaluation Report",
        "=" * 70,
    ]
    for stratum, metrics in subgroup_results.items():
        n = metrics.get("n_samples", "?")
        primary = metrics.get(primary_metric, float("nan"))
        lines.append(f"\n[{stratum}]  n={n}")
        if isinstance(primary, float) and not np.isnan(primary):
            lines.append(f"  {primary_metric}: {primary:.4f}")
        else:
            for k, v in metrics.items():
                if k != "n_samples" and isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: N/A")
    lines.append("=" * 70)
    return "\n".join(lines)
