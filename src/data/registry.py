"""
Dataset registry for the ACE-Net evaluation harness.

Provides metadata and a loader factory for the three datasets used in this
work: GoEmotions, SemEval-2018 Task 1 E-c, and ISEAR.

Usage:
  from src.data.registry import get_loader, get_registry
  loader = get_loader("goemotions")

Availability codes:
  'huggingface' : load via datasets.load_dataset()
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.data.task_type import TaskType


@dataclass
class DatasetInfo:
    """Metadata for one dataset in the registry."""
    name: str
    display_name: str
    task_type: TaskType
    n_labels: int
    label_names: List[str]
    availability: str
    source_url: str
    hf_path: Optional[str]
    hf_config: Optional[str]
    notes: str
    requires_extra_arg: bool = False
    extra_arg_name: Optional[str] = None


def _build_registry() -> Dict[str, DatasetInfo]:
    from src.data.goemotions_loader import GOEMOTIONS_LABELS
    from src.data.semeval2018_loader import SEMEVAL_LABELS
    from src.data.isear_loader import ISEAR_LABELS

    return {
        "goemotions": DatasetInfo(
            name="goemotions",
            display_name="GoEmotions (28-class)",
            task_type=TaskType.MULTILABEL,
            n_labels=28,
            label_names=GOEMOTIONS_LABELS,
            availability="huggingface",
            source_url="https://github.com/google-research/google-research/tree/master/goemotions",
            hf_path="go_emotions",
            hf_config="simplified",
            notes="Reddit comments, multi-label. Main benchmark. 43k examples.",
        ),
        "isear": DatasetInfo(
            name="isear",
            display_name="ISEAR (7-class, self-report)",
            task_type=TaskType.CLASSIFICATION,
            n_labels=7,
            label_names=ISEAR_LABELS,
            availability="huggingface",
            source_url="https://huggingface.co/datasets/dalopeza98/isear-cleaned-dataset",
            hf_path="dalopeza98/isear-cleaned-dataset",
            hf_config=None,
            notes=(
                "7.5k self-report sentences. HF mirror (dalopeza98). "
                "Val carved from train (10%, seed=42). "
                "Pass local_csv_path=<path> to use the legacy CSV loader instead."
            ),
            requires_extra_arg=False,
            extra_arg_name="local_csv_path",
        ),
        "semeval2018_ec": DatasetInfo(
            name="semeval2018_ec",
            display_name="SemEval-2018 Task1 E-c (11-class, multi-label)",
            task_type=TaskType.MULTILABEL,
            n_labels=11,
            label_names=SEMEVAL_LABELS,
            availability="huggingface",
            source_url="https://competitions.codalab.org/competitions/17751",
            hf_path="sem_eval_2018_task_1",
            hf_config="subtask5.english",
            notes="English tweets, 11-label multi-label. trust_remote_code=True required.",
        ),
    }


_REGISTRY: Optional[Dict[str, DatasetInfo]] = None


def get_registry() -> Dict[str, DatasetInfo]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_loader(name: str, cache_dir: Optional[str] = None, **kwargs: Any):
    """
    Return an initialized loader for dataset `name`.

    Args:
        name: dataset key from the registry
        cache_dir: HuggingFace cache dir
        **kwargs: additional args for loaders that require paths (e.g. local_csv_path)

    Raises:
        KeyError: if the dataset name is not in the registry
    """
    registry = get_registry()
    if name not in registry:
        raise KeyError(
            f"Unknown dataset: '{name}'. "
            f"Available: {sorted(registry.keys())}"
        )

    if name == "goemotions":
        from src.data.goemotions_loader import GoEmotionsLoader
        return GoEmotionsLoader(cache_dir=cache_dir)

    elif name == "isear":
        local_csv = kwargs.get("local_csv_path")
        if local_csv:
            from src.data.isear_loader import ISEARLoader
            return ISEARLoader(local_csv_path=local_csv)
        else:
            from src.data.isear_hf_loader import ISEARHFLoader
            return ISEARHFLoader(cache_dir=cache_dir)

    elif name == "semeval2018_ec":
        from src.data.semeval2018_loader import SemEval2018Loader
        return SemEval2018Loader(cache_dir=cache_dir)

    raise KeyError(f"No factory branch for dataset: '{name}'")


def list_datasets() -> str:
    """Return a formatted string listing all available datasets."""
    registry = get_registry()
    lines = [
        f"{'Name':<22} {'Task':<22} {'Labels':<8} {'Availability':<25} Notes",
        "-" * 110,
    ]
    for info in registry.values():
        lines.append(
            f"{info.name:<22} {info.task_type.value:<22} {info.n_labels:<8} "
            f"{info.availability:<25} {info.notes[:50]}"
        )
    return "\n".join(lines)
