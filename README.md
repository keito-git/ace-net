# ACE-Net: Allocation-based Context Encoding Network

Reproducibility code for ACE-Net, a context-injection extension of a
DeBERTa-v3 backbone for text emotion classification. ACE-Net learns a frozen,
label-free prototype dictionary and injects a selected context vector into an
intermediate encoder layer, so the upper layers re-encode the input
conditioned on the chosen context.

This repository contains only the code needed to reproduce the main
experiments. Datasets, the prototype dictionary, generated context data, and
result files are not included here (see "Data" below).

## Contents

```
train_ace.py                 ACE-Net training entry point
src/
  data/                      dataset registry and loaders
    registry.py              dataset metadata and loader factory
    base_loader.py           abstract loader interface
    goemotions_loader.py     GoEmotions (28-class, multi-label)
    semeval2018_loader.py    SemEval-2018 Task 1 E-c (11-class, multi-label)
    isear_loader.py          ISEAR CSV loader (7-class)
    isear_hf_loader.py       ISEAR HuggingFace loader (7-class)
    task_type.py             task-type definitions
  models/
    ace_net_classifier.py    ACE-Net model (prototype context injection)
    deberta_classifier.py    DeBERTa-v3 classification head
    dictionary_builder.py    label-free K-means prototype dictionary builder
  baselines/
    deberta_baseline.py      DeBERTa-v3 text-only baseline
    ace_net_baseline.py      ACE-Net training/evaluation loop
    self_training_baseline.py  self-training (pseudo-label) baseline
    rag_baseline.py          TF-IDF retrieval-augmented baseline
    llm_context_baseline.py  LLM-generated-context baseline
  eval/
    metrics.py               F1, aggregation, Welch's t-test
    subgroup_eval.py         context-deficiency subgroup evaluation
  utils/
    seed.py                  reproducible seeding
    stratify.py              context-deficiency stratification heuristics
    env_log.py               environment manifest for reproducibility
scripts/                     revision analysis scripts (p1a..p1e)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training.

## Usage

Train ACE-Net on GoEmotions with the canonical 5-seed protocol:

```bash
python train_ace.py --dataset goemotions
```

Single-seed run (no mixed precision):

```bash
python train_ace.py --dataset goemotions --seeds 42 --no_fp16
```

List the available datasets:

```bash
python train_ace.py --list_datasets --dataset goemotions
```

Supported dataset keys: `goemotions`, `semeval2018_ec`, `isear`.

## Revision analysis scripts

The `scripts/` directory holds the analysis scripts used for the revision.
They run on CPU and consume the saved probabilities / checkpoints (see "Data"),
not the raw datasets:

- `p1a_threshold_audit_conservative.py`: conservative threshold audit (ACE-Net
  at a fixed 0.5 threshold vs the baseline given a free per-class threshold).
- `p1b_extract_ace_probs.py`: extract ACE-Net sigmoid probabilities for the
  val/test splits from the per-seed checkpoints.
- `p1b_threshold_audit_fair.py`: symmetric threshold audit (both models with
  val-optimal per-class thresholds).
- `p1c_revision_reanalysis.py`: threshold, per-class, label-frequency-control,
  and threshold-free (macro average precision) re-analysis.
- `p1d_context_vector_analysis.py`: context-vector analysis and mean-vector
  substitution test for whether the mechanism behaves as a constant prior.
- `p1e_underdeterminacy_validation.py`: automated LLM-judge validation of the
  under-determinacy proxy (requires an `OPENAI_API_KEY` or `OPENROUTER_API_KEY`
  in the environment).

These scripts expect probability files, checkpoints, and result directories
under `results/`; those artifacts are provided in the separate Zenodo archive
and are not redistributed here.

## Data

Datasets must be obtained from their original sources and are not
redistributed here:

- GoEmotions: https://github.com/google-research/google-research/tree/master/goemotions
  (also available via the HuggingFace `go_emotions` dataset)
- SemEval-2018 Task 1 E-c: https://competitions.codalab.org/competitions/17751
  (HuggingFace `sem_eval_2018_task_1`, config `subtask5.english`)
- ISEAR: University of Geneva CISA, or the HuggingFace mirror referenced in
  `src/data/isear_hf_loader.py`

The prototype dictionary, the generated context data used to build it, and the
result files are deposited separately (Zenodo archive). The GoEmotions and
SemEval loaders fetch data through the HuggingFace `datasets` library; ISEAR
can be loaded from the HuggingFace mirror or a local CSV via `--local_csv_path`.

## License

Released under the MIT License (see `LICENSE`).
