"""
ACE-Net prototype dictionary builder.

Encodes context sentences with DeBERTa-v3-base [CLS] embeddings, then applies
MiniBatchKMeans to obtain K prototype centroids (the frozen dictionary D).
No emotion labels are used in this process (label-free); the dictionary
captures generic utterance-level context structure.

Returns an np.ndarray of shape (n_clusters, hidden_size=768), dtype float32,
to be registered as a frozen buffer in ACENetClassifier.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def build_prototype_dictionary(
    context_file: str,
    backbone: str = "microsoft/deberta-v3-base",
    n_clusters: int = 32,
    n_contexts: Optional[int] = None,
    encode_batch_size: int = 128,
    max_length: int = 128,
    device: Optional[str] = None,
    kmeans_seed: int = 42,
    kmeans_n_init: int = 10,
    kmeans_batch_size: int = 512,
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Build a K=n_clusters prototype dictionary from context sentences.

    Steps: load context sentences (JSON list of strings), encode each with a
    DeBERTa [CLS] embedding (no grad), fit MiniBatchKMeans, and return the
    centroids as a float32 array (n_clusters, 768).

    Raises:
        FileNotFoundError: if context_file does not exist
        ValueError: if context_file has fewer than n_clusters sentences
    """
    ctx_path = Path(context_file)
    if not ctx_path.exists():
        raise FileNotFoundError(f"Context file not found: {ctx_path}")

    with open(ctx_path, "r", encoding="utf-8") as f:
        contexts: List[str] = json.load(f)
    if not isinstance(contexts, list):
        raise ValueError(f"context_file must contain a JSON list, got {type(contexts)}")

    if n_contexts is not None:
        contexts = contexts[:n_contexts]

    if len(contexts) < n_clusters:
        raise ValueError(
            f"Too few contexts ({len(contexts)}) for n_clusters={n_clusters}. "
            "Need at least n_clusters context sentences."
        )

    logger.info(f"[DictBuilder] Encoding {len(contexts)} contexts with {backbone}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(backbone, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(backbone, cache_dir=cache_dir).to(device)
    model.eval()

    all_cls: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(contexts), encode_batch_size):
            batch = contexts[i : i + encode_batch_size]
            enc = tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_cls.append(cls_emb)
            if (i // encode_batch_size) % 10 == 0:
                logger.info(
                    f"[DictBuilder] Encoded {min(i + encode_batch_size, len(contexts))}"
                    f"/{len(contexts)} contexts"
                )

    del model
    torch.cuda.empty_cache()

    all_embeddings = np.concatenate(all_cls, axis=0)
    logger.info(
        f"[DictBuilder] Embedding matrix: {all_embeddings.shape}. "
        f"Running MiniBatchKMeans(K={n_clusters}, seed={kmeans_seed}) ..."
    )

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=kmeans_seed,
        n_init=kmeans_n_init,
        batch_size=kmeans_batch_size,
        max_iter=300,
        verbose=0,
    )
    kmeans.fit(all_embeddings)

    centroids = kmeans.cluster_centers_.astype(np.float32)
    logger.info(
        f"[DictBuilder] Done. Dictionary shape: {centroids.shape} "
        f"(inertia={kmeans.inertia_:.2f})"
    )
    return centroids


def load_or_build_dictionary(
    context_file: str,
    cache_path: str,
    n_clusters: int = 32,
    backbone: str = "microsoft/deberta-v3-base",
    force_rebuild: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Load a cached dictionary, or build it if the cache is missing or stale.

    cache_path should encode n_clusters/backbone/n_contexts so distinct
    configurations use distinct cache files.
    """
    cache = Path(cache_path)
    if cache.exists() and not force_rebuild:
        logger.info(f"[DictBuilder] Loading cached dictionary: {cache}")
        arr = np.load(str(cache))
        if arr.shape[0] != n_clusters:
            logger.warning(
                f"Cache shape {arr.shape} does not match n_clusters={n_clusters}. "
                "Rebuilding ..."
            )
        else:
            return arr

    logger.info("[DictBuilder] Cache not found or forced rebuild. Building ...")
    centroids = build_prototype_dictionary(
        context_file=context_file,
        n_clusters=n_clusters,
        backbone=backbone,
        **kwargs,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache), centroids)
    logger.info(f"[DictBuilder] Dictionary saved: {cache}")
    return centroids
