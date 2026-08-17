"""
Re-ID metrics: Rank-k, mAP and same/different-identity similarity statistics.

Gallery/query protocol
----------------------
For each query the gallery is ranked by cosine similarity. Standard Re-ID
practice — followed here — excludes gallery entries that are the *same image* as
the query, otherwise every query trivially retrieves itself at rank 1.

`sequence_ids` optionally also excludes gallery images from the query's own
capture sequence. Near-duplicate burst frames otherwise inflate Rank-1 without
demonstrating recognition across encounters.

Nothing here fabricates numbers: with no data, metrics come back as 0.0 and
`num_queries=0`, which callers must report honestly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SimilarityStats:
    same_identity_mean: float = 0.0
    same_identity_std: float = 0.0
    different_identity_mean: float = 0.0
    different_identity_std: float = 0.0
    same_pairs: int = 0
    different_pairs: int = 0
    separation: float = 0.0
    same_percentiles: Dict[str, float] = field(default_factory=dict)
    different_percentiles: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReIDMetrics:
    rank1: float = 0.0
    rank5: float = 0.0
    rank10: float = 0.0
    mean_ap: float = 0.0
    num_queries: int = 0
    num_gallery: int = 0
    num_identities: int = 0
    # "cross_sequence" excludes the query's own capture sequence from the gallery
    # (the honest protocol). "self_excluded_only" excludes just the query image,
    # so near-duplicate burst frames can be matched — an optimistic number.
    protocol: str = "cross_sequence"
    similarity: SimilarityStats = field(default_factory=SimilarityStats)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["similarity"] = self.similarity.to_dict()
        return data

    def format(self) -> str:
        if self.num_queries == 0:
            return (
                "Re-ID metrics: not computable — 0 valid queries.\n"
                "  Each identity needs at least 2 images across separate capture "
                "sequences for gallery/query evaluation."
            )
        s = self.similarity
        lines = [
            "Re-ID metrics" + (f"  [protocol: {self.protocol}]" if self.protocol else ""),
            f"  queries / gallery / identities : {self.num_queries} / {self.num_gallery} / {self.num_identities}",
            f"  Rank-1                        : {self.rank1:.4f}",
            f"  Rank-5                        : {self.rank5:.4f}",
            f"  Rank-10                       : {self.rank10:.4f}",
            f"  mAP                           : {self.mean_ap:.4f}",
            f"  cosine same-identity          : {s.same_identity_mean:.4f} ± {s.same_identity_std:.4f} (n={s.same_pairs})",
            f"  cosine different-identity     : {s.different_identity_mean:.4f} ± {s.different_identity_std:.4f} (n={s.different_pairs})",
            f"  separation (same − different) : {s.separation:.4f}",
        ]
        if self.protocol == "self_excluded_only":
            lines.append(
                "  NOTE: same-sequence frames were NOT excluded (no cross-sequence "
                "queries existed). These figures are optimistic."
            )
        return "\n".join(lines)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def cosine_similarity_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return l2_normalize(np.asarray(query, dtype=np.float64)) @ l2_normalize(
        np.asarray(gallery, dtype=np.float64)
    ).T


def compute_similarity_stats(
    embeddings: np.ndarray,
    labels: Sequence[Any],
    max_pairs: int = 2_000_000,
) -> SimilarityStats:
    """Cosine statistics over all distinct pairs (subsampled beyond `max_pairs`)."""
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    labels_arr = np.asarray(labels)
    n = len(labels_arr)
    if n < 2:
        return SimilarityStats()

    sim = embeddings @ embeddings.T
    same_mask = labels_arr[:, None] == labels_arr[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    same_scores = sim[same_mask & upper]
    diff_scores = sim[~same_mask & upper]

    rng = np.random.default_rng(0)
    if same_scores.size > max_pairs:
        same_scores = rng.choice(same_scores, max_pairs, replace=False)
    if diff_scores.size > max_pairs:
        diff_scores = rng.choice(diff_scores, max_pairs, replace=False)

    def percentiles(values: np.ndarray) -> Dict[str, float]:
        if values.size == 0:
            return {}
        qs = [1, 5, 25, 50, 75, 95, 99]
        return {f"p{q}": float(np.percentile(values, q)) for q in qs}

    same_mean = float(same_scores.mean()) if same_scores.size else 0.0
    diff_mean = float(diff_scores.mean()) if diff_scores.size else 0.0

    return SimilarityStats(
        same_identity_mean=same_mean,
        same_identity_std=float(same_scores.std()) if same_scores.size else 0.0,
        different_identity_mean=diff_mean,
        different_identity_std=float(diff_scores.std()) if diff_scores.size else 0.0,
        same_pairs=int(same_scores.size),
        different_pairs=int(diff_scores.size),
        separation=same_mean - diff_mean,
        same_percentiles=percentiles(same_scores),
        different_percentiles=percentiles(diff_scores),
    )


def evaluate_reid(
    embeddings: np.ndarray,
    labels: Sequence[Any],
    *,
    image_ids: Optional[Sequence[Any]] = None,
    sequence_ids: Optional[Sequence[Any]] = None,
    ranks: Sequence[int] = (1, 5, 10),
    allow_same_sequence_fallback: bool = True,
) -> ReIDMetrics:
    """
    Single-set (leave-one-out) evaluation: every image is a query against all
    others as gallery, with self — and where possible the query's own capture
    sequence — excluded.

    When no identity has images in two different sequences, cross-sequence
    evaluation is impossible. Rather than reporting 0 queries, this falls back to
    excluding only the query image and labels the result `self_excluded_only` so
    the weaker protocol is visible. Set `allow_same_sequence_fallback=False` to
    require the strict protocol.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels_arr = np.asarray(labels)
    n = len(labels_arr)
    if n < 2:
        return ReIDMetrics(similarity=compute_similarity_stats(embeddings, labels))

    image_ids_arr = np.asarray(image_ids) if image_ids is not None else np.arange(n)
    sequence_arr = np.asarray(sequence_ids) if sequence_ids is not None else None

    metrics = _rank_metrics(
        embeddings, labels_arr, image_ids_arr, sequence_arr, ranks, "cross_sequence"
    )
    if (
        metrics.num_queries == 0
        and sequence_arr is not None
        and allow_same_sequence_fallback
    ):
        metrics = _rank_metrics(
            embeddings, labels_arr, image_ids_arr, None, ranks, "self_excluded_only"
        )
    return metrics


def _rank_metrics(
    embeddings: np.ndarray,
    labels_arr: np.ndarray,
    image_ids_arr: np.ndarray,
    sequence_arr: Optional[np.ndarray],
    ranks: Sequence[int],
    protocol: str,
) -> ReIDMetrics:
    n = len(labels_arr)
    sim = cosine_similarity_matrix(embeddings, embeddings)

    hits = {k: 0 for k in ranks}
    average_precisions: List[float] = []
    valid_queries = 0

    for i in range(n):
        exclude = image_ids_arr == image_ids_arr[i]
        if sequence_arr is not None:
            exclude = exclude | (sequence_arr == sequence_arr[i])
        keep = ~exclude
        if not keep.any():
            continue

        gallery_labels = labels_arr[keep]
        relevant = gallery_labels == labels_arr[i]
        if not relevant.any():
            continue  # no gallery mate — unanswerable query, excluded

        valid_queries += 1
        order = np.argsort(-sim[i][keep], kind="stable")
        ranked_relevance = relevant[order]

        for k in ranks:
            if ranked_relevance[:k].any():
                hits[k] += 1

        positions = np.flatnonzero(ranked_relevance)
        precisions = (np.arange(len(positions)) + 1) / (positions + 1)
        average_precisions.append(float(precisions.mean()))

    similarity = compute_similarity_stats(embeddings, labels_arr)
    if valid_queries == 0:
        return ReIDMetrics(
            num_gallery=n,
            num_identities=int(len(np.unique(labels_arr))),
            protocol=protocol,
            similarity=similarity,
        )

    return ReIDMetrics(
        rank1=hits.get(1, 0) / valid_queries,
        rank5=hits.get(5, 0) / valid_queries,
        rank10=hits.get(10, 0) / valid_queries,
        mean_ap=float(np.mean(average_precisions)),
        num_queries=valid_queries,
        num_gallery=n,
        num_identities=int(len(np.unique(labels_arr))),
        protocol=protocol,
        similarity=similarity,
    )


def evaluate_query_gallery(
    query_embeddings: np.ndarray,
    query_labels: Sequence[Any],
    gallery_embeddings: np.ndarray,
    gallery_labels: Sequence[Any],
    *,
    ranks: Sequence[int] = (1, 5, 10),
) -> ReIDMetrics:
    """Explicit query/gallery evaluation (disjoint sets)."""
    query_labels_arr = np.asarray(query_labels)
    gallery_labels_arr = np.asarray(gallery_labels)
    if len(query_labels_arr) == 0 or len(gallery_labels_arr) == 0:
        return ReIDMetrics()

    sim = cosine_similarity_matrix(query_embeddings, gallery_embeddings)
    hits = {k: 0 for k in ranks}
    average_precisions: List[float] = []
    valid_queries = 0

    for i, label in enumerate(query_labels_arr):
        relevant = gallery_labels_arr == label
        if not relevant.any():
            continue
        valid_queries += 1
        order = np.argsort(-sim[i], kind="stable")
        ranked_relevance = relevant[order]
        for k in ranks:
            if ranked_relevance[:k].any():
                hits[k] += 1
        positions = np.flatnonzero(ranked_relevance)
        precisions = (np.arange(len(positions)) + 1) / (positions + 1)
        average_precisions.append(float(precisions.mean()))

    if valid_queries == 0:
        return ReIDMetrics(num_gallery=len(gallery_labels_arr))

    combined = np.vstack([np.asarray(query_embeddings), np.asarray(gallery_embeddings)])
    combined_labels = np.concatenate([query_labels_arr, gallery_labels_arr])

    return ReIDMetrics(
        rank1=hits.get(1, 0) / valid_queries,
        rank5=hits.get(5, 0) / valid_queries,
        rank10=hits.get(10, 0) / valid_queries,
        mean_ap=float(np.mean(average_precisions)),
        num_queries=valid_queries,
        num_gallery=len(gallery_labels_arr),
        num_identities=int(len(np.unique(combined_labels))),
        similarity=compute_similarity_stats(combined, combined_labels),
    )


def split_query_gallery(
    labels: Sequence[Any],
    *,
    sequence_ids: Optional[Sequence[Any]] = None,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Pick one query index per identity (preferring a sequence not otherwise used)
    and put the remainder in the gallery.
    """
    labels_arr = np.asarray(labels)
    rng = np.random.default_rng(seed)
    query: List[int] = []
    gallery: List[int] = []

    for label in np.unique(labels_arr):
        indices = np.flatnonzero(labels_arr == label)
        if len(indices) < 2:
            gallery.extend(int(i) for i in indices)
            continue

        chosen: Optional[int] = None
        if sequence_ids is not None:
            seq = np.asarray(sequence_ids)[indices]
            unique_sequences = np.unique(seq)
            if len(unique_sequences) > 1:
                held_out = rng.choice(unique_sequences)
                candidates = indices[seq == held_out]
                chosen = int(rng.choice(candidates))
        if chosen is None:
            chosen = int(rng.choice(indices))

        query.append(chosen)
        gallery.extend(int(i) for i in indices if int(i) != chosen)

    return query, gallery


def compute_roc(
    embeddings: np.ndarray,
    labels: Sequence[Any],
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """
    Verification ROC over all pairs: TAR (true accept) vs FAR (false accept).

    FAR is the operationally important number — it is the rate at which two
    different tigers would be merged into one identity.
    """
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    labels_arr = np.asarray(labels)
    n = len(labels_arr)
    if n < 2:
        return []

    sim = embeddings @ embeddings.T
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    same = (labels_arr[:, None] == labels_arr[None, :]) & upper
    diff = (labels_arr[:, None] != labels_arr[None, :]) & upper

    same_scores = sim[same]
    diff_scores = sim[diff]
    if same_scores.size == 0 or diff_scores.size == 0:
        return []

    if thresholds is None:
        thresholds = np.round(np.linspace(0.0, 0.99, 100), 4).tolist()

    curve: List[Dict[str, float]] = []
    for threshold in thresholds:
        tar = float((same_scores >= threshold).mean())
        far = float((diff_scores >= threshold).mean())
        curve.append({"threshold": float(threshold), "tar": tar, "far": far})
    return curve
