"""Canonical Pairwise Similarity Run domain objects and pure core logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PairwiseSegment:
    """One sentence segment participating in a pairwise similarity run."""

    index: int
    text: str
    start: int | None = None
    end: int | None = None


@dataclass(slots=True)
class PairwiseMetrics:
    """Aggregate metrics derived from one similarity matrix."""

    max_score: float
    mean_score: float
    median_score: float
    p95_score: float
    mean_best_a_to_b: float
    mean_best_b_to_a: float


@dataclass(slots=True)
class PairwiseMatchRecord:
    """One ranked match with rich left/right segment metadata."""

    rank: int
    score: float
    segment_a: PairwiseSegment
    segment_b: PairwiseSegment


@dataclass(slots=True)
class PairwiseRunResult:
    """Pure in-memory result of one pairwise similarity run."""

    segments_a: list[PairwiseSegment]
    segments_b: list[PairwiseSegment]
    similarity_matrix: np.ndarray
    matches: list[PairwiseMatchRecord]
    metrics: PairwiseMetrics


def cosine_similarity_matrix(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix with shape [len(A), len(B)]."""
    if embeddings_a.ndim != 2 or embeddings_b.ndim != 2:
        raise ValueError("Embeddings must be rank-2 arrays.")
    if embeddings_a.shape[1] != embeddings_b.shape[1]:
        raise ValueError("Embedding dimensions must match.")
    if embeddings_a.shape[0] == 0 or embeddings_b.shape[0] == 0:
        return np.empty((embeddings_a.shape[0], embeddings_b.shape[0]), dtype=np.float32)

    a_norm = _row_normalize(embeddings_a)
    b_norm = _row_normalize(embeddings_b)
    return (a_norm @ b_norm.T).astype(np.float32)


def run_pairwise_similarity_core(
    segments_a: list[PairwiseSegment],
    embeddings_a: np.ndarray,
    segments_b: list[PairwiseSegment],
    embeddings_b: np.ndarray,
    *,
    top_k: int,
) -> PairwiseRunResult:
    """Build one canonical pairwise result from rich segments and embeddings."""
    _validate_pairwise_inputs(segments_a, embeddings_a, segments_b, embeddings_b)
    matrix = cosine_similarity_matrix(embeddings_a, embeddings_b)
    matches = top_k_match_records(matrix, segments_a, segments_b, top_k)
    metrics = matrix_metrics(matrix)
    return PairwiseRunResult(
        segments_a=segments_a,
        segments_b=segments_b,
        similarity_matrix=matrix,
        matches=matches,
        metrics=metrics,
    )


def top_k_match_records(
    matrix: np.ndarray,
    segments_a: list[PairwiseSegment],
    segments_b: list[PairwiseSegment],
    k: int,
) -> list[PairwiseMatchRecord]:
    """Return globally highest-scoring segment pairs."""
    if matrix.ndim != 2:
        raise ValueError("Similarity matrix must be rank-2.")
    if matrix.shape != (len(segments_a), len(segments_b)):
        raise ValueError("Matrix shape must match segment list lengths.")
    if k <= 0 or matrix.size == 0:
        return []

    flat = matrix.ravel()
    k_eff = min(k, flat.size)
    if k_eff == flat.size:
        top_indices = np.argsort(flat)
    else:
        top_indices = np.argpartition(flat, -k_eff)[-k_eff:]

    ordered = sorted(
        top_indices.tolist(),
        key=lambda idx: (-float(flat[idx]), idx // matrix.shape[1], idx % matrix.shape[1]),
    )

    matches: list[PairwiseMatchRecord] = []
    for rank, flat_idx in enumerate(ordered, start=1):
        i = flat_idx // matrix.shape[1]
        j = flat_idx % matrix.shape[1]
        matches.append(
            PairwiseMatchRecord(
                rank=rank,
                score=float(matrix[i, j]),
                segment_a=segments_a[i],
                segment_b=segments_b[j],
            )
        )
    return matches


def matrix_metrics(matrix: np.ndarray) -> PairwiseMetrics:
    """Compute aggregate similarity metrics for one pairwise run."""
    if matrix.ndim != 2:
        raise ValueError("Similarity matrix must be rank-2.")
    if matrix.size == 0:
        return PairwiseMetrics(
            max_score=0.0,
            mean_score=0.0,
            median_score=0.0,
            p95_score=0.0,
            mean_best_a_to_b=0.0,
            mean_best_b_to_a=0.0,
        )

    flat = matrix.astype(np.float32, copy=False).ravel()
    return PairwiseMetrics(
        max_score=float(np.max(flat)),
        mean_score=float(np.mean(flat)),
        median_score=float(np.median(flat)),
        p95_score=float(np.percentile(flat, 95)),
        mean_best_a_to_b=float(np.mean(np.max(matrix, axis=1))) if matrix.shape[0] else 0.0,
        mean_best_b_to_a=float(np.mean(np.max(matrix, axis=0))) if matrix.shape[1] else 0.0,
    )


def make_segments(
    texts: list[str],
    *,
    spans: list[tuple[int, int]] | None = None,
) -> list[PairwiseSegment]:
    """Build canonical segment records from text and optional span metadata."""
    segments: list[PairwiseSegment] = []
    for index, text in enumerate(texts):
        start: int | None = None
        end: int | None = None
        if spans is not None:
            start, end = spans[index]
        segments.append(PairwiseSegment(index=index, text=text, start=start, end=end))
    return segments


def _validate_pairwise_inputs(
    segments_a: list[PairwiseSegment],
    embeddings_a: np.ndarray,
    segments_b: list[PairwiseSegment],
    embeddings_b: np.ndarray,
) -> None:
    if embeddings_a.ndim != 2 or embeddings_b.ndim != 2:
        raise ValueError("Embeddings must be rank-2 arrays.")
    if embeddings_a.shape[0] != len(segments_a):
        raise ValueError("Embedding row count for side A must match segment count.")
    if embeddings_b.shape[0] != len(segments_b):
        raise ValueError("Embedding row count for side B must match segment count.")
    if embeddings_a.size and embeddings_b.size and embeddings_a.shape[1] != embeddings_b.shape[1]:
        raise ValueError("Embedding dimensions must match.")


def _row_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms
