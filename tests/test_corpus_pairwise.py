"""Tests for corpus-level pairwise similarity workflows."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tibetan_pipeline.corpus_pairwise import run_corpus_pairwise_similarity
from tibetan_pipeline.pairwise import PairMatch
from tibetan_pipeline.pairwise_run import make_segments, run_pairwise_similarity_core
from tibetan_pipeline.sdk import EmbeddingView, PairwiseView, SegmentationView


class FakeSDK:
    """Small SDK double that avoids real segmenter/model loading."""

    instances: list["FakeSDK"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.embed_calls: list[tuple[tuple[str, ...], bool]] = []
        FakeSDK.instances.append(self)

    def segment_text(self, text: str) -> SegmentationView:
        raw_segments = [segment.strip() for segment in text.split("|") if segment.strip()]
        spans = []
        cursor = 0
        for segment in raw_segments:
            start = text.index(segment, cursor)
            end = start + len(segment)
            spans.append((start, end))
            cursor = end
        return SegmentationView(
            original_text=text,
            normalized_text=text,
            source_format="unicode",
            engine_name="fake",
            segments=raw_segments,
            spans=spans,
        )

    def embed_sentences(self, sentences: list[str], *, is_query: bool = False) -> EmbeddingView:
        self.embed_calls.append((tuple(sentences), is_query))
        embeddings = np.array([_embedding_for(sentence) for sentence in sentences], dtype=np.float32)
        return EmbeddingView(
            model_id="fake/model",
            device="cpu",
            sentences=sentences,
            embeddings=embeddings,
        )

    def pairwise_from_embedding_views(
        self,
        embedding_a: EmbeddingView,
        embedding_b: EmbeddingView,
        *,
        top_k: int = 100,
    ) -> PairwiseView:
        result = run_pairwise_similarity_core(
            make_segments(embedding_a.sentences),
            embedding_a.embeddings,
            make_segments(embedding_b.sentences),
            embedding_b.embeddings,
            top_k=top_k,
        )
        return PairwiseView(
            model_id=embedding_a.model_id,
            device=embedding_a.device,
            segments_a=embedding_a.sentences,
            segments_b=embedding_b.sentences,
            segment_records_a=result.segments_a,
            segment_records_b=result.segments_b,
            similarity_matrix=result.similarity_matrix,
            matches=[
                PairMatch(
                    rank=match.rank,
                    score=match.score,
                    i=match.segment_a.index,
                    j=match.segment_b.index,
                    sentence_a=match.segment_a.text,
                    sentence_b=match.segment_b.text,
                )
                for match in result.matches
            ],
            metrics=result.metrics,
        )


class CorpusPairwiseTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeSDK.instances.clear()

    def test_corpus_workflow_reuses_document_embeddings_for_all_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dir_a = root / "a"
            dir_b = root / "b"
            out = root / "out"
            dir_a.mkdir()
            dir_b.mkdir()
            (dir_a / "a1.txt").write_text("alpha|shared", encoding="utf-8")
            (dir_a / "a2.txt").write_text("beta", encoding="utf-8")
            (dir_b / "b1.txt").write_text("shared", encoding="utf-8")
            (dir_b / "b2.txt").write_text("gamma|alpha", encoding="utf-8")

            with patch("tibetan_pipeline.corpus_pairwise.TibetanResearchSDK", FakeSDK):
                artifacts = run_corpus_pairwise_similarity(
                    dir_a=dir_a,
                    dir_b=dir_b,
                    output_dir=out,
                    model_id="fake/model",
                    device="cpu",
                    top_k=2,
                )

            self.assertEqual(set(artifacts), {"documents_a_csv", "documents_b_csv", "summary_csv", "manifest_json"})
            sdk = FakeSDK.instances[0]
            self.assertEqual(len(sdk.embed_calls), 4)
            self.assertEqual([is_query for _, is_query in sdk.embed_calls], [True, True, False, False])

            manifest = json.loads(Path(artifacts["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["doc_count_a"], 2)
            self.assertEqual(manifest["doc_count_b"], 2)
            self.assertEqual(manifest["pair_count"], 4)

            with Path(artifacts["summary_csv"]).open(encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 4)
            self.assertTrue(all(row["similarity_npy"] for row in summary_rows))
            self.assertTrue(all(Path(row["topk_csv"]).exists() for row in summary_rows))


def _embedding_for(sentence: str) -> list[float]:
    vectors = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
        "gamma": [0.0, 0.0, 1.0],
        "shared": [1.0, 1.0, 0.0],
    }
    return vectors.get(sentence, [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
