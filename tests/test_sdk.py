"""Tests for the notebook-friendly research SDK."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from tibetan_pipeline.embeddings import EmbeddingResult
from tibetan_pipeline.segmenters.base import BaseSegmenter, Segment
from tibetan_pipeline.sdk import EmbeddingView, TibetanResearchSDK


class FakeSegmenter(BaseSegmenter):
    engine_name = "botok_ours"

    def segment(self, text: str) -> list[Segment]:
        return [
            Segment("ཀ་།", 0, 2),
            Segment("ཁ་།", 3, 5),
        ]


class SDKTests(unittest.TestCase):
    def test_segment_text_returns_view_and_dataframe(self) -> None:
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=FakeSegmenter()):
            sdk = TibetanResearchSDK(engine="botok_ours")
            view = sdk.segment_text("ཀ་། ཁ་།")

        self.assertEqual(view.engine_name, "botok_ours")
        self.assertEqual(view.segments, ["ཀ་།", "ཁ་།"])
        df = view.to_dataframe()
        self.assertEqual(list(df.columns), ["segment_index", "start", "end", "segment_text"])
        self.assertEqual(len(df), 2)

    def test_embed_sentences_passes_device_and_returns_dataframe(self) -> None:
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=FakeSegmenter()):
            sdk = TibetanResearchSDK(
                device="cpu",
                model_id="fake/model",
                batch_size=2,
                embedding_progress="batch",
            )
        with patch("tibetan_pipeline.sdk.TextEmbedder") as mock_embedder_cls:
            mock_embedder = mock_embedder_cls.return_value
            mock_embedder.encode_corpus.return_value = EmbeddingResult("fake/model", np.ones((2, 3), dtype=np.float32))
            view = sdk.embed_sentences(["a", "b"])

        mock_embedder_cls.assert_called_once_with(
            model_id="fake/model",
            batch_size=2,
            normalize_embeddings=True,
            device="cpu",
            embedding_progress="batch",
            torch_dtype=None,
            device_map=None,
            load_in_8bit=False,
            low_cpu_mem_usage=None,
        )

        self.assertEqual(view.model_id, "fake/model")
        self.assertEqual(view.device, "cpu")
        self.assertEqual(view.embeddings.shape, (2, 3))
        df = view.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn("vector_norm", df.columns)

    def test_embed_sentences_allows_progress_override(self) -> None:
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=FakeSegmenter()):
            sdk = TibetanResearchSDK(device="cpu", model_id="fake/model", batch_size=2, embedding_progress="off")
        with patch("tibetan_pipeline.sdk.TextEmbedder") as mock_embedder_cls:
            mock_embedder = mock_embedder_cls.return_value
            mock_embedder.encode_corpus.return_value = EmbeddingResult("fake/model", np.ones((1, 3), dtype=np.float32))
            sdk.embed_sentences(["a"], embedding_progress="sentence")

        mock_embedder_cls.assert_called_once_with(
            model_id="fake/model",
            batch_size=2,
            normalize_embeddings=True,
            device="cpu",
            embedding_progress="sentence",
            torch_dtype=None,
            device_map=None,
            load_in_8bit=False,
            low_cpu_mem_usage=None,
        )

    def test_embed_sentences_reuses_cached_embedder_for_same_model_load_config(self) -> None:
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=FakeSegmenter()):
            sdk = TibetanResearchSDK(device="cpu", model_id="fake/model", batch_size=2)
        with patch("tibetan_pipeline.sdk.TextEmbedder") as mock_embedder_cls:
            mock_embedder = mock_embedder_cls.return_value
            mock_embedder.encode_corpus.return_value = EmbeddingResult("fake/model", np.ones((1, 3), dtype=np.float32))

            sdk.embed_sentences(["a"])
            sdk.embed_sentences(["b"], batch_size=4, embedding_progress="batch")

        mock_embedder_cls.assert_called_once()
        self.assertEqual(mock_embedder.batch_size, 4)
        self.assertEqual(mock_embedder.embedding_progress, "batch")

    def test_pairwise_from_sentences_returns_ranked_dataframe(self) -> None:
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=FakeSegmenter()):
            sdk = TibetanResearchSDK(device="cpu", model_id="fake/model", batch_size=1)
        with patch("tibetan_pipeline.sdk.TextEmbedder.encode_queries", return_value=EmbeddingResult("fake/model", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))):
            with patch("tibetan_pipeline.sdk.TextEmbedder.encode_corpus", return_value=EmbeddingResult("fake/model", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))):
                view = sdk.pairwise_from_sentences(["a0", "a1"], ["b0", "b1"], top_k=2)

        self.assertEqual(view.similarity_matrix.shape, (2, 2))
        topk_df = view.topk_dataframe()
        self.assertEqual(len(topk_df), 2)
        self.assertEqual(topk_df.iloc[0]["rank"], 1)

    def test_pairwise_reuses_sdk_segmenter(self) -> None:
        fake_segmenter = FakeSegmenter()
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=fake_segmenter) as mock_resolve:
            sdk = TibetanResearchSDK(device="cpu", model_id="fake/model", batch_size=1)
            with patch("tibetan_pipeline.sdk.TextEmbedder.encode_queries", return_value=EmbeddingResult("fake/model", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))):
                with patch("tibetan_pipeline.sdk.TextEmbedder.encode_corpus", return_value=EmbeddingResult("fake/model", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))):
                    view = sdk.pairwise("ཀ་། ཁ་།", "ཀ་། ཁ་།", top_k=1)

        mock_resolve.assert_called_once()
        self.assertEqual(view.segments_a, ["ཀ་།", "ཁ་།"])

    def test_pairwise_from_embedding_views_reuses_precomputed_embeddings(self) -> None:
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=FakeSegmenter()):
            sdk = TibetanResearchSDK(device="cpu", model_id="fake/model", batch_size=1)

        view_a = EmbeddingView(
            model_id="fake/model",
            device="cpu",
            sentences=["a0", "a1"],
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        view_b = EmbeddingView(
            model_id="fake/model",
            device="cpu",
            sentences=["b0", "b1"],
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )

        with patch("tibetan_pipeline.sdk.TextEmbedder.encode_queries", side_effect=AssertionError("should not re-embed queries")):
            with patch("tibetan_pipeline.sdk.TextEmbedder.encode_corpus", side_effect=AssertionError("should not re-embed corpus")):
                pairwise_view = sdk.pairwise_from_embedding_views(view_a, view_b, top_k=2)

        self.assertEqual(pairwise_view.similarity_matrix.shape, (2, 2))
        self.assertEqual(pairwise_view.segments_a, ["a0", "a1"])
        self.assertEqual(pairwise_view.segments_b, ["b0", "b1"])
        self.assertEqual(len(pairwise_view.matches), 2)
        self.assertEqual(pairwise_view.segment_records_a[0].index, 0)
        self.assertEqual(pairwise_view.segment_records_b[1].index, 1)
        self.assertGreaterEqual(pairwise_view.metrics.max_score, 0.0)


if __name__ == "__main__":
    unittest.main()
