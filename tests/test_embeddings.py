"""Tests for Smart Fork embedding providers.

Tests the EmbeddingProvider ABC, VertexAIProvider, and related functionality.
Uses mocking to avoid actual API calls during testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from smart_fork.config import EmbeddingConfig
from smart_fork.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    VertexAIProvider,
)

if TYPE_CHECKING:
    from collections.abc import Generator


# -----------------------------------------------------------------------------
# EmbeddingError Tests
# -----------------------------------------------------------------------------


class TestEmbeddingError:
    """Tests for EmbeddingError exception class."""

    def test_error_with_message_and_provider(self) -> None:
        """Error stores message and provider."""
        error = EmbeddingError(
            message="API call failed",
            provider="vertex",
        )
        assert error.message == "API call failed"
        assert error.provider == "vertex"
        assert error.cause is None

    def test_error_with_cause(self) -> None:
        """Error stores the underlying cause."""
        cause = RuntimeError("Connection refused")
        error = EmbeddingError(
            message="API call failed",
            provider="ollama",
            cause=cause,
        )
        assert error.cause is cause

    def test_error_str_without_cause(self) -> None:
        """String representation without cause."""
        error = EmbeddingError(
            message="Rate limit exceeded",
            provider="vertex",
        )
        assert str(error) == "[vertex] Rate limit exceeded"

    def test_error_str_with_cause(self) -> None:
        """String representation includes cause."""
        cause = ValueError("Invalid input")
        error = EmbeddingError(
            message="Embedding failed",
            provider="vertex",
            cause=cause,
        )
        assert str(error) == "[vertex] Embedding failed: Invalid input"


# -----------------------------------------------------------------------------
# EmbeddingProvider ABC Tests
# -----------------------------------------------------------------------------


class TestEmbeddingProviderABC:
    """Tests for EmbeddingProvider abstract base class."""

    def test_cannot_instantiate_abc(self) -> None:
        """Cannot directly instantiate the ABC."""
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]

    def test_concrete_implementation_required(self) -> None:
        """Subclass must implement abstract methods."""

        class IncompleteProvider(EmbeddingProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore[abstract]

    def test_complete_implementation_works(self) -> None:
        """Subclass with all methods can be instantiated."""

        class MockProvider(EmbeddingProvider):
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 768 for _ in texts]

            def model_name(self) -> str:
                return "mock-model"

        provider = MockProvider()
        assert provider.model_name() == "mock-model"
        result = provider.embed(["test"])
        assert len(result) == 1
        assert len(result[0]) == 768


# -----------------------------------------------------------------------------
# VertexAIProvider Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def vertex_config() -> EmbeddingConfig:
    """Create a config for Vertex AI provider testing."""
    return EmbeddingConfig(
        provider="vertex",
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_model="text-embedding-004",
        batch_size=100,
        rate_limit_ms=0,  # No delay for tests
    )


def create_mock_model() -> MagicMock:
    """Create a mock TextEmbeddingModel with proper embedding responses."""
    mock_model = MagicMock()

    def create_mock_embeddings(inputs: list) -> list:
        """Create mock embedding objects for each input."""
        result = []
        for _ in inputs:
            mock_embedding = MagicMock()
            mock_embedding.values = [0.1] * 768
            result.append(mock_embedding)
        return result

    mock_model.get_embeddings.side_effect = create_mock_embeddings
    return mock_model


class TestVertexAIProviderInit:
    """Tests for VertexAIProvider initialization."""

    def test_requires_project_id(self) -> None:
        """Raises ValueError if vertex_project is None."""
        config = EmbeddingConfig(
            provider="vertex",
            vertex_project=None,  # Missing!
        )
        with pytest.raises(ValueError, match="vertex_project must be configured"):
            VertexAIProvider(config)

    def test_stores_config_values(self, vertex_config: EmbeddingConfig) -> None:
        """Provider stores configuration values."""
        provider = VertexAIProvider(vertex_config)
        assert provider._project == "test-project"
        assert provider._location == "us-central1"
        assert provider._model_id == "text-embedding-004"
        assert provider._batch_size == 100
        assert provider._rate_limit_ms == 0

    def test_respects_max_batch_size(self) -> None:
        """Batch size is capped at MAX_BATCH_SIZE (250)."""
        config = EmbeddingConfig(
            provider="vertex",
            vertex_project="test-project",
            batch_size=250,  # Max allowed in config
        )
        provider = VertexAIProvider(config)
        # Config allows 250, API limit is 250, so 250 is used
        assert provider._batch_size == 250

    def test_lazy_initialization(self, vertex_config: EmbeddingConfig) -> None:
        """Model is not loaded at init time."""
        provider = VertexAIProvider(vertex_config)
        assert provider._model is None
        assert provider._initialized is False


class TestVertexAIProviderEmbed:
    """Tests for VertexAIProvider.embed() method."""

    def test_empty_list_returns_empty(self, vertex_config: EmbeddingConfig) -> None:
        """Embedding empty list returns empty list without API call."""
        provider = VertexAIProvider(vertex_config)
        result = provider.embed([])
        assert result == []
        # Model should not be initialized for empty input
        assert provider._initialized is False

    def test_none_in_texts_raises_valueerror(
        self, vertex_config: EmbeddingConfig
    ) -> None:
        """Raises ValueError if texts contains None."""
        provider = VertexAIProvider(vertex_config)
        with pytest.raises(ValueError, match="texts\\[0\\] is None"):
            provider.embed([None])  # type: ignore[list-item]

    def test_none_at_index_raises_with_index(
        self, vertex_config: EmbeddingConfig
    ) -> None:
        """Error message includes the index of the None value."""
        provider = VertexAIProvider(vertex_config)
        with pytest.raises(ValueError, match="texts\\[2\\] is None"):
            provider.embed(["a", "b", None, "d"])  # type: ignore[list-item]

    def test_single_text_embedding(
        self,
        vertex_config: EmbeddingConfig,
    ) -> None:
        """Embeds a single text successfully."""
        mock_model = create_mock_model()

        with patch.dict(
            "sys.modules",
            {
                "vertexai": MagicMock(),
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=MagicMock(from_pretrained=lambda x: mock_model),
                    TextEmbeddingInput=MagicMock(side_effect=lambda **kwargs: kwargs),
                ),
            },
        ):
            provider = VertexAIProvider(vertex_config)
            result = provider.embed(["hello world"])

        assert len(result) == 1
        assert len(result[0]) == 768
        assert all(isinstance(v, float) for v in result[0])

    def test_multiple_texts_embedding(
        self,
        vertex_config: EmbeddingConfig,
    ) -> None:
        """Embeds multiple texts successfully."""
        mock_model = create_mock_model()

        with patch.dict(
            "sys.modules",
            {
                "vertexai": MagicMock(),
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=MagicMock(from_pretrained=lambda x: mock_model),
                    TextEmbeddingInput=MagicMock(side_effect=lambda **kwargs: kwargs),
                ),
            },
        ):
            texts = ["text one", "text two", "text three"]
            provider = VertexAIProvider(vertex_config)
            result = provider.embed(texts)

        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == 768

    def test_batches_large_input(self) -> None:
        """Splits large input into batches."""
        mock_model = create_mock_model()

        config = EmbeddingConfig(
            provider="vertex",
            vertex_project="test-project",
            batch_size=10,  # Small batch for testing
            rate_limit_ms=0,
        )

        with patch.dict(
            "sys.modules",
            {
                "vertexai": MagicMock(),
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=MagicMock(from_pretrained=lambda x: mock_model),
                    TextEmbeddingInput=MagicMock(side_effect=lambda **kwargs: kwargs),
                ),
            },
        ):
            provider = VertexAIProvider(config)

            # Create 25 texts to trigger 3 batches (10, 10, 5)
            texts = [f"text {i}" for i in range(25)]
            result = provider.embed(texts)

            # Should have called get_embeddings 3 times
            assert mock_model.get_embeddings.call_count == 3
            assert len(result) == 25

    @patch("time.sleep")
    def test_rate_limiting_between_batches(
        self,
        mock_sleep: MagicMock,
    ) -> None:
        """Applies rate limiting delay between batches."""
        mock_model = create_mock_model()

        config = EmbeddingConfig(
            provider="vertex",
            vertex_project="test-project",
            batch_size=10,
            rate_limit_ms=100,  # 100ms delay
        )

        with patch.dict(
            "sys.modules",
            {
                "vertexai": MagicMock(),
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=MagicMock(from_pretrained=lambda x: mock_model),
                    TextEmbeddingInput=MagicMock(side_effect=lambda **kwargs: kwargs),
                ),
            },
        ):
            provider = VertexAIProvider(config)

            # Create 25 texts to trigger 3 batches
            texts = [f"text {i}" for i in range(25)]
            provider.embed(texts)

            # Should sleep between batches (2 times for 3 batches)
            assert mock_sleep.call_count == 2
            # Each sleep should be 0.1 seconds (100ms)
            mock_sleep.assert_called_with(0.1)

    @patch("time.sleep")
    def test_no_rate_limit_after_last_batch(
        self,
        mock_sleep: MagicMock,
    ) -> None:
        """No sleep after the final batch."""
        mock_model = create_mock_model()

        config = EmbeddingConfig(
            provider="vertex",
            vertex_project="test-project",
            batch_size=10,
            rate_limit_ms=100,
        )

        with patch.dict(
            "sys.modules",
            {
                "vertexai": MagicMock(),
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=MagicMock(from_pretrained=lambda x: mock_model),
                    TextEmbeddingInput=MagicMock(side_effect=lambda **kwargs: kwargs),
                ),
            },
        ):
            provider = VertexAIProvider(config)

            # Exactly 10 texts = 1 batch, no sleep needed
            texts = [f"text {i}" for i in range(10)]
            provider.embed(texts)

            mock_sleep.assert_not_called()

    def test_api_error_wrapped_in_embedding_error(
        self,
        vertex_config: EmbeddingConfig,
    ) -> None:
        """API errors are wrapped in EmbeddingError."""
        mock_model = MagicMock()
        mock_model.get_embeddings.side_effect = RuntimeError("API quota exceeded")

        with patch.dict(
            "sys.modules",
            {
                "vertexai": MagicMock(),
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=MagicMock(from_pretrained=lambda x: mock_model),
                    TextEmbeddingInput=MagicMock(side_effect=lambda **kwargs: kwargs),
                ),
            },
        ):
            provider = VertexAIProvider(vertex_config)

            with pytest.raises(EmbeddingError) as exc_info:
                provider.embed(["test"])

            error = exc_info.value
            assert error.provider == "vertex"
            assert "batch 0" in error.message
            assert isinstance(error.cause, RuntimeError)


class TestVertexAIProviderModelName:
    """Tests for VertexAIProvider.model_name() method."""

    def test_returns_configured_model(self, vertex_config: EmbeddingConfig) -> None:
        """Returns the configured model name."""
        provider = VertexAIProvider(vertex_config)
        assert provider.model_name() == "text-embedding-004"

    def test_returns_custom_model_name(self) -> None:
        """Returns custom model name if configured."""
        config = EmbeddingConfig(
            provider="vertex",
            vertex_project="test-project",
            vertex_model="text-embedding-005",
        )
        provider = VertexAIProvider(config)
        assert provider.model_name() == "text-embedding-005"


class TestVertexAIProviderInitialization:
    """Tests for lazy initialization behavior."""

    def test_initialization_error_wrapped(self, vertex_config: EmbeddingConfig) -> None:
        """Initialization errors are wrapped in EmbeddingError."""
        mock_vertexai = MagicMock()
        mock_vertexai.init.side_effect = Exception("Auth failed")

        with patch.dict(
            "sys.modules",
            {
                "vertexai": mock_vertexai,
                "vertexai.language_models": MagicMock(),
            },
        ):
            provider = VertexAIProvider(vertex_config)

            with pytest.raises(EmbeddingError) as exc_info:
                provider.embed(["test"])

            error = exc_info.value
            assert error.provider == "vertex"
            assert "Failed to initialize" in error.message

    def test_initialized_only_once(
        self,
        vertex_config: EmbeddingConfig,
    ) -> None:
        """Model is initialized only on first embed call."""
        mock_model = create_mock_model()
        mock_vertexai = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "vertexai": mock_vertexai,
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=MagicMock(from_pretrained=lambda x: mock_model),
                    TextEmbeddingInput=MagicMock(side_effect=lambda **kwargs: kwargs),
                ),
            },
        ):
            provider = VertexAIProvider(vertex_config)

            # First call initializes
            provider.embed(["text 1"])
            assert provider._initialized is True
            init_call_count = mock_vertexai.init.call_count

            # Second call reuses
            provider.embed(["text 2"])
            assert mock_vertexai.init.call_count == init_call_count  # No new init


# -----------------------------------------------------------------------------
# Integration-style tests (still mocked but more realistic scenarios)
# -----------------------------------------------------------------------------


class TestVertexAIProviderIntegration:
    """Integration-style tests for realistic scenarios."""

    def test_large_batch_processing(self) -> None:
        """Handles 250+ texts correctly with batching."""
        mock_model = create_mock_model()

        config = EmbeddingConfig(
            provider="vertex",
            vertex_project="test-project",
            batch_size=250,  # Max batch size
            rate_limit_ms=0,
        )

        with patch.dict(
            "sys.modules",
            {
                "vertexai": MagicMock(),
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=MagicMock(from_pretrained=lambda x: mock_model),
                    TextEmbeddingInput=MagicMock(side_effect=lambda **kwargs: kwargs),
                ),
            },
        ):
            provider = VertexAIProvider(config)

            # Create 500 texts (2 full batches)
            texts = [f"Document content {i}" for i in range(500)]
            result = provider.embed(texts)

            assert len(result) == 500
            assert mock_model.get_embeddings.call_count == 2

    def test_preserves_input_order(self) -> None:
        """Embeddings are returned in same order as inputs."""
        mock_model = MagicMock()
        call_count = [0]

        def ordered_embeddings(inputs: list) -> list:
            result = []
            for i in range(len(inputs)):
                mock_emb = MagicMock()
                # Each embedding has a unique first value
                mock_emb.values = [float(call_count[0] + i)] + [0.0] * 767
                result.append(mock_emb)
            call_count[0] += len(inputs)
            return result

        mock_model.get_embeddings.side_effect = ordered_embeddings

        config = EmbeddingConfig(
            provider="vertex",
            vertex_project="test-project",
            batch_size=3,
            rate_limit_ms=0,
        )

        with patch.dict(
            "sys.modules",
            {
                "vertexai": MagicMock(),
                "vertexai.language_models": MagicMock(
                    TextEmbeddingModel=MagicMock(from_pretrained=lambda x: mock_model),
                    TextEmbeddingInput=MagicMock(side_effect=lambda **kwargs: kwargs),
                ),
            },
        ):
            provider = VertexAIProvider(config)

            # 7 texts across 3 batches
            texts = [f"text {i}" for i in range(7)]
            result = provider.embed(texts)

            # Verify order is preserved
            assert [r[0] for r in result] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


class TestEmbeddingProviderProtocol:
    """Verify VertexAIProvider satisfies EmbeddingProvider interface."""

    def test_is_embedding_provider(
        self,
        vertex_config: EmbeddingConfig,
    ) -> None:
        """VertexAIProvider is an instance of EmbeddingProvider."""
        provider = VertexAIProvider(vertex_config)
        assert isinstance(provider, EmbeddingProvider)

    def test_has_required_methods(
        self,
        vertex_config: EmbeddingConfig,
    ) -> None:
        """VertexAIProvider has all required methods."""
        provider = VertexAIProvider(vertex_config)
        assert callable(provider.embed)
        assert callable(provider.model_name)
