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
    OllamaProvider,
    VertexAIProvider,
    create_provider,
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


# -----------------------------------------------------------------------------
# OllamaProvider Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def ollama_config() -> EmbeddingConfig:
    """Create a config for Ollama provider testing."""
    return EmbeddingConfig(
        provider="ollama",
        ollama_host="http://localhost:11434",
        ollama_model="nomic-embed-text",
        dimensions=768,
    )


def create_mock_ollama_response(
    embeddings: list[list[float]],
    status_code: int = 200,
) -> MagicMock:
    """Create a mock httpx Response for Ollama API."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {"embeddings": embeddings}
    mock_response.text = ""
    return mock_response


class TestOllamaProviderInit:
    """Tests for OllamaProvider initialization."""

    def test_stores_config_values(self, ollama_config: EmbeddingConfig) -> None:
        """Provider stores configuration values."""
        provider = OllamaProvider(ollama_config)
        assert provider._host == "http://localhost:11434"
        assert provider._model_id == "nomic-embed-text"
        assert provider._dimensions == 768

    def test_strips_trailing_slash_from_host(self) -> None:
        """Trailing slash is stripped from host URL."""
        config = EmbeddingConfig(
            provider="ollama",
            ollama_host="http://localhost:11434/",
        )
        provider = OllamaProvider(config)
        assert provider._host == "http://localhost:11434"

    def test_custom_host_and_model(self) -> None:
        """Custom host and model can be configured."""
        config = EmbeddingConfig(
            provider="ollama",
            ollama_host="http://myserver:8080",
            ollama_model="custom-embed",
        )
        provider = OllamaProvider(config)
        assert provider._host == "http://myserver:8080"
        assert provider._model_id == "custom-embed"


class TestOllamaProviderEmbed:
    """Tests for OllamaProvider.embed() method."""

    def test_empty_list_returns_empty(self, ollama_config: EmbeddingConfig) -> None:
        """Embedding empty list returns empty list without API call."""
        provider = OllamaProvider(ollama_config)
        result = provider.embed([])
        assert result == []

    def test_none_in_texts_raises_valueerror(
        self, ollama_config: EmbeddingConfig
    ) -> None:
        """Raises ValueError if texts contains None."""
        provider = OllamaProvider(ollama_config)
        with pytest.raises(ValueError, match="texts\\[0\\] is None"):
            provider.embed([None])  # type: ignore[list-item]

    def test_none_at_index_raises_with_index(
        self, ollama_config: EmbeddingConfig
    ) -> None:
        """Error message includes the index of the None value."""
        provider = OllamaProvider(ollama_config)
        with pytest.raises(ValueError, match="texts\\[2\\] is None"):
            provider.embed(["a", "b", None, "d"])  # type: ignore[list-item]

    def test_single_text_embedding(self, ollama_config: EmbeddingConfig) -> None:
        """Embeds a single text successfully."""
        mock_response = create_mock_ollama_response([[0.1] * 768])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"httpx": MagicMock(Client=MagicMock(return_value=mock_client))},
        ):
            provider = OllamaProvider(ollama_config)
            result = provider.embed(["hello world"])

        assert len(result) == 1
        assert len(result[0]) == 768
        assert all(isinstance(v, float) for v in result[0])

    def test_multiple_texts_embedding(self, ollama_config: EmbeddingConfig) -> None:
        """Embeds multiple texts successfully."""
        mock_embeddings = [[0.1 * (i + 1)] * 768 for i in range(3)]
        mock_response = create_mock_ollama_response(mock_embeddings)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"httpx": MagicMock(Client=MagicMock(return_value=mock_client))},
        ):
            texts = ["text one", "text two", "text three"]
            provider = OllamaProvider(ollama_config)
            result = provider.embed(texts)

        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == 768

    def test_sends_correct_request(self, ollama_config: EmbeddingConfig) -> None:
        """Sends correct payload to Ollama API."""
        mock_response = create_mock_ollama_response([[0.1] * 768, [0.2] * 768])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"httpx": MagicMock(Client=MagicMock(return_value=mock_client))},
        ):
            provider = OllamaProvider(ollama_config)
            provider.embed(["hello", "world"])

        # Verify the request was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/embed"
        assert call_args[1]["json"]["model"] == "nomic-embed-text"
        assert call_args[1]["json"]["input"] == ["hello", "world"]

    def test_http_error_wrapped_in_embedding_error(
        self, ollama_config: EmbeddingConfig
    ) -> None:
        """HTTP errors are wrapped in EmbeddingError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.return_value = {"error": "Model not found"}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"httpx": MagicMock(Client=MagicMock(return_value=mock_client))},
        ):
            provider = OllamaProvider(ollama_config)

            with pytest.raises(EmbeddingError) as exc_info:
                provider.embed(["test"])

            error = exc_info.value
            assert error.provider == "ollama"
            assert "status 500" in error.message

    def test_connection_error_wrapped_in_embedding_error(
        self, ollama_config: EmbeddingConfig
    ) -> None:
        """Connection errors are wrapped in EmbeddingError."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("Connection refused")

        with patch.dict(
            "sys.modules",
            {"httpx": MagicMock(Client=MagicMock(return_value=mock_client))},
        ):
            provider = OllamaProvider(ollama_config)

            with pytest.raises(EmbeddingError) as exc_info:
                provider.embed(["test"])

            error = exc_info.value
            assert error.provider == "ollama"
            assert "Connection refused" in str(error)
            assert error.cause is not None

    def test_mismatched_embedding_count_raises_error(
        self, ollama_config: EmbeddingConfig
    ) -> None:
        """Raises error if response has wrong number of embeddings."""
        # Return 1 embedding when we sent 2 texts
        mock_response = create_mock_ollama_response([[0.1] * 768])
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"httpx": MagicMock(Client=MagicMock(return_value=mock_client))},
        ):
            provider = OllamaProvider(ollama_config)

            with pytest.raises(EmbeddingError) as exc_info:
                provider.embed(["text1", "text2"])

            error = exc_info.value
            assert error.provider == "ollama"
            assert "Expected 2 embeddings, got 1" in error.message

    def test_httpx_not_installed_raises_error(
        self, ollama_config: EmbeddingConfig
    ) -> None:
        """Raises EmbeddingError if httpx is not installed."""
        provider = OllamaProvider(ollama_config)

        # Simulate httpx not installed by making import fail
        import sys

        # Temporarily remove httpx from modules
        original_httpx = sys.modules.get("httpx")
        sys.modules["httpx"] = None  # type: ignore[assignment]

        try:
            with pytest.raises(EmbeddingError) as exc_info:
                provider.embed(["test"])

            error = exc_info.value
            # Note: When httpx is set to None, importing it raises TypeError
            # The error will be caught as a general exception
            assert error.provider == "ollama"
        finally:
            # Restore httpx
            if original_httpx is not None:
                sys.modules["httpx"] = original_httpx
            else:
                sys.modules.pop("httpx", None)


class TestOllamaProviderModelName:
    """Tests for OllamaProvider.model_name() method."""

    def test_returns_configured_model(self, ollama_config: EmbeddingConfig) -> None:
        """Returns the configured model name."""
        provider = OllamaProvider(ollama_config)
        assert provider.model_name() == "nomic-embed-text"

    def test_returns_custom_model_name(self) -> None:
        """Returns custom model name if configured."""
        config = EmbeddingConfig(
            provider="ollama",
            ollama_model="mxbai-embed-large",
        )
        provider = OllamaProvider(config)
        assert provider.model_name() == "mxbai-embed-large"


class TestOllamaProviderErrorHandling:
    """Tests for OllamaProvider error parsing."""

    def test_parse_error_from_json_response(
        self, ollama_config: EmbeddingConfig
    ) -> None:
        """Parses error message from JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = '{"error": "model not found"}'
        mock_response.json.return_value = {"error": "model not found"}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"httpx": MagicMock(Client=MagicMock(return_value=mock_client))},
        ):
            provider = OllamaProvider(ollama_config)

            with pytest.raises(EmbeddingError) as exc_info:
                provider.embed(["test"])

            error = exc_info.value
            assert "model not found" in error.message

    def test_parse_error_from_plain_text_response(
        self, ollama_config: EmbeddingConfig
    ) -> None:
        """Falls back to text when JSON parsing fails."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"
        mock_response.json.side_effect = Exception("Not JSON")

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"httpx": MagicMock(Client=MagicMock(return_value=mock_client))},
        ):
            provider = OllamaProvider(ollama_config)

            with pytest.raises(EmbeddingError) as exc_info:
                provider.embed(["test"])

            error = exc_info.value
            assert "Bad Gateway" in error.message


class TestOllamaProviderProtocol:
    """Verify OllamaProvider satisfies EmbeddingProvider interface."""

    def test_is_embedding_provider(self, ollama_config: EmbeddingConfig) -> None:
        """OllamaProvider is an instance of EmbeddingProvider."""
        provider = OllamaProvider(ollama_config)
        assert isinstance(provider, EmbeddingProvider)

    def test_has_required_methods(self, ollama_config: EmbeddingConfig) -> None:
        """OllamaProvider has all required methods."""
        provider = OllamaProvider(ollama_config)
        assert callable(provider.embed)
        assert callable(provider.model_name)


class TestOllamaProviderIntegration:
    """Integration-style tests for realistic scenarios."""

    def test_preserves_input_order(self, ollama_config: EmbeddingConfig) -> None:
        """Embeddings are returned in same order as inputs."""
        # Each embedding has unique first value to track order
        mock_embeddings = [[float(i)] + [0.0] * 767 for i in range(5)]
        mock_response = create_mock_ollama_response(mock_embeddings)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"httpx": MagicMock(Client=MagicMock(return_value=mock_client))},
        ):
            provider = OllamaProvider(ollama_config)
            texts = [f"text {i}" for i in range(5)]
            result = provider.embed(texts)

        # Verify order is preserved
        assert [r[0] for r in result] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_large_batch_single_request(self, ollama_config: EmbeddingConfig) -> None:
        """All texts sent in single request (no internal batching)."""
        # Ollama handles batching internally, so we send all at once
        mock_embeddings = [[0.1] * 768 for _ in range(100)]
        mock_response = create_mock_ollama_response(mock_embeddings)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"httpx": MagicMock(Client=MagicMock(return_value=mock_client))},
        ):
            provider = OllamaProvider(ollama_config)
            texts = [f"text {i}" for i in range(100)]
            result = provider.embed(texts)

        # Should have called post exactly once
        assert mock_client.post.call_count == 1
        assert len(result) == 100


# -----------------------------------------------------------------------------
# create_provider Factory Tests
# -----------------------------------------------------------------------------


class TestCreateProviderWithOllama:
    """Tests for create_provider() when Ollama is explicitly configured."""

    def test_ollama_provider_when_configured(
        self, ollama_config: EmbeddingConfig
    ) -> None:
        """Returns OllamaProvider when provider='ollama'."""
        provider = create_provider(ollama_config)
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name() == "nomic-embed-text"

    def test_ollama_ignores_vertex_config(self) -> None:
        """OllamaProvider returned even if vertex_project is set."""
        config = EmbeddingConfig(
            provider="ollama",
            vertex_project="some-project",  # Should be ignored
            ollama_model="custom-model",
        )
        provider = create_provider(config)
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name() == "custom-model"


class TestCreateProviderWithVertex:
    """Tests for create_provider() when Vertex AI is configured."""

    def test_vertex_provider_when_configured(
        self, vertex_config: EmbeddingConfig
    ) -> None:
        """Returns VertexAIProvider when provider='vertex' and project set."""
        provider = create_provider(vertex_config)
        assert isinstance(provider, VertexAIProvider)
        assert provider.model_name() == "text-embedding-004"

    def test_vertex_custom_model(self) -> None:
        """Returns VertexAIProvider with custom model."""
        config = EmbeddingConfig(
            provider="vertex",
            vertex_project="test-project",
            vertex_model="text-embedding-005",
        )
        provider = create_provider(config)
        assert isinstance(provider, VertexAIProvider)
        assert provider.model_name() == "text-embedding-005"


class TestCreateProviderFallback:
    """Tests for auto-fallback from Vertex to Ollama."""

    def test_fallback_when_vertex_project_missing(self) -> None:
        """Falls back to Ollama when vertex_project is None."""
        config = EmbeddingConfig(
            provider="vertex",
            vertex_project=None,  # No project configured
        )
        provider = create_provider(config)
        assert isinstance(provider, OllamaProvider)

    def test_fallback_disabled_raises_error(self) -> None:
        """Raises EmbeddingError when fallback disabled and Vertex unavailable."""
        config = EmbeddingConfig(
            provider="vertex",
            vertex_project=None,
        )
        with pytest.raises(EmbeddingError) as exc_info:
            create_provider(config, allow_fallback=False)

        error = exc_info.value
        assert error.provider == "vertex"
        assert "vertex_project" in error.message.lower()

    def test_fallback_uses_ollama_config(self) -> None:
        """Fallback uses Ollama settings from config."""
        config = EmbeddingConfig(
            provider="vertex",
            vertex_project=None,  # Triggers fallback
            ollama_host="http://custom:8080",
            ollama_model="custom-embed",
        )
        provider = create_provider(config)
        assert isinstance(provider, OllamaProvider)
        assert provider._host == "http://custom:8080"
        assert provider.model_name() == "custom-embed"


class TestCreateProviderEdgeCases:
    """Edge case tests for create_provider()."""

    def test_is_embedding_provider(self, ollama_config: EmbeddingConfig) -> None:
        """Returned provider is an EmbeddingProvider."""
        provider = create_provider(ollama_config)
        assert isinstance(provider, EmbeddingProvider)

    def test_vertex_provider_is_embedding_provider(
        self, vertex_config: EmbeddingConfig
    ) -> None:
        """VertexAI provider is an EmbeddingProvider."""
        provider = create_provider(vertex_config)
        assert isinstance(provider, EmbeddingProvider)

    def test_default_provider_is_vertex(self) -> None:
        """Default config uses Vertex provider if project is set."""
        config = EmbeddingConfig(vertex_project="default-project")
        provider = create_provider(config)
        assert isinstance(provider, VertexAIProvider)

    def test_default_config_falls_back_to_ollama(self) -> None:
        """Default config without project falls back to Ollama."""
        config = EmbeddingConfig()  # No vertex_project
        provider = create_provider(config)
        assert isinstance(provider, OllamaProvider)
