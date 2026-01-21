"""Embedding providers for Smart Fork.

This module provides an abstraction layer for embedding text into vectors.
All embedding providers implement the EmbeddingProvider ABC, enabling
seamless switching between Vertex AI (primary) and Ollama (fallback).

Provider architecture:
- EmbeddingProvider: Abstract base class defining the interface
- VertexAIProvider: Google Vertex AI text-embedding-004 (768-dim)
- OllamaProvider: Local Ollama with nomic-embed-text (768-dim)
- create_provider(): Factory function with auto-fallback

Why a single file for all providers:
- AGENTS.md specifies flat structure (no nested `embedding/` directory)
- Providers are tightly coupled to the same interface
- Simplifies imports: `from smart_fork.embeddings import create_provider`

Task types for Vertex AI embeddings:
- RETRIEVAL_DOCUMENT: Used when indexing session chunks (optimized for document storage)
- RETRIEVAL_QUERY: Used when embedding user search queries (optimized for query matching)
Using matching task types improves retrieval quality vs using the same type for both.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smart_fork.config import EmbeddingConfig


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding providers must implement this interface to enable
    consistent embedding generation across different backends (Vertex AI,
    Ollama, or future providers).

    Implementations must:
    - Return 768-dimensional vectors for compatibility with LanceDB schema
    - Handle batching internally if the underlying API has limits
    - Raise clear exceptions on API errors (not silently return empty)

    Example usage:
        >>> provider = create_provider(config)
        >>> embeddings = provider.embed(["hello world", "foo bar"])
        >>> len(embeddings)  # Number of input texts
        2
        >>> len(embeddings[0])  # Vector dimensions
        768
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of texts.

        Args:
            texts: List of text strings to embed. Empty list returns empty list.
                   Each text should be non-empty; behavior on empty strings
                   is provider-dependent but generally returns a zero vector.

        Returns:
            List of embedding vectors, one per input text. Each vector is a
            list of 768 floats. Order matches input order.

        Raises:
            EmbeddingError: If the embedding API call fails (network error,
                            rate limit exceeded, authentication failure, etc.)
            ValueError: If texts contains invalid data (e.g., None values)
        """
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the identifier of the embedding model.

        This is stored in SessionChunk.model_used to track which model
        produced each embedding. Useful for debugging and for detecting
        when re-indexing is needed after model changes.

        Returns:
            Model identifier string (e.g., "text-embedding-004", "nomic-embed-text")
        """
        ...


class EmbeddingError(Exception):
    """Raised when an embedding operation fails.

    This exception wraps underlying provider-specific errors (API errors,
    network timeouts, rate limits) into a consistent exception type for
    callers to handle.

    Attributes:
        message: Human-readable error description
        provider: Name of the provider that failed (e.g., "vertex", "ollama")
        cause: The underlying exception, if any
    """

    def __init__(
        self,
        message: str,
        provider: str,
        cause: Exception | None = None,
    ) -> None:
        """Initialize an EmbeddingError.

        Args:
            message: Human-readable error description
            provider: Name of the provider that failed
            cause: The underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.cause = cause

    def __str__(self) -> str:
        """Return a formatted error message including provider and cause."""
        base = f"[{self.provider}] {self.message}"
        if self.cause:
            return f"{base}: {self.cause}"
        return base


class VertexAIProvider(EmbeddingProvider):
    """Google Vertex AI text-embedding-004 embedding provider.

    Produces 768-dimensional embeddings using the text-embedding-004 model.
    Handles batching internally, respecting the 250 texts per API call limit.

    Authentication:
        Uses Application Default Credentials (ADC). Set up via:
        - gcloud auth application-default login (for local development)
        - Service account key (for production deployments)
        - Workload Identity (for GKE deployments)

    Rate Limiting:
        Implements configurable delay between batch API calls to avoid
        hitting Vertex AI quotas. Default is 100ms per the spec.

    Task Types:
        Uses RETRIEVAL_DOCUMENT for optimal document indexing. When querying,
        callers should use RETRIEVAL_QUERY task type for best results.

    Example usage:
        >>> from smart_fork.config import EmbeddingConfig
        >>> config = EmbeddingConfig(vertex_project="my-project")
        >>> provider = VertexAIProvider(config)
        >>> embeddings = provider.embed(["hello world"])
        >>> len(embeddings[0])  # Vector dimensions
        768
    """

    # API limits from Vertex AI documentation
    MAX_BATCH_SIZE = 250  # Maximum texts per API request
    DIMENSIONS = 768  # Output vector dimensions for text-embedding-004

    def __init__(self, config: "EmbeddingConfig") -> None:
        """Initialize the Vertex AI embedding provider.

        Args:
            config: Embedding configuration containing project ID, location,
                    model name, batch size, and rate limit settings.

        Raises:
            ValueError: If vertex_project is not configured.
        """
        if config.vertex_project is None:
            raise ValueError(
                "vertex_project must be configured for Vertex AI provider. "
                "Set it in config.json or via environment variable."
            )

        self._project = config.vertex_project
        self._location = config.vertex_location
        self._model_id = config.vertex_model
        self._batch_size = min(config.batch_size, self.MAX_BATCH_SIZE)
        self._rate_limit_ms = config.rate_limit_ms

        # Lazy-initialized model instance
        # Using Any type here because vertexai types are not well-typed
        self._model: object | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize Vertex AI SDK and load the model.

        This is done lazily on first embed() call to avoid unnecessary
        SDK initialization if the provider is never used.

        Raises:
            EmbeddingError: If initialization fails (auth issues, invalid project, etc.)
        """
        if self._initialized:
            return

        try:
            # Import here to avoid loading heavy SDK until needed
            import vertexai
            from vertexai.language_models import TextEmbeddingModel

            vertexai.init(project=self._project, location=self._location)
            self._model = TextEmbeddingModel.from_pretrained(self._model_id)
            self._initialized = True

        except ImportError as e:
            raise EmbeddingError(
                message="google-cloud-aiplatform package not installed",
                provider="vertex",
                cause=e,
            )
        except Exception as e:
            raise EmbeddingError(
                message=f"Failed to initialize Vertex AI: {e}",
                provider="vertex",
                cause=e,
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of texts.

        Handles batching internally - if more than batch_size texts are
        provided, they are split into multiple API calls with rate limiting
        between calls.

        Args:
            texts: List of text strings to embed. Empty list returns empty list.
                   Each text should be non-empty. Very long texts are truncated
                   by the API to 2048 tokens.

        Returns:
            List of embedding vectors (768 dimensions each). Order matches
            input order.

        Raises:
            EmbeddingError: If the API call fails (network error, rate limit,
                            authentication failure, etc.)
            ValueError: If texts contains None values.
        """
        if not texts:
            return []

        # Validate inputs before making any API calls
        for i, text in enumerate(texts):
            if text is None:
                raise ValueError(f"texts[{i}] is None; all texts must be strings")

        self._ensure_initialized()

        # Import TextEmbeddingInput for creating typed inputs
        from vertexai.language_models import TextEmbeddingInput

        all_embeddings: list[list[float]] = []

        # Process in batches
        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]

            try:
                # Create typed inputs with task type for optimal retrieval
                # RETRIEVAL_DOCUMENT is used for indexing; queries should use
                # RETRIEVAL_QUERY but that's handled at a higher level
                inputs = [
                    TextEmbeddingInput(text=text, task_type="RETRIEVAL_DOCUMENT")
                    for text in batch
                ]

                # Call the embedding API
                # Type ignore needed because _model is typed as object | None
                embeddings = self._model.get_embeddings(inputs)  # type: ignore[union-attr]

                # Extract vectors from response objects
                for embedding in embeddings:
                    # embedding.values is list[float]
                    all_embeddings.append(list(embedding.values))

            except Exception as e:
                batch_num = batch_start // self._batch_size
                raise EmbeddingError(
                    message=f"Embedding API call failed for batch {batch_num} "
                    f"(texts {batch_start}-{batch_start + len(batch) - 1})",
                    provider="vertex",
                    cause=e,
                )

            # Rate limiting: delay between batches (skip after last batch)
            if batch_start + self._batch_size < len(texts) and self._rate_limit_ms > 0:
                time.sleep(self._rate_limit_ms / 1000.0)

        return all_embeddings

    def model_name(self) -> str:
        """Return the identifier of the embedding model.

        Returns:
            Model identifier string (e.g., "text-embedding-004").
        """
        return self._model_id
