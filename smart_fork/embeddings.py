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
"""

from __future__ import annotations

from abc import ABC, abstractmethod


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
