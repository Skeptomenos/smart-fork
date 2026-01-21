"""Configuration loading for Smart Fork.

Provides type-safe configuration with Pydantic validation.
Config can be overridden via JSON file at:
    ~/.local/share/opencode/smart-fork/config.json

Why Pydantic for config (vs dataclasses for storage models):
- Pydantic excels at validation and parsing from JSON/dict
- Automatic type coercion (e.g., string "768" -> int 768)
- Better error messages for config issues
- Supports optional fields with defaults naturally
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# Default paths following XDG Base Directory spec
# Sessions source (read-only): ~/.local/share/opencode/sessions/
# Smart fork data: ~/.local/share/opencode/smart-fork/
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "opencode" / "smart-fork"
DEFAULT_SESSIONS_DIR = Path.home() / ".local" / "share" / "opencode" / "sessions"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers.

    Attributes:
        provider: Which embedding provider to use. "vertex" uses Google Vertex AI,
                  "ollama" uses local Ollama server. Vertex is primary, Ollama is fallback.
        vertex_project: GCP project ID for Vertex AI. Required if using vertex provider.
        vertex_location: GCP region for Vertex AI (default: us-central1).
        vertex_model: Vertex AI embedding model name.
        ollama_host: Ollama server URL.
        ollama_model: Ollama embedding model name.
        dimensions: Embedding vector dimensions. Must match the model output.
        batch_size: Maximum texts per embedding API call. Vertex supports up to 250.
        rate_limit_ms: Milliseconds to wait between API batch requests.
    """

    provider: Literal["vertex", "ollama"] = Field(
        default="vertex",
        description="Embedding provider: 'vertex' (primary) or 'ollama' (fallback)",
    )
    vertex_project: str | None = Field(
        default=None,
        description="GCP project ID for Vertex AI",
    )
    vertex_location: str = Field(
        default="us-central1",
        description="GCP region for Vertex AI",
    )
    vertex_model: str = Field(
        default="text-embedding-004",
        description="Vertex AI embedding model name",
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model name",
    )
    dimensions: int = Field(
        default=768,
        description="Embedding vector dimensions",
        ge=1,
    )
    batch_size: int = Field(
        default=100,
        description="Maximum texts per embedding API call",
        ge=1,
        le=250,
    )
    rate_limit_ms: int = Field(
        default=100,
        description="Milliseconds between API batch requests",
        ge=0,
    )


class ChunkingConfig(BaseModel):
    """Configuration for text chunking.

    Attributes:
        target_tokens: Target chunk size in tokens (aim for middle of range).
        min_tokens: Minimum tokens per chunk.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between consecutive chunks.
        encoding: tiktoken encoding name for token counting.
    """

    target_tokens: int = Field(
        default=768,
        description="Target chunk size in tokens",
        ge=1,
    )
    min_tokens: int = Field(
        default=512,
        description="Minimum tokens per chunk",
        ge=1,
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens per chunk",
        ge=1,
    )
    overlap_tokens: int = Field(
        default=50,
        description="Token overlap between chunks",
        ge=0,
    )
    encoding: str = Field(
        default="cl100k_base",
        description="tiktoken encoding name",
    )


class QueryConfig(BaseModel):
    """Configuration for query and search behavior.

    Attributes:
        top_chunks: Number of chunks to retrieve from vector search.
        top_results: Number of session results to return to user.
        recency_half_life_days: Half-life for recency decay scoring.
    """

    top_chunks: int = Field(
        default=20,
        description="Number of chunks to retrieve from vector search",
        ge=1,
    )
    top_results: int = Field(
        default=5,
        description="Number of session results to return",
        ge=1,
    )
    recency_half_life_days: int = Field(
        default=30,
        description="Half-life in days for recency decay scoring",
        ge=1,
    )


class ScoringWeights(BaseModel):
    """Weights for composite scoring algorithm.

    All weights should sum to 1.0 for normalized scoring.
    See spec section "Scoring Algorithm" for details.

    Attributes:
        best_similarity: Weight for highest matching chunk similarity.
        avg_similarity: Weight for mean similarity across matching chunks.
        chunk_ratio: Weight for matching chunks / total chunks ratio.
        recency: Weight for time-based recency decay.
        chain_quality: Weight for session chain bonus (has parent link).
    """

    best_similarity: float = Field(
        default=0.40,
        description="Weight for best chunk similarity",
        ge=0.0,
        le=1.0,
    )
    avg_similarity: float = Field(
        default=0.20,
        description="Weight for average chunk similarity",
        ge=0.0,
        le=1.0,
    )
    chunk_ratio: float = Field(
        default=0.05,
        description="Weight for chunk ratio",
        ge=0.0,
        le=1.0,
    )
    recency: float = Field(
        default=0.25,
        description="Weight for recency decay",
        ge=0.0,
        le=1.0,
    )
    chain_quality: float = Field(
        default=0.10,
        description="Weight for chain quality bonus",
        ge=0.0,
        le=1.0,
    )


class PathsConfig(BaseModel):
    """Configuration for file and directory paths.

    All paths support ~ expansion for home directory.

    Attributes:
        data_dir: Root directory for Smart Fork data (LanceDB, config, state).
        sessions_dir: Directory containing OpenCode session transcripts (read-only).
        lance_dir: Subdirectory under data_dir for LanceDB storage.
        sync_state_file: Filename for sync state JSON.
    """

    data_dir: Path = Field(
        default=DEFAULT_DATA_DIR,
        description="Root directory for Smart Fork data",
    )
    sessions_dir: Path = Field(
        default=DEFAULT_SESSIONS_DIR,
        description="Directory containing OpenCode sessions",
    )
    lance_dir: str = Field(
        default="lance",
        description="Subdirectory for LanceDB storage",
    )
    sync_state_file: str = Field(
        default="sync-state.json",
        description="Filename for sync state",
    )

    @property
    def lance_path(self) -> Path:
        """Full path to LanceDB storage directory."""
        return self.data_dir / self.lance_dir

    @property
    def sync_state_path(self) -> Path:
        """Full path to sync state JSON file."""
        return self.data_dir / self.sync_state_file

    @property
    def config_path(self) -> Path:
        """Full path to config JSON file."""
        return self.data_dir / "config.json"


class SmartForkConfig(BaseModel):
    """Root configuration for Smart Fork.

    Aggregates all sub-configurations. Load with `load_config()`.

    Attributes:
        embedding: Embedding provider configuration.
        chunking: Text chunking configuration.
        query: Query and search configuration.
        scoring: Scoring weights configuration.
        paths: File and directory paths configuration.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )


def load_config(config_path: Path | None = None) -> SmartForkConfig:
    """Load configuration from JSON file with defaults.

    Loads config from the specified path, or from the default location
    (~/.local/share/opencode/smart-fork/config.json). If the file doesn't
    exist, returns default configuration.

    Args:
        config_path: Optional path to config JSON file. If None, uses default.

    Returns:
        SmartForkConfig with values from file overlaid on defaults.

    Raises:
        ValueError: If config file exists but contains invalid JSON or
                    values that fail Pydantic validation.
    """
    if config_path is None:
        config_path = DEFAULT_DATA_DIR / "config.json"

    # Expand ~ in path
    config_path = config_path.expanduser()

    if not config_path.exists():
        # Return defaults if no config file
        return SmartForkConfig()

    try:
        with open(config_path) as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}") from e

    # Pydantic handles merging with defaults and validation
    return SmartForkConfig.model_validate(config_data)


def get_default_config() -> SmartForkConfig:
    """Get configuration with all default values.

    Useful for generating a template config file or for testing.

    Returns:
        SmartForkConfig with all default values.
    """
    return SmartForkConfig()
