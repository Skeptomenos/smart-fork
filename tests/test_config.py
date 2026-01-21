"""Tests for config module."""

import json
import tempfile
from pathlib import Path

import pytest

from smart_fork.config import (
    ChunkingConfig,
    EmbeddingConfig,
    PathsConfig,
    QueryConfig,
    ScoringWeights,
    SmartForkConfig,
    get_default_config,
    load_config,
)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_provider_is_vertex(self) -> None:
        """Default embedding provider should be vertex."""
        config = EmbeddingConfig()
        assert config.provider == "vertex"

    def test_default_dimensions(self) -> None:
        """Default dimensions should be 768."""
        config = EmbeddingConfig()
        assert config.dimensions == 768

    def test_default_batch_size(self) -> None:
        """Default batch size should be 100."""
        config = EmbeddingConfig()
        assert config.batch_size == 100

    def test_rate_limit_default(self) -> None:
        """Default rate limit should be 100ms."""
        config = EmbeddingConfig()
        assert config.rate_limit_ms == 100

    def test_ollama_defaults(self) -> None:
        """Ollama should have sensible defaults."""
        config = EmbeddingConfig()
        assert config.ollama_host == "http://localhost:11434"
        assert config.ollama_model == "nomic-embed-text"

    def test_vertex_defaults(self) -> None:
        """Vertex should have sensible defaults."""
        config = EmbeddingConfig()
        assert config.vertex_location == "us-central1"
        assert config.vertex_model == "text-embedding-004"


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_token_range(self) -> None:
        """Token range should be 512-1024 per spec."""
        config = ChunkingConfig()
        assert config.min_tokens == 512
        assert config.max_tokens == 1024

    def test_default_overlap(self) -> None:
        """Overlap should be 50 tokens per spec."""
        config = ChunkingConfig()
        assert config.overlap_tokens == 50

    def test_default_encoding(self) -> None:
        """Should use cl100k_base encoding."""
        config = ChunkingConfig()
        assert config.encoding == "cl100k_base"


class TestQueryConfig:
    """Tests for QueryConfig."""

    def test_default_top_chunks(self) -> None:
        """Should retrieve 20 chunks by default."""
        config = QueryConfig()
        assert config.top_chunks == 20

    def test_default_top_results(self) -> None:
        """Should return 5 results by default."""
        config = QueryConfig()
        assert config.top_results == 5

    def test_recency_half_life(self) -> None:
        """Recency half-life should be 30 days per spec."""
        config = QueryConfig()
        assert config.recency_half_life_days == 30


class TestScoringWeights:
    """Tests for ScoringWeights."""

    def test_weights_sum_to_one(self) -> None:
        """All weights should sum to 1.0."""
        weights = ScoringWeights()
        total = (
            weights.best_similarity
            + weights.avg_similarity
            + weights.chunk_ratio
            + weights.recency
            + weights.chain_quality
        )
        assert abs(total - 1.0) < 0.001

    def test_default_weights_match_spec(self) -> None:
        """Default weights should match spec values."""
        weights = ScoringWeights()
        assert weights.best_similarity == 0.40
        assert weights.avg_similarity == 0.20
        assert weights.chunk_ratio == 0.05
        assert weights.recency == 0.25
        assert weights.chain_quality == 0.10


class TestPathsConfig:
    """Tests for PathsConfig."""

    def test_default_data_dir(self) -> None:
        """Data dir should be in user's data directory."""
        config = PathsConfig()
        assert "opencode" in str(config.data_dir)
        assert "smart-fork" in str(config.data_dir)

    def test_default_sessions_dir(self) -> None:
        """Sessions dir should point to OpenCode sessions."""
        config = PathsConfig()
        assert "opencode" in str(config.sessions_dir)
        assert "sessions" in str(config.sessions_dir)

    def test_lance_path_property(self) -> None:
        """Lance path should be under data_dir."""
        config = PathsConfig()
        assert config.lance_path == config.data_dir / "lance"

    def test_sync_state_path_property(self) -> None:
        """Sync state path should be under data_dir."""
        config = PathsConfig()
        assert config.sync_state_path == config.data_dir / "sync-state.json"

    def test_config_path_property(self) -> None:
        """Config path should be under data_dir."""
        config = PathsConfig()
        assert config.config_path == config.data_dir / "config.json"


class TestSmartForkConfig:
    """Tests for SmartForkConfig root config."""

    def test_default_log_level(self) -> None:
        """Default log level should be INFO."""
        config = SmartForkConfig()
        assert config.log_level == "INFO"

    def test_all_sub_configs_have_defaults(self) -> None:
        """All sub-configs should be initialized with defaults."""
        config = SmartForkConfig()
        assert config.embedding is not None
        assert config.chunking is not None
        assert config.query is not None
        assert config.scoring is not None
        assert config.paths is not None


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_returns_defaults_when_file_missing(self) -> None:
        """Should return defaults when config file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent.json"
            config = load_config(nonexistent)
            assert config.embedding.provider == "vertex"
            assert config.log_level == "INFO"

    def test_load_from_json_file(self) -> None:
        """Should load config from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_data = {
                "log_level": "DEBUG",
                "embedding": {"provider": "ollama"},
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            config = load_config(config_path)
            assert config.log_level == "DEBUG"
            assert config.embedding.provider == "ollama"
            # Other settings should still have defaults
            assert config.query.top_results == 5

    def test_load_partial_config_merges_with_defaults(self) -> None:
        """Partial config should merge with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_data = {
                "chunking": {"max_tokens": 2048},
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            config = load_config(config_path)
            # Override should apply
            assert config.chunking.max_tokens == 2048
            # Defaults should still apply
            assert config.chunking.min_tokens == 512
            assert config.embedding.provider == "vertex"

    def test_load_invalid_json_raises_value_error(self) -> None:
        """Should raise ValueError on invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                f.write("{ invalid json }")

            with pytest.raises(ValueError, match="Invalid JSON"):
                load_config(config_path)

    def test_load_invalid_values_raises_validation_error(self) -> None:
        """Should raise error on invalid config values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_data = {
                "embedding": {"batch_size": 1000},  # exceeds max of 250
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            with pytest.raises(Exception):  # Pydantic ValidationError
                load_config(config_path)


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_config_with_defaults(self) -> None:
        """Should return config with all defaults."""
        config = get_default_config()
        assert isinstance(config, SmartForkConfig)
        assert config.embedding.provider == "vertex"
        assert config.chunking.min_tokens == 512


class TestConfigValidation:
    """Tests for config validation."""

    def test_invalid_provider_rejected(self) -> None:
        """Should reject invalid provider values."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            EmbeddingConfig(provider="invalid")  # type: ignore[arg-type]

    def test_negative_dimensions_rejected(self) -> None:
        """Should reject negative dimensions."""
        with pytest.raises(Exception):
            EmbeddingConfig(dimensions=-1)

    def test_batch_size_max_enforced(self) -> None:
        """Should enforce batch_size max of 250."""
        with pytest.raises(Exception):
            EmbeddingConfig(batch_size=300)

    def test_weight_range_enforced(self) -> None:
        """Should enforce weight range 0.0-1.0."""
        with pytest.raises(Exception):
            ScoringWeights(best_similarity=1.5)

    def test_invalid_log_level_rejected(self) -> None:
        """Should reject invalid log levels."""
        with pytest.raises(Exception):
            SmartForkConfig(log_level="TRACE")  # type: ignore[arg-type]
