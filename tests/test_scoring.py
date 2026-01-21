"""Tests for the composite scoring algorithm.

Tests cover:
- compute_session_score() with all weight components
- compute_recency_score() exponential decay behavior
- Edge cases and boundary conditions
- Weight configuration effects
- _get_session_has_parent() file I/O handling
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from smart_fork.config import ScoringWeights
from smart_fork.query import (
    compute_session_score,
    compute_recency_score,
    _get_session_has_parent,
)


# --- Fixtures ---


@pytest.fixture
def default_weights() -> ScoringWeights:
    """Default scoring weights from spec."""
    return ScoringWeights(
        best_similarity=0.40,
        avg_similarity=0.20,
        chunk_ratio=0.05,
        recency=0.25,
        chain_quality=0.10,
    )


@pytest.fixture
def sessions_dir() -> Generator[Path, None, None]:
    """Create a temporary sessions directory for testing parent lookups."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# --- compute_session_score Tests ---


class TestComputeSessionScore:
    """Tests for the composite scoring algorithm."""

    def test_perfect_score_components(self, default_weights: ScoringWeights) -> None:
        """Maximum score when all components are at their best values."""
        # All components maxed out
        score = compute_session_score(
            best_similarity=1.0,
            avg_similarity=1.0,
            matching_chunks=10,
            total_chunks=10,  # 100% chunk ratio
            session_timestamp=1700000000,
            query_timestamp=1700000000,  # Same time = recency 1.0
            has_parent=True,  # Chain quality bonus
            weights=default_weights,
        )

        # Should be 1.0 (all weights sum to 1.0)
        assert score == pytest.approx(1.0, abs=0.001)

    def test_zero_score_components(self, default_weights: ScoringWeights) -> None:
        """Minimum score when all components are at their worst values."""
        # All components at minimum (except very old for recency)
        score = compute_session_score(
            best_similarity=0.0,
            avg_similarity=0.0,
            matching_chunks=0,
            total_chunks=10,
            session_timestamp=0,  # Very old
            query_timestamp=1700000000,  # ~54 years later
            has_parent=False,
            weights=default_weights,
        )

        # Should be very close to 0
        assert score < 0.001

    def test_similarity_weight_contribution(
        self, default_weights: ScoringWeights
    ) -> None:
        """Test that similarity scores contribute correctly to final score."""
        # Only similarity components, everything else zeroed
        score = compute_session_score(
            best_similarity=0.8,
            avg_similarity=0.6,
            matching_chunks=0,
            total_chunks=1,
            session_timestamp=0,
            query_timestamp=1700000000,  # Very old, ~0 recency
            has_parent=False,
            weights=default_weights,
        )

        # Expected: best_sim (0.8 * 0.40) + avg_sim (0.6 * 0.20) = 0.32 + 0.12 = 0.44
        # (recency is nearly 0, chain is 0, ratio is 0)
        expected_sim_contribution = 0.8 * 0.40 + 0.6 * 0.20
        assert score == pytest.approx(expected_sim_contribution, abs=0.01)

    def test_chunk_ratio_calculation(self, default_weights: ScoringWeights) -> None:
        """Test that chunk_ratio is calculated as matching/total."""
        # Focus on chunk ratio
        weights_ratio_only = ScoringWeights(
            best_similarity=0.0,
            avg_similarity=0.0,
            chunk_ratio=1.0,  # Only chunk ratio
            recency=0.0,
            chain_quality=0.0,
        )

        score = compute_session_score(
            best_similarity=0.0,
            avg_similarity=0.0,
            matching_chunks=3,
            total_chunks=10,  # 30% ratio
            session_timestamp=0,
            query_timestamp=0,
            has_parent=False,
            weights=weights_ratio_only,
        )

        assert score == pytest.approx(0.3, abs=0.001)

    def test_chunk_ratio_caps_at_one(self, default_weights: ScoringWeights) -> None:
        """Chunk ratio should be capped at 1.0 even if matching > total."""
        weights_ratio_only = ScoringWeights(
            best_similarity=0.0,
            avg_similarity=0.0,
            chunk_ratio=1.0,
            recency=0.0,
            chain_quality=0.0,
        )

        score = compute_session_score(
            best_similarity=0.0,
            avg_similarity=0.0,
            matching_chunks=15,  # More than total (edge case)
            total_chunks=10,
            session_timestamp=0,
            query_timestamp=0,
            has_parent=False,
            weights=weights_ratio_only,
        )

        assert score == pytest.approx(1.0, abs=0.001)

    def test_chunk_ratio_zero_total_chunks(
        self, default_weights: ScoringWeights
    ) -> None:
        """Chunk ratio should be 0 when total_chunks is 0 (avoid division by zero)."""
        weights_ratio_only = ScoringWeights(
            best_similarity=0.0,
            avg_similarity=0.0,
            chunk_ratio=1.0,
            recency=0.0,
            chain_quality=0.0,
        )

        score = compute_session_score(
            best_similarity=0.0,
            avg_similarity=0.0,
            matching_chunks=5,
            total_chunks=0,  # Edge case
            session_timestamp=0,
            query_timestamp=0,
            has_parent=False,
            weights=weights_ratio_only,
        )

        assert score == pytest.approx(0.0, abs=0.001)

    def test_recency_decay_half_life(self, default_weights: ScoringWeights) -> None:
        """Recency score should be 0.5 at exactly one half-life."""
        weights_recency_only = ScoringWeights(
            best_similarity=0.0,
            avg_similarity=0.0,
            chunk_ratio=0.0,
            recency=1.0,  # Only recency
            chain_quality=0.0,
        )

        half_life_days = 30
        half_life_seconds = half_life_days * 24 * 60 * 60

        score = compute_session_score(
            best_similarity=0.0,
            avg_similarity=0.0,
            matching_chunks=0,
            total_chunks=1,
            session_timestamp=0,
            query_timestamp=half_life_seconds,  # Exactly 30 days later
            has_parent=False,
            weights=weights_recency_only,
            half_life_days=half_life_days,
        )

        assert score == pytest.approx(0.5, abs=0.001)

    def test_recency_decay_two_half_lives(
        self, default_weights: ScoringWeights
    ) -> None:
        """Recency score should be 0.25 at two half-lives."""
        weights_recency_only = ScoringWeights(
            best_similarity=0.0,
            avg_similarity=0.0,
            chunk_ratio=0.0,
            recency=1.0,
            chain_quality=0.0,
        )

        half_life_days = 30
        two_half_lives_seconds = 2 * half_life_days * 24 * 60 * 60

        score = compute_session_score(
            best_similarity=0.0,
            avg_similarity=0.0,
            matching_chunks=0,
            total_chunks=1,
            session_timestamp=0,
            query_timestamp=two_half_lives_seconds,  # 60 days later
            has_parent=False,
            weights=weights_recency_only,
            half_life_days=half_life_days,
        )

        assert score == pytest.approx(0.25, abs=0.001)

    def test_recency_same_timestamp(self, default_weights: ScoringWeights) -> None:
        """Recency should be 1.0 when query and session have same timestamp."""
        weights_recency_only = ScoringWeights(
            best_similarity=0.0,
            avg_similarity=0.0,
            chunk_ratio=0.0,
            recency=1.0,
            chain_quality=0.0,
        )

        score = compute_session_score(
            best_similarity=0.0,
            avg_similarity=0.0,
            matching_chunks=0,
            total_chunks=1,
            session_timestamp=1700000000,
            query_timestamp=1700000000,  # Same time
            has_parent=False,
            weights=weights_recency_only,
        )

        assert score == pytest.approx(1.0, abs=0.001)

    def test_recency_future_session(self, default_weights: ScoringWeights) -> None:
        """Recency should be 1.0 even if session is in the future (edge case)."""
        weights_recency_only = ScoringWeights(
            best_similarity=0.0,
            avg_similarity=0.0,
            chunk_ratio=0.0,
            recency=1.0,
            chain_quality=0.0,
        )

        score = compute_session_score(
            best_similarity=0.0,
            avg_similarity=0.0,
            matching_chunks=0,
            total_chunks=1,
            session_timestamp=1700000000,
            query_timestamp=1699999000,  # Query is before session
            has_parent=False,
            weights=weights_recency_only,
        )

        # age_seconds = max(0, ...) ensures non-negative
        assert score == pytest.approx(1.0, abs=0.001)

    def test_chain_quality_with_parent(self, default_weights: ScoringWeights) -> None:
        """Chain quality should be 1.0 when session has parent."""
        weights_chain_only = ScoringWeights(
            best_similarity=0.0,
            avg_similarity=0.0,
            chunk_ratio=0.0,
            recency=0.0,
            chain_quality=1.0,  # Only chain quality
        )

        score = compute_session_score(
            best_similarity=0.0,
            avg_similarity=0.0,
            matching_chunks=0,
            total_chunks=1,
            session_timestamp=0,
            query_timestamp=0,
            has_parent=True,
            weights=weights_chain_only,
        )

        assert score == pytest.approx(1.0, abs=0.001)

    def test_chain_quality_without_parent(
        self, default_weights: ScoringWeights
    ) -> None:
        """Chain quality should be 0.0 when session has no parent."""
        weights_chain_only = ScoringWeights(
            best_similarity=0.0,
            avg_similarity=0.0,
            chunk_ratio=0.0,
            recency=0.0,
            chain_quality=1.0,
        )

        score = compute_session_score(
            best_similarity=0.0,
            avg_similarity=0.0,
            matching_chunks=0,
            total_chunks=1,
            session_timestamp=0,
            query_timestamp=0,
            has_parent=False,
            weights=weights_chain_only,
        )

        assert score == pytest.approx(0.0, abs=0.001)

    def test_realistic_scoring_scenario(self, default_weights: ScoringWeights) -> None:
        """Test a realistic scoring scenario with mixed values."""
        # Simulate a moderately good match from a week ago with a parent
        one_week_seconds = 7 * 24 * 60 * 60
        query_timestamp = 1700000000

        score = compute_session_score(
            best_similarity=0.85,  # Good match
            avg_similarity=0.72,  # Decent average
            matching_chunks=3,
            total_chunks=10,  # 30% ratio
            session_timestamp=query_timestamp - one_week_seconds,  # 1 week old
            query_timestamp=query_timestamp,
            has_parent=True,  # Has parent
            weights=default_weights,
        )

        # Expected breakdown:
        # best_sim: 0.85 * 0.40 = 0.340
        # avg_sim: 0.72 * 0.20 = 0.144
        # chunk_ratio: 0.30 * 0.05 = 0.015
        # recency: 0.5^(7/30) * 0.25 ≈ 0.85 * 0.25 = 0.213
        # chain: 1.0 * 0.10 = 0.100
        # Total ≈ 0.812

        assert 0.7 < score < 0.9  # Should be a good score

    def test_score_clamped_to_01_range(self, default_weights: ScoringWeights) -> None:
        """Score should be clamped to [0, 1] range."""
        # Create weights that exceed 1.0 sum (invalid but we should handle it)
        oversized_weights = ScoringWeights(
            best_similarity=0.50,
            avg_similarity=0.50,
            chunk_ratio=0.50,
            recency=0.50,
            chain_quality=0.50,  # Total = 2.5
        )

        score = compute_session_score(
            best_similarity=1.0,
            avg_similarity=1.0,
            matching_chunks=10,
            total_chunks=10,
            session_timestamp=1700000000,
            query_timestamp=1700000000,
            has_parent=True,
            weights=oversized_weights,
        )

        # Should be clamped to 1.0 max
        assert score == pytest.approx(1.0, abs=0.001)


# --- compute_recency_score Tests ---


class TestComputeRecencyScore:
    """Tests for the recency scoring utility function."""

    def test_recency_at_zero_age(self) -> None:
        """Recency should be 1.0 when age is 0."""
        score = compute_recency_score(
            session_timestamp=1700000000,
            query_timestamp=1700000000,
            half_life_days=30,
        )
        assert score == pytest.approx(1.0, abs=0.001)

    def test_recency_at_one_half_life(self) -> None:
        """Recency should be 0.5 at exactly one half-life."""
        half_life_seconds = 30 * 24 * 60 * 60
        score = compute_recency_score(
            session_timestamp=0,
            query_timestamp=half_life_seconds,
            half_life_days=30,
        )
        assert score == pytest.approx(0.5, abs=0.001)

    def test_recency_decay_is_exponential(self) -> None:
        """Verify exponential decay formula: 0.5^(age/half_life)."""
        half_life_days = 30
        one_day_seconds = 24 * 60 * 60

        for days in [1, 7, 15, 30, 60, 90]:
            score = compute_recency_score(
                session_timestamp=0,
                query_timestamp=days * one_day_seconds,
                half_life_days=half_life_days,
            )
            expected = math.pow(0.5, days / half_life_days)
            assert score == pytest.approx(expected, abs=0.001), f"Failed at {days} days"

    def test_recency_different_half_life(self) -> None:
        """Recency should respect different half_life_days values."""
        one_week_seconds = 7 * 24 * 60 * 60

        # 7-day half-life
        score_7 = compute_recency_score(
            session_timestamp=0,
            query_timestamp=one_week_seconds,
            half_life_days=7,
        )
        assert score_7 == pytest.approx(0.5, abs=0.001)

        # 14-day half-life (same age, higher score)
        score_14 = compute_recency_score(
            session_timestamp=0,
            query_timestamp=one_week_seconds,
            half_life_days=14,
        )
        # At 7 days with 14-day half-life: 0.5^(7/14) = 0.5^0.5 ≈ 0.707
        assert score_14 == pytest.approx(0.707, abs=0.01)

    def test_recency_negative_age_treated_as_zero(self) -> None:
        """Future sessions should have recency of 1.0."""
        score = compute_recency_score(
            session_timestamp=1700000000,
            query_timestamp=1699000000,  # Query before session
            half_life_days=30,
        )
        assert score == pytest.approx(1.0, abs=0.001)


# --- _get_session_has_parent Tests ---


class TestGetSessionHasParent:
    """Tests for session parent lookup from metadata files."""

    def test_session_with_parent_session_id(self, sessions_dir: Path) -> None:
        """Should return True when session.json has parent_session_id."""
        session_id = "ses_test123"
        session_path = sessions_dir / session_id
        session_path.mkdir()

        metadata = {"parent_session_id": "ses_parent456", "repo_path": "/test"}
        with open(session_path / "session.json", "w") as f:
            json.dump(metadata, f)

        cache: dict[str, bool] = {}
        result = _get_session_has_parent(session_id, sessions_dir, cache)

        assert result is True
        assert cache[session_id] is True

    def test_session_with_forked_from(self, sessions_dir: Path) -> None:
        """Should return True when session.json has forked_from field."""
        session_id = "ses_test123"
        session_path = sessions_dir / session_id
        session_path.mkdir()

        metadata = {"forked_from": "ses_parent456", "repo_path": "/test"}
        with open(session_path / "session.json", "w") as f:
            json.dump(metadata, f)

        cache: dict[str, bool] = {}
        result = _get_session_has_parent(session_id, sessions_dir, cache)

        assert result is True

    def test_session_without_parent(self, sessions_dir: Path) -> None:
        """Should return False when no parent fields in session.json."""
        session_id = "ses_test123"
        session_path = sessions_dir / session_id
        session_path.mkdir()

        metadata = {"repo_path": "/test", "model": "claude-3"}
        with open(session_path / "session.json", "w") as f:
            json.dump(metadata, f)

        cache: dict[str, bool] = {}
        result = _get_session_has_parent(session_id, sessions_dir, cache)

        assert result is False

    def test_session_with_null_parent(self, sessions_dir: Path) -> None:
        """Should return False when parent_session_id is null."""
        session_id = "ses_test123"
        session_path = sessions_dir / session_id
        session_path.mkdir()

        metadata = {"parent_session_id": None, "repo_path": "/test"}
        with open(session_path / "session.json", "w") as f:
            json.dump(metadata, f)

        cache: dict[str, bool] = {}
        result = _get_session_has_parent(session_id, sessions_dir, cache)

        assert result is False

    def test_session_with_empty_parent(self, sessions_dir: Path) -> None:
        """Should return False when parent_session_id is empty string."""
        session_id = "ses_test123"
        session_path = sessions_dir / session_id
        session_path.mkdir()

        metadata = {"parent_session_id": "", "repo_path": "/test"}
        with open(session_path / "session.json", "w") as f:
            json.dump(metadata, f)

        cache: dict[str, bool] = {}
        result = _get_session_has_parent(session_id, sessions_dir, cache)

        assert result is False

    def test_nonexistent_session_directory(self, sessions_dir: Path) -> None:
        """Should return False when session directory doesn't exist."""
        cache: dict[str, bool] = {}
        result = _get_session_has_parent("ses_nonexistent", sessions_dir, cache)

        assert result is False

    def test_missing_session_json(self, sessions_dir: Path) -> None:
        """Should return False when session.json is missing."""
        session_id = "ses_test123"
        session_path = sessions_dir / session_id
        session_path.mkdir()
        # No session.json created

        cache: dict[str, bool] = {}
        result = _get_session_has_parent(session_id, sessions_dir, cache)

        assert result is False

    def test_invalid_json(self, sessions_dir: Path) -> None:
        """Should return False when session.json contains invalid JSON."""
        session_id = "ses_test123"
        session_path = sessions_dir / session_id
        session_path.mkdir()

        with open(session_path / "session.json", "w") as f:
            f.write("not valid json {{{")

        cache: dict[str, bool] = {}
        result = _get_session_has_parent(session_id, sessions_dir, cache)

        assert result is False

    def test_cache_hit(self, sessions_dir: Path) -> None:
        """Should use cached value and not re-read file."""
        session_id = "ses_test123"
        # Pre-populate cache without creating any files
        cache: dict[str, bool] = {session_id: True}

        result = _get_session_has_parent(session_id, sessions_dir, cache)

        assert result is True  # From cache, even though no file exists

    def test_cache_is_populated(self, sessions_dir: Path) -> None:
        """Cache should be populated after lookup."""
        session_id = "ses_test123"
        session_path = sessions_dir / session_id
        session_path.mkdir()

        metadata = {"parent_session_id": "ses_parent", "repo_path": "/test"}
        with open(session_path / "session.json", "w") as f:
            json.dump(metadata, f)

        cache: dict[str, bool] = {}
        _get_session_has_parent(session_id, sessions_dir, cache)

        assert session_id in cache
        assert cache[session_id] is True
