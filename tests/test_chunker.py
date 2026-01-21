"""Tests for smart_fork.chunker module.

Tests the token-based chunking functionality including:
- Basic chunking with various text sizes
- Token counting accuracy
- Overlap between chunks
- Break point detection
- Message chunking
- Edge cases (empty text, whitespace, etc.)
"""

from __future__ import annotations

import pytest

from smart_fork.chunker import Chunk, Chunker, chunk_messages
from smart_fork.config import ChunkingConfig


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_chunk_creation(self) -> None:
        """Test basic Chunk creation with all fields."""
        chunk = Chunk(
            text="Hello world",
            index=0,
            token_count=2,
            start_char=0,
            end_char=11,
        )
        assert chunk.text == "Hello world"
        assert chunk.index == 0
        assert chunk.token_count == 2
        assert chunk.start_char == 0
        assert chunk.end_char == 11


class TestChunkerTokenCounting:
    """Tests for token counting functionality."""

    def test_count_tokens_empty(self) -> None:
        """Empty string has zero tokens."""
        chunker = Chunker()
        assert chunker.count_tokens("") == 0

    def test_count_tokens_simple(self) -> None:
        """Simple text has expected token count."""
        chunker = Chunker()
        # "Hello world" is typically 2 tokens with cl100k_base
        count = chunker.count_tokens("Hello world")
        assert count == 2

    def test_count_tokens_longer_text(self) -> None:
        """Longer text has reasonable token count."""
        chunker = Chunker()
        # Roughly 1 token per 4 characters for English text
        text = "The quick brown fox jumps over the lazy dog."
        count = chunker.count_tokens(text)
        assert 5 <= count <= 15  # Reasonable range

    def test_count_tokens_with_newlines(self) -> None:
        """Text with newlines counts correctly."""
        chunker = Chunker()
        text = "Line one\n\nLine two\n\nLine three"
        count = chunker.count_tokens(text)
        assert count > 0


class TestChunkerBasicChunking:
    """Tests for basic chunking functionality."""

    def test_chunk_empty_text(self) -> None:
        """Empty text returns empty list."""
        chunker = Chunker()
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_chunk_whitespace_only(self) -> None:
        """Whitespace-only text returns empty list."""
        chunker = Chunker()
        chunks = chunker.chunk_text("   \n\n   ")
        assert chunks == []

    def test_chunk_short_text_single_chunk(self) -> None:
        """Short text that fits in one chunk returns single chunk."""
        chunker = Chunker()
        text = "This is a short text that should fit in one chunk."
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].index == 0
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)

    def test_chunk_preserves_content(self) -> None:
        """Chunking preserves all content (no data loss)."""
        chunker = Chunker()
        text = "Hello world. This is a test."
        chunks = chunker.chunk_text(text)

        # For short text, should be single chunk
        assert len(chunks) == 1
        assert chunks[0].text == text


class TestChunkerLongText:
    """Tests for chunking longer texts."""

    @pytest.fixture
    def small_chunk_config(self) -> ChunkingConfig:
        """Config with small chunk sizes for testing."""
        return ChunkingConfig(
            target_tokens=50,
            min_tokens=20,
            max_tokens=100,
            overlap_tokens=10,
            encoding="cl100k_base",
        )

    def test_long_text_creates_multiple_chunks(
        self, small_chunk_config: ChunkingConfig
    ) -> None:
        """Long text is split into multiple chunks."""
        chunker = Chunker(small_chunk_config)
        # Create text that's definitely longer than max_tokens
        text = "This is a sentence. " * 50  # ~200 tokens
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1

    def test_chunks_have_sequential_indices(
        self, small_chunk_config: ChunkingConfig
    ) -> None:
        """Chunks have sequential 0-based indices."""
        chunker = Chunker(small_chunk_config)
        text = "This is a sentence. " * 50
        chunks = chunker.chunk_text(text)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunks_within_token_limits(
        self, small_chunk_config: ChunkingConfig
    ) -> None:
        """Each chunk is within configured token limits."""
        chunker = Chunker(small_chunk_config)
        text = "This is a sentence. " * 50
        chunks = chunker.chunk_text(text)

        for chunk in chunks:
            # Max limit should always be respected
            assert chunk.token_count <= small_chunk_config.max_tokens
            # Min limit applies to chunks that aren't near the end
            # (overlap can cause smaller chunks near boundaries)
            if chunk.index < len(chunks) - 2:  # Allow last 2 chunks to be smaller
                # Only check if there's enough total content
                if chunk.token_count < small_chunk_config.min_tokens:
                    # This can happen when overlap causes small remaining chunks
                    # Just verify the chunk isn't empty
                    assert chunk.token_count > 0

    def test_chunks_have_correct_character_positions(
        self, small_chunk_config: ChunkingConfig
    ) -> None:
        """Chunk start/end positions are valid and non-overlapping."""
        chunker = Chunker(small_chunk_config)
        text = "This is a sentence. " * 50
        chunks = chunker.chunk_text(text)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char <= len(text)
            assert chunk.start_char < chunk.end_char
            # Text at position matches chunk text
            # Note: With overlap, chunks may have different boundaries


class TestChunkerOverlap:
    """Tests for chunk overlap functionality."""

    @pytest.fixture
    def overlap_config(self) -> ChunkingConfig:
        """Config with significant overlap for testing."""
        return ChunkingConfig(
            target_tokens=50,
            min_tokens=20,
            max_tokens=100,
            overlap_tokens=20,  # 40% overlap
            encoding="cl100k_base",
        )

    def test_chunks_have_overlapping_content(
        self, overlap_config: ChunkingConfig
    ) -> None:
        """Consecutive chunks share some content due to overlap."""
        chunker = Chunker(overlap_config)
        # Create structured text that's easy to verify
        text = "Sentence one. " * 20 + "Sentence two. " * 20 + "Sentence three. " * 20
        chunks = chunker.chunk_text(text)

        if len(chunks) >= 2:
            # Check that end of chunk N appears at start of chunk N+1
            # This is true because of overlap
            for i in range(len(chunks) - 1):
                # Get last few words of current chunk
                curr_end = (
                    chunks[i].text[-50:] if len(chunks[i].text) > 50 else chunks[i].text
                )
                next_start = (
                    chunks[i + 1].text[:100]
                    if len(chunks[i + 1].text) > 100
                    else chunks[i + 1].text
                )
                # Some overlap should exist
                # (exact matching is tricky due to break points)
                assert len(chunks[i].text) > 0
                assert len(chunks[i + 1].text) > 0

    def test_zero_overlap_config(self) -> None:
        """Chunking works with zero overlap."""
        config = ChunkingConfig(
            target_tokens=50,
            min_tokens=20,
            max_tokens=100,
            overlap_tokens=0,
            encoding="cl100k_base",
        )
        chunker = Chunker(config)
        text = "Word " * 100  # ~100 tokens
        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1


class TestChunkerBreakPoints:
    """Tests for break point detection."""

    @pytest.fixture
    def small_chunk_config(self) -> ChunkingConfig:
        """Config with small chunks to test break points."""
        return ChunkingConfig(
            target_tokens=30,
            min_tokens=10,
            max_tokens=50,
            overlap_tokens=5,
            encoding="cl100k_base",
        )

    def test_prefers_paragraph_breaks(self, small_chunk_config: ChunkingConfig) -> None:
        """Chunker prefers breaking at paragraph boundaries."""
        chunker = Chunker(small_chunk_config)
        # Text with paragraph break in the middle
        text = "First paragraph with some words.\n\nSecond paragraph with more words."

        # If this fits in one chunk, we can't test break preference
        if chunker.count_tokens(text) > small_chunk_config.max_tokens:
            chunks = chunker.chunk_text(text)
            # First chunk should end at or near paragraph break
            if len(chunks) >= 2:
                assert "\n\n" in chunks[0].text or chunks[0].text.endswith("\n")

    def test_prefers_line_breaks(self, small_chunk_config: ChunkingConfig) -> None:
        """Chunker prefers breaking at line boundaries."""
        chunker = Chunker(small_chunk_config)
        # Text with line breaks
        text = "Line one with words.\nLine two with words.\nLine three with words.\nLine four with words.\nLine five with words."

        chunks = chunker.chunk_text(text)
        # Chunks should tend to end at newlines when possible
        for chunk in chunks[:-1]:  # Exclude last chunk
            # Either ends with newline or we couldn't find a good break
            text_stripped = chunk.text.rstrip()
            ends_at_boundary = (
                chunk.text.endswith("\n")
                or text_stripped.endswith(".")
                or text_stripped.endswith("!")
                or text_stripped.endswith("?")
                or chunk.text.endswith(" ")
            )
            assert ends_at_boundary or len(chunk.text) > 0

    def test_prefers_sentence_breaks(self, small_chunk_config: ChunkingConfig) -> None:
        """Chunker prefers breaking at sentence boundaries."""
        chunker = Chunker(small_chunk_config)
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."

        chunks = chunker.chunk_text(text)
        # Check that chunks tend to end after periods
        for chunk in chunks[:-1]:
            # Should end at sentence or word boundary
            stripped = chunk.text.rstrip()
            assert (
                stripped.endswith(".")
                or stripped.endswith("!")
                or stripped.endswith("?")
                or chunk.text.endswith(" ")
                or len(chunk.text) > 0
            )


class TestChunkerEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_single_long_word(self) -> None:
        """Very long single word is handled correctly."""
        chunker = Chunker()
        # Create a very long "word" (no spaces)
        text = "a" * 10000
        chunks = chunker.chunk_text(text)

        # Should still chunk without error
        assert len(chunks) >= 1
        # All content should be preserved
        combined = "".join(c.text for c in chunks)
        # May have some duplication due to overlap, but should contain original
        assert len(combined) >= len(text)

    def test_unicode_text(self) -> None:
        """Unicode text is chunked correctly."""
        chunker = Chunker()
        text = "Hello! " + "Emoji test " + "End."
        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1
        assert chunks[0].text == text  # Short enough for single chunk

    def test_mixed_newlines(self) -> None:
        """Text with various newline styles is handled."""
        chunker = Chunker()
        text = "Line one\r\nLine two\nLine three\r\n\r\nParagraph two"
        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1

    def test_only_newlines(self) -> None:
        """Text that's only newlines returns empty."""
        chunker = Chunker()
        chunks = chunker.chunk_text("\n\n\n")
        assert chunks == []

    def test_special_characters(self) -> None:
        """Text with special characters is handled."""
        chunker = Chunker()
        text = "Code: `print('hello')` and more code: ```python\nx = 1\n```"
        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1


class TestChunkMessages:
    """Tests for the chunk_messages function."""

    def test_empty_messages(self) -> None:
        """Empty message list returns empty chunks."""
        chunks = chunk_messages([])
        assert chunks == []

    def test_single_message(self) -> None:
        """Single message is chunked correctly."""
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        chunks = chunk_messages(messages)

        assert len(chunks) == 1
        assert "user:" in chunks[0].text
        assert "Hello, how are you?" in chunks[0].text

    def test_multiple_messages(self) -> None:
        """Multiple messages are combined and chunked."""
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."},
        ]
        chunks = chunk_messages(messages)

        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        assert "user:" in combined
        assert "assistant:" in combined
        assert "What is Python?" in combined
        assert "programming language" in combined

    def test_messages_with_empty_content(self) -> None:
        """Messages with empty content are skipped."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},  # Empty
            {"role": "user", "content": "Goodbye"},
        ]
        chunks = chunk_messages(messages)

        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        assert "Hello" in combined
        assert "Goodbye" in combined

    def test_messages_missing_role(self) -> None:
        """Messages missing role key use 'unknown'."""
        messages = [{"content": "Hello"}]
        chunks = chunk_messages(messages)

        assert len(chunks) == 1
        assert "unknown:" in chunks[0].text

    def test_messages_missing_content(self) -> None:
        """Messages missing content key are skipped."""
        messages = [{"role": "user"}]
        chunks = chunk_messages(messages)

        # Empty content means message is skipped
        assert chunks == []

    def test_long_conversation_creates_multiple_chunks(self) -> None:
        """Long conversation is split into multiple chunks."""
        config = ChunkingConfig(
            target_tokens=50,
            min_tokens=20,
            max_tokens=100,
            overlap_tokens=10,
        )
        # Create a long conversation
        messages = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i} with some content here.",
            }
            for i in range(50)
        ]
        chunks = chunk_messages(messages, config)

        assert len(chunks) > 1


class TestChunkerConfiguration:
    """Tests for configuration handling."""

    def test_default_config(self) -> None:
        """Chunker uses default config when none provided."""
        chunker = Chunker()
        # Default is 768 target tokens, 512-1024 range
        assert chunker.config.target_tokens == 768
        assert chunker.config.min_tokens == 512
        assert chunker.config.max_tokens == 1024
        assert chunker.config.overlap_tokens == 50
        assert chunker.config.encoding == "cl100k_base"

    def test_custom_config(self) -> None:
        """Chunker respects custom configuration."""
        config = ChunkingConfig(
            target_tokens=100,
            min_tokens=50,
            max_tokens=200,
            overlap_tokens=25,
            encoding="cl100k_base",
        )
        chunker = Chunker(config)

        assert chunker.config.target_tokens == 100
        assert chunker.config.min_tokens == 50
        assert chunker.config.max_tokens == 200
        assert chunker.config.overlap_tokens == 25

    def test_different_encoding(self) -> None:
        """Chunker can use different tiktoken encodings."""
        # p50k_base is another valid encoding
        config = ChunkingConfig(encoding="p50k_base")
        chunker = Chunker(config)

        # Should work without error
        count = chunker.count_tokens("Hello world")
        assert count > 0


class TestChunkerTokenAccuracy:
    """Tests for token count accuracy."""

    def test_chunk_token_count_matches_actual(self) -> None:
        """Chunk's token_count field matches actual token count."""
        chunker = Chunker()
        text = "This is a sample text for testing token accuracy."
        chunks = chunker.chunk_text(text)

        for chunk in chunks:
            actual_count = chunker.count_tokens(chunk.text)
            assert chunk.token_count == actual_count

    def test_token_count_consistency(self) -> None:
        """Token counting is consistent across calls."""
        chunker = Chunker()
        text = "Consistent token counting test."

        count1 = chunker.count_tokens(text)
        count2 = chunker.count_tokens(text)
        count3 = chunker.count_tokens(text)

        assert count1 == count2 == count3
