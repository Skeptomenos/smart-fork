"""Token-based text chunking for Smart Fork.

Splits session transcripts into chunks suitable for embedding and vector search.
Uses tiktoken for accurate token counting with configurable chunk sizes.

Design principles:
- Token-based chunking ensures consistent embedding input sizes
- Overlap between chunks preserves context at boundaries
- Message-aware splitting respects conversation structure
- Configurable via ChunkingConfig for tuning
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from smart_fork.config import ChunkingConfig, get_default_config


@dataclass
class Chunk:
    """A single chunk of text with metadata.

    Attributes:
        text: The chunk content.
        index: Zero-based position within the original text.
        token_count: Number of tokens in this chunk.
        start_char: Character offset where this chunk starts in original text.
        end_char: Character offset where this chunk ends in original text.
    """

    text: str
    index: int
    token_count: int
    start_char: int
    end_char: int


class Chunker:
    """Token-based text chunker using tiktoken.

    Splits text into chunks within the configured token range, with overlap
    between consecutive chunks to preserve context at boundaries.

    The chunking algorithm:
    1. Encode full text to tokens
    2. Take target_tokens from current position
    3. Decode to text and find a good break point (newline, sentence, word)
    4. Move position back by overlap_tokens for next chunk
    5. Repeat until all text is chunked

    Why token-based chunking (vs character-based):
    - Embedding models have token limits, not character limits
    - Consistent semantic density per chunk
    - More predictable embedding quality
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize chunker with configuration.

        Args:
            config: Chunking configuration. Uses defaults if None.
        """
        if config is None:
            config = get_default_config().chunking
        self.config = config
        self._encoding = tiktoken.get_encoding(config.encoding)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured encoding.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        return len(self._encoding.encode(text))

    def chunk_text(self, text: str) -> list[Chunk]:
        """Split text into overlapping chunks within token limits.

        Args:
            text: Text to chunk. Can be any length.

        Returns:
            List of Chunk objects, ordered by position in original text.
            Empty list if text is empty.
        """
        if not text or not text.strip():
            return []

        # Encode full text to tokens
        tokens = self._encoding.encode(text)
        total_tokens = len(tokens)

        # If text fits in a single chunk, return as-is
        if total_tokens <= self.config.max_tokens:
            return [
                Chunk(
                    text=text,
                    index=0,
                    token_count=total_tokens,
                    start_char=0,
                    end_char=len(text),
                )
            ]

        chunks: list[Chunk] = []
        token_pos = 0
        chunk_index = 0

        while token_pos < total_tokens:
            # Determine how many tokens to take for this chunk
            remaining = total_tokens - token_pos
            chunk_token_count = min(self.config.target_tokens, remaining)

            # Extract tokens for this chunk
            chunk_tokens = tokens[token_pos : token_pos + chunk_token_count]
            chunk_text = self._encoding.decode(chunk_tokens)

            # Find a good break point if we're not at the end
            if token_pos + chunk_token_count < total_tokens:
                chunk_text, actual_token_count = self._find_break_point(
                    chunk_text, chunk_tokens
                )
            else:
                actual_token_count = len(chunk_tokens)

            # Calculate character positions in original text
            # We need to find where this chunk starts and ends in the original
            prefix_tokens = tokens[:token_pos]
            prefix_text = self._encoding.decode(prefix_tokens)
            start_char = len(prefix_text)
            end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=chunk_index,
                    token_count=actual_token_count,
                    start_char=start_char,
                    end_char=end_char,
                )
            )

            # Move to next position with overlap
            # Ensure we make forward progress even with overlap
            advance = max(1, actual_token_count - self.config.overlap_tokens)
            token_pos += advance
            chunk_index += 1

        return chunks

    def _find_break_point(self, text: str, tokens: list[int]) -> tuple[str, int]:
        """Find a good break point in text, preferring natural boundaries.

        Looks for break points in order of preference:
        1. Double newline (paragraph boundary)
        2. Single newline (line boundary)
        3. Sentence ending (. ! ?)
        4. Word boundary (space)
        5. Falls back to original text if no good break found

        Only considers break points that keep chunk above min_tokens.

        Args:
            text: The chunk text to find a break in.
            tokens: The tokens for this text (for accurate counting).

        Returns:
            Tuple of (text up to break point, token count of that text).
        """
        min_chars = len(self._encoding.decode(tokens[: self.config.min_tokens]))

        # Try break patterns in order of preference
        break_patterns = [
            "\n\n",  # Paragraph boundary
            "\n",  # Line boundary
        ]

        for pattern in break_patterns:
            # Find last occurrence after min_chars
            pos = text.rfind(pattern, min_chars)
            if pos > 0:
                break_text = text[: pos + len(pattern)]
                break_tokens = self._encoding.encode(break_text)
                if len(break_tokens) >= self.config.min_tokens:
                    return break_text, len(break_tokens)

        # Try sentence boundaries (. ! ?)
        for punct in [".", "!", "?"]:
            # Look for punctuation followed by space or newline
            search_pos = len(text) - 1
            while search_pos >= min_chars:
                pos = text.rfind(punct, min_chars, search_pos)
                if pos < 0:
                    break
                # Check if followed by space or newline (sentence end)
                if pos + 1 < len(text) and text[pos + 1] in " \n":
                    break_text = text[: pos + 1]
                    break_tokens = self._encoding.encode(break_text)
                    if len(break_tokens) >= self.config.min_tokens:
                        return break_text, len(break_tokens)
                search_pos = pos - 1

        # Try word boundary (last space)
        pos = text.rfind(" ", min_chars)
        if pos > 0:
            break_text = text[: pos + 1]
            break_tokens = self._encoding.encode(break_text)
            if len(break_tokens) >= self.config.min_tokens:
                return break_text, len(break_tokens)

        # No good break found, return original
        return text, len(tokens)


def chunk_messages(
    messages: list[dict[str, str]], config: ChunkingConfig | None = None
) -> list[Chunk]:
    """Chunk a list of conversation messages.

    Formats messages as "role: content" and chunks the combined text.
    This preserves the conversation structure while enabling efficient embedding.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        config: Chunking configuration. Uses defaults if None.

    Returns:
        List of Chunk objects for the combined message text.
    """
    if not messages:
        return []

    # Format messages as "role: content" with double newlines between
    formatted_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            formatted_parts.append(f"{role}: {content}")

    combined_text = "\n\n".join(formatted_parts)

    chunker = Chunker(config)
    return chunker.chunk_text(combined_text)
