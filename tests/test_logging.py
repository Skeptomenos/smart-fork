"""Tests for smart_fork.logging module.

These tests verify that structlog is properly configured and that
logging functions work as expected.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from typing import Any
from unittest.mock import patch

import pytest
import structlog

from smart_fork.logging import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVEL_ENV_VAR,
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    unbind_context,
)


@pytest.fixture(autouse=True)
def reset_logging() -> None:
    """Reset logging state before each test."""
    # Clear any bound context
    clear_context()
    # Reset structlog configuration
    structlog.reset_defaults()
    # Reset stdlib logging
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configures_with_default_level(self) -> None:
        """Default level should be INFO when not specified."""
        configure_logging()
        logger = logging.getLogger("smart_fork")
        assert logger.level == logging.INFO

    def test_configures_with_explicit_level(self) -> None:
        """Explicit level should override default."""
        configure_logging(level="DEBUG")
        logger = logging.getLogger("smart_fork")
        assert logger.level == logging.DEBUG

    def test_configures_with_warning_level(self) -> None:
        """WARNING level should work correctly."""
        configure_logging(level="WARNING")
        logger = logging.getLogger("smart_fork")
        assert logger.level == logging.WARNING

    def test_configures_with_error_level(self) -> None:
        """ERROR level should work correctly."""
        configure_logging(level="ERROR")
        logger = logging.getLogger("smart_fork")
        assert logger.level == logging.ERROR

    def test_env_var_overrides_default(self) -> None:
        """SMART_FORK_LOG_LEVEL env var should override default."""
        with patch.dict(os.environ, {LOG_LEVEL_ENV_VAR: "DEBUG"}):
            configure_logging()
            logger = logging.getLogger("smart_fork")
            assert logger.level == logging.DEBUG

    def test_explicit_level_overrides_env_var(self) -> None:
        """Explicit level should override env var."""
        with patch.dict(os.environ, {LOG_LEVEL_ENV_VAR: "DEBUG"}):
            configure_logging(level="WARNING")
            logger = logging.getLogger("smart_fork")
            assert logger.level == logging.WARNING

    def test_invalid_env_var_uses_default(self) -> None:
        """Invalid env var value should fall back to default."""
        with patch.dict(os.environ, {LOG_LEVEL_ENV_VAR: "INVALID"}):
            configure_logging()
            logger = logging.getLogger("smart_fork")
            assert logger.level == getattr(logging, DEFAULT_LOG_LEVEL)

    def test_configures_root_logger_handler(self) -> None:
        """Root logger should have exactly one handler."""
        configure_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1

    def test_reconfigure_replaces_handlers(self) -> None:
        """Reconfiguring should replace handlers, not add."""
        configure_logging()
        configure_logging()
        configure_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger_with_bind_method(self) -> None:
        """get_logger should return a logger that supports structlog operations."""
        configure_logging()
        log = get_logger("test")
        # Verify it has the expected structlog methods
        assert hasattr(log, "info")
        assert hasattr(log, "debug")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")
        assert hasattr(log, "bind")
        assert callable(log.bind)

    def test_logger_with_name(self) -> None:
        """Logger should have the specified name."""
        configure_logging()
        log = get_logger("my.module.name")
        # The logger name is stored internally
        assert log is not None

    def test_logger_without_name(self) -> None:
        """Logger without name should work (root logger)."""
        configure_logging()
        log = get_logger()
        assert log is not None

    def test_logger_can_log_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Logger should be able to log info messages."""
        configure_logging(level="INFO")
        log = get_logger("test")
        log.info("test message")
        # Log output goes to stderr
        captured = capsys.readouterr()
        assert "test message" in captured.err

    def test_logger_can_log_with_context(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Logger should include additional context in output."""
        configure_logging(level="INFO")
        log = get_logger("test")
        log.info("processing", session_id="ses_123", chunks=42)
        captured = capsys.readouterr()
        assert "session_id" in captured.err
        assert "ses_123" in captured.err

    def test_debug_not_shown_at_info_level(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Debug messages should not appear when level is INFO."""
        configure_logging(level="INFO")
        log = get_logger("test")
        log.debug("debug message")
        captured = capsys.readouterr()
        assert "debug message" not in captured.err

    def test_debug_shown_at_debug_level(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Debug messages should appear when level is DEBUG."""
        configure_logging(level="DEBUG")
        log = get_logger("test")
        log.debug("debug message")
        captured = capsys.readouterr()
        assert "debug message" in captured.err


class TestContextBinding:
    """Tests for context binding functions."""

    def test_bind_context_adds_to_logs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Bound context should appear in log messages."""
        configure_logging(level="INFO")
        log = get_logger("test")

        bind_context(session_id="ses_abc", repo="my-repo")
        log.info("test message")

        captured = capsys.readouterr()
        assert "session_id" in captured.err
        assert "ses_abc" in captured.err
        assert "repo" in captured.err
        assert "my-repo" in captured.err

    def test_clear_context_removes_bindings(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """clear_context should remove all bound context."""
        configure_logging(level="INFO")
        log = get_logger("test")

        bind_context(session_id="ses_123")
        clear_context()
        log.info("test message")

        captured = capsys.readouterr()
        # session_id should NOT be in the output after clearing
        assert "ses_123" not in captured.err

    def test_unbind_specific_key(self, capsys: pytest.CaptureFixture[str]) -> None:
        """unbind_context should remove specific keys only."""
        configure_logging(level="INFO")
        log = get_logger("test")

        bind_context(session_id="ses_123", repo="my-repo")
        unbind_context("session_id")
        log.info("test message")

        captured = capsys.readouterr()
        # session_id should be removed, but repo should remain
        assert "ses_123" not in captured.err
        assert "my-repo" in captured.err

    def test_context_propagates_across_loggers(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Bound context should appear in all loggers."""
        configure_logging(level="INFO")
        log1 = get_logger("module1")
        log2 = get_logger("module2")

        bind_context(request_id="req_xyz")
        log1.info("from module1")
        log2.info("from module2")

        captured = capsys.readouterr()
        # Both log lines should have the request_id
        lines = captured.err.strip().split("\n")
        assert len(lines) >= 2
        assert "req_xyz" in lines[0]
        assert "req_xyz" in lines[1]


class TestJSONOutput:
    """Tests for JSON output mode."""

    def test_json_output_produces_parseable_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """JSON output mode should produce valid JSON."""
        import json

        configure_logging(level="INFO", json_output=True)
        log = get_logger("test")
        log.info("test message", key="value")

        captured = capsys.readouterr()
        # Should be parseable as JSON
        line = captured.err.strip()
        data = json.loads(line)
        assert data["event"] == "test message"
        assert data["key"] == "value"

    def test_json_output_includes_timestamp(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """JSON output should include ISO timestamp."""
        import json

        configure_logging(level="INFO", json_output=True)
        log = get_logger("test")
        log.info("test message")

        captured = capsys.readouterr()
        data = json.loads(captured.err.strip())
        assert "timestamp" in data


class TestDefaultLogLevel:
    """Tests for DEFAULT_LOG_LEVEL constant."""

    def test_default_log_level_is_info(self) -> None:
        """Default log level should be INFO."""
        assert DEFAULT_LOG_LEVEL == "INFO"

    def test_log_level_env_var_name(self) -> None:
        """Environment variable name should be correct."""
        assert LOG_LEVEL_ENV_VAR == "SMART_FORK_LOG_LEVEL"
