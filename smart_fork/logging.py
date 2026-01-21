"""Structured logging configuration for Smart Fork.

This module configures structlog for consistent, structured logging across
the entire smart-fork application. It supports:

- Console output with human-readable formatting for development
- JSON output for production/log aggregation (when configured)
- Log level configuration via environment variable or config
- Context binding for request-scoped data (session_id, repo_path, etc.)

Usage:
    from smart_fork.logging import get_logger, configure_logging

    # Configure once at startup (usually in cli.py or __init__.py)
    configure_logging(level="DEBUG")

    # Get a logger for your module
    log = get_logger(__name__)
    log.info("Processing session", session_id="ses_abc123", chunks=42)

Why structlog?
    - Automatic context propagation (session IDs follow through call chains)
    - Structured output enables log parsing and aggregation
    - Pretty console output during development
    - Type-safe and testable
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Literal

import structlog
from structlog.typing import Processor


# Type alias for log levels
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Default log level if not configured
DEFAULT_LOG_LEVEL: LogLevel = "INFO"

# Environment variable for log level override
LOG_LEVEL_ENV_VAR = "SMART_FORK_LOG_LEVEL"


def _get_log_level_from_env() -> LogLevel:
    """Get log level from environment variable if set."""
    level = os.environ.get(LOG_LEVEL_ENV_VAR, "").upper()
    valid_levels: tuple[LogLevel, ...] = (
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    )
    if level in valid_levels:
        return level  # type: ignore[return-value]
    return DEFAULT_LOG_LEVEL


def _get_console_renderer(json_output: bool = False) -> Processor:
    """Get appropriate renderer for console output.

    Args:
        json_output: If True, use JSON renderer for machine parsing.
                     If False, use human-readable console renderer.

    Returns:
        A structlog processor for rendering log output.
    """
    if json_output:
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer(
        colors=True,
        exception_formatter=structlog.dev.plain_traceback,
    )


def configure_logging(
    level: LogLevel | None = None,
    json_output: bool = False,
) -> None:
    """Configure structlog for the application.

    This should be called once at application startup, typically in the CLI
    entry point. Subsequent calls will reconfigure logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to INFO, or SMART_FORK_LOG_LEVEL env var if set.
        json_output: If True, output logs as JSON for machine parsing.
                     If False, use human-readable console output.

    Example:
        # In cli.py main():
        configure_logging(level="DEBUG")

        # Or let it use defaults/env var:
        configure_logging()
    """
    # Resolve log level: explicit > env var > default
    resolved_level = level or _get_log_level_from_env()
    numeric_level = getattr(logging, resolved_level)

    # Shared processors for both stdlib and structlog
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure the stdlib logging to use structlog for formatting
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                _get_console_renderer(json_output),
            ],
        )
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Also configure the smart_fork logger specifically
    smart_fork_logger = logging.getLogger("smart_fork")
    smart_fork_logger.setLevel(numeric_level)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__). If None, returns root logger.

    Returns:
        A bound logger that supports structured logging.

    Example:
        log = get_logger(__name__)
        log.info("Processing started", session_id="ses_123")
        log.debug("Chunk processed", chunk_index=0, tokens=512)
        log.error("Failed to embed", error=str(e), chunks_remaining=10)
    """
    # structlog.get_logger() returns Any, but we've configured it to use
    # stdlib.BoundLogger via structlog.configure(). Cast is safe.
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_context(**kwargs: Any) -> None:
    """Bind context variables that will be included in all subsequent logs.

    This is useful for adding request-scoped data like session IDs or
    repository paths that should appear in all log messages.

    Args:
        **kwargs: Key-value pairs to bind to the logging context.

    Example:
        # At the start of processing a session:
        bind_context(session_id="ses_abc123", repo="my-project")

        # All subsequent logs will include these fields
        log.info("Processing chunks")  # Includes session_id and repo

        # Clear when done:
        clear_context()
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables.

    Call this at the end of a request/operation to prevent context
    from leaking into unrelated log messages.
    """
    structlog.contextvars.clear_contextvars()


def unbind_context(*keys: str) -> None:
    """Remove specific keys from the logging context.

    Args:
        *keys: Names of context variables to remove.

    Example:
        unbind_context("session_id")  # Remove just session_id
    """
    structlog.contextvars.unbind_contextvars(*keys)
