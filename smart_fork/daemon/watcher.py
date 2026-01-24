"""File system watcher for Smart Fork daemon."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Timer
from typing import Callable

import structlog
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from smart_fork.ingest import sync_sessions

logger = structlog.get_logger()


class SessionEventHandler(FileSystemEventHandler):
    """Handles file system events for the session storage directory.

    Implements a debounce mechanism to aggregate multiple file events (common when
    sessions are active) into a single sync trigger.
    """

    def __init__(
        self,
        debounce_seconds: float = 5.0,
        sync_callback: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the event handler.

        Args:
            debounce_seconds: Time to wait after last event before syncing.
            sync_callback: Optional callback to run instead of default sync.
                           Useful for testing.
        """
        self.debounce_seconds = debounce_seconds
        self.sync_callback = sync_callback
        self._timer: Timer | None = None
        self._last_path: str | None = None

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Handle any file system event."""
        # We only care about file modifications/creations
        if event.event_type not in ("created", "modified", "moved"):
            return

        # We only care about JSON files
        if event.is_directory or not event.src_path.endswith(".json"):
            return

        self._last_path = event.src_path
        logger.debug("file_changed", path=event.src_path, type=event.event_type)
        self._schedule_sync()

    def _schedule_sync(self) -> None:
        """Schedule a sync operation with debounce."""
        if self._timer:
            self._timer.cancel()

        self._timer = Timer(self.debounce_seconds, self._trigger_sync)
        self._timer.start()

    def _trigger_sync(self) -> None:
        """Execute the sync operation."""
        logger.info("triggering_sync", reason="file_change_detected")

        try:
            if self.sync_callback:
                self.sync_callback()
            else:
                # Run incremental sync
                result = sync_sessions(force=False)
                logger.info(
                    "sync_complete",
                    sessions_synced=result.sessions_processed,
                    chunks_added=result.chunks_added,
                    errors=len(result.errors or []),
                )
        except Exception as e:
            logger.error("sync_failed", error=str(e))
        finally:
            self._timer = None


class SessionWatcher:
    """Manages the file system observer."""

    def __init__(self, watch_path: Path) -> None:
        self.watch_path = watch_path
        self.observer = Observer()
        self.handler = SessionEventHandler()

    def start(self) -> None:
        """Start watching for changes."""
        logger.info("watcher_starting", path=str(self.watch_path))
        self.observer.schedule(self.handler, str(self.watch_path), recursive=True)
        self.observer.start()

    def stop(self) -> None:
        """Stop watching."""
        logger.info("watcher_stopping")
        self.observer.stop()
        self.observer.join()

    def run_forever(self) -> None:
        """Run the watcher until interrupted."""
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
