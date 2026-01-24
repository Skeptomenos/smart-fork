"""Daemon process manager for Smart Fork."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

import structlog

from smart_fork.config import SmartForkConfig, load_config

logger = structlog.get_logger()


class DaemonManager:
    """Manages the lifecycle of the Smart Fork background daemon.

    Handles starting, stopping, and checking the status of the daemon process.
    Uses a PID file to track the running process.
    """

    def __init__(self, config: SmartForkConfig | None = None) -> None:
        if config is None:
            config = load_config()

        self.config = config

        # Daemon files are stored in the config directory
        # e.g., ~/.local/share/opencode/smart-fork/
        self.data_dir = config.paths.sync_state_path.parent
        self.pid_file = self.data_dir / "daemon.pid"
        self.log_file = self.data_dir / "daemon.log"

    def start(self) -> None:
        """Start the daemon in the background."""
        if self.is_running():
            pid = self.get_pid()
            logger.info("daemon_already_running", pid=pid)
            print(f"Daemon is already running (PID: {pid})")
            return

        logger.info("starting_daemon")

        # Ensure log file exists and is writable
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Open log file for stdout/stderr redirection
        with open(self.log_file, "a") as log_f:
            # Construct the command to run 'smart-fork watch'
            # We use sys.executable to ensure we use the same Python environment
            cmd = [sys.executable, "-m", "smart_fork.cli", "watch"]

            # Start the process detached
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                start_new_session=True,  # Detach from terminal
            )

            # Write PID file
            with open(self.pid_file, "w") as pid_f:
                pid_f.write(str(process.pid))

            logger.info("daemon_started", pid=process.pid, log_file=str(self.log_file))
            print(f"Started Smart Fork daemon (PID: {process.pid})")
            print(f"Logs: {self.log_file}")

    def stop(self) -> None:
        """Stop the running daemon."""
        pid = self.get_pid()
        if not pid:
            logger.info("daemon_not_running")
            print("Daemon is not running")
            return

        try:
            logger.info("stopping_daemon", pid=pid)
            os.kill(pid, signal.SIGTERM)

            # Wait for process to exit
            for _ in range(10):
                if not self.is_process_running(pid):
                    break
                time.sleep(0.5)
            else:
                # Force kill if still running
                logger.warning("daemon_force_kill", pid=pid)
                os.kill(pid, signal.SIGKILL)

            print(f"Stopped daemon (PID: {pid})")

        except ProcessLookupError:
            logger.warning("daemon_process_not_found", pid=pid)
            print(f"Process {pid} not found (stale PID file?)")
        except OSError as e:
            logger.error("daemon_stop_failed", error=str(e))
            print(f"Error stopping daemon: {e}")
        finally:
            # Clean up PID file
            if self.pid_file.exists():
                self.pid_file.unlink()

    def restart(self) -> None:
        """Restart the daemon."""
        self.stop()
        time.sleep(1)
        self.start()

    def status(self) -> None:
        """Print daemon status."""
        pid = self.get_pid()
        if pid and self.is_process_running(pid):
            print(f"Status: Running")
            print(f"PID:    {pid}")
            print(f"Log:    {self.log_file}")

            # Check last log line if possible
            if self.log_file.exists():
                try:
                    # simplistic tail
                    with open(self.log_file, "rb") as f:
                        f.seek(-1024, 2)  # Go to end
                        tail = f.read().decode("utf-8", errors="replace")
                        last_line = tail.strip().split("\n")[-1]
                        print(f"Latest: {last_line}")
                except (OSError, IndexError):
                    pass
        else:
            print("Status: Stopped")
            # Clean up stale PID file if it exists
            if pid:
                print("Note: Removing stale PID file")
                self.pid_file.unlink()

    def get_pid(self) -> int | None:
        """Read PID from file."""
        if not self.pid_file.exists():
            return None
        try:
            with open(self.pid_file) as f:
                return int(f.read().strip())
        except (ValueError, OSError):
            return None

    def is_running(self) -> bool:
        """Check if daemon is currently running."""
        pid = self.get_pid()
        if not pid:
            return False
        return self.is_process_running(pid)

    @staticmethod
    def is_process_running(pid: int) -> bool:
        """Check if a process with the given PID is running."""
        try:
            # signal 0 does not send a signal but checks if process exists
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def enable_autostart(self) -> None:
        """Enable auto-start on login (macOS/launchd only)."""
        if sys.platform != "darwin":
            print("Auto-start is only supported on macOS currently.")
            return

        plist_path = Path.home() / "Library/LaunchAgents/com.opencode.smartfork.plist"
        python_path = sys.executable
        # Assuming smart-fork is installed in the same venv as python
        # We invoke it via python -m smart_fork.cli to be safe

        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.opencode.smartfork</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>smart_fork.cli</string>
        <string>watch</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{self.log_file}</string>
    <key>StandardErrorPath</key>
    <string>{self.log_file}</string>
    <key>WorkingDirectory</key>
    <string>{os.getcwd()}</string>
</dict>
</plist>
"""
        with open(plist_path, "w") as f:
            f.write(plist_content)

        print(f"Created launchd agent: {plist_path}")
        print("Loading service...")

        try:
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                check=False,
                capture_output=True,
            )
            subprocess.run(["launchctl", "load", str(plist_path)], check=True)
            print("Service enabled and started.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to load service: {e}")

    def disable_autostart(self) -> None:
        """Disable auto-start."""
        if sys.platform != "darwin":
            return

        plist_path = Path.home() / "Library/LaunchAgents/com.opencode.smartfork.plist"

        if plist_path.exists():
            try:
                subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
                plist_path.unlink()
                print("Service disabled and removed.")
            except Exception as e:
                print(f"Error disabling service: {e}")
        else:
            print("Service is not enabled.")
