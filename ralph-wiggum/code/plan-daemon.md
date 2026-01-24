# Smart Fork Daemon Implementation Plan

**Goal**: Automatically index new OpenCode sessions in the background without manual intervention.
**Approach**: Phased implementation starting with a simple foreground watcher, then adding background management, and finally system persistence.

## Phase 1: The Watcher (Foreground) - COMPLETE
*Goal: Validate file monitoring and sync triggering logic without process management complexity.*

- [x] **Dependencies**: Add `watchdog` to `pyproject.toml`.
- [x] **Event Handler**: Implement `SessionEventHandler` in `smart_fork/daemon/watcher.py`.
    -   Monitor `~/.local/share/opencode/storage/`.
    -   **Debounce Logic**: Wait for 5 seconds of silence after a file system event before triggering sync.
    -   **Action**: Call `sync_sessions(force=False)` (incremental sync) when stabilized.
- [x] **CLI Command**: Add `smart-fork watch` to `cli.py`.
- [x] **Verification**: Verified `smart-fork watch` triggers sync on file changes.

## Phase 2: Background Service - COMPLETE
*Goal: Run the watcher as a detached background process.*

- [x] **Process Manager**: Create `smart_fork/daemon/manager.py`.
    -   **PID File**: Store process ID in `~/.local/share/opencode/smart-fork/daemon.pid`.
    -   **Log File**: Redirect stdout/stderr to `~/.local/share/opencode/smart-fork/daemon.log`.
- [x] **CLI Commands**: Add `smart-fork daemon` group.
    -   `start`: Check PID, if not running, spawn `smart-fork watch` in background, write PID.
    -   `stop`: Read PID, send SIGTERM, remove PID file.
    -   `status`: Check if PID exists and process is running.
    -   `logs`: Tail the log file.
- [x] **Verification**: Verified `start`, `status`, and `stop` lifecycle.

## Phase 3: System Persistence - COMPLETE
*Goal: Start automatically on login.*

- [x] **Launchd Agent**: Generate `~/Library/LaunchAgents/com.opencode.smartfork.plist`.
- [x] **CLI Commands**:
    -   `enable`: Write plist, load with `launchctl`.
    -   `disable`: Unload with `launchctl`, remove plist.

## Technical Details

### 1. Watch Strategy
Watching the `storage` directory recursively.
Filtering for `*.json` events in `storage/session`, `storage/message`, `storage/part`.
**Debounce**: 5.0 seconds.

### 2. Concurrency Safety
- [x] Added `file locking` to `sync_sessions` using `fcntl` to prevent race conditions between daemon and manual CLI.

## Next Steps
- [ ] User can enable auto-start if desired: `smart-fork daemon enable`
