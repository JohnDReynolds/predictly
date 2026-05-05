"""
Module: modal_predictly.py

Predictly worker functions for Modal.

This module owns:
- Durable per-user storage in a Modal Volume
- Upload handling (train.csv / test.csv)

Design goals
------------
- Keep the web layer (modal_server.py) lightweight: it only forwards requests.
- Keep durable state in the worker Volume under each user's directory.
- Ensure polling sees fresh state by using:
    - users_vol.commit() after writes (publish changes)
    - users_vol.reload() before reads (refresh snapshot)
  Without reload() in read functions, warm containers can serve stale results.

Notes on async policy
---------------------
- "Always rerun training when requested" UNLESS a job is already in progress
  (QUEUED or RUNNING) for the same user_id.
- No queue of "next runs". If a user requests a new run while one is running,
  they must call again after completion.
"""

# Errors to ignore
# pylint: disable=broad-exception-caught, disable=import-outside-toplevel
# pylint: disable=superfluous-parens, disable=ungrouped-imports
# pylint: disable=unused-import, disable=wrong-import-order
# pyright: reportUnknownMemberType=false
# pyright: reportUnusedImport=false

from __future__ import annotations

# MUST be first real import: sets env vars / limits before numpy, autogluon, flaml, etc.
import tabular.env_setup

# Python imports
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterable, Tuple

# Third-party imports
import modal

# Project Imports
import tabular.utilities as util
from tabular.utilities import AppError, MetricType, Option, TaskType

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
_APP_NAME = "predictly-worker"
_NO_USER_ID = {"status": "error", "error_type": "missing_user_id", "message": "Missing user_id."}
_PROJECT_ROOT = Path(__file__).parent

_STATE_UNKNOWN = "UNKNOWN"
_STATE_QUEUED = "QUEUED"
_STATE_RUNNING = "RUNNING"
_STATE_SUCCEEDED = "SUCCEEDED"
_STATE_FAILED = "FAILED"

_TRAIN_JOB_STATE_FILENAME = "train_job_state.json"
_TRAIN_JOB_RESULT_FILENAME = "train_job_result.json"


# -----------------------------------------------------------------------------
# Modal setup
# -----------------------------------------------------------------------------
app = modal.App(_APP_NAME)
control_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements_control.txt")
    .add_local_dir(_PROJECT_ROOT / "tabular", remote_path="/root/tabular")
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(_PROJECT_ROOT / "tabular", remote_path="/root/tabular")
)
users_vol = modal.Volume.from_name("predictly-users", create_if_missing=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _delete_old_subdirs(do_delete: bool = False) -> None:
    """
    Delete immediate sub-directories under `base_dir` whose mtime is older than `max_age_seconds`.
    """
    base_dir = Path(util.USERS_DIR)

    if not _modal_dir_exists(base_dir):
        util.print_modal(f"Cleanup base directory does not exist: {base_dir}")
        return

    if not base_dir.is_dir():
        util.print_modal(f"Cleanup base path is not a directory: {base_dir}")
        return

    now = time.time()
    max_age_seconds = util.MODAL_HOURS_TO_KEEP_USER_DIRECTORIES * 60 * 60
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue

        try:
            stat = entry.stat()
        except OSError as e:
            util.print_modal(f"Cleanup skipping {entry} (stat failed): {e}")
            continue

        age_seconds = now - stat.st_mtime

        if age_seconds > max_age_seconds:
            try:
                if do_delete:
                    shutil.rmtree(entry)
                    msg = "deleted"
                else:
                    msg = "will delete"
                util.print_modal(f"Cleanup {msg} {entry} (age: {age_seconds / 3600:.1f}h)")
            except OSError as e:
                util.print_modal(f"Cleanup failed to delete {entry}: {e}")


def _epoch_now() -> int:
    """Return current epoch seconds as int."""
    return int(time.time())


def _get_predict():
    import tabular.predict as predict

    return predict


def _get_user_options(user_id: str) -> dict[str, Any] | None:
    """Load options.json for user_id, if present and valid."""
    path = _options_path(user_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else None  # pyright: ignore
    except Exception:
        return None


def _http_status_for_app_error(error_type: str | None) -> int:
    """Map known AppError types to HTTP status codes."""
    mapping: dict[str, int] = {
        # State conflicts / ordering issues:
        "no_user_directory": 409,
        # Input / validation issues:
        "invalid_csv_file": 422,
        "non_unique_id": 422,
        "train_test_mismatch": 422,
        "unknown_metric": 422,
        "unknown_option": 422,
        "unknown_processors": 422,
        "unknown_task": 422,
    }
    if not error_type:
        return 400
    return mapping.get(error_type, 400)


def _log_error(user_id: str, message: str, error_type: str) -> dict[str, str]:
    """Log a structured error line (Cloud Logging friendly)."""
    error = {"status": "error", "user_id": user_id, "error_type": error_type, "message": message}
    util.print_modal(json.dumps(error))
    return error


def _modal_dir_exists(path: str | Path, *, retries: int = 30, delay: float = 2.0) -> bool:
    """Return True if a directory exists on a Modal Volume, with small retries.

    Modal volumes can lag slightly in warm containers or across invocations,
    so a single is_dir() check can occasionally report False even though the
    directory is actually there.
    """
    directory = Path(path)
    for _ in range(retries):
        if directory.is_dir():
            return True
        time.sleep(delay)

    # Informational only.  The caller will decide if this is fatal or not.
    util.print_modal(f"################################# _modal_dir_exists() failed for {path}")
    return False


def _options_path(user_id: str) -> str:
    """Return per-user options.json path."""
    return os.path.join(_user_directory(user_id), util.OPTIONS_FILE_NAME)


def _read_json(path: str) -> dict[str, Any] | None:
    """Read JSON dict from path, returning None if missing/invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else None  # pyright: ignore
    except Exception:
        return None


def _set_user_options(user_id: str, options: dict[str, Any]) -> None:
    """Overwrite options.json for user_id."""
    path = _options_path(user_id)

    # util.print_modal("_set_user_options()")
    # util.print_modal(options["uid_column_name"])

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(options, handle, ensure_ascii=False, indent=2)
    users_vol.commit()


def _status_payload(
    state: str, *, message: str = "", updated_at_epoch: int | None = None
) -> dict[str, Any]:
    """Build a stable state payload for _train_job_state.json."""
    return {
        "state": state,
        "updated_at_epoch": _epoch_now() if updated_at_epoch is None else int(updated_at_epoch),
        "message": message,
    }


def _train_and_predict_base(user_id: str) -> Dict[str, str]:
    """Background async training + prediction.

    This function is spawned (fire-and-forget) by the web app after a successful
    claim_train_slot_remote() call.

    Key correctness rule (race guard):
      - We read the current run_id from _train_job_state.json at the start.
      - Every write we perform (RUNNING, FAILED, SUCCEEDED, result) is tagged
        with that same run_id.
      - Before writing, we re-read state to ensure the run_id still matches.
        If it doesn't, we stop writing to avoid overwriting a newer run.

    Returns:
        The same dict payload as predict.train_and_predict(user_id).
        (Return value is mainly for logs; the UI reads result from the Volume.)
    """
    if not user_id:
        return _NO_USER_ID

    state_path = _train_job_state_path(user_id)
    result_path = _train_job_result_path(user_id)

    # Always unlink state and result json
    Path(state_path).unlink(missing_ok=True)
    Path(result_path).unlink(missing_ok=True)
    users_vol.commit()  # commit to disk

    def _load_current_run_id() -> int:
        """Read run_id from state; if missing, create one and persist it."""
        state0 = _read_json(state_path) or {}
        run_id0 = state0.get("run_id")
        if isinstance(run_id0, int):
            return int(run_id0)

        # Defensive fallback: if someone spawned without a run_id in state, create one.
        new_run_id = _epoch_now()
        patched = _status_payload(
            str(state0.get("state", _STATE_RUNNING)), message=str(state0.get("message", "") or "")
        )
        patched["run_id"] = new_run_id
        _write_json(state_path, patched)
        return new_run_id

    def _still_our_run(run_id: int) -> bool:
        """Return True if the state file still refers to the same run_id."""
        cur = _read_json(state_path) or {}
        return isinstance(cur.get("run_id"), int) and int(cur["run_id"]) == int(run_id)

    def _write_state_if_current(run_id: int, state: str, message: str) -> bool:
        """Write state only if run_id still matches; return True if written."""
        if not _still_our_run(run_id):
            return False
        payload = _status_payload(state, message=message)
        payload["run_id"] = int(run_id)
        _write_json(state_path, payload)
        return True

    try:
        # Refresh snapshot to reduce stale reads in warm containers.
        users_vol.reload()

        run_id = _load_current_run_id()

        ok, msg = _training_files_exist(user_id)
        if not ok:
            _write_state_if_current(run_id, _STATE_FAILED, msg)
            return {"status": "error", "error_type": "missing_files", "message": msg}

        # RUNNING (only if still current)
        if not _write_state_if_current(run_id, _STATE_RUNNING, ""):
            return {
                "status": "error",
                "error_type": "run_superseded",
                "message": "This run was superseded before it started.",
            }

        # Run training
        util.log_modal_memory("In _train_and_predict_base() before importing predict")
        predict = _get_predict()
        util.log_modal_memory("In _train_and_predict_base() after importing predict")
        result: Dict[str, Any] = predict.train_and_predict(user_id)

        # If pipeline returns {"status":"error",...}, treat as FAILED.
        status = str(result.get("status", "")).lower()
        if status == "error":
            msg2 = str(result.get("message", "Predicting failed."))
            _write_state_if_current(run_id, _STATE_FAILED, msg2)

            # Tag the returned result for logs/debugging (doesn't affect UI envelope).
            result["run_id"] = int(run_id)
            return result

        # Before writing result, ensure we are still the current run.
        if not _still_our_run(run_id):
            return {
                "status": "error",
                "error_type": "run_superseded",
                "message": "This run was superseded before writing results.",
            }

        # Write result WITH run_id (do not delete old results anymore).
        result_with_run: Dict[str, Any] = dict(result)
        result_with_run["run_id"] = int(run_id)
        _write_json(result_path, result_with_run)

        # SUCCEEDED
        _write_state_if_current(run_id, _STATE_SUCCEEDED, "")

        # Delete old sub-directories for all users.
        _delete_old_subdirs(do_delete=True)

        util.log_modal_memory("Finish _train_and_predict_base()")
        return result_with_run

    except Exception as exc:
        # Best-effort state update (only if current).
        try:
            users_vol.reload()
            run_id = _load_current_run_id()
            _write_state_if_current(run_id, _STATE_FAILED, str(exc))
        except Exception:
            pass

        # Force retraining
        util.remove_training_results(_user_directory(user_id))
        users_vol.commit()  # commit to disk

        error_type = exc.error_type if isinstance(exc, AppError) else "train_background_failed"

        return _log_error(user_id, str(exc), error_type=error_type)


def _train_job_result_path(user_id: str) -> str:
    """Return per-user path for async training result JSON."""
    return os.path.join(_user_directory(user_id), _TRAIN_JOB_RESULT_FILENAME)


def _train_job_state_path(user_id: str) -> str:
    """Return per-user path for async training state JSON."""
    return os.path.join(_user_directory(user_id), _TRAIN_JOB_STATE_FILENAME)


def _training_files_exist(user_id: str) -> tuple[bool, str]:
    """Check required train/test files exist for this user."""
    directory = _user_directory(user_id)
    train_path = os.path.join(directory, util.TRAIN_FILE_NAME)
    test_path = os.path.join(directory, util.TEST_FILE_NAME)
    if not os.path.exists(train_path):
        return False, "Missing train.csv (upload training file first)."
    if not os.path.exists(test_path):
        return False, "Missing test.csv (upload prediction file first)."
    return True, ""


def _unlink(directory: str, file_name: str | Iterable[str]) -> None:
    base = Path(directory)
    names = (file_name,) if isinstance(file_name, str) else file_name
    for name in names:
        (base / name).unlink(missing_ok=True)
    users_vol.commit()  # commit to disk


def _user_directory(user_id: str, *, ok_to_mkdir: bool = False) -> str:
    """Return the per-user directory path inside the container.

    Args:
        user_id: Predictly job/user id.
        ok_to_mkdir: If True, create directory if missing.

    Returns:
        Absolute directory path inside the Volume mount.

    Raises:
        ValueError: If directory does not exist after retries.
    """
    directory = Path(util.USERS_DIR) / user_id

    if ok_to_mkdir:
        directory.mkdir(parents=True, exist_ok=True)

    if not _modal_dir_exists(directory):
        raise ValueError(f"{util.SESSION_EXPIRED_MESSAGE} (user_id = '{user_id}')")

    return str(directory)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON payload to path (overwrite)."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    users_vol.commit()


# -----------------------------------------------------------------------------
# Endpoints function
# -----------------------------------------------------------------------------
@app.function(
    cpu=0.20,  # Uses 0.01, Startup 0.2 to 0.5, only called once
    memory=256,  # Uses 145
    image=control_image,
    volumes={util.USERS_DIR: users_vol},
)
def claim_train_slot_remote(
    user_id: str,
    task: TaskType,
    metric: MetricType,
    speed: int,
    uid_column_name: str,
) -> Dict[str, Any]:
    """Claim the right to start a new async training run for this user_id.

    Policy:
      - ALWAYS rerun training when requested, unless a job is already QUEUED/RUNNING.
      - Duplicate-click protection: only one request can claim; others return current state.
      - On claim: write QUEUED durably and stamp a new run_id.
    """
    if not user_id:
        return _NO_USER_ID

    # Refresh snapshot so we don't decide based on stale state.
    users_vol.reload()

    ok, msg = _training_files_exist(user_id)
    if not ok:
        return {
            "status": "error",
            "error_type": "missing_files",
            "message": msg,
            "state": _STATE_UNKNOWN,
        }

    state_path = _train_job_state_path(user_id)

    # Always unlink state json in case there is a stale one left from a failed run
    Path(state_path).unlink(missing_ok=True)
    users_vol.commit()  # commit to disk

    existing = _read_json(state_path) or {}
    existing_state = str(existing.get("state", _STATE_UNKNOWN))

    # Single-flight: if already queued/running, do not claim.
    if existing_state in (_STATE_QUEUED, _STATE_RUNNING):
        out: Dict[str, Any] = {
            "status": "ok",
            "claimed": False,
            "state": existing_state,
            "updated_at_epoch": int(existing.get("updated_at_epoch", 0) or 0),
            "message": str(existing.get("message", "") or ""),
        }
        # Preserve run_id if present (helps the UI later).
        if isinstance(existing.get("run_id"), int):
            out["run_id"] = int(existing["run_id"])
        return out

    # Not running -> start a NEW run.

    # Persist chosen params (same behavior as sync path).
    options: dict[str, Any] = _get_user_options(user_id) or {}
    options[Option.TASK] = task
    options[Option.METRIC] = metric
    options[Option.SPEED] = speed
    options[Option.UID_COLUMN_NAME] = uid_column_name if uid_column_name else util.STRING_MISSING
    _set_user_options(user_id, options)

    # New run identifier + fresh timestamp.
    now_epoch = _epoch_now()
    run_id = now_epoch

    # Durable QUEUED + run_id + updated_at_epoch
    queued = _status_payload(_STATE_QUEUED, message="")
    queued["run_id"] = run_id
    queued["updated_at_epoch"] = now_epoch

    _write_json(state_path, queued)

    util.log_modal_memory("Finish claim_train_slot_remote() reserved memory=256")
    return {"status": "ok", "claimed": True, **queued}


@app.function(
    cpu=0.20,  # Uses 0.02, Startup 0.1 to 0.4, only called once
    memory=256,  # Uses 147
    image=control_image,
    volumes={util.USERS_DIR: users_vol},
)
def get_train_job_result_remote(user_id: str) -> Dict[str, Any]:
    """Get async training result for user_id if ready.

    IMPORTANT:
      - This function reads from the Volume, so it calls users_vol.reload() first.

    Returns:
        If result exists:
            The full training result dict (same payload as Phase 1 /ui/train).

        If not ready:
            {
              "status": "error",
              "error_type": "not_ready",
              "message": "Result is not ready.",
              "state": "QUEUED"|"RUNNING"|"FAILED"|"UNKNOWN"
            }
    """
    if not user_id:
        return _NO_USER_ID

    try:
        users_vol.reload()

        # Do not call _user_directory().  Need to return "not_ready" instead of throwing.  This
        # keeps polling safe and idempotent.
        directory = Path(util.USERS_DIR) / user_id
        if not _modal_dir_exists(directory):
            return {
                "status": "error",
                "error_type": "not_ready",
                "message": "Result is not ready.",
                "state": _STATE_UNKNOWN,
            }

        result = _read_json(_train_job_result_path(user_id))
        if result:
            util.log_modal_memory("Finish get_train_job_result_remote() reserved memory=256")
            return result

        state = _read_json(_train_job_state_path(user_id)) or {}
        cur_state = str(state.get("state", _STATE_UNKNOWN))

        return {
            "status": "error",
            "error_type": "not_ready",
            "message": "Result is not ready.",
            "state": cur_state,
        }
    except Exception as exc:
        return _log_error(user_id, str(exc), error_type="result_read_failed")


# @app.function(
#     cpu=0.125,  # Uses 0.04, Startup 0.1 to 0.3, called every 6 seconds
#     memory=256,  # Uses 145
#     image=control_image,
#     volumes={util.USERS_DIR: users_vol},
# )
# def get_train_job_status_remote(user_id: str) -> Dict[str, Any]:
#     """Get durable async training status for a user_id.

#     IMPORTANT:
#       - This function reads from the Volume, so it calls users_vol.reload() first.

#     Returns:
#         {
#           "status": "ok",
#           "state": "QUEUED"|"RUNNING"|"SUCCEEDED"|"FAILED"|"UNKNOWN",
#           "updated_at_epoch": <int>,
#           "message": <optional str>
#         }
#     """
#     if not user_id:
#         return _NO_USER_ID

#     try:
#         users_vol.reload()

#         directory = Path(util.USERS_DIR) / user_id
#         if not _modal_dir_exists(directory):
#             return {"status": "ok", "state": _STATE_UNKNOWN, "updated_at_epoch": 0, "message": ""

#         state = _read_json(_train_job_state_path(user_id))
#         if not state:
#             return {"status": "ok", "state": _STATE_UNKNOWN, "updated_at_epoch": 0, "message": ""

#         # util.log_modal_memory("finish get_train_job_status_remote() reserved memory=256")
#         return {
#             "status": "ok",
#             "state": str(state.get("state", _STATE_UNKNOWN)),
#             "updated_at_epoch": int(state.get("updated_at_epoch", 0) or 0),
#             "message": str(state.get("message", "") or ""),
#         }
#     except Exception as exc:
#         return _log_error(user_id, str(exc), error_type="status_read_failed")


@app.function(
    cpu=0.125,  # Uses 0.04, Startup 0.1 to 0.3, called every 6 seconds
    memory=256,  # Uses 145
    image=control_image,
    volumes={util.USERS_DIR: users_vol},
)
def get_train_job_status_remote(user_id: str) -> Dict[str, Any]:
    """Get durable async training status for a user_id.

    IMPORTANT:
      - This function reads from the Volume, so it calls users_vol.reload() first.

    Returns:
        {
          "status": "ok" | "error",
          "state": "QUEUED"|"RUNNING"|"SUCCEEDED"|"FAILED"|"UNKNOWN",
          "updated_at_epoch": <int>,
          "message": <optional str>,
          "error_type": <optional str>
        }
    """
    if not user_id:
        return _NO_USER_ID

    try:
        users_vol.reload()

        directory = Path(util.USERS_DIR) / user_id
        if not _modal_dir_exists(directory):
            return {"status": "ok", "state": _STATE_UNKNOWN, "updated_at_epoch": 0, "message": ""}

        state = _read_json(_train_job_state_path(user_id))
        if not state:
            return {"status": "ok", "state": _STATE_UNKNOWN, "updated_at_epoch": 0, "message": ""}

        cur_state = str(state.get("state", _STATE_UNKNOWN))
        updated_at = int(state.get("updated_at_epoch", 0) or 0)

        # --- NEW: derive timeout from Option.SPEED so we know which training function was used ---
        options = _get_user_options(user_id) or {}
        raw_speed = options.get(Option.SPEED, 0)

        speed: int
        if isinstance(raw_speed, int):
            speed = raw_speed
        else:
            # Defensive fallback: default to the first speed index
            speed = 0

        # Map speed -> configured timeout minutes
        timeout_minutes = util.MODAL_TIMEOUT_MINUTES[speed]
        # try:
        #     timeout_minutes = util.MODAL_TIMEOUT_MINUTES[speed]
        # except Exception:
        #     timeout_minutes = max(util.MODAL_TIMEOUT_MINUTES)

        timeout_seconds = int(timeout_minutes * 60)
        cushion_seconds = 30  # Buffer so we don't race Modal timeout.  > 2 * App.POLL_INTERVAL_MS

        now = int(time.time())

        # If it's been QUEUED/RUNNING longer than the appropriate timeout + cushion,
        # treat this as a timed-out job rather than letting the UI rely on a long stale window.
        if (
            cur_state in (_STATE_QUEUED, _STATE_RUNNING)
            and updated_at > 0
            and now - updated_at > timeout_seconds + cushion_seconds
        ):
            return {
                "status": "error",
                "state": _STATE_FAILED,
                "updated_at_epoch": updated_at,
                "message": (
                    "Your job timed out. Please try Predicting again.  If it still times-out, "
                    "then try using a smaller dataset."
                ),
                "error_type": "timeout",
            }

        return {
            "status": "ok",
            "state": cur_state,
            "updated_at_epoch": updated_at,
            "message": str(state.get("message", "") or ""),
        }
    except Exception as exc:
        return _log_error(user_id, str(exc), error_type="status_read_failed")


@app.function(
    cpu=util.MODAL_CPUS[0],
    memory=util.MODAL_MEMORY[0],
    timeout=util.MODAL_TIMEOUT_MINUTES[0] * 60,  # defaults to 300secs/5mins
    image=image,
    volumes={util.USERS_DIR: users_vol},
)
def train_and_predict_speed0(user_id: str) -> Dict[str, Any]:
    """Run train_and_predict for speed == 0"""
    # util.log_modal_memory(
    #     f"Start train_and_predict_speed0 for {user_id}, reserved memory={util.MODAL_MEMORY[0]}"
    # )
    return _train_and_predict_base(user_id)


@app.function(
    cpu=util.MODAL_CPUS[1],
    memory=util.MODAL_MEMORY[1],
    timeout=util.MODAL_TIMEOUT_MINUTES[1] * 60,  # defaults to 300secs/5mins
    image=image,
    volumes={util.USERS_DIR: users_vol},
)
def train_and_predict_speed1(user_id: str) -> Dict[str, Any]:
    """Run train_and_predict for speed == 1"""
    # util.log_modal_memory(
    #     f"Start train_and_predict_speed1 for {user_id}, reserved memory={util.MODAL_MEMORY[1]}"
    # )
    return _train_and_predict_base(user_id)


@app.function(
    cpu=1,  # It does need 1 cpu
    memory=1024,  # Uses 750 consistently
    image=image,
    volumes={util.USERS_DIR: users_vol},
)
def upload_file_remote(
    user_id: str, file_name: str, file_bytes: bytes
) -> Tuple[Dict[str, Any], int]:
    """Store a user file in the Modal Volume and return inferred metadata.

    This mirrors the legacy Flask upload flow.

    Args:
        user_id: Predictly user/job ID.
        file_name: "train.csv" or "test.csv".
        file_bytes: File content.

    Returns:
        (payload, http_status_code)
    """
    if file_name not in (util.TRAIN_FILE_NAME, util.TEST_FILE_NAME):
        return (
            {
                "status": "error",
                "error_type": "invalid_filename",
                "message": "Only 'train.csv' and 'test.csv' are supported.",
                "file_name": file_name,
                "user_id": user_id,
            },
            400,
        )

    try:
        directory = _user_directory(user_id, ok_to_mkdir=(file_name == util.TRAIN_FILE_NAME))
        filepath = Path(directory) / file_name

        # Bounded retries for transient volume / I/O issues.
        last_exc: Exception | None = None
        for _ in range(3):
            try:
                filepath.write_bytes(file_bytes)
                users_vol.commit()
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # Small backoff for transient NFS / volume hiccups.
                time.sleep(0.3)

        if last_exc is not None:
            # After a few attempts, fail hard with the underlying exception message.
            raise last_exc

    except Exception as exc:
        return _log_error(user_id, str(exc), error_type="upload_save_failed"), 500

    response: Dict[str, Any] = {"status": "ok", "user_id": user_id}
    options: Dict[str, Any] = {util.Option.DATA_DIRECTORY: directory}

    # Always unlink state and result json
    _unlink(directory, (_TRAIN_JOB_RESULT_FILENAME, _TRAIN_JOB_STATE_FILENAME))
    # Force retraining
    util.remove_training_results(directory)
    users_vol.commit()  # commit to disk

    error_type = ""  # just for lint
    try:
        if file_name == util.TRAIN_FILE_NAME:
            error_type = "get_xtrain_json_failed"
            _unlink(directory, (util.OPTIONS_FILE_NAME, util.TEST_FILE_NAME))

            response["data"], response["data_description"] = util.get_xtrain_json(**options)
            util.log_modal_memory("Finish train upload_file_remote(): reserved memory=1024")
            return response, 200
        else:  # file_name == "test.csv"
            error_type = "infer_and_validate_options_failed"
            (
                options,
                response["valid_task_metrics"],
                response["unique_columns"],
                response["data"],
                # response["data_health"],
                response["data_description"],
            ) = util.infer_and_validate_options(ready_to_train=False, **options)

            # Convert for proper UI Display
            response[str(util.Option.TASK)] = options[util.Option.TASK].capitalize()
            response[str(util.Option.METRIC)] = util.METRICS_DISPLAY[options[util.Option.METRIC]]

            # Simple column names.
            response[str(util.Option.Y_COLUMN_NAME)] = options[util.Option.Y_COLUMN_NAME]
            response[str(util.Option.UID_COLUMN_NAME)] = options.get(
                util.Option.UID_COLUMN_NAME, ""
            )
            util.log_modal_memory("Finish test upload_file_remote(): reserved memory=1024")
            return response, 200

    except Exception as exc:
        _unlink(directory, file_name)
        if isinstance(exc, AppError):
            error_type = exc.error_type
            status_code = _http_status_for_app_error(error_type)
        else:
            status_code = 422
        response = _log_error(user_id, str(exc), error_type)
        if isinstance(exc, AppError):
            response["data"] = exc.data
            response["data_description"] = exc.data_description
            # response["data_health"] = exc.data_health
        return response, status_code
