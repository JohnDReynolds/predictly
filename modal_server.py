"""
Module: modal_server.py

HTTP API for Predictly running on Modal.

This module exposes a small, UI-facing HTTP surface area using FastAPI,
mounted as a Modal ASGI web app.

Key design principles
---------------------
- This file is ONLY a thin HTTP adapter.
- Heavy ML logic and durable storage live in the worker app (modal_predictly.py).
- We do NOT import project code here (keeps the web container lightweight).
- Response envelope is ALWAYS: { "result": {...} }.
  (The UI expects this exact shape.)

Modal layout
------------
Two deployed Modal apps:

1) predictly-worker (modal_predictly.py)
   - upload_file_remote(user_id, file_name, file_bytes) -> (payload, http_status)
   - train_and_predict_remote(user_id, metric, uid_column_name) -> payload

   Phase 2 worker functions used by this web app:
   - claim_train_slot_remote(user_id, metric, uid_column_name) -> {status, claimed, state, ...}
   - get_train_job_status_remote(user_id) -> {status, state, updated_at_epoch, message}
   - get_train_job_result_remote(user_id) ->
         result dict OR {status:"error", error_type:"not_ready", ...}
   - train_and_predict_background(user_id) -> runs in background (spawned)

2) predictly-web (this file)
   - Hosts FastAPI endpoints used by the UI/curl clients.

Endpoints
---------
Phase 1 (synchronous; must remain stable):
- POST /ui/upload
- POST /ui/train

Phase 2 (asynchronous; durable state + polling):
- POST /ui/train_async
- GET  /ui/train_status/{user_id}
- GET  /ui/train_result/{user_id}

Important Phase 2 policy
------------------------
- "Always rerun training when requested" UNLESS a job is already in progress
  (QUEUED or RUNNING) for the same user_id.
- Duplicate-click protection is enforced in the worker via claim_train_slot_remote:
  only one request can "claim" a new run; all others return the in-flight state.
"""

# Errors to ignore
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from typing import Any

import modal
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# -----------------------------------------------------------------------------
# Modal application identity
# -----------------------------------------------------------------------------

_WEB_APP_NAME = "predictly-web"
_WORKER_APP_NAME = "predictly-worker"


# -----------------------------------------------------------------------------
# Web container image (INTENTIONALLY LIGHTWEIGHT)
# -----------------------------------------------------------------------------
"""
This image is used ONLY for the HTTP / FastAPI layer.

Key point:
- It does NOT include your ML stack.
- It does NOT include your project code.
- It ONLY needs enough to:
    * parse HTTP requests
    * forward them to existing Modal functions

Your heavy image (AutoGluon, pandas, etc.) lives in modal_predictly.py.
"""
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi",
    "pydantic",
    "python-multipart",
)


# -----------------------------------------------------------------------------
# Modal App + FastAPI app
# -----------------------------------------------------------------------------

app = modal.App(_WEB_APP_NAME)

#: The FastAPI application that defines HTTP routes.
web = FastAPI()

# This is required for the browser to read responses from the Modal domain.
web.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://unique-dusk-0a4d6a.netlify.app",  # Netlify
        "https://predictly.cloud",  # GoDaddy custom domain
        "https://www.predictly.cloud",  # www (Netlify redirects, but safe to allow)
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Lookup existing Modal functions (CRITICAL DESIGN POINT)
# We intentionally DO NOT import modal_predictly here.
# Why?

# Each Modal function runs in its own container with its own filesystem/image.
# Importing modal_predictly inside this web container would require:
# - duplicating project code
# - duplicating heavy dependencies
# - tightly coupling the web layer to the ML layer

# Instead, we look up already-deployed Modal functions by name.
# This is the cleanest and most stable pattern for Modal web frontends.
# -----------------------------------------------------------------------------

# Worker functions
claim_train_slot_remote = modal.Function.from_name(_WORKER_APP_NAME, "claim_train_slot_remote")
get_train_job_status_remote = modal.Function.from_name(
    _WORKER_APP_NAME, "get_train_job_status_remote"
)
get_train_job_result_remote = modal.Function.from_name(
    _WORKER_APP_NAME, "get_train_job_result_remote"
)
train_and_predict_speed0 = modal.Function.from_name(_WORKER_APP_NAME, "train_and_predict_speed0")
train_and_predict_speed1 = modal.Function.from_name(_WORKER_APP_NAME, "train_and_predict_speed1")
upload_file_remote = modal.Function.from_name(_WORKER_APP_NAME, "upload_file_remote")


# -----------------------------------------------------------------------------
# Small helpers / schemas
# -----------------------------------------------------------------------------

_DATASET_TO_FILENAME: dict[str, str] = {
    "training": "train.csv",  # must agree with util.TRAIN_FILE_NAME
    "train": "train.csv",  # must agree with util.TRAIN_FILE_NAME
    "test": "test.csv",  # must agree with util.TEST_FILE_NAME
}


class TrainBody(BaseModel):
    """Request body for training endpoints.

    Used by synchronous and asynchronous training routes.

    Attributes:
        user_id: Predictly job/user identifier.
        task: Task type selected by the user.
        metric: Metric name selected by the user.
        speed: Training speed preset.
        uid_column_name: Optional unique-id column name. Empty string is allowed.
    """

    user_id: str
    task: str
    metric: str
    speed: int = 0
    uid_column_name: str = ""  # Could be None


def _missing_user_id_result() -> dict[str, Any]:
    """Build the standard missing-user-id response envelope.

    Returns:
        A UI response envelope containing a stable ``missing_user_id`` error.
    """
    return {
        "result": {
            "status": "error",
            "error_type": "missing_user_id",
            "message": "Missing user_id.",
        }
    }


# -----------------------------------------------------------------------------
# Routes / Endpoints
# -----------------------------------------------------------------------------
@app.function(image=web_image)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    """Expose the FastAPI app as a Modal web endpoint.

    This function runs in the lightweight web image and receives all
    HTTP traffic from the browser/UI.

    Returns:
        The FastAPI application instance.
    """
    return web


@web.post("/ui/train_async")
def ui_train_async(body: TrainBody) -> dict[str, Any]:
    """Start async training and prediction, then return immediately.

    If no job is currently running for the user, this endpoint claims a new
    run in the worker, spawns background training, and returns queued state.
    If a job is already queued or running, it returns the current in-flight
    state without spawning another job.

    Args:
        body: Parsed JSON request body.

    Returns:
        Standard UI response envelope containing either queued/running state
        or a structured error payload.

    Raises:
        Exception: Propagates unexpected Modal worker lookup, remote-call, or
            spawn failures. Normal user/data errors are returned inside the
            response envelope instead of raised.
    """
    user_id = body.user_id.strip()
    if not user_id:
        return _missing_user_id_result()

    # The worker enforces "single flight" per user_id and writes durable state.
    claim = claim_train_slot_remote.remote(
        user_id, body.task.strip(), body.metric.strip(), body.speed, body.uid_column_name.strip()
    )
    # claim = claim_train_slot_remote.remote(
    #     user_id, body.metric.strip(), body.speed, body.uid_column_name.strip()
    # )
    if str(claim.get("status", "")).lower() == "error":
        return {"result": claim}

    claimed = bool(claim.get("claimed", False))
    state = str(claim.get("state", "UNKNOWN"))
    updated_at_epoch = claim.get("updated_at_epoch", 0)
    message = claim.get("message", "")

    # Only spawn when the worker says we successfully claimed a fresh run.
    if claimed:
        if body.speed == 1:
            train_and_predict_speed1.spawn(user_id)
        else:
            train_and_predict_speed0.spawn(user_id)

    payload: dict[str, Any] = {"status": "ok", "state": state, "user_id": user_id}
    if isinstance(updated_at_epoch, int):
        payload["updated_at_epoch"] = updated_at_epoch
    if isinstance(message, str) and message.strip():
        payload["message"] = message.strip()

    return {"result": payload}


@web.get("/ui/train_result/{user_id}")
def ui_train_result(user_id: str) -> dict[str, Any]:
    """Return the async training result if it is ready.

    Args:
        user_id: Predictly job/user identifier.

    Returns:
        Standard UI response envelope containing either the completed training
        result or a structured ``not_ready`` / worker error payload.

    Raises:
        Exception: Propagates unexpected Modal remote-call failures. Expected
            polling states such as ``not_ready`` are returned in the envelope.
    """
    user_id = user_id.strip()
    if not user_id:
        return _missing_user_id_result()

    payload = get_train_job_result_remote.remote(user_id)
    return {"result": payload}


@web.get("/ui/train_status/{user_id}")
def ui_train_status(user_id: str) -> dict[str, Any]:
    """Return durable async training status for polling.

    This endpoint is safe to call repeatedly. The worker reads state from
    persistent Volume storage.

    Args:
        user_id: Predictly job/user identifier.

    Returns:
        Standard UI response envelope containing current job state, timestamp,
        and optional message/error details.

    Raises:
        Exception: Propagates unexpected Modal remote-call failures. Missing
            or unknown job state is returned in the envelope instead of raised.
    """
    user_id = user_id.strip()
    if not user_id:
        return _missing_user_id_result()

    payload = get_train_job_status_remote.remote(user_id)
    return {"result": payload}


@web.post("/ui/upload")
async def ui_upload(
    user_id: str = Form(...),
    dataset_kind: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload a training or prediction CSV file.

    Validates ``dataset_kind``, reads the uploaded file bytes, forwards the
    file to the worker, and wraps the worker payload in the standard UI
    response envelope.

    Args:
        user_id: Predictly job/user identifier.
        dataset_kind: One of ``"training"``, ``"train"``, or ``"test"``.
        file: Uploaded CSV file.

    Returns:
        Standard UI response envelope containing the worker payload and,
        when available, the worker HTTP-style status code.

    Raises:
        Exception: Propagates unexpected upload read failures or Modal
            remote-call failures. Expected validation errors are returned in
            the response envelope instead of raised.
    """
    safe_filename = _DATASET_TO_FILENAME.get(dataset_kind.strip().lower())
    if not safe_filename:
        return {
            "result": {
                "status": "error",
                "error_type": "invalid_dataset_kind",
                "message": f"Unknown dataset_kind '{dataset_kind}' (expected training|test).",
                "user_id": user_id,
            }
        }

    file_bytes = await file.read()

    payload, status = upload_file_remote.remote(
        user_id.strip(),
        safe_filename,
        file_bytes,
    )

    # IMPORTANT: The UI expects {"result": ...} — do not change this envelope.
    return {"result": payload, "http_status": status}
