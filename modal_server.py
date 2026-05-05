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

    Used by:
      - POST /ui/train        (sync)
      - POST /ui/train_async  (async)

    Attributes:
        user_id: Predictly job/user identifier.
        metric: Metric name selected by the user.
        uid_column_name: Optional unique-id column name ("" allowed).
        speed: Currently only 0.
    """

    user_id: str
    task: str
    metric: str
    speed: int = 0
    uid_column_name: str = ""  # Could be None


def _missing_user_id_result() -> dict[str, Any]:
    """Return a stable missing_user_id error in the standard UI envelope."""
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

    This function:
      - Runs in the lightweight web_image
      - Hosts the FastAPI app
      - Receives all HTTP traffic from the browser/UI

    Returns:
        The FastAPI application instance.
    """
    return web


@web.post("/ui/train_async")
def ui_train_async(body: TrainBody) -> dict[str, Any]:
    """Start async training + prediction and return immediately.

    Policy:
      - If NO job is currently running for user_id:
          - Claim a new run in the worker (durably writes QUEUED + clears prior result)
          - Spawn background training
          - Return {status:"ok", state:"QUEUED", ...}
      - If a job IS already running (QUEUED/RUNNING):
          - Do not spawn another job (duplicate-click protection)
          - Return the current in-flight state

    Important:
      - This endpoint does NOT block.
      - It does NOT "queue a second job for later".
        If the user wants a new run after completion, they call /ui/train_async again.

    Args:
        body: Parsed JSON request body (TrainBody).

    Returns:
        Success (always):
            {
              "result": {
                "status": "ok",
                "state": "QUEUED" | "RUNNING",
                "user_id": "<user_id>",
                "updated_at_epoch": <int>,
                "message": <optional>
              }
            }

        Error:
            { "result": {"status":"error","error_type":...,"message":...} }
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
    """Get async training result if ready.

    Args:
        user_id: Predictly job/user identifier.

    Returns:
        If ready (SUCCEEDED):
            { "result": { ... same payload as Phase 1 /ui/train result ... } }

        If not ready:
            {
              "result": {
                "status": "error",
                "error_type": "not_ready",
                "message": "...",
                "state": "QUEUED"|"RUNNING"|"FAILED"|"UNKNOWN"
              }
            }

        If worker returns an error:
            { "result": {"status":"error","error_type":...,"message":...} }
    """
    user_id = user_id.strip()
    if not user_id:
        return _missing_user_id_result()

    payload = get_train_job_result_remote.remote(user_id)
    return {"result": payload}


@web.get("/ui/train_status/{user_id}")
def ui_train_status(user_id: str) -> dict[str, Any]:
    """Get durable async training status (pollable).

    This endpoint is safe to call repeatedly (polling).
    The worker function should read from the persistent Volume state.

    Args:
        user_id: Predictly job/user identifier.

    Returns:
        {
          "result": {
            "status": "ok",
            "state": "QUEUED"|"RUNNING"|"SUCCEEDED"|"FAILED"|"UNKNOWN",
            "updated_at_epoch": <int>,
            "message": <optional str>
          }
        }
        or an error envelope { "result": {...} }.
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
    """Upload a training or test CSV file.

    Behavior:
      - Validates dataset_kind -> maps to "train.csv" or "test.csv"
      - Reads file bytes
      - Calls worker: upload_file_remote(user_id, file_name, file_bytes)
      - Returns payload wrapped in { "result": ... } with http_status for debugging

    Args:
        user_id: Predictly job/user identifier.
        dataset_kind: One of {"training", "train", "test"}.
        file: Uploaded CSV file.

    Returns:
        {
          "result": <payload returned by upload_file_remote>,
          "http_status": <int>
        }
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
