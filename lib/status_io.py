from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

TERMINAL_STATES = {"succeeded", "failed", "canceled"}
VALID_STATES = {"queued", "running", "succeeded", "failed", "canceled"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_status(job_dir: Path) -> dict[str, Any]:
    return read_json(job_dir / "status.json")


def save_status(job_dir: Path, payload: dict[str, Any]) -> None:
    state = payload.get("state")
    if state not in VALID_STATES:
        raise ValueError(f"invalid state: {state}")
    atomic_write_json(job_dir / "status.json", payload)


def save_meta(job_dir: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(job_dir / "meta.json", payload)


def touch_heartbeat(job_dir: Path) -> dict[str, Any]:
    payload = load_status(job_dir)
    payload["last_heartbeat_at"] = now_iso()
    save_status(job_dir, payload)
    return payload


def update_state(
    job_dir: Path,
    *,
    state: str,
    error_message: str | None = None,
    exit_code: int | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    pid: int | None = None,
    remote_job_id: str | None = None,
    host: str | None = None,
    mode: str | None = None,
    gpu_ids: list[int] | None = None,
    gpu_lock_path: str | None = None,
) -> dict[str, Any]:
    payload = load_status(job_dir)
    payload["state"] = state
    if error_message is not None:
        payload["error_message"] = error_message
    if exit_code is not None or state in TERMINAL_STATES:
        payload["exit_code"] = exit_code
    if started_at is not None:
        payload["started_at"] = started_at
    if ended_at is not None:
        payload["ended_at"] = ended_at
    if pid is not None:
        payload["pid"] = pid
    if remote_job_id is not None:
        payload["remote_job_id"] = remote_job_id
    if host is not None:
        payload["host"] = host
    if mode is not None:
        payload["mode"] = mode
    if gpu_ids is not None:
        payload["gpu_ids"] = gpu_ids
    if gpu_lock_path is not None:
        payload["gpu_lock_path"] = gpu_lock_path
    payload["last_heartbeat_at"] = now_iso()
    save_status(job_dir, payload)
    return payload


def build_kv_lines(status: dict[str, Any]) -> list[str]:
    return [
        f"JOB_ID={status.get('job_id', '')}",
        f"STATE={status.get('state', '')}",
        f"MODE={status.get('mode', '')}",
        f"HOST={status.get('host', '')}",
        f"WORK_DIR={status.get('work_dir', '')}",
        f"JOB_DIR={status.get('job_dir', '')}",
        f"STATUS_PATH={status.get('job_dir', '')}/status.json",
        f"LOG_PATH={status.get('log_path', '')}",
        f"SUMMARY_PATH={status.get('summary_path', '')}",
        f"METRICS_PATH={status.get('metrics_path', '')}",
        f"PID={'' if status.get('pid') is None else status.get('pid')}",
        f"REMOTE_JOB_ID={status.get('remote_job_id', '')}",
        f"ENV_KIND={status.get('env_kind', '')}",
        f"CONDA_ENV={status.get('conda_env', '')}",
        f"CONDA_PREFIX={status.get('conda_prefix', '')}",
        f"PYTHON_BIN={status.get('python_bin', '')}",
    ]


def summarize_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": status.get("job_id", ""),
        "state": status.get("state", ""),
        "mode": status.get("mode", ""),
        "host": status.get("host", ""),
        "log_path": status.get("log_path", ""),
        "summary_path": status.get("summary_path", ""),
        "metrics_path": status.get("metrics_path", ""),
        "exit_code": status.get("exit_code"),
        "error_message": status.get("error_message", ""),
    }
