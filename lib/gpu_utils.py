from __future__ import annotations

import json
import os
import socket
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MIN_FREE_MEM_MB = 1024
DEFAULT_MAX_UTIL = 10
DEFAULT_REMOTE_PREFERRED_MIN_FREE_MEM_MB = 8 * 1024
DEFAULT_MAX_BAD_UTIL = 90
DEFAULT_LEASE_TIMEOUT_SECONDS = 180
BASE_DIR = Path(__file__).resolve().parent.parent
LOCKS_DIR = BASE_DIR / "locks"


@dataclass
class GPUInfo:
    index: int
    uuid: str
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_gpu: int

    @property
    def memory_free_mb(self) -> int:
        return max(self.memory_total_mb - self.memory_used_mb, 0)

    @property
    def capacity_rank(self) -> int:
        if self.memory_total_mb >= 46000:
            return 0
        if self.memory_total_mb >= 22000:
            return 1
        return 2

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        return (self.capacity_rank, self.memory_used_mb, self.utilization_gpu, self.index)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
            "utilization_gpu": self.utilization_gpu,
        }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_nvidia_smi() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi not found") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        raise RuntimeError(stderr or stdout or "nvidia-smi failed") from exc
    return result.stdout


def query_local_gpus() -> list[GPUInfo]:
    output = _run_nvidia_smi().strip()
    if not output:
        return []
    gpus: list[GPUInfo] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        gpus.append(
            GPUInfo(
                index=int(parts[0]),
                uuid=parts[1],
                name=parts[2],
                memory_total_mb=int(float(parts[3])),
                memory_used_mb=int(float(parts[4])),
                utilization_gpu=int(float(parts[5])),
            )
        )
    return gpus


def local_host() -> str:
    return socket.gethostname()


def gpu_lock_path(host: str, gpu_index: int) -> Path:
    return LOCKS_DIR / f"{host.replace('/', '_')}__gpu{gpu_index}.lock"


def read_lock_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with suppress(json.JSONDecodeError, OSError):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return None


def _pid_alive_local(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_pid_alive(host: str, pid: int | None) -> bool:
    if not pid:
        return False
    if host in {local_host(), "localhost", "127.0.0.1"}:
        return _pid_alive_local(pid)
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            host,
            f"ps -p {int(pid)} >/dev/null 2>&1",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _parse_iso(text: str | None) -> datetime | None:
    if not text:
        return None
    with suppress(ValueError):
        return datetime.fromisoformat(text)
    return None


def lock_is_stale(payload: dict[str, Any], lease_timeout_seconds: int = DEFAULT_LEASE_TIMEOUT_SECONDS) -> bool:
    state = payload.get("state")
    if state in {"succeeded", "failed", "canceled", "finished"}:
        return True
    worker_host = payload.get("worker_host") or payload.get("host") or payload.get("target_host")
    worker_pid = payload.get("worker_pid") or payload.get("remote_pid")
    if worker_host and worker_pid and is_pid_alive(str(worker_host), int(worker_pid)):
        return False
    owner_host = payload.get("owner_host") or payload.get("entry_host") or payload.get("host")
    owner_pid = payload.get("owner_pid")
    if worker_pid and not is_pid_alive(str(worker_host), int(worker_pid)):
        return True
    if owner_host and owner_pid and is_pid_alive(str(owner_host), int(owner_pid)):
        return False
    heartbeat_at = _parse_iso(payload.get("last_heartbeat_at") or payload.get("updated_at") or payload.get("created_at"))
    if heartbeat_at is None:
        return True
    age_seconds = (datetime.now(timezone.utc) - heartbeat_at.astimezone(timezone.utc)).total_seconds()
    return age_seconds > lease_timeout_seconds


def cleanup_stale_gpu_lock(path: Path, lease_timeout_seconds: int = DEFAULT_LEASE_TIMEOUT_SECONDS) -> bool:
    payload = read_lock_payload(path)
    if not payload:
        path.unlink(missing_ok=True)
        return True
    if lock_is_stale(payload, lease_timeout_seconds=lease_timeout_seconds):
        path.unlink(missing_ok=True)
        return True
    return False


def write_lock_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_lock_payload(
    *,
    job_id: str,
    host: str,
    gpu_index: int,
    job_dir: str,
    owner_host: str,
    owner_pid: int,
    mode: str,
) -> dict[str, Any]:
    now = now_utc_iso()
    return {
        "job_id": job_id,
        "host": host,
        "target_host": host,
        "gpu_index": gpu_index,
        "job_dir": job_dir,
        "mode": mode,
        "state": "queued",
        "owner_host": owner_host,
        "owner_pid": owner_pid,
        "worker_host": host,
        "worker_pid": None,
        "created_at": now,
        "last_heartbeat_at": now,
        "updated_at": now,
    }


def acquire_gpu_lock(payload: dict[str, Any]) -> Path | None:
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    path = gpu_lock_path(str(payload["host"]), int(payload["gpu_index"]))
    if path.exists() and not cleanup_stale_gpu_lock(path):
        return None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def refresh_gpu_lock(
    lock_path: Path | str | None,
    *,
    state: str | None = None,
    owner_pid: int | None = None,
    worker_pid: int | None = None,
    worker_host: str | None = None,
) -> dict[str, Any] | None:
    if not lock_path:
        return None
    path = Path(lock_path)
    payload = read_lock_payload(path)
    if not payload:
        return None
    if state is not None:
        payload["state"] = state
    if owner_pid is not None:
        payload["owner_pid"] = owner_pid
    if worker_pid is not None:
        payload["worker_pid"] = worker_pid
    if worker_host is not None:
        payload["worker_host"] = worker_host
    now = now_utc_iso()
    payload["last_heartbeat_at"] = now
    payload["updated_at"] = now
    write_lock_payload(path, payload)
    return payload


def release_gpu_lock(lock_path: Path | str | None, final_state: str | None = None) -> None:
    if not lock_path:
        return
    path = Path(lock_path)
    payload = read_lock_payload(path)
    if payload and final_state is not None:
        payload["state"] = final_state
        payload["last_heartbeat_at"] = now_utc_iso()
        payload["updated_at"] = payload["last_heartbeat_at"]
        write_lock_payload(path, payload)
    path.unlink(missing_ok=True)


def list_reserved_gpus(host: str | None = None) -> set[int]:
    reserved: set[int] = set()
    target_host = host or local_host()
    if not LOCKS_DIR.exists():
        return reserved
    for path in LOCKS_DIR.glob(f"{target_host.replace('/', '_')}__gpu*.lock"):
        if path.exists() and not cleanup_stale_gpu_lock(path):
            payload = read_lock_payload(path) or {}
            gpu_index = payload.get("gpu_index")
            if isinstance(gpu_index, int):
                reserved.add(gpu_index)
    return reserved


def _gpu_passes_local_filters(gpu: GPUInfo, min_free_mem_mb: int, max_util: int) -> bool:
    if gpu.memory_free_mb < min_free_mem_mb:
        return False
    if gpu.utilization_gpu >= max_util:
        return False
    if gpu.utilization_gpu > DEFAULT_MAX_BAD_UTIL and gpu.memory_used_mb <= 16:
        return False
    return True


def list_free_gpus(min_free_mem_mb: int = DEFAULT_MIN_FREE_MEM_MB, max_util: int = DEFAULT_MAX_UTIL) -> list[int]:
    free: list[GPUInfo] = []
    reserved = list_reserved_gpus()
    for gpu in query_local_gpus():
        if gpu.index in reserved:
            continue
        if not _gpu_passes_local_filters(gpu, min_free_mem_mb=min_free_mem_mb, max_util=max_util):
            continue
        free.append(gpu)
    free.sort(key=lambda item: item.sort_key)
    return [gpu.index for gpu in free]


def has_preferred_local_gpu(min_free_mem_mb: int = DEFAULT_REMOTE_PREFERRED_MIN_FREE_MEM_MB, max_util: int = DEFAULT_MAX_UTIL) -> bool:
    reserved = list_reserved_gpus()
    for gpu in query_local_gpus():
        if gpu.index in reserved:
            continue
        if gpu.capacity_rank != 0:
            continue
        if not _gpu_passes_local_filters(gpu, min_free_mem_mb=min_free_mem_mb, max_util=max_util):
            continue
        return True
    return False


def pick_best_gpu(min_free_mem_mb: int = DEFAULT_MIN_FREE_MEM_MB, max_util: int = DEFAULT_MAX_UTIL) -> int | None:
    free = list_free_gpus(min_free_mem_mb=min_free_mem_mb, max_util=max_util)
    return free[0] if free else None


def pick_candidate_gpus(min_free_mem_mb: int = DEFAULT_MIN_FREE_MEM_MB, max_util: int = DEFAULT_MAX_UTIL) -> list[int]:
    return list_free_gpus(min_free_mem_mb=min_free_mem_mb, max_util=max_util)


def write_gpu_probe(path: Path) -> None:
    payload = {
        "host": local_host(),
        "gpus": [gpu.to_dict() for gpu in query_local_gpus()],
    }
    path.write_text(json.dumps(payload, indent=2) + os.linesep)
