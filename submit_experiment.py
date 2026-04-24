#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lib.gpu_utils import acquire_gpu_lock, build_lock_payload, cleanup_stale_gpu_lock, gpu_lock_path, is_pid_alive, read_lock_payload, refresh_gpu_lock, release_gpu_lock

BASE_DIR = Path(__file__).resolve().parent
HOSTS_FILE = BASE_DIR / "hosts.txt"
GPU_PROBE = BASE_DIR / "gpu_probe.py"
REMOTE_LAUNCH = BASE_DIR / "remote_launch.sh"
JOBS_DIR = BASE_DIR / "jobs"
LOCKS_DIR = BASE_DIR / "locks"
SHARED_ROOT = Path("/zssd/cs")


@dataclass
class Candidate:
    host: str
    gpu: dict

    @property
    def capacity_rank(self) -> int:
        memory_total_mb = int(self.gpu.get("memory_total_mb", 0))
        if memory_total_mb >= 46000:
            return 0
        if memory_total_mb >= 22000:
            return 1
        return 2

    @property
    def sort_key(self) -> tuple[int, int, int, int, str, int]:
        return (
            self.capacity_rank,
            self.gpu["process_count"],
            self.gpu["memory_used_mb"],
            self.gpu["utilization_gpu"],
            self.host,
            self.gpu["index"],
        )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-._")
    return value or "job"


def load_hosts() -> list[str]:
    if not HOSTS_FILE.exists():
        raise FileNotFoundError(f"hosts file not found: {HOSTS_FILE}")
    hosts = [line.strip() for line in HOSTS_FILE.read_text().splitlines() if line.strip()]
    if not hosts:
        raise RuntimeError("hosts.txt is empty")
    return hosts


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(stderr or stdout or f"command failed: {' '.join(command)}")
    return result


def ssh(host: str, remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            host,
            remote_command,
        ],
        check=check,
    )


def ensure_shared_workdir(workdir: Path) -> Path:
    resolved = workdir.resolve()
    try:
        resolved.relative_to(SHARED_ROOT)
    except ValueError as exc:
        raise ValueError(f"workdir must live under {SHARED_ROOT}: {resolved}") from exc
    return resolved


def probe_host(host: str) -> dict:
    probe_path = shlex.quote(str(GPU_PROBE))
    result = ssh(host, f"python3 {probe_path}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid probe output from {host}: {result.stdout!r}") from exc
    if "error" in payload:
        raise RuntimeError(f"probe failed on {host}: {payload['error']}")
    return payload


def passes_filters(gpu: dict, args: argparse.Namespace) -> bool:
    if args.require_no_proc and gpu["process_count"] > 0:
        return False
    if gpu["memory_free_mb"] < int(args.min_free_mem_gb * 1024):
        return False
    if gpu["utilization_gpu"] > args.max_util:
        return False
    if gpu["utilization_gpu"] > 90 and gpu["memory_used_mb"] <= 16:
        return False
    return True


def collect_candidates(hosts: list[str], args: argparse.Namespace) -> tuple[list[Candidate], list[dict]]:
    candidates: list[Candidate] = []
    reports: list[dict] = []
    for host in hosts:
        report = {"host": host}
        try:
            payload = probe_host(host)
            report["gpus"] = payload["gpus"]
            for gpu in payload["gpus"]:
                if passes_filters(gpu, args):
                    candidates.append(Candidate(host=host, gpu=gpu))
        except Exception as exc:  # noqa: BLE001
            report["error"] = str(exc)
        reports.append(report)
    candidates.sort(key=lambda item: item.sort_key)
    return candidates, reports


def lock_path(host: str, gpu_index: int) -> Path:
    return LOCKS_DIR / f"{host.replace('/', '_')}__gpu{gpu_index}.lock"


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with suppress(json.JSONDecodeError):
        return json.loads(path.read_text())
    return None


def is_remote_pid_alive(host: str, pid: int | None) -> bool:
    if not pid:
        return False
    result = ssh(host, f"ps -p {pid} >/dev/null 2>&1", check=False)
    return result.returncode == 0


def cleanup_stale_lock(path: Path) -> bool:
    return cleanup_stale_gpu_lock(path)


def acquire_lock(candidate: Candidate, job_id: str, owner_pid: int, job_dir: Path) -> Path | None:
    payload = build_lock_payload(
        job_id=job_id,
        host=candidate.host,
        gpu_index=candidate.gpu["index"],
        job_dir=str(job_dir),
        owner_host=socket.gethostname(),
        owner_pid=owner_pid,
        mode="remote",
    )
    return acquire_gpu_lock(payload)


def refresh_candidate(host: str, gpu_index: int, args: argparse.Namespace) -> bool:
    payload = probe_host(host)
    for gpu in payload["gpus"]:
        if gpu["index"] == gpu_index:
            return passes_filters(gpu, args)
    return False


def make_job_dir(job_name: str) -> tuple[str, Path]:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"{stamp}_{job_name}_{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_id, job_dir


def write_text(path: Path, content: str) -> None:
    path.write_text(content)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def encode(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def build_env_payload(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.conda_activate and args.venv_activate:
        raise ValueError("choose only one of --conda-activate or --venv-activate")
    if args.conda_activate:
        snippet = args.conda_activate.strip()
        prefix_match = re.fullmatch(r"conda run -p (?P<value>.+) true", snippet)
        if prefix_match:
            return ("conda_prefix", prefix_match.group("value"), "")
        env_match = re.fullmatch(r"conda run -n (?P<value>.+) true", snippet)
        if env_match:
            return ("conda_env", env_match.group("value"), "")
        return ("shell", snippet, "")
    if args.venv_activate:
        return ("venv", args.venv_activate, "")
    return ("", "", "")


def launch_remote(candidate: Candidate, job_dir: Path, args: argparse.Namespace) -> int:
    stdout_log = job_dir / "stdout.log"
    stderr_log = job_dir / "stderr.log"
    pid_file = job_dir / "pid"
    status_file = job_dir / "status"
    exit_code_file = job_dir / "exit_code"
    env_mode, env_value, extra = build_env_payload(args)
    cmd_b64 = encode(args.cmd)
    if env_mode in {"conda_env", "conda_prefix"}:
        conda_bin = os.environ.get("MULTI_TASK_CONDA_BIN") or shutil.which("conda") or os.path.expanduser("~/mambaforge/bin/conda")
        remote_command = " ".join(
            [
                "bash",
                shlex.quote(str(REMOTE_LAUNCH)),
                shlex.quote(str(args.workdir)),
                shlex.quote(str(candidate.gpu["index"])),
                shlex.quote(str(job_dir)),
                shlex.quote(str(stdout_log)),
                shlex.quote(str(stderr_log)),
                shlex.quote(str(pid_file)),
                shlex.quote(str(status_file)),
                shlex.quote(str(exit_code_file)),
                shlex.quote(conda_bin),
                shlex.quote("prefix" if env_mode == "conda_prefix" else "env"),
                shlex.quote(env_value),
                shlex.quote(cmd_b64),
            ]
        )
    else:
        env_snippet = args.conda_activate or (f"source {shlex.quote(args.venv_activate)}" if args.venv_activate else "")
        remote_command = " ".join(
            [
                "bash",
                shlex.quote(str(REMOTE_LAUNCH)),
                shlex.quote(str(args.workdir)),
                shlex.quote(str(candidate.gpu["index"])),
                shlex.quote(str(job_dir)),
                shlex.quote(str(stdout_log)),
                shlex.quote(str(stderr_log)),
                shlex.quote(str(pid_file)),
                shlex.quote(str(status_file)),
                shlex.quote(str(exit_code_file)),
                shlex.quote(shutil.which("conda") or os.path.expanduser("~/mambaforge/bin/conda")),
                shlex.quote("shell"),
                shlex.quote(env_snippet),
                shlex.quote(cmd_b64),
            ]
        )
    result = ssh(candidate.host, remote_command)
    remote_pid = int(result.stdout.strip().splitlines()[-1])
    return remote_pid


def update_lock(lock_file: Path, remote_pid: int, host: str) -> None:
    refresh_gpu_lock(lock_file, state="running", worker_pid=remote_pid, worker_host=host)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find an idle GPU host and launch a remote experiment.")
    parser.add_argument("--workdir", default=os.getcwd(), help="Shared project directory under /zssd/cs")
    parser.add_argument("--cmd", required=True, help="Command to run remotely")
    parser.add_argument("--name", default="experiment", help="Job name")
    parser.add_argument("--conda-activate", help="Shell snippet to activate conda or modules")
    parser.add_argument("--venv-activate", help="Path to a venv activate script")
    parser.add_argument("--min-free-mem-gb", type=float, default=8.0, help="Minimum free GPU memory in GiB")
    parser.add_argument("--max-util", type=int, default=10, help="Maximum allowed GPU utilization percent")
    parser.add_argument("--allow-proc", action="store_true", help="Allow GPUs with existing compute processes")
    parser.add_argument("--dry-run", action="store_true", help="Show the selected target without launching")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.require_no_proc = not args.allow_proc
    args.workdir = ensure_shared_workdir(Path(args.workdir))
    hosts = load_hosts()
    job_name = slugify(args.name)
    job_id, job_dir = make_job_dir(job_name)
    launcher_log = job_dir / "launcher.out"
    write_text(job_dir / "status", "queued\n")

    reports: list[dict] = []
    lock_file: Path | None = None
    chosen: Candidate | None = None

    try:
        candidates, reports = collect_candidates(hosts, args)
        if not candidates:
            raise RuntimeError("no idle GPU matched the current filters")

        for candidate in candidates:
            maybe_lock = acquire_lock(candidate, job_id, os.getpid(), job_dir)
            if maybe_lock is None:
                continue
            if not refresh_candidate(candidate.host, candidate.gpu["index"], args):
                maybe_lock.unlink(missing_ok=True)
                continue
            lock_file = maybe_lock
            chosen = candidate
            break

        if chosen is None or lock_file is None:
            raise RuntimeError("all matching GPUs were claimed concurrently; retry soon")

        metadata = {
            "job_id": job_id,
            "created_at": now_iso(),
            "entry_host": socket.gethostname(),
            "workdir": str(args.workdir),
            "cmd": args.cmd,
            "job_name": job_name,
            "host": chosen.host,
            "gpu_index": chosen.gpu["index"],
            "gpu_uuid": chosen.gpu["uuid"],
            "lock_file": str(lock_file),
            "status": "queued",
            "filters": {
                "min_free_mem_gb": args.min_free_mem_gb,
                "max_util": args.max_util,
                "require_no_proc": args.require_no_proc,
            },
            "probe_reports": reports,
        }
        write_json(job_dir / "job.json", metadata)
        write_text(launcher_log, json.dumps({"selected_host": chosen.host, "selected_gpu": chosen.gpu["index"]}, indent=2) + "\n")

        if args.dry_run:
            print(json.dumps({"job_id": job_id, "job_dir": str(job_dir), "host": chosen.host, "gpu_index": chosen.gpu["index"]}, indent=2))
            return 0

        remote_pid = launch_remote(chosen, job_dir, args)
        update_lock(lock_file, remote_pid, chosen.host)
        metadata["remote_pid"] = remote_pid
        metadata["status"] = "running"
        metadata["started_at"] = now_iso()
        write_json(job_dir / "job.json", metadata)
        write_text(job_dir / "status", "running\n")
        print(json.dumps({
            "job_id": job_id,
            "job_dir": str(job_dir),
            "host": chosen.host,
            "gpu_index": chosen.gpu["index"],
            "remote_pid": remote_pid,
            "stdout_log": str(job_dir / "stdout.log"),
            "stderr_log": str(job_dir / "stderr.log"),
        }, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        write_text(launcher_log, str(exc) + "\n")
        write_text(job_dir / "status", "failed\n")
        payload = {
            "job_id": job_id,
            "created_at": now_iso(),
            "entry_host": socket.gethostname(),
            "workdir": str(args.workdir),
            "cmd": args.cmd,
            "job_name": job_name,
            "status": "failed",
            "error": str(exc),
            "probe_reports": reports,
        }
        if chosen is not None:
            payload["host"] = chosen.host
            payload["gpu_index"] = chosen.gpu["index"]
        if lock_file is not None:
            payload["lock_file"] = str(lock_file)
        write_json(job_dir / "job.json", payload)
        if lock_file is not None:
            release_gpu_lock(lock_file, final_state="failed")
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
