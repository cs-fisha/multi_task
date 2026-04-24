#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from lib.gpu_utils import refresh_gpu_lock, release_gpu_lock

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ssh(host: str, remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            host,
            remote_command,
        ],
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"ssh failed for {host}")
    return result


def load_job(job_dir: Path) -> dict:
    return json.loads((job_dir / "job.json").read_text())


def save_job(job_dir: Path, payload: dict) -> None:
    (job_dir / "job.json").write_text(json.dumps(payload, indent=2) + "\n")


def check_status(payload: dict) -> str:
    host = payload.get("host")
    pid = payload.get("remote_pid")
    if not host or not pid:
        return payload.get("status", "unknown")
    result = ssh(host, f"ps -p {int(pid)} >/dev/null 2>&1", check=False)
    return "running" if result.returncode == 0 else "finished"


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh job status from remote hosts.")
    parser.add_argument("job", nargs="?", help="Optional job id")
    args = parser.parse_args()

    targets = []
    if args.job:
        targets = [JOBS_DIR / args.job]
    else:
        targets = sorted([path for path in JOBS_DIR.iterdir() if path.is_dir()]) if JOBS_DIR.exists() else []

    results = []
    for job_dir in targets:
        try:
            payload = load_job(job_dir)
            status = check_status(payload)
            payload["status"] = status
            payload["checked_at"] = now_iso()
            save_job(job_dir, payload)
            (job_dir / "status").write_text(status + "\n")
            lock_file = payload.get("lock_file")
            if status == "running":
                refresh_gpu_lock(lock_file, state="running", worker_pid=payload.get("remote_pid"), worker_host=payload.get("host"))
            elif status in {"finished", "failed"}:
                release_gpu_lock(lock_file, final_state=status)
            results.append({"job_id": payload.get("job_id", job_dir.name), "status": status, "job_dir": str(job_dir)})
        except Exception as exc:  # noqa: BLE001
            results.append({"job_id": job_dir.name, "error": str(exc)})

    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
