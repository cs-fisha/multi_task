#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HOSTS_FILE = BASE_DIR / "hosts.txt"
GPU_PROBE = BASE_DIR / "gpu_probe.py"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "command failed")
    return result


def ssh(host: str, remote_command: str) -> subprocess.CompletedProcess[str]:
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            host,
            remote_command,
        ]
    )


def load_hosts() -> list[str]:
    return [line.strip() for line in HOSTS_FILE.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="List currently idle GPUs across hosts.")
    parser.add_argument("--min-free-mem-gb", type=float, default=8.0)
    parser.add_argument("--max-util", type=int, default=10)
    parser.add_argument("--allow-proc", action="store_true")
    args = parser.parse_args()

    require_no_proc = not args.allow_proc
    results = []
    for host in load_hosts():
        try:
            payload = json.loads(ssh(host, f"python3 {GPU_PROBE}").stdout)
            idle = []
            for gpu in payload["gpus"]:
                if require_no_proc and gpu["process_count"] > 0:
                    continue
                if gpu["memory_free_mb"] < int(args.min_free_mem_gb * 1024):
                    continue
                if gpu["utilization_gpu"] > args.max_util:
                    continue
                idle.append(gpu)
            results.append({"host": host, "idle_gpus": idle})
        except Exception as exc:  # noqa: BLE001
            results.append({"host": host, "error": str(exc)})

    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
