#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict

GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]
APP_QUERY = [
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,used_memory",
    "--format=csv,noheader,nounits",
]


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"command failed: {' '.join(command)}")
    return result.stdout.strip()


def parse_csv_lines(raw: str, expected_fields: int) -> list[list[str]]:
    rows: list[list[str]] = []
    if not raw:
        return rows
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < expected_fields:
            continue
        rows.append(parts[:expected_fields])
    return rows


def build_probe() -> dict:
    gpu_rows = parse_csv_lines(run_command(GPU_QUERY), 5)
    app_rows = parse_csv_lines(run_command(APP_QUERY), 3)

    apps_by_uuid: dict[str, list[dict]] = defaultdict(list)
    for gpu_uuid, pid, used_memory_mb in app_rows:
        apps_by_uuid[gpu_uuid].append(
            {
                "pid": int(pid),
                "used_memory_mb": int(used_memory_mb),
            }
        )

    gpus = []
    for index, uuid, memory_total_mb, memory_used_mb, utilization_gpu in gpu_rows:
        processes = apps_by_uuid.get(uuid, [])
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_total_mb": int(memory_total_mb),
                "memory_used_mb": int(memory_used_mb),
                "memory_free_mb": int(memory_total_mb) - int(memory_used_mb),
                "utilization_gpu": int(utilization_gpu),
                "process_count": len(processes),
                "processes": processes,
            }
        )

    return {"gpus": gpus}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe local GPU state via nvidia-smi.")
    parser.parse_args()

    try:
        payload = build_probe()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
