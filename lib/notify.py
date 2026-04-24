from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .status_io import load_status


def _cc_connect_path() -> str | None:
    return shutil.which("cc-connect")


def _run_notify(message: str) -> None:
    binary = _cc_connect_path()
    if not binary:
        return
    try:
        subprocess.run([binary, "send", "--message", message], check=False, capture_output=True, text=True)
    except Exception:
        return


def _message_prefix(status: dict) -> str:
    return "实验已完成" if status.get("state") == "succeeded" else "实验失败"


def build_message(status: dict) -> str:
    lines = [
        _message_prefix(status),
        "",
        f"- job_id: {status.get('job_id', '')}",
        f"- state: {status.get('state', '')}",
        f"- mode: {status.get('mode', '')}",
        f"- host: {status.get('host', '')}",
        f"- summary: {status.get('summary_path', '')}",
    ]
    metrics_path = status.get("metrics_path", "")
    if status.get("state") == "succeeded":
        lines.append(f"- metrics: {metrics_path}")
        lines.append("")
        lines.append("请读取 summary.md 和 metrics.json 后，用中文向我总结最终结论。")
    else:
        lines.append(f"- log: {status.get('log_path', '')}")
        lines.append("")
        lines.append("请读取 summary.md 和日志后，用中文总结失败原因和下一步建议。")
    return "\n".join(lines)


def notify_completion(job_info: dict) -> None:
    _run_notify(build_message(job_info))


def notify_failure(job_info: dict) -> None:
    _run_notify(build_message(job_info))


def notify_from_job_dir(job_dir: Path) -> None:
    status = load_status(job_dir)
    if status.get("state") == "succeeded":
        notify_completion(status)
    elif status.get("state") in {"failed", "canceled"}:
        notify_failure(status)


def cli_payload(job_dir: Path) -> str:
    return json.dumps(load_status(job_dir), indent=2, ensure_ascii=False)
