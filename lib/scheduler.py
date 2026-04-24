from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .gpu_utils import DEFAULT_REMOTE_PREFERRED_MIN_FREE_MEM_MB, has_preferred_local_gpu, local_host, pick_best_gpu, pick_candidate_gpus, refresh_gpu_lock, release_gpu_lock
from .notify import notify_completion, notify_failure
from .status_io import load_status, now_iso, touch_heartbeat, update_state

BASE_DIR = Path(__file__).resolve().parent.parent
SUBMIT_EXPERIMENT = BASE_DIR / "submit_experiment.py"
JOB_STATUS = BASE_DIR / "job_status.py"
CC_JOB_WAIT = BASE_DIR / "cc_job_wait"


@dataclass
class JobInfo:
    job_id: str
    mode: str
    host: str
    pid: int | None
    remote_job_id: str
    gpu_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "mode": self.mode,
            "host": self.host,
            "pid": self.pid,
            "remote_job_id": self.remote_job_id,
            "gpu_ids": self.gpu_ids,
        }


def _run(command: list[str], cwd: str | None = None, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "command failed").strip())
    return result


def ensure_conda_command() -> str:
    candidates = [
        shutil.which("conda"),
        str(Path.home() / "mambaforge" / "bin" / "conda"),
        str(Path.home() / "miniconda3" / "bin" / "conda"),
        str(Path.home() / "anaconda3" / "bin" / "conda"),
        "/opt/conda/bin/conda",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    result = subprocess.run(["bash", "-lc", "command -v conda"], capture_output=True, text=True)
    path = result.stdout.strip()
    if path:
        return path
    raise RuntimeError("conda not found")


def validate_env(conda_env: str, conda_prefix: str) -> None:
    if conda_prefix:
        if not Path(conda_prefix).exists():
            raise RuntimeError(f"conda prefix does not exist: {conda_prefix}")
        return
    if not conda_env:
        raise RuntimeError("one of --conda-env or --conda-prefix is required")
    conda = ensure_conda_command()
    result = _run([conda, "env", "list", "--json"], check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "failed to list conda envs").strip())
    payload = json.loads(result.stdout)
    envs = payload.get("envs", [])
    names = {Path(env).name for env in envs}
    if conda_env not in names:
        raise RuntimeError(f"conda env not found: {conda_env}")


def pick_execution_mode(min_free_mem_mb: int = 1024, max_util: int = 10) -> tuple[str, int | None]:
    gpu_id = pick_best_gpu(min_free_mem_mb=min_free_mem_mb, max_util=max_util)
    if gpu_id is None:
        return "remote", None
    return "local", gpu_id


def pick_local_candidate_gpus(min_free_mem_mb: int = 1024, max_util: int = 10) -> list[int]:
    return pick_candidate_gpus(min_free_mem_mb=min_free_mem_mb, max_util=max_util)


def _build_conda_command(conda_env: str, conda_prefix: str, command: list[str]) -> list[str]:
    conda = ensure_conda_command()
    if conda_prefix:
        return [conda, "run", "-p", conda_prefix, *command]
    return [conda, "run", "-n", conda_env, *command]


def _extract_metrics_from_log(job_dir: Path) -> dict[str, Any] | None:
    status = load_status(job_dir)
    log_path = Path(status["log_path"])
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9_./-]{1,63})\s*[:=]\s*(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    metrics: dict[str, float | int] = {}
    for match in pattern.finditer(text):
        key = match.group("key")
        raw = match.group("value")
        if any(ch in raw for ch in ".eE"):
            value: float | int = float(raw)
        else:
            value = int(raw)
        metrics[key] = value
    if not metrics:
        return None
    primary_name = None
    for candidate in ["val_accuracy", "accuracy", "acc", "f1", "loss", "val_loss"]:
        if candidate in metrics:
            primary_name = candidate
            break
    if primary_name is None:
        primary_name = next(iter(metrics))
    primary_value = metrics[primary_name]
    payload = {
        "job_id": status["job_id"],
        "task_type": status["task_type"],
        "primary_metric": {
            "name": primary_name,
            "value": primary_value,
            "higher_is_better": "loss" not in primary_name.lower(),
        },
        "metrics": metrics,
        "artifacts": {},
    }
    metrics_path = Path(status["metrics_path"])
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def submit_local(job_dir: Path, command: list[str], conda_env: str, conda_prefix: str, gpu_id: int, gpu_lock_path: str = "") -> JobInfo:
    log_path = job_dir / "train.log"
    exit_code_path = job_dir / "exit_code"
    status = load_status(job_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    wrapped = _build_conda_command(conda_env, conda_prefix, command)
    launcher = [
        sys.executable,
        "-c",
        (
            "import subprocess,sys,time,pathlib; "
            "cmd = sys.argv[1:]; "
            f"log = open({str(log_path)!r}, 'a', encoding='utf-8'); "
            f"pathlib.Path({str(job_dir / 'started_at')!r}).write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z')); "
            "p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True); "
            f"open({str(exit_code_path)!r}, 'w', encoding='utf-8').write(str(p.returncode)); "
            "sys.exit(p.returncode)"
        ),
        *wrapped,
    ]
    process = subprocess.Popen(launcher, cwd=status["work_dir"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    refresh_gpu_lock(gpu_lock_path, state="running", owner_pid=os.getpid(), worker_pid=process.pid, worker_host=local_host())
    update_state(job_dir, state="running", started_at=now_iso(), pid=process.pid, host=local_host(), mode="local", gpu_ids=[gpu_id], gpu_lock_path=gpu_lock_path)
    subprocess.Popen([sys.executable, str(CC_JOB_WAIT), status["job_id"], "--monitor-local", "--poll-interval", "5"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return JobInfo(job_id=status["job_id"], mode="local", host=local_host(), pid=process.pid, remote_job_id="", gpu_ids=[gpu_id])


def submit_remote(job_dir: Path, command: list[str], conda_env: str, conda_prefix: str, task_name: str) -> JobInfo:
    status = load_status(job_dir)
    cmd_str = shlex.join(command)
    args = [sys.executable, str(SUBMIT_EXPERIMENT), "--workdir", status["work_dir"], "--cmd", cmd_str, "--name", task_name]
    if conda_prefix:
        args.extend(["--conda-activate", f"conda run -p {shlex.quote(conda_prefix)} true"])
    elif conda_env:
        args.extend(["--conda-activate", f"conda run -n {shlex.quote(conda_env)} true"])
    env = os.environ.copy()
    env["MULTI_TASK_CONDA_BIN"] = ensure_conda_command()
    result = _run(args, env=env)
    payload = json.loads(result.stdout)
    host = payload.get("host", "")
    remote_job_id = payload.get("job_id", "")
    update_state(job_dir, state="queued", host=host, mode="remote", remote_job_id=remote_job_id, gpu_ids=[payload.get("gpu_index")] if payload.get("gpu_index") is not None else [], gpu_lock_path=payload.get("lock_file", ""))
    remote_meta = {
        "remote_submit_payload": payload,
        "submitted_at": now_iso(),
    }
    (job_dir / "remote_submit.json").write_text(json.dumps(remote_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    subprocess.Popen([sys.executable, str(CC_JOB_WAIT), status["job_id"], "--monitor-remote", "--poll-interval", "15"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return JobInfo(job_id=status["job_id"], mode="remote", host=host, pid=None, remote_job_id=remote_job_id, gpu_ids=[payload.get("gpu_index")] if payload.get("gpu_index") is not None else [])


def query_remote_status(remote_job_id: str) -> dict[str, Any]:
    result = _run([sys.executable, str(JOB_STATUS), remote_job_id], check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "remote status query failed").strip())
    payload = json.loads(result.stdout)
    if isinstance(payload, list):
        if not payload:
            return {}
        return payload[0]
    return payload


def cancel_remote_job(remote_job_id: str) -> None:
    raise NotImplementedError("cancel_remote_job adapter is not implemented yet")


def sync_remote_to_local(job_dir: Path) -> dict[str, Any]:
    status = load_status(job_dir)
    remote_job_id = status.get("remote_job_id", "")
    if not remote_job_id:
        return status
    remote_dir = BASE_DIR / "jobs" / remote_job_id
    if not remote_dir.exists():
        return status

    remote_status_path = remote_dir / "status"
    remote_stdout = remote_dir / "stdout.log"
    remote_stderr = remote_dir / "stderr.log"
    remote_exit_code = remote_dir / "exit_code"
    remote_started_at = remote_dir / "started_at"

    if remote_stdout.exists():
        (job_dir / "train.log").write_text(remote_stdout.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    elif remote_stderr.exists():
        (job_dir / "train.log").write_text(remote_stderr.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    if remote_started_at.exists() and not status.get("started_at"):
        raw = remote_started_at.read_text(encoding="utf-8").strip()
        if raw:
            if len(raw) >= 24 and raw[-5:].isdigit():
                raw = raw[:-5] + raw[-5:-2] + ":" + raw[-2:]
            update_state(job_dir, state=status["state"], started_at=raw)
            status = load_status(job_dir)

    remote_state = remote_status_path.read_text(encoding="utf-8").strip() if remote_status_path.exists() else "queued"
    stderr_text = remote_stderr.read_text(encoding="utf-8", errors="replace") if remote_stderr.exists() else ""
    if status.get("gpu_lock_path"):
        refresh_gpu_lock(status.get("gpu_lock_path"), state="running" if remote_state in {"queued", "running"} else remote_state)

    if remote_state == "running":
        touch_heartbeat(job_dir)
        return load_status(job_dir)

    if remote_state == "finished":
        exit_code = 0
        if remote_exit_code.exists():
            try:
                exit_code = int(remote_exit_code.read_text(encoding="utf-8").strip())
            except ValueError:
                exit_code = 0
        update_state(job_dir, state="succeeded", ended_at=now_iso(), exit_code=exit_code, error_message="")
    elif remote_state == "failed":
        exit_code = 1
        if remote_exit_code.exists():
            try:
                exit_code = int(remote_exit_code.read_text(encoding="utf-8").strip())
            except ValueError:
                exit_code = 1
        error_message = stderr_text.strip().splitlines()[-1] if stderr_text.strip() else status.get("error_message", "remote job failed")
        update_state(job_dir, state="failed", ended_at=now_iso(), exit_code=exit_code, error_message=error_message)
    else:
        touch_heartbeat(job_dir)

    return load_status(job_dir)


def build_summary(job_dir: Path) -> Path:
    status = load_status(job_dir)
    metrics_path = Path(status["metrics_path"])
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
    elif status.get("state") == "succeeded":
        extracted = _extract_metrics_from_log(job_dir)
        if extracted:
            metrics = extracted
    status = load_status(job_dir)
    log_path = Path(status["log_path"])
    tail_text = ""
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail_text = "\n".join(lines[-10:])
    duration = ""
    started_at = status.get("started_at")
    ended_at = status.get("ended_at")
    if started_at and ended_at:
        try:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(ended_at)
            duration = str(end_dt - start_dt)
        except Exception:
            duration = ""
    key_metrics = []
    primary = metrics.get("primary_metric") if isinstance(metrics, dict) else None
    primary_name = None
    if isinstance(primary, dict):
        primary_name = primary.get("name")
        key_metrics.append(f"- {primary_name or ''}: {primary.get('value', '')}")
    metric_items = metrics.get("metrics") if isinstance(metrics, dict) else None
    if isinstance(metric_items, dict):
        for key, value in list(metric_items.items())[:8]:
            if primary_name and key == primary_name:
                continue
            key_metrics.append(f"- {key}: {value}")
    if not key_metrics:
        key_metrics.append("- (none)")
    result_bullets = [f"- final_state: {status.get('state', '')}"]
    if status.get("state") == "succeeded":
        result_bullets.extend([
            "- 任务已正常结束。",
            "- 如有结构化指标，优先以 metrics.json 为准。",
            "- 请继续核对产出文件是否完整。",
        ])
    else:
        result_bullets.extend([
            f"- 失败原因: {status.get('error_message', '') or '请查看日志末尾'}",
            "- 建议先检查环境参数、命令参数和日志末尾错误。",
            "- 修复后可复用同一命令重新提交。",
        ])
    output_files = [
        f"- log: {status.get('log_path', '')}",
        f"- metrics: {status.get('metrics_path', '')}",
    ]
    if isinstance(metrics.get("artifacts"), dict):
        for key, value in metrics["artifacts"].items():
            output_files.append(f"- {key}: {value}")
    lines = [
        "# Experiment Summary",
        "",
        "## Job",
        f"- job_id: {status.get('job_id', '')}",
        f"- name: {status.get('name', '')}",
        f"- task_type: {status.get('task_type', '')}",
        f"- mode: {status.get('mode', '')}",
        f"- host: {status.get('host', '')}",
        f"- gpu_ids: {status.get('gpu_ids', [])}",
        f"- work_dir: {status.get('work_dir', '')}",
        f"- env_kind: {status.get('env_kind', '')}",
        f"- conda_env: {status.get('conda_env', '')}",
        f"- conda_prefix: {status.get('conda_prefix', '')}",
        f"- started_at: {started_at}",
        f"- ended_at: {ended_at}",
        f"- duration: {duration}",
        f"- exit_code: {status.get('exit_code')}",
        f"- final_state: {status.get('state', '')}",
        "",
        "## Command",
        status.get("command", ""),
        "",
        "## Result",
        *result_bullets[:8],
        "",
        "## Key Metrics",
        *key_metrics,
        "",
        "## Output Files",
        *output_files,
        "",
        "## Next Suggestions",
        "1. 先读取 summary.md 和 metrics.json，再决定是否继续下一轮实验。",
        "2. 如果失败，优先查看 train.log 末尾错误并确认环境参数是否正确。",
        "3. 如果成功，补充结构化指标输出以便 Claude 自动总结。",
    ]
    if tail_text:
        lines.extend(["", "## Log Tail", "```", tail_text, "```"])
    summary_path = Path(status["summary_path"])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def finalize_job(job_dir: Path) -> dict[str, Any]:
    status = load_status(job_dir)
    release_gpu_lock(status.get("gpu_lock_path"), final_state=status.get("state"))
    build_summary(job_dir)
    status = load_status(job_dir)
    if status.get("state") == "succeeded":
        notify_completion(status)
    elif status.get("state") in {"failed", "canceled"}:
        notify_failure(status)
    return status
