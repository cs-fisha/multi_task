# multi_task

A lightweight experiment launcher for multi-GPU, multi-host research workflows.

`multi_task` provides a single entrypoint for running training, evaluation, and inference jobs across local and remote GPU machines. It is designed for researchers or individual developers who want a simple, file-based workflow instead of a full cluster scheduler.

The core idea is straightforward:

1. submit every job through one command
2. prefer an idle GPU on the current machine
3. fall back to a remote machine when needed
4. record status, logs, summary, and metrics in a stable job directory
5. let humans or Claude Code read those artifacts afterward

---

## Why this project exists

When experiments are launched ad hoc, it becomes hard to answer basic questions:

- Where did the job run?
- Is it still running?
- What command was actually executed?
- Did it succeed or fail?
- What was the final metric?

`multi_task` solves that by wrapping job submission in a thin orchestration layer. It does not try to be Kubernetes, Slurm, or Ray. Instead, it keeps the workflow simple:

- a small set of CLI entrypoints
- plain files as the source of truth
- optional SSH-based remote dispatch
- minimal third-party dependencies

---

## Main features

- One unified entrypoint for `train`, `eval`, and `infer`
- Local GPU preferred before remote dispatch
- File-based job state tracking
- Stable, machine-readable stdout protocol after submission
- Per-job output directory with logs and metadata
- Automatic `summary.md` generation
- Optional `metrics.json` extraction from logs
- Optional notification integration
- Lightweight SSH bootstrap helper for passwordless access

---

## How it works

At a high level, a job goes through this flow:

```text
User / Claude Code
        |
        v
cc_run_experiment
        |
        +--> inspect local GPU availability
        |
        +--> local path: launch process on current machine
        |
        \--> remote path: submit through SSH-based adapter
                         to another machine

Both paths write to:
jobs/<job_id>/
    - meta.json
    - status.json
    - command.sh
    - train.log
    - summary.md
    - metrics.json (optional)
```

The result is that local and remote jobs look the same from the outside: they always produce a local job directory that can be inspected later.

---

## Repository structure

```text
multi_task/
├── cc_run_experiment          # main unified launcher
├── cc_job_status              # print the current status of a job
├── cc_job_wait                # wait until a job reaches a terminal state
├── cc_job_notify              # send a completion/failure notification
├── gpu_probe.py               # inspect local GPU state via nvidia-smi
├── list_idle.py               # list idle GPUs across configured hosts
├── submit_experiment.py       # remote submission adapter
├── job_status.py              # refresh status for remote jobs
├── remote_launch.sh           # shell wrapper used on remote workers
├── setup_passwordless_ssh.py  # generic SSH bootstrap helper
├── lib/
│   ├── __init__.py
│   ├── gpu_utils.py           # GPU discovery, filtering, lease/lock handling
│   ├── notify.py              # notification integration
│   ├── scheduler.py           # local/remote dispatch and summary orchestration
│   └── status_io.py           # status/meta read-write helpers
└── jobs/                      # runtime output, ignored by git
```

---

## Key files explained

### `cc_run_experiment`
The main entrypoint. It:

- validates arguments
- validates the conda environment
- creates a job directory
- picks local vs remote execution
- starts the job
- emits stable key-value output for downstream parsing

### `cc_job_status`
Reads `jobs/<job_id>/status.json` and prints a compact JSON summary.

### `cc_job_wait`
Polls a job until it finishes, then prints the summary and metrics paths.

### `cc_job_notify`
Sends a notification after a job succeeds or fails.

### `gpu_probe.py`
Uses `nvidia-smi` to inspect GPU memory and utilization on the current machine.

### `list_idle.py`
Uses SSH plus `gpu_probe.py` to inspect multiple hosts and show which GPUs are idle.

### `submit_experiment.py`
Handles remote submission and compatibility with the existing SSH-based remote execution path.

### `job_status.py`
Refreshes remote job state and synchronizes it back to the local job record.

### `remote_launch.sh`
Shell wrapper executed on a remote worker. It prepares environment activation, writes status files, and captures exit codes.

### `setup_passwordless_ssh.py`
A generic helper that installs an SSH public key onto a set of hosts and optionally appends an SSH config block.

### `lib/gpu_utils.py`
Responsible for:

- querying local GPUs
- applying GPU filtering rules
- choosing candidates
- maintaining lease/lock files so concurrent submissions do not grab the same GPU

### `lib/status_io.py`
Responsible for:

- atomic status writes
- metadata writes
- state updates
- generating stable key-value submission output

### `lib/scheduler.py`
Responsible for:

- local submission
- remote submission
- remote-to-local status synchronization
- metric extraction
- summary generation

### `lib/notify.py`
Wraps the optional notification backend.

---

## Installation assumptions

This repository is intentionally lightweight and expects your environment to provide the basics.

Typical assumptions are:

- Linux-based machines
- Python 3
- `nvidia-smi` available on worker machines
- Conda or compatible environment manager
- SSH access between machines
- a shared or otherwise coordinated filesystem layout for code and job artifacts

This project is not plug-and-play for every infrastructure setup. You should expect to adapt host inventory, SSH behavior, and directory conventions to your own environment.

---

## Quick start

### 1. Submit a training job

```bash
./cc_run_experiment \
  --cwd "$PWD" \
  --task-type train \
  --name "demo-train" \
  --conda-env your-env \
  --async \
  -- python train.py --config configs/foo.yaml
```

### 2. Submit using a conda prefix instead of an env name

```bash
./cc_run_experiment \
  --cwd "$PWD" \
  --task-type eval \
  --name "demo-eval" \
  --conda-prefix /path/to/conda/env \
  --async \
  -- python eval.py --config configs/foo.yaml
```

### 3. Query status

```bash
./cc_job_status <job_id>
```

### 4. Wait for completion

```bash
./cc_job_wait <job_id> --timeout 3600 --poll-interval 15
```

### 5. Send a manual notification

```bash
./cc_job_notify <job_id>
```

---

## Submission output contract

After a submission, `cc_run_experiment` prints stable key-value lines such as:

```text
JOB_ID=exp_20260421_153012_ab12cd
STATE=queued
MODE=local
HOST=my-machine
WORK_DIR=/path/to/repo
JOB_DIR=/path/to/multi_task/jobs/exp_20260421_153012_ab12cd
STATUS_PATH=/path/to/multi_task/jobs/exp_20260421_153012_ab12cd/status.json
LOG_PATH=/path/to/multi_task/jobs/exp_20260421_153012_ab12cd/train.log
SUMMARY_PATH=/path/to/multi_task/jobs/exp_20260421_153012_ab12cd/summary.md
METRICS_PATH=/path/to/multi_task/jobs/exp_20260421_153012_ab12cd/metrics.json
PID=12345
REMOTE_JOB_ID=
ENV_KIND=conda
CONDA_ENV=your-env
CONDA_PREFIX=
PYTHON_BIN=
```

This makes it easy for other tools, wrappers, or agents to parse job creation results.

---

## Job artifacts

Each job writes to `jobs/<job_id>/`.

Common files include:

- `meta.json` — metadata about the declared task
- `status.json` — current state, paths, environment info, and execution mode
- `command.sh` — the business command that was launched
- `train.log` — captured stdout/stderr log stream
- `summary.md` — generated human-readable summary
- `metrics.json` — structured metrics, if available or extracted

This directory is the main debugging surface for both humans and automation.

---

## Metrics extraction

If your training script does not write `metrics.json`, `multi_task` can extract simple scalar values from logs.

Examples:

```text
val_accuracy=0.9123
train_loss: 0.321
best_epoch=4
```

The extracted metrics are written to `metrics.json`, and a primary metric is selected automatically when possible.

---

## SSH setup helper

The repository includes `setup_passwordless_ssh.py` as a generic helper for installing public keys on multiple hosts.

Typical usage pattern:

```bash
python3 setup_passwordless_ssh.py \
  --hosts-file hosts.txt \
  --user your-ssh-user \
  --host-pattern 'gpu-*'
```

This helper is intentionally generic. You should review and adapt it before using it in a production or shared environment.

---

## What is intentionally not included in this public version

This repository does **not** include:

- private host inventories
- historical jobs
- runtime lock files
- internal handoff notes
- private experiment logs
- private machine addresses or credentials

---

## Limitations

- The project assumes a Linux/SSH/Conda style workflow
- Remote execution is adapter-based, not a full scheduler abstraction
- Some infrastructure behavior is environment-specific and may require customization
- It is optimized for simple research workflows, not for multi-tenant production clusters

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

---

## Disclaimer

This repository is provided as a lightweight research workflow utility, not as a hardened production orchestration system.

You are responsible for reviewing and adapting:

- SSH configuration
- host inventory management
- filesystem layout
- credential handling
- environment activation
- access control and operational safety

Use it only in environments where you understand and accept those tradeoffs.
