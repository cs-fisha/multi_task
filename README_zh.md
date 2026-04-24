# multi_task

English version: [README.md](README.md)

一个面向多 GPU、多机器研究工作流的轻量实验启动与状态跟踪工具。

`multi_task` 提供了一个统一入口，用来在本机和远程 GPU 机器上运行训练、评测和推理任务。它更适合研究者或个人开发者：不想引入完整的集群调度系统，但又希望实验提交、状态记录、日志管理和结果追踪都更规范、更稳定。

这个项目的核心思路很简单：

1. 所有任务统一从一个命令提交
2. 优先使用当前机器上的空闲 GPU
3. 当前机器资源不足时回退到远程机器
4. 把状态、日志、summary 和 metrics 固定写到 job 目录
5. 让人或 Claude Code 都能在事后读取这些产物

---

## 这个项目解决什么问题

实验如果是临时直接跑起来的，后面很容易遇到这些问题：

- 这个任务到底跑在哪台机器上？
- 现在还在跑吗？
- 当时真正执行的命令是什么？
- 最终是成功了还是失败了？
- 指标结果是多少？

`multi_task` 的做法不是去替代 Slurm、Kubernetes 或 Ray，而是在现有研究环境上增加一层很薄的调度与记录层，让实验工作流保持简单：

- CLI 入口少且稳定
- 以普通文件作为状态真相来源
- 可选使用 SSH 做远程分发
- 尽量少依赖第三方库

---

## 主要特性

- 一个统一入口，覆盖 `train` / `eval` / `infer`
- 优先本机 GPU，不够时再远程分发
- 基于文件的任务状态跟踪
- 提交后输出稳定、机器可解析的键值协议
- 每个任务都有独立输出目录，保存日志和元信息
- 自动生成 `summary.md`
- 可选从日志中提取 `metrics.json`
- 可选接入通知能力
- 提供轻量的 SSH 免密初始化脚本

---

## 整体架构

一个任务的大致流程如下：

```text
用户 / Claude Code
        |
        v
cc_run_experiment
        |
        +--> 检查本机 GPU 可用性
        |
        +--> 本地路径：在当前机器启动任务
        |
        \--> 远程路径：通过 SSH 适配层提交到其他机器

两条路径最终都会把结果写到：
jobs/<job_id>/
    - meta.json
    - status.json
    - command.sh
    - train.log
    - summary.md
    - metrics.json（可选）
```

也就是说，不管任务最终是在本机运行还是远程运行，从外部看都会得到一个统一结构的本地 job 目录，方便后续调试、总结和自动化读取。

---

## 仓库结构

```text
multi_task/
├── cc_run_experiment          # 主入口：统一提交 train/eval/infer 任务
├── cc_job_status              # 查询任务当前状态
├── cc_job_wait                # 等待任务结束
├── cc_job_notify              # 发送任务完成/失败通知
├── gpu_probe.py               # 通过 nvidia-smi 检查本机 GPU 状态
├── list_idle.py               # 查看配置主机上的空闲 GPU
├── submit_experiment.py       # 远程提交适配层
├── job_status.py              # 刷新远程任务状态
├── remote_launch.sh           # 远程机器上的执行包装脚本
├── setup_passwordless_ssh.py  # 通用 SSH 免密初始化脚本
├── lib/
│   ├── __init__.py
│   ├── gpu_utils.py           # GPU 查询、过滤、锁/lease 管理
│   ├── notify.py              # 通知后端封装
│   ├── scheduler.py           # 本地/远程调度、summary、metrics 协调
│   └── status_io.py           # 状态与元信息的读写工具
└── jobs/                      # 运行期输出目录，默认不纳入 git
```

---

## 各核心文件作用说明

### `cc_run_experiment`
这是主入口，负责：

- 校验命令行参数
- 校验 conda 环境
- 创建 job 目录
- 选择本地执行还是远程执行
- 启动任务
- 输出稳定的键值对结果，方便后续脚本或 Claude Code 解析

### `cc_job_status`
读取 `jobs/<job_id>/status.json`，输出一个简洁的 JSON 状态摘要。

### `cc_job_wait`
轮询任务，直到任务进入终态，然后输出 summary 和 metrics 的路径。

### `cc_job_notify`
在任务成功或失败后发送通知。

### `gpu_probe.py`
通过 `nvidia-smi` 查询当前机器 GPU 的显存占用和利用率。

### `list_idle.py`
通过 SSH 在多个主机上调用 `gpu_probe.py`，列出空闲 GPU。

### `submit_experiment.py`
负责远程提交逻辑，以及与已有 SSH 远程执行链路做兼容。

### `job_status.py`
刷新远程任务状态，并把结果同步回本地 job 记录。

### `remote_launch.sh`
运行在远程 worker 上的 shell 包装器，负责环境激活、状态写入和退出码记录。

### `setup_passwordless_ssh.py`
一个通用辅助脚本，用于把 SSH 公钥安装到多个主机上，并可选地追加 SSH config 配置块。

### `lib/gpu_utils.py`
负责：

- 查询本机 GPU
- 应用 GPU 过滤规则
- 选择候选 GPU
- 维护 lease/lock 文件，避免并发提交时重复抢占同一块 GPU

### `lib/status_io.py`
负责：

- 原子写入状态文件
- 写入 meta 信息
- 更新任务状态
- 生成稳定的键值协议输出

### `lib/scheduler.py`
负责：

- 本地任务提交
- 远程任务提交
- 远程状态同步回本地
- 指标提取
- summary 生成

### `lib/notify.py`
封装可选的通知后端。

---

## 环境假设

这个仓库刻意保持轻量，因此默认假设你的环境已经具备这些基础条件：

- Linux 机器
- Python 3
- worker 机器上可使用 `nvidia-smi`
- 已安装 Conda 或兼容环境管理器
- 机器之间能通过 SSH 访问
- 代码目录和 job 产物目录有共享存储，或至少有一致的路径组织方式

它不是一个对所有基础设施都开箱即用的系统。你通常需要根据自己的环境调整：

- host inventory
- SSH 行为
- 目录规范
- 环境激活方式

---

## 快速开始

### 1. 提交一个训练任务

```bash
./cc_run_experiment \
  --cwd "$PWD" \
  --task-type train \
  --name "demo-train" \
  --conda-env your-env \
  --async \
  -- python train.py --config configs/foo.yaml
```

### 2. 使用 conda prefix 提交

```bash
./cc_run_experiment \
  --cwd "$PWD" \
  --task-type eval \
  --name "demo-eval" \
  --conda-prefix /path/to/conda/env \
  --async \
  -- python eval.py --config configs/foo.yaml
```

### 3. 查询状态

```bash
./cc_job_status <job_id>
```

### 4. 等待结束

```bash
./cc_job_wait <job_id> --timeout 3600 --poll-interval 15
```

### 5. 手动发送通知

```bash
./cc_job_notify <job_id>
```

---

## 提交后的标准输出协议

`cc_run_experiment` 在提交任务后会输出稳定的键值对，例如：

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

这样其他工具、包装脚本或 Claude Code 就可以稳定解析任务启动结果。

---

## Job 产物说明

每个任务都会在 `jobs/<job_id>/` 下写出文件。

常见文件包括：

- `meta.json`：任务的声明信息和元数据
- `status.json`：当前状态、路径、环境信息和执行模式
- `command.sh`：真正被执行的业务命令
- `train.log`：stdout/stderr 日志流
- `summary.md`：自动生成的人类可读总结
- `metrics.json`：结构化指标文件

这个目录是人和自动化程序排查问题、查看结果时最重要的入口。

---

## 指标提取

如果你的训练脚本没有主动写 `metrics.json`，`multi_task` 可以从日志中提取简单的数值指标。

例如：

```text
val_accuracy=0.9123
train_loss: 0.321
best_epoch=4
```

提取后的结果会写入 `metrics.json`，并尽可能自动选择一个主指标。

---

## SSH 初始化脚本

仓库里包含 `setup_passwordless_ssh.py`，它是一个通用辅助脚本，用于给多台主机安装 SSH 公钥。

典型用法：

```bash
python3 setup_passwordless_ssh.py \
  --hosts-file hosts.txt \
  --user your-ssh-user \
  --host-pattern 'gpu-*'
```

这个脚本是通用模板，不是针对任何特定环境的硬编码版本。在生产环境或共享环境中使用前，建议你先检查并按需调整。

---

## 这个公开版本刻意不包含什么

这个仓库 **不包含** 以下内容：

- 私有 host 清单
- 历史 jobs
- 运行时 lock 文件
- 内部 handoff 笔记
- 私有实验日志
- 私有机器地址或凭据

---

## 局限性

- 这个项目默认是 Linux / SSH / Conda 风格工作流
- 远程执行仍然是适配层方案，不是完整调度系统抽象
- 某些基础设施行为与环境强相关，通常需要自行定制
- 它更适合简单研究工作流，不适合多租户生产集群

---

## License

本项目采用 MIT 协议，详见 [LICENSE](LICENSE)。

---

## 免责声明

这个仓库定位为轻量研究工作流工具，不是一个经过强化设计的生产级任务编排系统。

你需要自行评估并调整以下内容：

- SSH 配置
- host inventory 管理
- 文件系统布局
- 凭据管理
- 环境激活方式
- 访问控制与运行安全

请只在你理解并接受这些权衡的环境中使用它。
