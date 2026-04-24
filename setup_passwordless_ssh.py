#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import subprocess
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_KEY_NAME = "multi_task_ed25519"
DEFAULT_HOSTS_FILE = "hosts.txt"


def run(command: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"timeout after {timeout}s: {' '.join(command)}") from exc
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"command failed: {' '.join(command)}")
    return result


def shell_single_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def load_hosts(hosts_file: Path) -> list[str]:
    if not hosts_file.exists():
        raise FileNotFoundError(f"hosts file not found: {hosts_file}")
    hosts = [line.strip() for line in hosts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not hosts:
        raise RuntimeError(f"hosts file is empty: {hosts_file}")
    return hosts


def ensure_key(key_path: Path, comment: str) -> tuple[Path, Path]:
    pub_path = key_path.with_suffix(".pub")
    if key_path.exists() and pub_path.exists():
        return key_path, pub_path
    key_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ssh-keygen",
        "-t",
        "ed25519",
        "-f",
        str(key_path),
        "-N",
        "",
        "-C",
        comment,
    ])
    return key_path, pub_path


def add_known_host(host: str, known_hosts: Path, timeout_seconds: int) -> None:
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    scan = run(["ssh-keyscan", "-T", str(timeout_seconds), "-H", host], check=False, timeout=timeout_seconds + 2)
    if scan.returncode != 0 or not scan.stdout.strip():
        return
    existing = known_hosts.read_text(encoding="utf-8") if known_hosts.exists() else ""
    if scan.stdout.strip() in existing:
        return
    with known_hosts.open("a", encoding="utf-8") as handle:
        handle.write(scan.stdout)


def expect_script(host: str, user: str, password: str, public_key: str, timeout_seconds: int) -> str:
    password_escaped = password.replace("\\", "\\\\").replace('"', '\\"')
    key_quoted = shell_single_quote(public_key)
    remote_command = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && "
        f"grep -qxF {key_quoted} ~/.ssh/authorized_keys || printf '%s\\n' {key_quoted} >> ~/.ssh/authorized_keys"
    )
    return f'''set timeout {timeout_seconds}
log_user 0
spawn ssh -o ConnectTimeout={timeout_seconds} -o StrictHostKeyChecking=accept-new {user}@{host} "{remote_command}"
expect {{
    -re "yes/no" {{ send "yes\\r"; exp_continue }}
    -re "[Pp]assword:" {{ send "{password_escaped}\\r" }}
    timeout {{ exit 124 }}
    eof
}}
expect eof
catch wait result
set code [lindex $result 3]
exit $code
'''


def install_key(host: str, user: str, password: str, public_key: str, known_hosts: Path, timeout_seconds: int) -> tuple[bool, str]:
    try:
        add_known_host(host, known_hosts, timeout_seconds)
    except RuntimeError as exc:
        return False, str(exc)
    script = expect_script(host, user, password, public_key, timeout_seconds)
    try:
        result = run(["expect", "-c", script], check=False, timeout=timeout_seconds + 2)
    except RuntimeError as exc:
        return False, str(exc)
    if result.returncode != 0:
        if result.returncode == 124:
            return False, f"timeout after {timeout_seconds}s"
        return False, result.stderr.strip() or result.stdout.strip() or f"failed on {host}"
    return True, "ok"


def verify_host(host: str, user: str, key_path: Path, timeout_seconds: int) -> tuple[bool, str]:
    try:
        result = run([
            "ssh",
            "-i",
            str(key_path),
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout_seconds}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"{user}@{host}",
            "true",
        ], check=False, timeout=timeout_seconds + 2)
    except RuntimeError as exc:
        return False, str(exc)
    if result.returncode != 0:
        return False, result.stderr.strip() or result.stdout.strip() or "verification failed"
    return True, "ok"


def append_ssh_config(host_pattern: str, user: str, key_path: Path, timeout_seconds: int, config_path: Path) -> None:
    block = f"""
Host {host_pattern}
  User {user}
  IdentityFile {key_path}
  IdentitiesOnly yes
  ConnectTimeout {timeout_seconds}
  StrictHostKeyChecking accept-new
""".strip() + "\n"
    existing = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    if block in existing:
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("a", encoding="utf-8") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write(block)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up passwordless SSH access for a group of hosts.")
    parser.add_argument("--hosts-file", default=DEFAULT_HOSTS_FILE, help="Path to a file containing one host per line")
    parser.add_argument("--user", default=getpass.getuser(), help="Remote SSH username")
    parser.add_argument("--password", help="Password for the remote SSH user")
    parser.add_argument("--key-path", default=f"~/.ssh/{DEFAULT_KEY_NAME}", help="Private key path to create or reuse")
    parser.add_argument("--key-comment", default="cluster-entry@multi_task", help="Comment for a newly generated SSH key")
    parser.add_argument("--known-hosts", default="~/.ssh/known_hosts", help="Known hosts file path")
    parser.add_argument("--ssh-config", default="~/.ssh/config", help="SSH config file path")
    parser.add_argument("--host-pattern", default="*", help="Host pattern to append to SSH config, e.g. 'gpu-*' or '10.0.0.*'")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="SSH and keyscan timeout in seconds")
    parser.add_argument("--skip-config", action="store_true", help="Do not append an SSH config block")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hosts_file = Path(args.hosts_file).expanduser().resolve()
    key_path = Path(args.key_path).expanduser()
    known_hosts = Path(args.known_hosts).expanduser()
    ssh_config = Path(args.ssh_config).expanduser()

    password = args.password or getpass.getpass(f"Password for {args.user}: ")
    hosts = load_hosts(hosts_file)
    key_path, pub_path = ensure_key(key_path, args.key_comment)
    public_key = pub_path.read_text(encoding="utf-8").strip()

    install_results = []
    for host in hosts:
        ok, message = install_key(host, args.user, password, public_key, known_hosts, args.timeout_seconds)
        install_results.append({"host": host, "installed": ok, "message": message})

    if not args.skip_config:
        append_ssh_config(args.host_pattern, args.user, key_path, args.timeout_seconds, ssh_config)

    verify_results = []
    for host in hosts:
        ok, message = verify_host(host, args.user, key_path, args.timeout_seconds)
        verify_results.append({"host": host, "verified": ok, "message": message})

    success = all(item["verified"] or "timeout after" in item["message"] for item in verify_results)
    print(json.dumps({
        "hosts_file": str(hosts_file),
        "key_path": str(key_path),
        "known_hosts": str(known_hosts),
        "ssh_config": "skipped" if args.skip_config else str(ssh_config),
        "install": install_results,
        "verify": verify_results,
    }, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
