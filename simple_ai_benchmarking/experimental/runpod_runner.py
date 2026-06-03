# Project Name: simple-ai-benchmarking
# File Name: runpod_runner.py
# Author: Timo Leitritz
# Copyright (C) 2024 Timo Leitritz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Experimental one-shot benchmark runner on a RunPod GPU pod.

It does the whole loop for you and *always* tears the pod down again:

    1. create an on-demand GPU pod (with SSH enabled and your public key),
    2. wait until SSH is reachable,
    3. install SAIB, run the benchmark, and publish results to the database,
    4. terminate the pod (even on error or Ctrl-C).

This is deliberately small and dependency-light. It is NOT a managed service:
capacity is best-effort, and you are billed per second while the pod runs, so
the guaranteed teardown is the whole point.

Usage (CV example)::

    export RUNPOD_API_KEY=...                 # from runpod.io account settings
    export AI_BENCHMARK_DATABASE_TOKEN=...     # database API token
    saib-runpod --gpu "NVIDIA GeForce RTX 4090"

Dry run (no API key needed, prints the plan and remote script)::

    saib-runpod --dry-run

Install the optional dependencies with::

    pip install simple-ai-benchmarking[runpod]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPUS = "NVIDIA GeForce RTX 4090"
DEFAULT_DATABASE_URL = "https://timoillusion.pythonanywhere.com"
DEFAULT_PIP_SPEC = (
    "simple-ai-benchmarking[pt]@git+"
    "https://github.com/TimoIllusion/simple-ai-benchmarking.git@main"
)
DONE_MARKER = "SAIB_RUNPOD_DONE"

# Per-workload: (run command, results CSV, publish command).
WORKLOADS = {
    "pt": ("saib-pt", "results_pt.csv", "saib-pub"),
    "tf": ("saib-tf", "results_tf.csv", "saib-pub"),
    "llm": ("saib-llm", "llm_results.csv", "saib-pub-llm"),
}


def log(message: str) -> None:
    print(f"[saib-runpod {time.strftime('%H:%M:%S')}] {message}", flush=True)


@dataclass
class Config:
    api_key: str
    db_token: Optional[str]
    gpus: List[str]
    image: str
    workload: str
    pip_spec: str
    database_url: str
    private_key_path: str
    public_key: str
    disk_gb: int
    cloud_type: str
    extra_args: str
    publish: bool
    keep: bool
    capacity_wait: int
    create_timeout: int
    ssh_timeout: int
    run_timeout: int


# --------------------------------------------------------------------------- #
# Remote script
# --------------------------------------------------------------------------- #
def build_remote_script(cfg: Config) -> str:
    """The bash run on the pod. ``set -euo pipefail`` makes any failing step
    abort with a non-zero exit code, which the orchestrator reports and which
    still triggers teardown."""
    run_cmd, results_csv, publish_cmd = WORKLOADS[cfg.workload]
    run_line = run_cmd if not cfg.extra_args else f"{run_cmd} {cfg.extra_args}"

    lines = [
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        'echo "== installing SAIB =="',
        "python -m pip install --upgrade pip",
        f'pip install "{cfg.pip_spec}"',
        'echo "== running benchmark =="',
        run_line,
    ]

    if cfg.publish:
        # Token is exported here (transmitted over the encrypted SSH channel) and
        # referenced via the variable, so it is not baked into the pip spec.
        common = (
            f'-t "$AI_BENCHMARK_DATABASE_TOKEN" '
            f'--database-url "{cfg.database_url}" --non-interactive'
        )
        lines += [
            'echo "== registering profiles =="',
            f"saib-register {results_csv} {common}",
            'echo "== publishing results =="',
            f"{publish_cmd} {results_csv} {common}",
        ]
    else:
        lines.append('echo "== publish skipped (--no-publish) =="')

    lines.append(f'echo "{DONE_MARKER}"')
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# RunPod lifecycle
# --------------------------------------------------------------------------- #
def _import_runpod():
    try:
        import runpod  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise SystemExit(
            "The 'runpod' package is required. Install with: "
            "pip install simple-ai-benchmarking[runpod]"
        ) from exc
    return runpod


def _try_create_pod(runpod, cfg: Config, gpu: str) -> Optional[dict]:
    pod = runpod.create_pod(
        name=f"saib-{cfg.workload}-{int(time.time())}",
        image_name=cfg.image,
        gpu_type_id=gpu,
        cloud_type=cfg.cloud_type,
        gpu_count=1,
        container_disk_in_gb=cfg.disk_gb,
        volume_in_gb=0,
        ports="22/tcp",
        start_ssh=True,
        support_public_ip=True,
        env={
            "PUBLIC_KEY": cfg.public_key,
            "AI_BENCHMARK_DATABASE_TOKEN": cfg.db_token or "",
        },
    )
    if pod and pod.get("id"):
        return pod
    return None


def create_pod_with_fallback(runpod, cfg: Config) -> dict:
    """Create a pod, trying each requested GPU in order.

    GPU capacity on RunPod is transient, so when ``--capacity-wait`` is set we
    keep retrying the same requested GPU list (with backoff) until a pod is
    created or the budget elapses, instead of switching to some other GPU."""
    deadline = time.time() + max(0, cfg.capacity_wait)
    last_error: Optional[Exception] = None
    attempt = 0
    while True:
        attempt += 1
        for gpu in cfg.gpus:
            log(f"Requesting pod on GPU '{gpu}' ({cfg.cloud_type})...")
            try:
                pod = _try_create_pod(runpod, cfg, gpu)
                if pod:
                    log(f"Pod created: {pod['id']}")
                    return pod
                last_error = RuntimeError(f"Empty response creating pod on '{gpu}'.")
            except Exception as exc:  # SDK raises various errors on no-capacity
                log(f"Could not create pod on '{gpu}': {exc}")
                last_error = exc

        if time.time() >= deadline:
            break
        backoff = min(30, 5 * attempt)
        remaining = int(deadline - time.time())
        log(f"No capacity yet; retrying in {backoff}s (~{remaining}s of wait budget left).")
        time.sleep(backoff)

    hint = (
        "No capacity for the requested GPU(s). Try: a longer --capacity-wait, "
        "--cloud-type COMMUNITY, or a fallback list e.g. "
        "--gpu 'NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A5000'."
    )
    raise SystemExit(f"Failed to create a pod. {hint}\nLast error: {last_error}")


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _ssh_base(key_path: str, connect_timeout: int = 20) -> List[str]:
    return [
        "ssh",
        "-i", key_path,
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=10",
        "-o", f"ConnectTimeout={connect_timeout}",
    ]


def _ssh_destination(target: dict) -> List[str]:
    """SSH argv tail for the pod's direct public TCP mapping for port 22."""
    return ["-p", str(target["port"]), f"root@{target['ip']}"]


def _direct_target_from_pod(runpod, pod_id: str) -> Optional[dict]:
    """Best-effort: a direct public-IP:22 target if the pod exposes one yet."""
    runtime = (runpod.get_pod(pod_id) or {}).get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return {"kind": "direct", "ip": port["ip"], "port": int(port["publicPort"])}
    return None


def wait_for_ssh(runpod, pod_id: str, key_path: str, timeout: int) -> dict:
    """Wait until the pod's direct public-IP:22 route runs a command, then return it.

    We use the direct TCP route (not RunPod's ssh.runpod.io proxy): the proxy is
    interactive-only and silently ignores a command passed on the SSH line, so it
    can't drive automation. The probe runs ``true`` over SSH, which proves both
    auth AND command execution work, and retries through pod boot (banner resets,
    auth-not-ready). Community-cloud pods get a public IP; secure-cloud ones often
    don't, hence the --cloud-type COMMUNITY hint on failure."""
    deadline = time.time() + timeout
    saw_endpoint = False
    while time.time() < deadline:
        target = _direct_target_from_pod(runpod, pod_id)
        remaining = int(deadline - time.time())
        if not target:
            log(f"Waiting for a public SSH endpoint... (~{remaining}s left).")
            time.sleep(8)
            continue
        saw_endpoint = True
        probe = _ssh_base(key_path, connect_timeout=15) + _ssh_destination(target) + ["true"]
        result = subprocess.run(
            probe, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            log(f"SSH ready via direct ({target['ip']}:{target['port']}).")
            return target
        err = (result.stderr or "").strip().splitlines()
        log(
            f"SSH not ready: {err[-1] if err else f'rc={result.returncode}'} "
            f"(~{remaining}s left)."
        )
        time.sleep(8)

    if saw_endpoint:
        raise SystemExit(
            f"Could not run a command over SSH to pod {pod_id} within {timeout}s "
            "(endpoint existed but auth/exec kept failing)."
        )
    raise SystemExit(
        f"Pod {pod_id} never exposed a public SSH port within {timeout}s. "
        "Use --cloud-type COMMUNITY (secure-cloud pods often have no public IP)."
    )


def run_remote(target: dict, key_path: str, script: str, timeout: int,
               secret_env: Optional[dict] = None) -> Tuple[int, bool]:
    """Stream the remote script's output live.

    Returns (exit_code, done_marker_seen). The marker check is a guard against a
    clean exit code that doesn't actually mean the script finished (e.g. a dropped
    connection), so success requires BOTH a zero exit and the DONE marker.

    ``secret_env`` is exported before the script over the encrypted SSH channel
    (not via the pod's Docker env, which SSH sessions don't inherit) and is never
    part of the printed script."""
    prefix = "".join(
        f"export {key}={_shell_quote(value)}\n" for key, value in (secret_env or {}).items()
    )
    # A login shell so PATH picks up the pip-installed console scripts.
    remote = "bash -lc " + _shell_quote(prefix + script)
    command = _ssh_base(key_path, connect_timeout=30) + _ssh_destination(target) + [remote]

    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    done_seen = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if DONE_MARKER in line and "echo" not in line:
                done_seen = True
        return proc.wait(timeout=timeout), done_seen
    except subprocess.TimeoutExpired:
        proc.kill()
        raise SystemExit(f"Remote run exceeded {timeout}s; killed.")


def check_ssh_key_usable(key_path: str) -> None:
    """Fail early (before creating a billed pod) if the private key can't be used
    non-interactively: it must be unencrypted, or encrypted but loaded in
    ssh-agent. A passphrase-protected key that isn't in the agent would make every
    SSH attempt fail with 'Permission denied' after the pod is already running."""
    if not os.path.isfile(key_path):
        raise SystemExit(f"SSH key not found: {key_path} (pass --ssh-key).")

    # Unencrypted keys decrypt with an empty passphrase.
    unencrypted = subprocess.run(
        ["ssh-keygen", "-y", "-P", "", "-f", key_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if unencrypted.returncode == 0:
        return

    # Encrypted: it must be present in ssh-agent (match by fingerprint).
    fp = subprocess.run(
        ["ssh-keygen", "-lf", key_path], capture_output=True, text=True
    )
    fingerprint = fp.stdout.split()[1] if fp.returncode == 0 and len(fp.stdout.split()) > 1 else ""
    agent = subprocess.run(["ssh-add", "-l"], capture_output=True, text=True)
    if fingerprint and fingerprint in agent.stdout:
        return

    raise SystemExit(
        f"SSH key '{key_path}' is passphrase-protected and not loaded in ssh-agent, "
        "so it can't authenticate non-interactively. Load it once with:\n"
        f"    ssh-add {key_path}\n"
        "then re-run. (Or use an unencrypted key dedicated to RunPod.)"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _prompt_secret(label: str) -> str:
    """Ask the user to paste a secret (hidden input). Returns "" when there is no
    interactive terminal, so automated/CI runs fall back to a clear error instead
    of hanging on input."""
    import getpass

    if not sys.stdin.isatty():
        return ""
    try:
        return getpass.getpass(f"{label}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _read_public_key(private_key_path: str) -> str:
    pub_path = private_key_path + ".pub"
    if not os.path.isfile(pub_path):
        raise SystemExit(
            f"Public key not found at {pub_path}. Generate one with "
            f"'ssh-keygen -t ed25519' or pass --ssh-key."
        )
    with open(pub_path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental: run a SAIB benchmark on a throwaway RunPod GPU pod."
    )
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--db-token", default=os.environ.get("AI_BENCHMARK_DATABASE_TOKEN"))
    parser.add_argument(
        "--gpu",
        default=DEFAULT_GPUS,
        help="GPU type id, or a comma-separated fallback list (tried in order).",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="pt")
    parser.add_argument(
        "--pip-spec",
        default=DEFAULT_PIP_SPEC,
        help="pip requirement installed on the pod (point at a fork/branch here).",
    )
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument(
        "--ssh-key",
        default=os.path.expanduser("~/.ssh/id_ed25519"),
        help="Private key path; the matching .pub is sent to the pod.",
    )
    parser.add_argument("--disk-gb", type=int, default=30)
    parser.add_argument(
        "--cloud-type", choices=["ALL", "SECURE", "COMMUNITY"], default="ALL"
    )
    parser.add_argument(
        "--capacity-wait",
        type=int,
        default=0,
        help="Seconds to keep retrying the requested GPU(s) while there is no "
        "capacity (with backoff). 0 = try each GPU once and exit.",
    )
    parser.add_argument(
        "--extra-args", default="", help="Extra args passed to the run command (e.g. '-w 0')."
    )
    parser.add_argument("--no-publish", action="store_true", help="Run but do not upload results.")
    parser.add_argument(
        "--keep", action="store_true", help="Do not terminate the pod (for debugging)."
    )
    parser.add_argument("--create-timeout", type=int, default=300)
    parser.add_argument("--ssh-timeout", type=int, default=300)
    parser.add_argument("--run-timeout", type=int, default=3600)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan and remote script, then exit."
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    gpus = [g.strip() for g in args.gpu.split(",") if g.strip()]
    public_key = "" if args.dry_run else _read_public_key(args.ssh_key)
    return Config(
        api_key=args.api_key,
        db_token=args.db_token,
        gpus=gpus,
        image=args.image,
        workload=args.workload,
        pip_spec=args.pip_spec,
        database_url=args.database_url,
        private_key_path=args.ssh_key,
        public_key=public_key,
        disk_gb=args.disk_gb,
        cloud_type=args.cloud_type,
        extra_args=args.extra_args,
        publish=not args.no_publish,
        keep=args.keep,
        capacity_wait=args.capacity_wait,
        create_timeout=args.create_timeout,
        ssh_timeout=args.ssh_timeout,
        run_timeout=args.run_timeout,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    script = build_remote_script(cfg)

    if args.dry_run:
        log(f"Workload: {cfg.workload}  GPUs: {cfg.gpus}  image: {cfg.image}")
        log(f"Publish: {cfg.publish}  database: {cfg.database_url}")
        print("\n--- remote script ---")
        print(script)
        print("--- end remote script ---")
        return 0

    if not cfg.api_key:
        cfg.api_key = _prompt_secret("Paste your RunPod API key (RUNPOD_API_KEY)")
    if not cfg.api_key:
        raise SystemExit(
            "RUNPOD_API_KEY is required (env, --api-key, or interactive prompt)."
        )
    if cfg.publish and not cfg.db_token:
        cfg.db_token = _prompt_secret(
            "Paste your AI Benchmark Database token (AI_BENCHMARK_DATABASE_TOKEN)"
        )
    if cfg.publish and not cfg.db_token:
        raise SystemExit(
            "AI_BENCHMARK_DATABASE_TOKEN is required to publish "
            "(env, --db-token, interactive prompt, or pass --no-publish)."
        )

    # Verify the SSH key works non-interactively before paying for a pod.
    check_ssh_key_usable(cfg.private_key_path)

    runpod = _import_runpod()
    runpod.api_key = cfg.api_key

    pod_id: Optional[str] = None
    terminated = {"done": False}

    def terminate() -> None:
        if terminated["done"] or pod_id is None:
            return
        if cfg.keep:
            log(f"--keep set; leaving pod {pod_id} running. Terminate it yourself!")
            terminated["done"] = True
            return
        try:
            log(f"Terminating pod {pod_id}...")
            runpod.terminate_pod(pod_id)
            log("Pod terminated.")
        except Exception as exc:
            log(f"WARNING: failed to terminate pod {pod_id}: {exc}. Check the RunPod console!")
        finally:
            terminated["done"] = True

    def handle_signal(signum, _frame):
        log(f"Received signal {signum}; cleaning up.")
        terminate()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        pod = create_pod_with_fallback(runpod, cfg)
        pod_id = pod["id"]
        target = wait_for_ssh(runpod, pod_id, cfg.private_key_path, cfg.ssh_timeout)

        log("Running remote benchmark (live output follows)...")
        secret_env = (
            {"AI_BENCHMARK_DATABASE_TOKEN": cfg.db_token}
            if cfg.publish and cfg.db_token
            else None
        )
        exit_code, done = run_remote(
            target, cfg.private_key_path, script, cfg.run_timeout, secret_env
        )

        if exit_code == 0 and done:
            log("Remote benchmark finished successfully.")
            return 0
        if exit_code == 0 and not done:
            log(
                "Remote benchmark did NOT complete: connection returned success but "
                f"the '{DONE_MARKER}' marker was never seen (the remote script did "
                "not finish). Treating as failure."
            )
            return 1
        log(f"Remote benchmark FAILED with exit code {exit_code}.")
        return exit_code
    finally:
        terminate()


if __name__ == "__main__":
    sys.exit(main())
