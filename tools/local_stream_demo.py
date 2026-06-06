#!/usr/bin/env python3
# Project Name: simple-ai-benchmarking
# File Name: local_stream_demo.py
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

"""End-to-end local smoke test for the dashboard live-log streaming feature.

It reproduces what a RunPod pod does in ``saib-runpod --debug`` -- minus RunPod
and Docker -- entirely on localhost:

    dummy_workload.py  ->  logfile  ->  saib-logship  ->  local dashboard DB
                                                  |
                              this script polls the read endpoint and verifies
                              the lines actually streamed through.

Typical use (two terminals)::

    # terminal 1, in the ai-benchmark-database repo:
    tools/serve_local.sh            # prints AI_BENCHMARK_DATABASE_TOKEN=...

    # terminal 2, in this repo:
    AI_BENCHMARK_DATABASE_TOKEN=<token> tools/local_stream_demo.py

Or let this script boot the server too (needs the sibling DB repo + uv)::

    tools/local_stream_demo.py --start-server --db-repo ../ai-benchmark-database

Then open the printed live-console URL in a browser to watch it in real time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGSHIP_MODULE = "simple_ai_benchmarking.experimental.log_shipper"


def info(msg: str) -> None:
    print(f"[demo] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Optional: boot the local dashboard server from the sibling DB repo
# --------------------------------------------------------------------------- #
def _uv_manage(db_repo: Path) -> List[str]:
    return [
        "uv", "run", "--with", "django>=5.0,<6.0", "--with", "djangorestframework",
        "python", "manage.py",
    ]


def start_server(db_repo: Path, host: str, port: int) -> tuple[subprocess.Popen, str]:
    """Migrate, mint a token, and run the dev server. Returns (process, token)."""
    base = _uv_manage(db_repo)
    info("applying migrations ...")
    subprocess.run(base + ["migrate", "--no-input"], cwd=db_repo, check=True)

    info("minting API token ...")
    mint = (
        "from django.contrib.auth.models import User\n"
        "from rest_framework.authtoken.models import Token\n"
        "u,_ = User.objects.get_or_create(username='local')\n"
        "t,_ = Token.objects.get_or_create(user=u)\n"
        "print(t.key)\n"
    )
    out = subprocess.run(
        base + ["shell", "-c", mint], cwd=db_repo, check=True,
        capture_output=True, text=True,
    )
    token = out.stdout.strip().splitlines()[-1]

    info(f"starting dev server on {host}:{port} ...")
    proc = subprocess.Popen(
        base + ["runserver", f"{host}:{port}", "--noreload"], cwd=db_repo
    )
    return proc, token


def wait_for_server(database_url: str, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(database_url + "/dashboard/", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


# --------------------------------------------------------------------------- #
# Read endpoint polling (the same primitive the browser live console uses)
# --------------------------------------------------------------------------- #
def fetch_lines(database_url: str, run_id: str, after: int = 0) -> dict:
    url = f"{database_url}/dashboard/api/runs/{run_id}/logs/?after={after}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode())


# --------------------------------------------------------------------------- #
# The demo run
# --------------------------------------------------------------------------- #
def run_demo(args: argparse.Namespace, database_url: str, token: str) -> int:
    run_id = args.run_id or f"local-demo-{int(time.time())}"
    workdir = Path(tempfile.mkdtemp(prefix="saib-stream-demo-"))
    log_file = workdir / "run.log"
    stop_file = workdir / "run.done"
    log_file.write_text("")

    console_url = f"{database_url}/dashboard/runs/{run_id}/"
    info(f"run_id = {run_id}")
    info(f"live console: {console_url}")
    info("(open that URL now to watch the stream live)")

    # 1) Start the shipper tailing the (currently empty) log file.
    ship_cmd = [
        sys.executable, "-m", LOGSHIP_MODULE, str(log_file),
        "--run-id", run_id, "--database-url", database_url, "--token", token,
        "--stop-file", str(stop_file), "--report", "--provider", "local",
        "--host", "localhost", "--accelerator", "CPU",
        "--benchmark-family", "demo", "--interval", "1",
    ]
    info("starting saib-logship ...")
    shipper = subprocess.Popen(ship_cmd, cwd=REPO_ROOT)

    # 2) Run the dummy workload, appending its console to the same log file.
    work_cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "dummy_workload.py"),
        "--steps", str(args.steps), "--delay", str(args.delay),
    ]
    if args.fail:
        work_cmd.append("--fail")
    info("running dummy workload ...")
    with open(log_file, "a") as sink:
        rc = subprocess.run(work_cmd, stdout=sink, stderr=subprocess.STDOUT).returncode
    info(f"workload exited rc={rc}")

    # 3) Signal completion (exit code -> completed/failed) and let the shipper drain.
    stop_file.write_text("ok" if rc == 0 else str(rc))
    try:
        shipper.wait(timeout=30)
    except subprocess.TimeoutExpired:
        shipper.terminate()

    # 4) Verify the lines made it through the server.
    info("verifying via read endpoint ...")
    data = fetch_lines(database_url, run_id)
    lines = data.get("lines", [])
    info(f"server returned {len(lines)} line(s); status={data.get('status')!r}")
    print("------ streamed console (from the database) ------")
    for entry in lines:
        marker = "!" if entry["stream"] == "stderr" else " "
        print(f"  {marker} {entry['text']}")
    print("--------------------------------------------------")

    expected_status = "failed" if args.fail else "completed"
    ok = bool(lines) and data.get("status") == expected_status
    if ok:
        info(f"PASS: lines streamed and run reported as {expected_status}.")
        info(f"Open {console_url} to view it.")
    else:
        info(
            "FAIL: expected streamed lines and "
            f"status={expected_status!r}, got {len(lines)} line(s) / "
            f"status={data.get('status')!r}."
        )
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=os.environ.get("AI_BENCHMARK_DATABASE_TOKEN"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--fail", action="store_true", help="Simulate a failing run.")
    parser.add_argument("--start-server", action="store_true", help="Boot the dashboard from --db-repo.")
    parser.add_argument("--db-repo", default="../ai-benchmark-database")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    database_url = args.database_url.rstrip("/")
    token = args.token
    server: Optional[subprocess.Popen] = None
    try:
        if args.start_server:
            db_repo = Path(args.db_repo).resolve()
            if not (db_repo / "manage.py").exists():
                raise SystemExit(f"--db-repo has no manage.py: {db_repo}")
            database_url = f"http://{args.host}:{args.port}"
            server, token = start_server(db_repo, args.host, args.port)
            if not wait_for_server(database_url):
                raise SystemExit("server did not become ready in time")

        if not token:
            raise SystemExit(
                "No token. Set AI_BENCHMARK_DATABASE_TOKEN (printed by "
                "serve_local.sh) or pass --start-server."
            )
        return run_demo(args, database_url, token)
    finally:
        if server is not None:
            info("stopping dev server ...")
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    sys.exit(main())
