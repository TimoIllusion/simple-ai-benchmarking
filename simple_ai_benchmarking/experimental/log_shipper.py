# Project Name: simple-ai-benchmarking
# File Name: log_shipper.py
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

"""Stream a growing log file to the AI Benchmark Database for live debugging.

This is the client half of the dashboard's log-streaming feature. A remote
benchmark host (typically a RunPod pod launched with ``saib-runpod --debug``)
tees its whole console output to a file and runs ``saib-logship`` against it in
the background. The shipper tails the file and POSTs new lines to the
authenticated ingest endpoint (``/dashboard/api/runs/logs/``), keyed by the
run's ``run_id``, so the lines show up on the dashboard's live console.

Design goals, in order:

* **Never disturb the benchmark.** Shipping is strictly best-effort: any network
  or server error is swallowed and retried on the next tick. The shipper is a
  side-car, not part of the measured workload.
* **No heavy deps.** Like ``runpod_runner``, this uses stdlib ``urllib`` only, so
  it works even in the torch-free vLLM pod image without pulling ``requests``.
* **Lossless ordering.** Only complete lines are shipped (a half-written final
  line is carried to the next tick); a final drain flushes whatever remains when
  a stop file appears or the runtime cap is hit.

Usage on the pod::

    saib-logship /workspace/run.log --run-id "$RUNPOD_POD_ID" \\
        --database-url "$URL" --token "$AI_BENCHMARK_DATABASE_TOKEN" \\
        --stop-file /workspace/run.done &
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_DATABASE_URL = "https://benchmarks.timoleitritz.dev"
LOGS_ENDPOINT = "/dashboard/api/runs/logs/"
REPORT_ENDPOINT = "/dashboard/api/runs/report/"

DEFAULT_INTERVAL = 3.0          # seconds between tail passes
DEFAULT_BATCH_LINES = 500       # lines per POST (server cap is higher)
DEFAULT_MAX_RUNTIME = 6 * 3600  # hard stop so a forgotten shipper can't run forever
MAX_PENDING_LINES = 20000       # bound memory if the server is unreachable for long
POST_TIMEOUT = 15               # seconds per ingest request
FINAL_FLUSH_RETRIES = 4         # extra attempts to ship the tail before giving up


def log(message: str) -> None:
    print(f"[saib-logship {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _post_json(url: str, payload: dict, token: Optional[str], label: str) -> bool:
    """POST ``payload`` as JSON with optional Token auth. Return True on a 2xx.

    Best-effort: every failure mode (HTTP error, timeout, DNS, bad JSON) returns
    False instead of raising, so the caller simply retries on the next tick."""
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=POST_TIMEOUT) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        log(f"{label} rejected (HTTP {exc.code}); will retry")
        return False
    except Exception as exc:  # noqa: BLE001 -- side-car must never crash the run
        log(f"{label} failed ({exc.__class__.__name__}); will retry")
        return False


def post_lines(
    database_url: str,
    run_id: str,
    lines: List[str],
    token: Optional[str],
    stream: str = "stdout",
    start_seq: Optional[int] = None,
) -> bool:
    """POST a batch of lines to the ingest endpoint. Return True on a 2xx.

    When ``start_seq`` is given, each line is tagged with a monotonically
    increasing ``seq`` (its absolute index in the shipped stream) so the server
    can drop re-sent duplicates: a slow/timed-out POST that actually succeeded
    server-side is retried with the *same* seqs and ignored the second time,
    instead of storing the whole batch again (the cause of the duplicated
    console). ``seq`` is also stable across a shipper restart (offset resets to
    0, lines re-read from the top get their original seqs)."""
    url = database_url.rstrip("/") + LOGS_ENDPOINT
    if start_seq is None:
        payload_lines: list = lines
    else:
        payload_lines = [
            {"seq": start_seq + i, "text": text} for i, text in enumerate(lines)
        ]
    payload = {"run_id": run_id, "stream": stream, "lines": payload_lines}
    return _post_json(url, payload, token, "ingest")


def post_run_report(
    database_url: str, token: Optional[str], fields: dict
) -> bool:
    """POST a run status/heartbeat to the report endpoint. Best-effort like
    ``post_lines`` -- used to make the run appear (and finish) in the dashboard's
    run table, separate from the streamed console lines."""
    url = database_url.rstrip("/") + REPORT_ENDPOINT
    payload = {k: v for k, v in fields.items() if v not in (None, "")}
    return _post_json(url, payload, token, "run report")


def _terminal_status_from_stop_file(stop_file: Optional[str]) -> str:
    """Map an optional stop-file body to a terminal run status. The launcher
    writes the benchmark's exit code into the stop file; ``0``/``ok``/empty means
    success, anything else means the run failed."""
    if not stop_file:
        return "completed"
    try:
        with open(stop_file, "r", encoding="utf-8", errors="replace") as handle:
            marker = handle.read().strip()
    except OSError:
        return "completed"
    if marker in ("", "0", "ok", "OK"):
        return "completed"
    return "failed"


@dataclass
class LogShipper:
    """Tails ``path`` and ships new lines to the database, keyed by ``run_id``."""

    path: str
    run_id: str
    database_url: str = DEFAULT_DATABASE_URL
    token: Optional[str] = None
    stream: str = "stdout"
    stop_file: Optional[str] = None
    interval: float = DEFAULT_INTERVAL
    batch_lines: int = DEFAULT_BATCH_LINES
    max_runtime: float = DEFAULT_MAX_RUNTIME

    # Optional run-table reporting (separate from streamed console lines).
    report: bool = False
    provider: str = ""
    host: str = ""
    accelerator: str = ""
    benchmark_family: str = ""
    benchmark_name: str = ""
    runner_name: str = ""

    _offset: int = field(default=0, init=False)
    _carry: str = field(default="", init=False)
    _pending: List[str] = field(default_factory=list, init=False)
    # Absolute index of ``_pending[0]`` in the shipped stream. Carried alongside
    # the (string) pending buffer so each line keeps a stable ``seq`` for the
    # server's idempotent ingest, while ``_pending`` stays a plain list of text.
    _seq_base: int = field(default=0, init=False)

    def _read_new_text(self) -> str:
        """Return file bytes appended since the last read, advancing the offset.

        Handles the file not existing yet (returns "") and truncation/rotation
        (offset past EOF -> restart from the top)."""
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return ""
        if size < self._offset:
            # File shrank (truncated/rotated): start over from the beginning.
            self._offset = 0
            self._carry = ""
        if size == self._offset:
            return ""
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
        except OSError:
            return ""
        self._offset += len(chunk)
        return chunk.decode("utf-8", errors="replace")

    def _split_complete_lines(self, text: str, *, final: bool) -> List[str]:
        """Turn newly read text into complete lines.

        A trailing fragment without a newline is held in ``_carry`` until the
        rest of the line arrives -- unless ``final`` is set, in which case any
        leftover fragment is flushed too (end-of-run with no trailing newline)."""
        combined = self._carry + text
        if not combined:
            return []
        parts = combined.split("\n")
        if final:
            self._carry = ""
            # Drop a trailing "" produced by a final newline, keep real content.
            if parts and parts[-1] == "":
                parts = parts[:-1]
            return parts
        self._carry = parts[-1]
        return parts[:-1]

    def _enqueue(self, lines: List[str]) -> None:
        if not lines:
            return
        self._pending.extend(lines)
        # Bound memory if the server has been unreachable: keep the newest lines.
        if len(self._pending) > MAX_PENDING_LINES:
            dropped = len(self._pending) - MAX_PENDING_LINES
            self._pending = self._pending[-MAX_PENDING_LINES:]
            # Advance the base by the dropped count so seqs stay tied to the
            # line's absolute position even after trimming the oldest.
            self._seq_base += dropped
            log(f"dropping {dropped} oldest buffered lines (server unreachable)")

    def _flush_pending(self) -> None:
        """Ship buffered lines in batches; stop on the first failed batch so the
        unsent remainder is retried (in order) on the next tick."""
        while self._pending:
            batch = self._pending[: self.batch_lines]
            if not post_lines(
                self.database_url, self.run_id, batch, self.token, self.stream,
                start_seq=self._seq_base,
            ):
                return
            del self._pending[: len(batch)]
            self._seq_base += len(batch)

    def poll_once(self, *, final: bool = False) -> None:
        """One tail pass: read new text, queue complete lines, flush to server."""
        text = self._read_new_text()
        self._enqueue(self._split_complete_lines(text, final=final))
        self._flush_pending()

    def _final_flush(self) -> None:
        """Last drain at end of run: capture anything written after the stop, then
        retry shipping the tail a few times. Unlike a normal tick (which just waits
        for the next pass), this is the last chance to deliver those lines, so a
        transient blip on the final batch must not silently drop the end of the
        console."""
        self.poll_once(final=True)
        for _ in range(FINAL_FLUSH_RETRIES):
            if not self._pending:
                return
            time.sleep(self.interval)
            self._flush_pending()
        if self._pending:
            log(f"{len(self._pending)} line(s) undelivered at exit")

    def _should_stop(self) -> bool:
        return bool(self.stop_file) and os.path.exists(self.stop_file)

    def _report(self, status: str) -> None:
        if not self.report:
            return
        post_run_report(
            self.database_url,
            self.token,
            {
                "run_id": self.run_id,
                "status": status,
                "provider": self.provider,
                "host": self.host,
                "accelerator": self.accelerator,
                "benchmark_family": self.benchmark_family,
                "benchmark_name": self.benchmark_name,
                "runner_name": self.runner_name,
            },
        )

    def run(self) -> int:
        log(
            f"shipping '{self.path}' -> {self.database_url}{LOGS_ENDPOINT} "
            f"(run_id={self.run_id})"
        )
        self._report("running")
        deadline = time.time() + self.max_runtime
        while True:
            self.poll_once()
            if self._should_stop():
                # Final drain: capture anything written after the stop file.
                self._final_flush()
                self._report(_terminal_status_from_stop_file(self.stop_file))
                log("stop file present; final flush done, exiting")
                return 0
            if time.time() >= deadline:
                self._final_flush()
                self._report("failed")
                log("max runtime reached; exiting")
                return 0
            time.sleep(self.interval)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tail a log file and stream new lines to the AI Benchmark "
        "Database dashboard for live debugging."
    )
    parser.add_argument("logfile", help="Path to the growing log file to tail.")
    parser.add_argument("--run-id", required=True, help="Run identifier (keys the dashboard console).")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument(
        "--token",
        default=os.environ.get("AI_BENCHMARK_DATABASE_TOKEN"),
        help="Database API token (default: $AI_BENCHMARK_DATABASE_TOKEN).",
    )
    parser.add_argument("--stream", default="stdout", choices=["stdout", "stderr", "system"])
    parser.add_argument(
        "--stop-file", default=None,
        help="Exit (after a final flush) once this file exists.",
    )
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--batch-lines", type=int, default=DEFAULT_BATCH_LINES)
    parser.add_argument("--max-runtime", type=float, default=DEFAULT_MAX_RUNTIME)
    parser.add_argument(
        "--report", action="store_true",
        help="Also report run status (running -> completed/failed) to the run table.",
    )
    parser.add_argument("--provider", default="")
    parser.add_argument("--host", default="")
    parser.add_argument("--accelerator", default="")
    parser.add_argument("--benchmark-family", default="")
    parser.add_argument("--benchmark-name", default="")
    parser.add_argument("--runner-name", default="")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.token:
        log("WARNING: no token provided; ingest will be rejected (401). Continuing best-effort.")
    shipper = LogShipper(
        path=args.logfile,
        run_id=args.run_id,
        database_url=args.database_url,
        token=args.token,
        stream=args.stream,
        stop_file=args.stop_file,
        interval=args.interval,
        batch_lines=args.batch_lines,
        max_runtime=args.max_runtime,
        report=args.report,
        provider=args.provider,
        host=args.host,
        accelerator=args.accelerator,
        benchmark_family=args.benchmark_family,
        benchmark_name=args.benchmark_name,
        runner_name=args.runner_name,
    )
    try:
        return shipper.run()
    except KeyboardInterrupt:
        shipper.poll_once(final=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
