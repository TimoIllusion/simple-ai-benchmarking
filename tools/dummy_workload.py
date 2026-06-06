#!/usr/bin/env python3
# Project Name: simple-ai-benchmarking
# File Name: dummy_workload.py
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

"""A fake benchmark run that prints a believable console over a few seconds.

This stands in for a real SAIB benchmark when exercising the live-log streaming
path locally (see ``tools/local_stream_demo.py``): tee its output to a file and
point ``saib-logship`` at that file. It prints to both stdout and stderr, flushes
each line, and sleeps between lines so a tailing shipper observes incremental
growth -- exactly like an actual pod's console.
"""

from __future__ import annotations

import argparse
import sys
import time


def emit(text: str, *, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    print(text, file=stream, flush=True)


def run(steps: int, delay: float, fail: bool) -> int:
    emit("== installing SAIB ==")
    time.sleep(delay)
    emit("Collecting simple-ai-benchmarking ...")
    emit("Successfully installed simple-ai-benchmarking")
    emit("== loading model 'dummy/tiny-7B' ==")
    time.sleep(delay)
    emit("WARNING: running on CPU; numbers are illustrative only", err=True)

    for i in range(1, steps + 1):
        pct = 100 * i // steps
        tps = 120.0 + i  # pretend tokens/sec creeps up as caches warm
        emit(f"[step {i}/{steps}] {pct:3d}%  throughput={tps:.1f} tok/s")
        time.sleep(delay)

    if fail:
        emit("ERROR: simulated benchmark failure (--fail)", err=True)
        return 1

    emit("== benchmark complete: mean throughput 126.5 tok/s ==")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10, help="Number of progress lines.")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between lines.")
    parser.add_argument("--fail", action="store_true", help="Exit non-zero at the end.")
    args = parser.parse_args(argv)
    return run(args.steps, args.delay, args.fail)


if __name__ == "__main__":
    sys.exit(main())
