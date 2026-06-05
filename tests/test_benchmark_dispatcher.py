# Project Name: simple-ai-benchmarking
# File Name: test_default_execution.py
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


from simple_ai_benchmarking.entrypoints import BenchmarkDispatcher
from simple_ai_benchmarking.config_structures import AIFramework


# Smallest possible smoke test of the full dispatch pipeline. The default config
# builds 3 models x (inference + training); ResNet50/ViT-B-16 are heavy and may
# download weights, so restrict to the lightweight SIMPLE_CLASSIFICATION_CNN via
# "-w 0 1" (index 0 = inference, 1 = training) and run a single tiny batch.
_FAST_ARGS = ["-w", "0", "1"]


def _make_fast_dispatcher(framework: AIFramework) -> BenchmarkDispatcher:
    dispatcher = BenchmarkDispatcher(framework)
    dispatcher.BATCH_SIZE = 1
    dispatcher.REPETITIONS = 1
    dispatcher.NUM_BATCHES_INFERENCE = 1
    dispatcher.NUM_BATCHES_TRAINING = 1
    return dispatcher


def test_pt_benchmark() -> None:
    _make_fast_dispatcher(AIFramework.PYTORCH).run(args=_FAST_ARGS)


def test_tf_benchmark() -> None:
    _make_fast_dispatcher(AIFramework.TENSORFLOW).run(args=_FAST_ARGS)


def test_publish_each_requires_token(monkeypatch):
    monkeypatch.delenv("AI_BENCHMARK_DATABASE_TOKEN", raising=False)
    dispatcher = BenchmarkDispatcher(AIFramework.PYTORCH)
    args = dispatcher.parser.parse_args(["--publish-each"])

    import pytest

    with pytest.raises(SystemExit, match="requires"):
        dispatcher._build_publisher(args)


def test_dispatcher_wires_incremental_publisher(monkeypatch):
    import simple_ai_benchmarking.entrypoints as entrypoints

    publisher = object()
    captured = {}
    monkeypatch.setattr(entrypoints, "initialize_logger", lambda path: None)
    monkeypatch.setattr(entrypoints, "build_default_pt_workload_configs", lambda *a, **k: [])
    monkeypatch.setattr(
        entrypoints.WorkloadFactory, "build_multiple_workloads", lambda *a, **k: ["workload"]
    )
    monkeypatch.setattr(
        entrypoints, "process_workloads", lambda *a, **k: captured.update(k)
    )
    monkeypatch.setattr(BenchmarkDispatcher, "_build_publisher", lambda self, args: publisher)

    BenchmarkDispatcher(AIFramework.PYTORCH).run(args=["--publish-each", "-t", "tok"])

    assert captured["on_workload_logged"] is publisher


if __name__ == "__main__":
    test_pt_benchmark()
    test_tf_benchmark()
