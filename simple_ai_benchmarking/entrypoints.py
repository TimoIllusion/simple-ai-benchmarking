# Project Name: simple-ai-benchmarking
# File Name: entrypoints.py
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


from argparse import ArgumentParser
from typing import List

from loguru import logger

from simple_ai_benchmarking.benchmark import process_workloads
from simple_ai_benchmarking.results import initialize_logger
from simple_ai_benchmarking.config_structures import AIFramework, AIWorkloadBaseConfig
from simple_ai_benchmarking.config_pt_tf import build_default_pt_workload_configs
from simple_ai_benchmarking.workloads.factory import WorkloadFactory

REPETITIONS = 3


def header() -> None:
    print("############## SIMPLE AI BENCHMARKING ##############")
    print()


def override_workload_cfg(
    workload_cfgs: List[AIWorkloadBaseConfig],
) -> List[AIWorkloadBaseConfig]:

    workload_info = [f"[{i}] {w}" for i, w in enumerate(workload_cfgs)]

    logger.info("Available workloads:")
    [logger.info(x) for x in workload_info]

    parser = ArgumentParser()
    parser.add_argument(
        "-w",
        "--workload-id-selection-override",
        type=int,
        default=None,
        nargs="+",
        help=f"Insert indices for workloads to run: {workload_info}. Default: None (run all workloads)",
    )
    parser.add_argument(
        "-b",
        "--batch-size-override",
        type=int,
        default=None,
        help="Override batch size for all workloads. Default: None (no override)",
    )
    parser.add_argument(
        "-n",
        "--num-batches-override",
        type=int,
        default=None,
        help="Override amount of batches to process for all workloads. Default: None (no override)",
    )
    args = parser.parse_args()

    if args.workload_id_selection_override is not None:
        workload_cfgs = [workload_cfgs[i] for i in args.workload_id_selection_override]
        logger.warning("Selected workloads: {}", [str(x) for x in workload_cfgs])

    if args.batch_size_override is not None:
        for cfg in workload_cfgs:
            cfg.dataset_cfg.batch_size = args.batch_size_override
        logger.warning("Batch size override: {}", args.batch_size_override)

    if args.num_batches_override is not None:
        for cfg in workload_cfgs:
            cfg.dataset_cfg.num_batches = args.num_batches_override
        logger.warning("Num batches override: {}", args.num_batches_override)

    return workload_cfgs


def run_tf_benchmarks() -> None:
    run_benchmarks(AIFramework.TENSORFLOW, "benchmark_tf.log", "benchmark_results_tf")


def run_pt_benchmarks() -> None:
    run_benchmarks(AIFramework.PYTORCH, "benchmark_pt.log", "benchmark_results_pt")


def run_benchmarks(
    framework: AIFramework, log_file_path: str, results_name: str
) -> None:

    header()

    initialize_logger(log_file_path)

    workload_configs = build_default_pt_workload_configs(framework)

    workload_configs = override_workload_cfg(workload_configs)

    workloads = WorkloadFactory.build_multiple_workloads(workload_configs, framework)

    process_workloads(workloads, results_name, repetitions=REPETITIONS)


def publish() -> None:
    from simple_ai_benchmarking.database import publish_results_cli

    publish_results_cli()
