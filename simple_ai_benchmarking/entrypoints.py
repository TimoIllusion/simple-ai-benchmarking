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


class BenchmarkDispatcher:
    REPETITIONS = 3
    BATCH_SIZE = 32
    NUM_BATCHES_INFERENCE = 150
    NUM_BATCHES_TRAINING = 50
    LOG_FILE_PATH = "benchmark.log"

    def __init__(self, framework: AIFramework, results_name: str = "results"):

        self.parser = self._setup_parser()
        self.framework = framework
        self.results_name = results_name

    def _setup_parser(self):
        parser = ArgumentParser()
        parser.add_argument(
            "-w",
            "--workload-id-selection-override",
            type=int,
            default=None,
            nargs="+",
            help="Insert indices for workloads to run. Default: None (run all workloads)",
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
        return parser

    def _header(self):
        print("############## SIMPLE AI BENCHMARKING ##############")
        print()

    def run(self):
        self._header()
        initialize_logger(self.LOG_FILE_PATH)
        workload_configs = build_default_pt_workload_configs(
            self.framework,
            batch_size=self.BATCH_SIZE,
            num_batches_inference=self.NUM_BATCHES_INFERENCE,
            num_batches_training=self.NUM_BATCHES_TRAINING,
        )
        workload_configs = self._override_workload_cfg(workload_configs)
        workloads = WorkloadFactory.build_multiple_workloads(
            workload_configs, self.framework
        )
        process_workloads(workloads, self.results_name, repetitions=self.REPETITIONS)

    def _override_workload_cfg(self, workload_cfgs: List[AIWorkloadBaseConfig]):
        args = self.parser.parse_args()
        workload_info = [f"[{i}] {w}" for i, w in enumerate(workload_cfgs)]
        logger.info("Available workloads:")
        for x in workload_info:
            logger.info(x)

        if args.workload_id_selection_override is not None:
            workload_cfgs = [
                workload_cfgs[i] for i in args.workload_id_selection_override
            ]
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


def run_pt_benchmarks():
    dispatcher = BenchmarkDispatcher(AIFramework.PYTORCH, "results_pt")
    dispatcher.run()


def run_tf_benchmarks():
    dispatcher = BenchmarkDispatcher(AIFramework.TENSORFLOW, "results_tf")
    dispatcher.run()


def publish():
    from simple_ai_benchmarking.database import publish_results_cli

    publish_results_cli()
