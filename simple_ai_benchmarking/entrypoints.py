from argparse import ArgumentParser
from typing import List

from loguru import logger

from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.benchmark import process_workloads
from simple_ai_benchmarking.results import initialize_logger
from simple_ai_benchmarking.config_structures import AIFramework
from simple_ai_benchmarking.config_pt_tf import build_default_pt_workloads

REPETITIONS = 1

def header() -> None:
    print("############## SIMPLE AI BENCHMARKING ##############")
    print()


def workload_overrides(workloads: List[AIWorkload]) -> List[AIWorkload]:

    workload_info = [f"[{i}] {w}" for i, w in enumerate(workloads)]

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
        workloads = [workloads[i] for i in args.workload_id_selection_override]
        logger.warning("Selected workloads: {}", [str(x) for x in workloads])

    if args.batch_size_override is not None:
        for workload in workloads:
            workload.cfg.batch_size = args.batch_size_override
        logger.warning("Batch size override: {}", args.batch_size_override)

    if args.num_batches_override is not None:
        for workload in workloads:
            workload.cfg.num_batches = args.num_batches_override
        logger.warning("Num batches override: {}", args.num_batches_override)

    return workloads


def run_tf_benchmarks() -> None:

    header()

    initialize_logger("benchmark_tf.log")

    workloads = build_default_pt_workloads(AIFramework.TENSORFLOW)
    
    workloads = workload_overrides(workloads)
    
    process_workloads(workloads, "benchmark_results_tf", repetitions=REPETITIONS)


# TODO: rework config management and cli, directly select models by name and use registry
def run_pt_benchmarks() -> None:

    header()

    initialize_logger("benchmark_pt.log")

    workloads = build_default_pt_workloads(AIFramework.PYTORCH)
    
    workloads = workload_overrides(workloads)

    process_workloads(workloads, "benchmark_results_pt", repetitions=REPETITIONS)


def publish() -> None:
    from simple_ai_benchmarking.database import main

    main()
