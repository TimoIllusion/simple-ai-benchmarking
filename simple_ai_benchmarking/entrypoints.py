from simple_ai_benchmarking.benchmark import proccess_workloads


def run_tf_benchmarks() -> None:

    from simple_ai_benchmarking.config_tf import build_default_tf_workloads

    workloads = build_default_tf_workloads()
    proccess_workloads(workloads, "benchmark_results_tf")


def run_pt_benchmarks() -> None:

    from simple_ai_benchmarking.config_pt import build_default_pt_workloads

    workloads = build_default_pt_workloads()
    proccess_workloads(workloads, "benchmark_results_pt")
