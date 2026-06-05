# Project Name: simple-ai-benchmarking
# File Name: llm_database.py

import csv
from dataclasses import dataclass

from simple_ai_benchmarking.database import (
    build_publish_parser,
    enrich_benchmark_data,
    get_git_commit_hash_from_package_version,
    handle_token_pw_user,
    submit_results,
)
from simple_ai_benchmarking.version_and_metadata import REPO_URL, VERSION


@dataclass
class LLMBenchmarkData:
    ai_framework_name: str
    ai_framework_version: str
    ai_framework_extra_info: str
    python_version: str
    cpu_name: str
    accelerator: str
    backend: str
    model: str
    model_params: int
    benchmark_type: str
    benchmark_precision: str
    quantization: str
    context_length: int
    prompt_tokens: int
    generated_tokens: int
    concurrency: int
    requests: int
    duration_s: float
    prompt_tokens_per_second: float
    generated_tokens_per_second: float
    total_tokens_per_second: float
    time_to_first_token_s: float
    power_usage_watts: float
    operating_system: str
    benchmark_github_repo_url: str
    benchmark_version: str
    benchmark_commit_id: str
    benchmark_date: str
    benchmark_family: str
    benchmark_spec_name: str
    benchmark_spec_version: str
    benchmark_profile_id: str
    benchmark_profile_hash: str
    benchmark_config_hash: str
    benchmark_runner_id: str
    benchmark_runner_hash: str
    benchmark_payload_hash: str
    serving_engine: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


def read_csv_and_create_llm_benchmark_dataset(
    csv_file_path: str, extra_info: str = None
):
    benchmark_datasets = []
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames or []
        if "bench_info_backend" not in fieldnames:
            if "bench_info_workload_type" in fieldnames:
                raise ValueError(
                    f"{csv_file_path} looks like a CV results CSV "
                    "(no 'bench_info_backend' column). Use 'saib-pub' to publish CV results."
                )
            raise ValueError(
                f"{csv_file_path} is missing the required 'bench_info_backend' "
                "column; is this a SAIB LLM results CSV?"
            )
        for row in reader:
            if extra_info is None:
                row_extra_info = row["sw_info_ai_framework_extra_info"]
            else:
                row_extra_info = extra_info

            benchmark_datasets.append(
                LLMBenchmarkData(
                    ai_framework_name=row["sw_info_ai_framework_name"],
                    ai_framework_version=row["sw_info_ai_framework_version"],
                    ai_framework_extra_info=row_extra_info,
                    python_version=row["sw_info_python_version"],
                    cpu_name=row["hw_info_cpu"],
                    accelerator=row["hw_info_accelerator"],
                    backend=row["bench_info_backend"],
                    model=row["bench_info_model"],
                    model_params=int(row["bench_info_model_params"]),
                    benchmark_type=row["bench_info_benchmark_type"].lower(),
                    benchmark_precision=row.get("bench_info_compute_precision", ""),
                    quantization=row["bench_info_quantization"],
                    context_length=int(row["bench_info_context_length"]),
                    prompt_tokens=int(row["bench_info_prompt_tokens"]),
                    generated_tokens=int(row["bench_info_generated_tokens"]),
                    concurrency=int(row["bench_info_concurrency"]),
                    requests=int(row["performance_requests"]),
                    duration_s=float(row["performance_duration_s"]),
                    prompt_tokens_per_second=float(
                        row["performance_prompt_tokens_per_second"]
                    ),
                    generated_tokens_per_second=float(
                        row["performance_generated_tokens_per_second"]
                    ),
                    total_tokens_per_second=float(
                        row["performance_total_tokens_per_second"]
                    ),
                    time_to_first_token_s=float(
                        row["performance_time_to_first_token_s"]
                    ),
                    power_usage_watts=-1.0,
                    operating_system=row["sw_info_os_version"],
                    benchmark_github_repo_url=REPO_URL,
                    benchmark_version=VERSION,
                    benchmark_commit_id=get_git_commit_hash_from_package_version(),
                    benchmark_date=row["bench_info_date"],
                    benchmark_family=row["bench_info_benchmark_family"],
                    benchmark_spec_name=row["bench_info_benchmark_spec_name"],
                    benchmark_spec_version=row["bench_info_benchmark_spec_version"],
                    benchmark_profile_id=row["bench_info_benchmark_profile_id"],
                    benchmark_profile_hash=row["bench_info_benchmark_profile_hash"],
                    benchmark_config_hash=row["bench_info_benchmark_config_hash"],
                    benchmark_runner_id=row["bench_info_benchmark_runner_id"],
                    benchmark_runner_hash=row["bench_info_benchmark_runner_hash"],
                    benchmark_payload_hash=row["bench_info_benchmark_payload_hash"],
                    # Spec-2.3 identity dimension; absent in pre-2.3 CSVs -> "".
                    serving_engine=row.get("bench_info_serving_engine", ""),
                )
            )

    print(f"Loaded {len(benchmark_datasets)} LLM benchmark results from {csv_file_path}.")
    return benchmark_datasets


def parse_arguments():
    return build_publish_parser(
        "Submit LLM benchmark results to the AI Benchmark Database."
    ).parse_args()


def read_and_enrich_llm_benchmark_data(args):
    benchmark_datasets = read_csv_and_create_llm_benchmark_dataset(
        args.results_csv_path, args.extra_info
    )
    return enrich_benchmark_data(
        benchmark_datasets,
        LLMBenchmarkData,
        "llm_benchmark_dataset.json",
        args.non_interactive,
    )


def publish_llm_results_cli():
    args = parse_arguments()
    submit_url = args.database_url + "/benchmarks/llm/submit/"

    api_token = handle_token_pw_user(args)
    benchmark_datasets = read_and_enrich_llm_benchmark_data(args)
    submit_results(
        benchmark_datasets,
        submit_url,
        api_token,
        user=args.user,
        password=args.password,
    )
