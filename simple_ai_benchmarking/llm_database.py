# Project Name: simple-ai-benchmarking
# File Name: llm_database.py

import argparse
import csv
import json
import os
from dataclasses import dataclass

from simple_ai_benchmarking.database import (
    get_git_commit_hash_from_package_version,
    handle_token_pw_user,
    prompt_for_updates,
    submit_benchmark_result_token_auth,
    submit_benchmark_result_user_pw_auth,
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

    def to_dict(self) -> dict:
        return self.__dict__


def read_csv_and_create_llm_benchmark_dataset(
    csv_file_path: str, extra_info: str = None
):
    benchmark_datasets = []
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
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
                )
            )

    print(f"Loaded {len(benchmark_datasets)} LLM benchmark results from {csv_file_path}.")
    return benchmark_datasets


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Submit LLM benchmark results to the AI Benchmark Database."
    )
    parser.add_argument("results_csv_path", type=str)
    parser.add_argument(
        "--database-url", type=str, default="https://timoillusion.pythonanywhere.com"
    )
    parser.add_argument("--non-interactive", action="store_true", default=False)
    parser.add_argument("-t", "--token", type=str, default=None)
    parser.add_argument("-p", "--password", type=str, default=None)
    parser.add_argument("-u", "--user", type=str, default=None)
    parser.add_argument("-e", "--extra-info", type=str, default=None)
    return parser.parse_args()


def read_and_enrich_llm_benchmark_data(args):
    benchmark_datasets = read_csv_and_create_llm_benchmark_dataset(
        args.results_csv_path, args.extra_info
    )

    json_file_path = "llm_benchmark_dataset.json"
    with open(json_file_path, "w") as f:
        json.dump([x.to_dict() for x in benchmark_datasets], f, indent=4)

    if not args.non_interactive:
        input(
            f"You may now edit the file {json_file_path} to change meta data. Press Enter to continue after reviewing the json ..."
        )

    with open(json_file_path, "r") as f:
        benchmark_datasets = [LLMBenchmarkData(**x) for x in json.load(f)]

    if not args.non_interactive:
        benchmark_datasets = prompt_for_updates(benchmark_datasets)

    return benchmark_datasets


def submit_llm_results(benchmark_datasets, args, submit_url, api_token):
    for benchmark_data in benchmark_datasets:
        print("Publishing LLM benchmark...")

        if api_token:
            success = submit_benchmark_result_token_auth(
                benchmark_data, submit_url, api_token
            )
        else:
            success = submit_benchmark_result_user_pw_auth(
                benchmark_data, submit_url, args.user, args.password
            )

        if not success:
            print("Submission failed. Exiting...")
            break


def publish_llm_results_cli():
    args = parse_arguments()
    submit_url = args.database_url + "/benchmarks/llm/submit/"

    api_token = handle_token_pw_user(args)
    benchmark_datasets = read_and_enrich_llm_benchmark_data(args)
    submit_llm_results(benchmark_datasets, args, submit_url, api_token)
