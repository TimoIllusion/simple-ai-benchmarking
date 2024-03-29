import csv
import requests
import os
import subprocess
import argparse
import pprint
import json
from copy import deepcopy

import dataclasses
from dataclasses import dataclass


@dataclass
class BenchmarkData:
    ai_framework_name: str
    ai_framework_version: str
    ai_framework_extra_info: str
    python_version: str
    cpu_name: str
    accelerator: str
    model: str
    benchmark_type: str
    score_iterations_per_second: float
    benchmark_precision: str
    power_usage_watts: float
    batch_size: int
    operating_system: str
    benchmark_github_repo_url: str
    benchmark_version: str
    benchmark_commit_id: str
    benchmark_date: str

    def to_dict(self) -> dict:
        """Converts the data class instance to a dictionary."""
        return self.__dict__


def read_version():
    with open(os.path.join("simple_ai_benchmarking", "VERSION"), encoding="utf-8") as f:
        return f.read().strip()


def get_git_commit_hash():
    try:
        # Run the git command to get the current commit ID
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        return commit_hash
    except subprocess.CalledProcessError:
        # Handle cases where the git command fails
        print("An error occurred while trying to fetch the commit ID.")
        return "N/A"
    except FileNotFoundError:
        # Handle the case where git is not installed or not found in the system's PATH
        print("Git is not installed or not found in PATH.")
        return "N/A"


def get_git_repository_url():
    try:
        # Run the git command to get the remote origin URL
        repo_url = (
            subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            .decode("ascii")
            .strip()
        )
        return repo_url
    except subprocess.CalledProcessError:
        # Handle cases where the git command fails
        print("An error occurred while trying to fetch the repository URL.")
        return "N/A"
    except FileNotFoundError:
        # Handle the case where git is not installed or not found in the system's PATH
        print("Git is not installed or not found in PATH.")
        return "N/A"


def submit_benchmark_result(
    data: BenchmarkData, submit_url: str, api_token: str
) -> bool:
    """Submit a single benchmark result to the API."""
    headers = {
        "Authorization": f"Token {api_token}",  # Include the token in the header
        "Content-Type": "application/json",
    }

    response = requests.post(submit_url, json=data.to_dict(), headers=headers)
    if response.status_code == 201:
        print("Successfully added:", data)
        return True
    else:
        print("Failed to add:", data, "Response:", response.text)
        return False



def prompt_for_updates(benchmark_data_list, keys_to_update=['accelerator', 'cpu_name']):
    """
    Allows user to update specific fields across all benchmark data items.
    :param benchmark_data_list: List of BenchmarkData instances.
    :param keys_to_update: List of keys to potentially update.
    """

    # Collect unique values for specified keys across all items
    unique_values = {key: set() for key in keys_to_update}
    for data in benchmark_data_list:
        for key in keys_to_update:
            value = getattr(data, key, None)
            if value:
                unique_values[key].add(value)

    # Show current unique values and prompt for changes
    for key, values in unique_values.items():
        print(f"\nCurrent unique values for {key}: {', '.join(values)}")
        if input(f"Do you want to change all occurrences of '{key}'? (y/n): ").strip().lower() == 'y':
            new_value = input(f"Enter new value for all {key}: ").strip()
            if new_value:
                # Update all occurrences
                for data in benchmark_data_list:
                    setattr(data, key, new_value)

    # Optionally, review changes for one or more items
    if input("\nWould you like to review changes to any item? (y/n): ").strip().lower() == 'y':
        for data in benchmark_data_list:
            pprint.pprint(data.to_dict())
            if input("\nContinue reviewing? (y/n): ").strip().lower() != 'y':
                break

    return benchmark_data_list


def read_csv_and_create_benchmark_dataset(csv_file_path: str):

    benchmark_datasets = []
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            benchmark_data = BenchmarkData(
                ai_framework_name=f"{row['sw_info_ai_framework_name']}",
                ai_framework_version=row["sw_info_ai_framework_version"],
                ai_framework_extra_info=row["sw_info_ai_framework_extra_info"],
                python_version=row["sw_info_python_version"],
                cpu_name=row["hw_info_cpu"],
                accelerator=row["hw_info_accelerator"],
                model=row["bench_info_model"],
                benchmark_type="inference",
                score_iterations_per_second=float(row["infer_performance_throughput"]),
                benchmark_precision=row["bench_info_compute_precision"],
                power_usage_watts=-1.0,
                batch_size=int(row["bench_info_batch_size_inference"]),
                operating_system=row["sw_info_os_version"],
                benchmark_github_repo_url=get_git_repository_url(),
                benchmark_version=read_version(),
                benchmark_commit_id=get_git_commit_hash(),
                benchmark_date=row["bench_info_date"],
            )

            benchmark_datasets.append(benchmark_data)

            benchmark_data = deepcopy(benchmark_data)
            benchmark_data.benchmark_type = "training"
            benchmark_data.score_iterations_per_second = float(
                row["train_performance_throughput"]
            )
            benchmark_data.batch_size = int(row["bench_info_batch_size_training"])

            benchmark_datasets.append(benchmark_data)

    print(f"Loaded {len(benchmark_datasets)} benchmark results from {csv_file_path}.")

    return benchmark_datasets


def main():

    parser = argparse.ArgumentParser(
        description="Submit benchmark results to the AI Benchmark Database."
    )
    parser.add_argument(
        "results_csv_path",
        type=str,
        help="Path to the CSV file containing the benchmark results.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="https://timoillusion.pythonanywhere.com",
        help="The URL of the AI Benchmark Database.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Whether to run the script in non-interactive mode.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="The API token to authenticate with the database.",
    )

    args = parser.parse_args()

    submit_url_path = "/benchmarks/submit/"  # Change to your actual endpoint
    submit_url = args.database_url + submit_url_path

    if args.token:
        api_token = args.token
    else:
        api_token = os.environ.get("AI_BENCHMARK_DATABASE_TOKEN")

    print("Token:", api_token)

    if not api_token:
        raise ValueError(
            "API token not set. Please set the AI_BENCHMARK_DATABASE_TOKEN environment variable OR use the --token arg. Maybe also restart your IDE or terminal session. Also maybe restart your pc."
        )

    benchmark_datasets = read_csv_and_create_benchmark_dataset(args.results_csv_path)

    # write to json
    json_file_path = "benchmark_dataset.json"
    with open(json_file_path, "w") as f:
        data = [x.to_dict() for x in benchmark_datasets]
        json.dump(data, f, indent=4)

    if not args.non_interactive:
        # wait for user input
        input(
            f"You may now edit the file {json_file_path} to change meta data. Press Enter to continue..."
        )

    # reload json
    with open(json_file_path, "r") as f:
        benchmark_datasets = [BenchmarkData(**x) for x in json.load(f)]

    if not args.non_interactive:
        benchmark_datasets = prompt_for_updates(benchmark_datasets)

    for benchmark_data in benchmark_datasets:

        print("Publishing...")

        success = submit_benchmark_result(benchmark_data, submit_url, api_token)

        if not success:
            print("Submission failed. Exiting...")
            break


if __name__ == "__main__":
    main()
