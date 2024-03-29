import csv
import requests
import os
import subprocess
import argparse
import pprint
import json

import dataclasses
from dataclasses import dataclass


@dataclass
class BenchmarkData:
    ai_framework: str
    python_version: str
    cpu_name: str
    accelerator_name: str
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


def submit_benchmark_result(data: BenchmarkData, submit_url: str, api_token: str):
    """Submit a single benchmark result to the API."""
    headers = {
        "Authorization": f"Token {api_token}",  # Include the token in the header
        "Content-Type": "application/json",
    }

    response = requests.post(submit_url, json=data.to_dict(), headers=headers)
    if response.status_code == 201:
        print("Successfully added:", data)
    else:
        print("Failed to add:", data, "Response:", response.text)


def prompt_for_updates(benchmark_data: BenchmarkData) -> BenchmarkData:
    while True:
        print("Current result data overview:")
        pprint.pprint(benchmark_data.to_dict())

        print("Do you want to change anything? (y/n)")
        user_input = input().strip().lower()
        if user_input != "y":
            return benchmark_data

        print(
            "\nPlease provide the keys to update (comma-separated), or press Enter to go over all keys step by step:"
        )
        user_input = input().strip()

        if user_input:
            field_names_to_update = [
                field_name.strip() for field_name in user_input.split(",")
            ]
            fields_to_update = [
                f
                for f in dataclasses.fields(BenchmarkData)
                if f.name in field_names_to_update
            ]
        else:
            fields_to_update = dataclasses.fields(BenchmarkData)

        for field in fields_to_update:
            current_value = getattr(benchmark_data, field.name)
            print(
                f"\nCurrent value of {field.name} ({field.type.__name__}): {current_value}"
            )
            new_value = input(
                f"Enter new value for {field.name} (leave blank to keep current): "
            ).strip()

            if new_value:
                try:
                    if field.type == int:
                        new_value = int(new_value)
                    elif field.type == float:
                        new_value = float(new_value)
                    setattr(benchmark_data, field.name, new_value)
                except ValueError as e:
                    print(
                        f"Error converting value: {e}. Skipping update for this field."
                    )

        print("\nReview your changes:")
        pprint.pprint(benchmark_data.to_dict())
        user_input = (
            input("Are you satisfied with the changes? (y/n): ").strip().lower()
        )
        if user_input == "y":
            break  # Exit the loop if the user is satisfied
        else:
            print("\nContinuing editing...")

    return benchmark_data


# TODO: fix automatic generation of benchmark data and meta infos, so that no manual input is needed
def read_csv_and_submit(csv_file_path: str, submit_url: str, api_token: str):

    benchmark_datasets = []
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            benchmark_data = BenchmarkData(
                ai_framework=f"{row['sw_info_ai_framework_name']}{row['sw_info_ai_framework_version']}+{row['sw_info_ai_framework_extra_info']}",
                python_version=row["sw_info_python_version"],
                cpu_name=row["hw_info_cpu"],
                accelerator_name=row["hw_info_accelerator"],
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

    print(f"Loaded {len(benchmark_datasets)} benchmark results from {csv_file_path}.")

    # write to json
    json_file_path = "benchmark_dataset.json"
    with open(json_file_path, "w") as f:
        data = [x.to_dict() for x in benchmark_datasets]
        json.dump(data, f, indent=4)

    # wait for user input
    input(
        f"You may now edit the file {json_file_path} to change values. Press Enter to continue..."
    )
    # reload json
    with open(json_file_path, "r") as f:
        benchmark_datasets = [BenchmarkData(**x) for x in json.load(f)]

    for benchmark_data in benchmark_datasets:

        benchmark_data = prompt_for_updates(benchmark_data)

        print("Publishing...")
        print(benchmark_data.to_dict())

        submit_benchmark_result(benchmark_data, submit_url, api_token)

        benchmark_data.benchmark_type = "training"
        benchmark_data.score_iterations_per_second = float(
            row["train_performance_throughput"]
        )
        benchmark_data.batch_size = int(row["bench_info_batch_size_training"])

        submit_benchmark_result(benchmark_data, submit_url, api_token)


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
        default="http://localhost:8000",
        help="The URL of the AI Benchmark Database.",
    )
    args = parser.parse_args()

    submit_url_path = "/benchmarks/submit/"  # Change to your actual endpoint
    submit_url = args.database_url + submit_url_path

    api_token = os.environ.get("AI_BENCHMARK_DATABASE_TOKEN")
    print("Token:", api_token)
    if not api_token:
        raise ValueError(
            "API token not set. Please set the AI_BENCHMARK_DATABASE_TOKEN environment variable. Maybe also restart your IDE or terminal session. Also maybe restart your pc."
        )

    read_csv_and_submit(args.results_csv_path, submit_url, api_token)


if __name__ == "__main__":
    main()
