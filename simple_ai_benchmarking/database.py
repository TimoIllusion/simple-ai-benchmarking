import csv
import requests
from requests.auth import HTTPBasicAuth
import os
import subprocess
import argparse
import pprint
import json
from copy import deepcopy

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
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        return commit_hash
    except subprocess.CalledProcessError:
        print("An error occurred while trying to fetch the commit ID.")
        return "N/A"
    except FileNotFoundError:
        print("Git is not installed or not found in PATH.")
        return "N/A"


def get_git_repository_url():
    try:
        repo_url = (
            subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            .decode("ascii")
            .strip()
        )
        return repo_url
    except subprocess.CalledProcessError:
        print("An error occurred while trying to fetch the repository URL.")
        return "N/A"
    except FileNotFoundError:
        print("Git is not installed or not found in PATH.")
        return "N/A"


def submit_benchmark_result_user_pw_auth(
    data: BenchmarkData, submit_url: str, user: str, pw: str
) -> bool:
    """Submit a single benchmark result to the API."""

    response = requests.post(
        submit_url, json=data.to_dict(), auth=HTTPBasicAuth(user, pw)
    )

    if response.status_code == 201:
        print("Successfully added:")
        print(data.to_dict())
        return True
    else:
        print("Failed to add:")
        print(data.to_dict())
        print("Response:", response.text)
        return False


def submit_benchmark_result_token_auth(
    data: BenchmarkData, submit_url: str, api_token: str
) -> bool:
    """Submit a single benchmark result to the API."""
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }
    response = requests.post(submit_url, json=data.to_dict(), headers=headers)

    if response.status_code == 201:
        print("Successfully added:")
        print(data.to_dict())
        return True
    else:
        print("Failed to add:")
        print(data.to_dict())
        print("Response:", response.text)
        return False


def prompt_for_updates(
    benchmark_data_list,
    keys_to_update=[
        "accelerator",
        "cpu_name",
        "ai_framework_version",
        "ai_framework_extra_info",
    ],
):

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
        if (
            input(f"Do you want to change all occurrences of '{key}'? (y/N): ")
            .strip()
            .lower()
            == "y"
        ):
            new_value = input(f"Enter new value for all {key}: ").strip()
            if new_value:
                # Update all occurrences
                for data in benchmark_data_list:
                    setattr(data, key, new_value)

    # Optionally, review changes for one or more items
    if (
        input("\nWould you like to review changes to any item? (y/N): ").strip().lower()
        == "y"
    ):
        for data in benchmark_data_list:

            pprint.pprint(data.to_dict())

            if input("\nContinue reviewing? (y/N): ").strip().lower() != "y":
                break

    # Finally ask to publish or not
    if (
        input("\nWould you like to publish the benchmark results? (y/N): ")
        .strip()
        .lower()
        == "y"
    ):
        return benchmark_data_list
    else:
        print("Exiting without publishing.")
        exit(1)


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
        "-t",
        "--token",
        type=str,
        default=None,
        help="The API token to authenticate with the database.",
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        default=None,
        help="The password to authenticate with the database.",
    )
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        default=None,
        help="The user to authenticate with the database.",
    )

    args = parser.parse_args()

    submit_endpoint = "/benchmarks/submit/"
    submit_url = args.database_url + submit_endpoint

    if args.token:
        api_token = args.token
    else:
        api_token = os.environ.get("AI_BENCHMARK_DATABASE_TOKEN")

    if not api_token:

        if args.user and args.password:
            print("User:", args.user)
            print("Password:", "*" * len(args.password))
        else:
            print("User and/or password not provided.")

        raise ValueError(
            "No suitable authentication provided. Please provide a token or user and password. Check README.md for details."
        )

    else:
        print("API Token:", "*" * (len(api_token) - 3) + api_token[-3:])

    benchmark_datasets = read_csv_and_create_benchmark_dataset(args.results_csv_path)

    # write data to json for review
    json_file_path = "benchmark_dataset.json"
    with open(json_file_path, "w") as f:
        data = [x.to_dict() for x in benchmark_datasets]
        json.dump(data, f, indent=4)

    if not args.non_interactive:
        input(
            f"You may now edit the file {json_file_path} to change meta data. Press Enter to continue after reviewing the json ..."
        )

    # reload json
    with open(json_file_path, "r") as f:
        benchmark_datasets = [BenchmarkData(**x) for x in json.load(f)]

    if not args.non_interactive:
        benchmark_datasets = prompt_for_updates(benchmark_datasets)

    for benchmark_data in benchmark_datasets:

        print("Publishing...")

        if api_token is not None:
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


if __name__ == "__main__":
    main()
