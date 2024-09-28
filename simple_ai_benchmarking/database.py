# Project Name: simple-ai-benchmarking
# File Name: database.py
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


import csv
import os
import argparse
import pprint
import json

import requests
from requests.auth import HTTPBasicAuth

from dataclasses import dataclass

from simple_ai_benchmarking.version_and_metadata import VERSION, REPO_URL


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
    input_shape: str
    model_params: int
    model_num_classes: int

    def to_dict(self) -> dict:
        """Converts the data class instance to a dictionary."""
        return self.__dict__


def get_git_commit_hash_from_package_version():
    version = get_package_version("simple-ai-benchmarking")

    if "git" in version:
        git_commit_hash = version.split("+")[1].split(".")[1]
    else:
        git_commit_hash = "N/A"

    return git_commit_hash


def get_version_importlib(package_name):
    try:
        from importlib.metadata import version, PackageNotFoundError

        return version(package_name)
    except PackageNotFoundError:
        return None
    except ImportError:
        # This should not occur since we're checking Python version before calling
        return None


def get_version_pkg_resources(package_name):
    try:
        import pkg_resources

        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None
    except ImportError:
        return None


def get_package_version(package_name):
    import sys

    if sys.version_info >= (3, 8):
        return get_version_importlib(package_name)
    else:
        return get_version_pkg_resources(package_name)


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


def read_csv_and_create_benchmark_dataset(csv_file_path: str, extra_info: str = None):

    benchmark_datasets = []
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            if "training" in row["bench_info_workload_type"].lower():
                benchmark_type = "training"
            elif "inference" in row["bench_info_workload_type"].lower():
                benchmark_type = "inference"
            else:
                raise ValueError(
                    f"Unknown benchmark type: {row['bench_info_workload_type']}"
                )

            if extra_info is None:
                extra_info = row["sw_info_ai_framework_extra_info"]

            benchmark_data = BenchmarkData(
                ai_framework_name=row["sw_info_ai_framework_name"],
                ai_framework_version=row["sw_info_ai_framework_version"],
                ai_framework_extra_info=extra_info,
                python_version=row["sw_info_python_version"],
                cpu_name=row["hw_info_cpu"],
                accelerator=row["hw_info_accelerator"],
                model=row["bench_info_model"],
                benchmark_type=benchmark_type,
                score_iterations_per_second=float(row["performance_throughput"]),
                benchmark_precision=row["bench_info_compute_precision"],
                power_usage_watts=-1.0,
                batch_size=int(row["bench_info_batch_size"]),
                operating_system=row["sw_info_os_version"],
                benchmark_github_repo_url=REPO_URL,
                benchmark_version=VERSION,
                benchmark_commit_id=get_git_commit_hash_from_package_version(),
                benchmark_date=row["bench_info_date"],
                input_shape=row["bench_info_sample_shape"],
                model_params=int(row["bench_info_num_parameters"]),
                model_num_classes=int(row["bench_info_num_classes"]),
            )

            benchmark_datasets.append(benchmark_data)

    print(f"Loaded {len(benchmark_datasets)} benchmark results from {csv_file_path}.")

    return benchmark_datasets


def publish_results_cli():

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
    parser.add_argument(
        "-e",
        "--extra-info",
        type=str,
        default=None,
        help="Extra information to add to the benchmark results.",
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
            raise ValueError(
                "No suitable authentication provided. Please provide a token or user and password. Check README.md for details."
            )

    else:
        print("API Token:", "*" * (len(api_token) - 3) + api_token[-3:])

    benchmark_datasets = read_csv_and_create_benchmark_dataset(
        args.results_csv_path, args.extra_info
    )

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
