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
from loguru import logger

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


# Possible outcomes of a single submission, used to drive the publish loop.
SUBMIT_CREATED = "created"
SUBMIT_DUPLICATE = "duplicate"
SUBMIT_FAILED = "failed"


def _classify_submission(data: "BenchmarkData", response) -> str:
    """Map an HTTP response to a submission outcome.

    A 409 means the server already has this exact result; that is not a hard
    error, so the caller can skip it and keep publishing the remaining runs."""
    if response.status_code == 201:
        print("Successfully added:")
        print(data.to_dict())
        return SUBMIT_CREATED
    if response.status_code == 409:
        print("Skipping duplicate (identical result already on server):")
        print(data.to_dict())
        return SUBMIT_DUPLICATE
    _report_submission_failure(data, response)
    return SUBMIT_FAILED


def _report_submission_failure(data: "BenchmarkData", response) -> None:
    """Print a failed submission and, when the server rejected an unregistered
    profile, point the user at the registration step they are missing."""
    print("Failed to add:")
    print(data.to_dict())
    print("Response:", response.text)
    if "Unknown benchmark profile hash" in response.text:
        print(
            "Hint: this benchmark profile is not registered on the server yet. "
            "Run 'saib-register' (or 'saib-register-llm' for LLM results) against "
            "the same database and CSV before publishing."
        )


def submit_benchmark_result_user_pw_auth(
    data: BenchmarkData, submit_url: str, user: str, pw: str
) -> str:
    """Submit a single benchmark result to the API. Returns a SUBMIT_* outcome."""

    response = requests.post(
        submit_url, json=data.to_dict(), auth=HTTPBasicAuth(user, pw)
    )

    return _classify_submission(data, response)


def submit_benchmark_result_token_auth(
    data: BenchmarkData, submit_url: str, api_token: str
) -> str:
    """Submit a single benchmark result to the API. Returns a SUBMIT_* outcome."""
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }
    response = requests.post(submit_url, json=data.to_dict(), headers=headers)

    return _classify_submission(data, response)


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
        fieldnames = reader.fieldnames or []
        if "bench_info_workload_type" not in fieldnames:
            if "bench_info_backend" in fieldnames:
                raise ValueError(
                    f"{csv_file_path} looks like an LLM results CSV "
                    "(no 'bench_info_workload_type' column). Use 'saib-pub-llm' to publish LLM results."
                )
            raise ValueError(
                f"{csv_file_path} is missing the required 'bench_info_workload_type' "
                "column; is this a SAIB CV results CSV?"
            )
        for row in reader:

            if "training" in row["bench_info_workload_type"].lower():
                benchmark_type = "training"
            elif "inference" in row["bench_info_workload_type"].lower():
                benchmark_type = "inference"
            else:
                raise ValueError(
                    f"Unknown benchmark type: {row['bench_info_workload_type']}"
                )

            # Per-row, not a reassignment of the `extra_info` parameter: otherwise
            # the first row's value would leak onto every subsequent row (e.g. a
            # multi-device CSV would publish every row with row 0's device).
            if extra_info is None:
                row_extra_info = row["sw_info_ai_framework_extra_info"]
            else:
                row_extra_info = extra_info

            benchmark_data = BenchmarkData(
                ai_framework_name=row["sw_info_ai_framework_name"],
                ai_framework_version=row["sw_info_ai_framework_version"],
                ai_framework_extra_info=row_extra_info,
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

            benchmark_datasets.append(benchmark_data)

    print(f"Loaded {len(benchmark_datasets)} benchmark results from {csv_file_path}.")

    return benchmark_datasets


def build_publish_parser(description: str) -> argparse.ArgumentParser:
    """Build the argument parser shared by every publish CLI (CV and LLM)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "results_csv_path",
        type=str,
        help="Path to the CSV file containing the benchmark results.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="https://benchmarks.timoleitritz.dev",
        help="The URL of the AI Benchmark Database.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Run the script in non-interactive mode.",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        default=None,
        help="API token to authenticate with the database.",
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        default=None,
        help="Password to authenticate with the database.",
    )
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        default=None,
        help="User to authenticate with the database.",
    )
    parser.add_argument(
        "-e",
        "--extra-info",
        type=str,
        default=None,
        help="Extra information to add to the benchmark results.",
    )
    return parser


def parse_arguments():
    """Parse command-line arguments."""
    return build_publish_parser(
        "Submit benchmark results to the AI Benchmark Database."
    ).parse_args()


def handle_token_pw_user(args):
    """Get token for authentication."""
    if args.token:
        api_token = args.token
    else:
        api_token = os.environ.get("AI_BENCHMARK_DATABASE_TOKEN")

    if not api_token:
        if args.user and args.password:
            print("User:", args.user)
            print("Password:", "*" * len(args.password))
            return None
        else:
            raise ValueError(
                "No suitable authentication provided. Please provide a token or user and password. Check README.md for details."
            )
    else:
        print("API Token:", "*" * (len(api_token) - 3) + api_token[-3:])
        return api_token


def enrich_benchmark_data(benchmark_datasets, data_class, json_file_path, non_interactive):
    """Round-trip results through an editable JSON file for optional metadata fixes.

    Shared by the CV and LLM publish flows; only the dataclass and json file name
    differ between families."""
    with open(json_file_path, "w") as f:
        json.dump([x.to_dict() for x in benchmark_datasets], f, indent=4)

    if not non_interactive:
        input(
            f"You may now edit the file {json_file_path} to change meta data. Press Enter to continue after reviewing the json ..."
        )

    with open(json_file_path, "r") as f:
        benchmark_datasets = [data_class(**x) for x in json.load(f)]

    if not non_interactive:
        benchmark_datasets = prompt_for_updates(benchmark_datasets)

    return benchmark_datasets


def read_and_enrich_benchmark_data(args):
    """Read and prepare benchmark data from the provided CSV."""
    benchmark_datasets = read_csv_and_create_benchmark_dataset(
        args.results_csv_path, args.extra_info
    )
    return enrich_benchmark_data(
        benchmark_datasets, BenchmarkData, "benchmark_dataset.json", args.non_interactive
    )


def submit_results(
    benchmark_datasets,
    submit_url,
    api_token=None,
    user=None,
    password=None,
) -> int:
    """Submit benchmark results to the database and return the failure count."""
    failures = 0
    for benchmark_data in benchmark_datasets:
        print("Publishing...")

        try:
            if api_token:
                outcome = submit_benchmark_result_token_auth(
                    benchmark_data, submit_url, api_token
                )
            else:
                outcome = submit_benchmark_result_user_pw_auth(
                    benchmark_data, submit_url, user, password
                )
        except Exception as e:
            print(f"Submission failed: {e}")
            failures += 1
            break

        if outcome == SUBMIT_DUPLICATE:
            # Already on the server; skip this run and keep publishing the rest.
            continue
        if outcome == SUBMIT_FAILED:
            print("Submission failed. Exiting...")
            failures += 1
            break
    return failures


def publish_results_cli():
    """Main function that coordinates the publishing of results."""

    args = parse_arguments()
    submit_endpoint = "/benchmarks/submit/"
    submit_url = args.database_url + submit_endpoint

    api_token = handle_token_pw_user(args)

    benchmark_datasets = read_and_enrich_benchmark_data(args)

    submit_results(
        benchmark_datasets,
        submit_url,
        api_token,
        user=args.user,
        password=args.password,
    )


def register_profiles_from_csv(
    csv_path,
    database_url,
    *,
    token=None,
    user=None,
    password=None,
) -> int:
    """Register unique profiles from a CSV and return the failure count."""
    url = database_url.rstrip("/") + "/benchmarks/profiles/register/"

    headers = {}
    auth = None
    if token:
        headers["Authorization"] = f"Token {token}"
    elif user and password:
        auth = HTTPBasicAuth(user, password)
    else:
        raise ValueError("Provide a token or user and password.")

    fields = {
        "benchmark_family": "bench_info_benchmark_family",
        "benchmark_spec_name": "bench_info_benchmark_spec_name",
        "benchmark_spec_version": "bench_info_benchmark_spec_version",
        "benchmark_profile_id": "bench_info_benchmark_profile_id",
        "benchmark_profile_hash": "bench_info_benchmark_profile_hash",
        "benchmark_runner_id": "bench_info_benchmark_runner_id",
        "benchmark_runner_hash": "bench_info_benchmark_runner_hash",
    }

    seen = set()
    failures = 0

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phash = row.get(fields["benchmark_profile_hash"])
            if not phash or phash in seen:
                continue
            seen.add(phash)
            payload = {k: row.get(src, "") for k, src in fields.items()}
            try:
                r = requests.post(
                    url, json=payload, headers=headers, auth=auth, timeout=30
                )
                if r.status_code in (200, 201):
                    print(
                        f"[{r.status_code}] {payload['benchmark_profile_id']} "
                        f"({phash[:12]})"
                    )
                else:
                    failures += 1
                    print(f"[{r.status_code}] FAILED {phash[:12]}: {r.text}")
            except Exception as e:
                failures += 1
                print(f"FAILED {phash[:12]}: {e}")

    if not seen:
        raise ValueError("No profile hashes found in CSV.")
    return failures


def build_incremental_publisher(
    *,
    csv_path,
    read_csv_fn,
    submit_url,
    database_url,
    token,
):
    """Build a callback that registers profiles and publishes newly added rows."""
    published_count = 0

    def publish(result_logger=None):
        nonlocal published_count
        try:
            datasets = read_csv_fn(csv_path)
            new_datasets = datasets[published_count:]
            if not new_datasets:
                return
            registration_failures = register_profiles_from_csv(
                csv_path, database_url, token=token
            )
            if registration_failures:
                raise RuntimeError(
                    f"Failed to register {registration_failures} benchmark profile(s)."
                )
            submission_failures = submit_results(new_datasets, submit_url, token)
            if submission_failures:
                raise RuntimeError(
                    f"Failed to submit {submission_failures} benchmark result(s)."
                )
            published_count = len(datasets)
        except Exception as e:
            logger.warning(f"Could not incrementally publish benchmark results: {e}")

    return publish


def register_profiles_cli():
    """Register benchmark profiles found in a results CSV with the database."""
    parser = argparse.ArgumentParser(
        description="Register benchmark profiles found in a results CSV with the AI Benchmark Database."
    )
    parser.add_argument(
        "results_csv_path",
        type=str,
        help="Path to the CSV file containing the benchmark results.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="https://benchmarks.timoleitritz.dev",
        help="The URL of the AI Benchmark Database.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Run the script in non-interactive mode. Accepted for parity with "
        "saib-pub; registration is already non-interactive when a token "
        "(-t) or AI_BENCHMARK_DATABASE_TOKEN is provided.",
    )
    parser.add_argument("-t", "--token", type=str, default=None)
    parser.add_argument("-u", "--user", type=str, default=None)
    parser.add_argument("-p", "--password", type=str, default=None)
    args = parser.parse_args()

    api_token = handle_token_pw_user(args)
    try:
        failures = register_profiles_from_csv(
            args.results_csv_path,
            args.database_url,
            token=api_token,
            user=args.user,
            password=args.password,
        )
    except (FileNotFoundError, ValueError) as e:
        import sys
        sys.exit(str(e))
    if failures:
        import sys
        sys.exit(1)
