# Project Name: simple-ai-benchmarking
# File Name: test_publish.py
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

import os
import unittest
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

from simple_ai_benchmarking.database import (
    build_incremental_publisher,
    build_publish_parser,
    enrich_benchmark_data,
    get_git_commit_hash_from_package_version,
    read_csv_and_create_benchmark_dataset,
    register_profiles_from_csv,
)
from simple_ai_benchmarking.llm_database import read_csv_and_create_llm_benchmark_dataset


def _write_csv(content: str) -> str:
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
        csv_file.write(content)
        return csv_file.name


class TestPublishDatabase(unittest.TestCase):

    def test_get_git_commit_hash_from_package_version(self):

        git_commit = get_git_commit_hash_from_package_version()
        self.assertIsNotNone(git_commit)

    def test_read_llm_csv_and_create_dataset(self):
        csv_content = (
            "sw_info_ai_framework_name,sw_info_ai_framework_version,sw_info_ai_framework_extra_info,sw_info_python_version,"
            "sw_info_os_version,hw_info_cpu,hw_info_accelerator,bench_info_benchmark_type,bench_info_backend,bench_info_model,"
            "bench_info_model_params,bench_info_compute_precision,bench_info_quantization,bench_info_context_length,"
            "bench_info_prompt_tokens,bench_info_generated_tokens,bench_info_concurrency,bench_info_date,performance_requests,"
            "performance_duration_s,performance_prompt_tokens_per_second,performance_generated_tokens_per_second,"
            "performance_total_tokens_per_second,performance_time_to_first_token_s,bench_info_benchmark_family,"
            "bench_info_benchmark_spec_name,bench_info_benchmark_spec_version,bench_info_benchmark_profile_id,"
            "bench_info_benchmark_profile_hash,bench_info_benchmark_config_hash,bench_info_benchmark_runner_id,"
            "bench_info_benchmark_runner_hash,bench_info_benchmark_payload_hash\n"
            "llama.cpp,0.0.1,cuda,3.11.0,Linux,AMD Ryzen,NVIDIA RTX 4090,inference,llama.cpp,Meta-Llama-3-8B,"
            "8000000000,FP16,Q4_K_M,4096,128,256,1,2026-05-11T12:00:00,10,60.0,21.3,42.5,63.8,0.24,"
            "llm,saib_llm_generation,1.0,llm_llama_v1,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,"
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,saib.llm.generation.v1,"
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc,"
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\n"
        )
        # Use delete=False and close the handle before reading: Windows does not
        # allow reopening a NamedTemporaryFile by path while its handle is open.
        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
            csv_file.write(csv_content)
            csv_path = csv_file.name
        try:
            benchmark_data = read_csv_and_create_llm_benchmark_dataset(csv_path)
        finally:
            os.remove(csv_path)

        self.assertEqual(len(benchmark_data), 1)
        self.assertEqual(benchmark_data[0].backend, "llama.cpp")
        self.assertEqual(benchmark_data[0].generated_tokens_per_second, 42.5)
        self.assertEqual(benchmark_data[0].benchmark_spec_name, "saib_llm_generation")

    def test_cv_reader_rejects_llm_csv_with_clear_error(self):
        # An LLM CSV (has bench_info_backend, no bench_info_workload_type).
        csv_path = _write_csv(
            "bench_info_backend,bench_info_model\nollama,llama3\n"
        )
        try:
            with self.assertRaises(ValueError) as ctx:
                read_csv_and_create_benchmark_dataset(csv_path)
        finally:
            os.remove(csv_path)
        self.assertIn("saib-pub-llm", str(ctx.exception))

    def test_llm_reader_rejects_cv_csv_with_clear_error(self):
        # A CV CSV (has bench_info_workload_type, no bench_info_backend).
        csv_path = _write_csv(
            "bench_info_workload_type,bench_info_model\nInferenceWorkload,ResNet50\n"
        )
        try:
            with self.assertRaises(ValueError) as ctx:
                read_csv_and_create_llm_benchmark_dataset(csv_path)
        finally:
            os.remove(csv_path)
        self.assertIn("saib-pub", str(ctx.exception))

    def test_build_publish_parser_parses_common_args(self):
        parser = build_publish_parser("desc")

        args = parser.parse_args(
            ["results.csv", "-t", "tok", "--non-interactive"]
        )

        self.assertEqual(args.results_csv_path, "results.csv")
        self.assertEqual(args.token, "tok")
        self.assertTrue(args.non_interactive)
        self.assertEqual(
            args.database_url, "https://timoillusion.pythonanywhere.com"
        )

    def test_enrich_benchmark_data_roundtrips_non_interactively(self):
        @dataclass
        class Tiny:
            a: int
            b: str

            def to_dict(self):
                return self.__dict__

        items = [Tiny(1, "x"), Tiny(2, "y")]
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as json_file:
            json_path = json_file.name
        try:
            result = enrich_benchmark_data(
                items, Tiny, json_path, non_interactive=True
            )
        finally:
            os.remove(json_path)

        self.assertEqual(result, items)

    def test_register_profiles_cli_mocked(self):
        from unittest.mock import patch, MagicMock
        from simple_ai_benchmarking.database import register_profiles_cli

        csv_content = (
            "bench_info_benchmark_family,bench_info_benchmark_spec_name,bench_info_benchmark_spec_version,"
            "bench_info_benchmark_profile_id,bench_info_benchmark_profile_hash,bench_info_benchmark_runner_id,"
            "bench_info_benchmark_runner_hash\n"
            "cv,saib_cv_classification,1.0,cv_resnet50_v1,hash123,runner_id,runner_hash\n"
        )

        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
            csv_file.write(csv_content)
            csv_path = csv_file.name

        try:
            with patch("argparse.ArgumentParser.parse_args") as mock_args, \
                 patch("requests.post") as mock_post:
                
                mock_args.return_value = MagicMock(
                    results_csv_path=csv_path,
                    database_url="https://timoillusion.pythonanywhere.com",
                    token="test_token",
                    user=None,
                    password=None
                )
                
                mock_response = MagicMock()
                mock_response.status_code = 201
                mock_post.return_value = mock_response

                register_profiles_cli()
                
                self.assertEqual(mock_post.call_count, 1)
                args, kwargs = mock_post.call_args
                self.assertEqual(kwargs["json"]["benchmark_profile_hash"], "hash123")
                self.assertEqual(kwargs["headers"]["Authorization"], "Token test_token")
        finally:
            os.remove(csv_path)

    def test_register_profiles_from_csv_deduplicates_profiles(self):
        from unittest.mock import MagicMock, patch

        csv_path = _write_csv(
            "bench_info_benchmark_family,bench_info_benchmark_spec_name,"
            "bench_info_benchmark_spec_version,bench_info_benchmark_profile_id,"
            "bench_info_benchmark_profile_hash,bench_info_benchmark_runner_id,"
            "bench_info_benchmark_runner_hash\n"
            "cv,spec,1,p1,hash1,r1,rhash\n"
            "cv,spec,1,p1,hash1,r1,rhash\n"
            "cv,spec,1,p2,hash2,r1,rhash\n"
        )
        response = MagicMock(status_code=201)
        try:
            with patch("simple_ai_benchmarking.database.requests.post", return_value=response) as post:
                failures = register_profiles_from_csv(
                    csv_path, "http://database.test/", token="tok"
                )
        finally:
            os.remove(csv_path)

        self.assertEqual(failures, 0)
        self.assertEqual(post.call_count, 2)

    def test_incremental_publisher_submits_only_new_tail_and_retries_failures(self):
        from unittest.mock import patch
        from simple_ai_benchmarking import database as db

        rows = ["a"]
        submitted = []
        outcomes = iter([1, 0, 0])

        def submit(datasets, submit_url, token):
            submitted.append(list(datasets))
            return next(outcomes)

        publisher = build_incremental_publisher(
            csv_path="results.csv",
            read_csv_fn=lambda path: list(rows),
            submit_url="http://database.test/benchmarks/submit/",
            database_url="http://database.test",
            token="tok",
        )
        with patch.object(db, "register_profiles_from_csv", return_value=0), patch.object(
            db, "submit_results", side_effect=submit
        ):
            publisher()
            rows.append("b")
            publisher()
            rows.append("c")
            publisher()

        self.assertEqual(submitted, [["a"], ["a", "b"], ["c"]])


class TestSubmissionOutcomes(unittest.TestCase):

    class _Resp:
        def __init__(self, status_code, text=""):
            self.status_code = status_code
            self.text = text

    class _Data:
        def to_dict(self):
            return {"benchmark_profile_hash": "abc"}

    def test_classify_submission_outcomes(self):
        from simple_ai_benchmarking import database as db

        self.assertEqual(
            db._classify_submission(self._Data(), self._Resp(201)), db.SUBMIT_CREATED
        )
        self.assertEqual(
            db._classify_submission(self._Data(), self._Resp(409)), db.SUBMIT_DUPLICATE
        )
        self.assertEqual(
            db._classify_submission(self._Data(), self._Resp(400, "boom")),
            db.SUBMIT_FAILED,
        )

    def test_unknown_profile_failure_prints_register_hint(self):
        import io
        from contextlib import redirect_stdout
        from simple_ai_benchmarking import database as db

        buf = io.StringIO()
        with redirect_stdout(buf):
            db._report_submission_failure(
                self._Data(), self._Resp(400, "Unknown benchmark profile hash.")
            )
        self.assertIn("saib-register", buf.getvalue())

    def test_submit_results_skips_duplicates_and_continues(self):
        from unittest.mock import patch
        from simple_ai_benchmarking import database as db

        outcomes = iter([db.SUBMIT_CREATED, db.SUBMIT_DUPLICATE, db.SUBMIT_CREATED])
        attempted = []

        def fake_submit(data, submit_url, api_token):
            attempted.append(data)
            return next(outcomes)

        with patch.object(db, "submit_benchmark_result_token_auth", fake_submit):
            db.submit_results(["a", "b", "c"], "http://x", "tok")

        # The duplicate ("b") is skipped without aborting; "c" still gets published.
        self.assertEqual(attempted, ["a", "b", "c"])

    def test_submit_results_stops_on_hard_failure(self):
        from unittest.mock import patch
        from simple_ai_benchmarking import database as db

        outcomes = iter([db.SUBMIT_CREATED, db.SUBMIT_FAILED, db.SUBMIT_CREATED])
        attempted = []

        def fake_submit(data, submit_url, api_token):
            attempted.append(data)
            return next(outcomes)

        with patch.object(db, "submit_benchmark_result_token_auth", fake_submit):
            db.submit_results(["a", "b", "c"], "http://x", "tok")

        # A genuine failure still aborts: "c" is never attempted.
        self.assertEqual(attempted, ["a", "b"])


# Entry point for running the tests
if __name__ == "__main__":
    unittest.main()
