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
from tempfile import NamedTemporaryFile

from simple_ai_benchmarking.database import get_git_commit_hash_from_package_version
from simple_ai_benchmarking.llm_database import read_csv_and_create_llm_benchmark_dataset


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


# Entry point for running the tests
if __name__ == "__main__":
    unittest.main()
