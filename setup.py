# Project Name: simple-ai-benchmarking
# File Name: setup.py
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

import subprocess
import setuptools

from simple_ai_benchmarking.version_and_metadata import VERSION


def get_git_revision_short_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return None


git_revision = get_git_revision_short_hash()

if git_revision:
    version = f"{VERSION}+git.{git_revision}"
else:
    version = VERSION

with open("README.md", "r") as fh:
    long_description = fh.read()

# get version from VERSION file
setuptools.setup(
    name="simple-ai-benchmarking",
    version=version,
    author="Timo Leitritz",
    author_email="placeholder@example.com",
    description="A package for benchmarking various AI models in a simple way.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TimoIllusion/simple-ai-benchmarking",
    packages=setuptools.find_packages(),
    # Ship bundled HF model configs (e.g. Qwen3) so the LLM backends build their
    # architecture offline instead of fetching config.json from huggingface.co.
    package_data={"simple_ai_benchmarking": ["model_configs/*/config.json"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "tqdm",
        "psutil",
        "pandas",
        "tabulate",
        "loguru",
        "pytest",
        "requests",
        "py-cpuinfo",
    ],
    extras_require={
        "pt": [
            "torch>=1.0.0",
            "torchvision",
            "torchaudio",
            # transformers 5.x imports torch.distributed.tensor.device_mesh, which
            # only exists in torch>=2.5; cap at <5 so the huggingface-causal backend
            # works across the wide torch range SAIB targets. Use torch>=2.5 if you
            # want transformers 5.x.
            "transformers<5",
        ],
        # Optional, EXPERIMENTAL: real FP8 / int8 / int4 low-precision kernels via
        # torchao. Use ONLY on torch >= 2.6 and a recent GPU (FP8: Ada/Hopper SM 8.9+,
        # FP4/NVFP4: Blackwell SM100). torchao tracks recent torch closely and is
        # unpinned here, so on older torch (e.g. the CUDA 12.4 / torch 2.4 images)
        # the install can FAIL and take the whole environment down -- use [pt] there
        # instead. This extra may break as torchao evolves.
        "lowbit": ["torchao"],
        # Optional: experimental one-shot benchmarking on a RunPod GPU pod.
        # (SSH uses the system OpenSSH client, so only the RunPod SDK is needed.)
        "runpod": ["runpod"],
        "tfdml": ["tensorflow-cpu==2.10.0", "tensorflow-directml-plugin"],
        "tf": ["tensorflow>=2.3.0"],
        "xlsx": ["openpyxl"],
    },
    entry_points={
        "console_scripts": [
            "saib-tf = simple_ai_benchmarking.entrypoints:run_tf_benchmarks",
            "saib-pt = simple_ai_benchmarking.entrypoints:run_pt_benchmarks",
            "saib-llm = simple_ai_benchmarking.entrypoints:run_llm_benchmarks",
            "saib-pub = simple_ai_benchmarking.entrypoints:publish",
            "saib-pub-llm = simple_ai_benchmarking.entrypoints:publish_llm",
            "saib-register = simple_ai_benchmarking.entrypoints:register_profiles",
            "saib-runpod = simple_ai_benchmarking.experimental.runpod_runner:main",
        ]
    },
)
