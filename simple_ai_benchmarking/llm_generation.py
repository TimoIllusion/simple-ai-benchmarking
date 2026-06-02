# Project Name: simple-ai-benchmarking
# File Name: llm_generation.py
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


import argparse
import os

from simple_ai_benchmarking.benchmark import process_workloads
from simple_ai_benchmarking.config_structures import (
    AIFramework,
    GenerationModelConfig,
    LLMGenerationConfig,
)
from simple_ai_benchmarking.llm_results import LLMBenchmarkLogger
from simple_ai_benchmarking.workloads.factory import WorkloadFactory
from simple_ai_benchmarking.workloads.llm_workload import (
    OLLAMA_BACKEND,
    OPENAI_COMPATIBLE_BACKEND,
    PYTORCH_GENERATION_BACKEND,
)

SUPPORTED_LLM_BACKENDS = (
    OPENAI_COMPATIBLE_BACKEND,
    OLLAMA_BACKEND,
    PYTORCH_GENERATION_BACKEND,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LLM generation benchmark.")
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_LLM_BACKENDS,
        default=PYTORCH_GENERATION_BACKEND,
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="SimpleTransformerLM")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generated-tokens", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--compute-precision", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--model-params", type=int, default=0)
    parser.add_argument("--ai-framework-version", default="")
    parser.add_argument("--ai-framework-extra-info", default="")
    parser.add_argument("--accelerator", default="unknown")
    parser.add_argument("--weight-source", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--out-file-base", default="llm_results")
    return parser.parse_args()


def _default_base_url(backend: str) -> str:
    if backend == OLLAMA_BACKEND:
        return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if backend == PYTORCH_GENERATION_BACKEND:
        return ""
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")


def build_generation_config_from_args(
    args: argparse.Namespace,
) -> LLMGenerationConfig:
    api_key = args.api_key
    if api_key is None and args.api_key_env:
        api_key = os.environ.get(args.api_key_env)

    # Default to the best local device (cuda/mps/cpu), matching how the CV
    # benchmarks pick their device, unless one was explicitly requested.
    device_name = args.device
    if device_name is None:
        if args.backend == PYTORCH_GENERATION_BACKEND:
            from simple_ai_benchmarking.config_pt_tf import get_device_name_pytorch

            device_name = get_device_name_pytorch()
        else:
            device_name = "cpu"

    ai_framework_version = args.ai_framework_version
    ai_framework_extra_info = args.ai_framework_extra_info
    accelerator = args.accelerator
    weight_source = args.weight_source
    if args.backend == PYTORCH_GENERATION_BACKEND:
        import torch

        ai_framework_version = ai_framework_version or torch.__version__
        ai_framework_extra_info = ai_framework_extra_info or device_name
        compute_precision = args.compute_precision or "FP32"
        quantization = args.quantization or "none"
        weight_source = weight_source or "random_weights"
        if accelerator == "unknown":
            if device_name.startswith("cuda") and torch.cuda.is_available():
                accelerator = torch.cuda.get_device_name(None)
            elif device_name == "mps":
                accelerator = "Apple MPS"
            else:
                accelerator = "CPU"
    else:
        compute_precision = args.compute_precision
        quantization = args.quantization

    return LLMGenerationConfig(
        backend=args.backend,
        device_name=device_name,
        model=args.model,
        requests=args.requests,
        warmup_requests=args.warmup_requests,
        concurrency=args.concurrency,
        prompt_tokens=args.prompt_tokens,
        generated_tokens=args.generated_tokens,
        context_length=args.context_length,
        compute_precision=compute_precision,
        quantization=quantization,
        weight_source=weight_source,
        accelerator=accelerator,
        ai_framework_version=ai_framework_version,
        ai_framework_extra_info=ai_framework_extra_info,
        model_params=args.model_params,
        base_url=args.base_url or _default_base_url(args.backend),
        api_key=api_key,
        timeout_s=args.timeout_s,
        model_cfg=GenerationModelConfig(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            attention_heads=args.attention_heads,
            transformer_layers=args.transformer_layers,
        ),
    )


def run_llm_generation_cli() -> None:
    args = parse_arguments()
    config = build_generation_config_from_args(args)
    workload = WorkloadFactory.create_workload(config, AIFramework.PYTORCH)
    process_workloads(
        [workload],
        out_file_base=args.out_file_base,
        repetitions=args.repetitions,
        result_logger=LLMBenchmarkLogger(),
    )
