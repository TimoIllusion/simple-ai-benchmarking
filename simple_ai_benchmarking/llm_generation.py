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
    HF_CAUSAL_BACKEND,
    OLLAMA_BACKEND,
    OPENAI_COMPATIBLE_BACKEND,
    PYTORCH_GENERATION_BACKEND,
)

# The Hugging Face backend builds a model from a Hugging Face repo id and
# substitutes the default repo when --model is left at its placeholder.
HF_BACKENDS = (HF_CAUSAL_BACKEND,)

SUPPORTED_LLM_BACKENDS = (
    OPENAI_COMPATIBLE_BACKEND,
    OLLAMA_BACKEND,
    PYTORCH_GENERATION_BACKEND,
    HF_CAUSAL_BACKEND,
)

# Local PyTorch backends that build and run a model in-process (vs HTTP serving
# backends). They share device auto-detection and framework metadata defaults.
LOCAL_PYTORCH_BACKENDS = (
    PYTORCH_GENERATION_BACKEND,
    HF_CAUSAL_BACKEND,
)

# Backends `saib-llm` runs when no --backend is given: the lightweight reference
# transformer and a real Hugging Face causal LM (random-init from its config).
# Mirrors how the CV benchmark runs several models in a single invocation. Low-bit
# (FP8/FP4) benchmarking is intentionally out of scope for the local backends; run
# it through a serving engine (vLLM) via the openai-compatible backend instead.
DEFAULT_LLM_BACKENDS = (
    PYTORCH_GENERATION_BACKEND,
    HF_CAUSAL_BACKEND,
)

# Default Hugging Face causal LM for the huggingface-causal backend. Only the
# model config is fetched and the weights are randomly initialised (approach B),
# so no large checkpoint is downloaded. Qwen2.5 is open (Apache-2.0, ungated); the
# 0.5B size keeps the default run portable (fits modest GPUs, builds in seconds).
# For the headline cross-engine comparison run the 7B explicitly, e.g.
# `--model Qwen/Qwen2.5-7B-Instruct`, and benchmark the same model under vLLM via
# `saib-runpod --workload vllm`. Override with --model <hf_repo_id>.
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LLM generation benchmark.")
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_LLM_BACKENDS,
        default=None,
        help="Generation backend to run. Default: None, which runs the default "
        "local workloads (simple transformer + Hugging Face causal LM). For low-bit "
        "(FP8/FP4) benchmarking, serve the model with vLLM and use --backend "
        "openai-compatible.",
    )
    parser.add_argument(
        "-w",
        "--workloads",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help="Indices (0-based) of the default workloads to run, e.g. `-w 0` for "
        "only the first (simple transformer) or `-w 1` for only the second "
        "(Hugging Face causal LM). Default: None (run all). Cannot be combined "
        "with --backend.",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="SimpleTransformerLM")
    # Defaults put real load on the device: for the local backends concurrency is
    # the batch size of one forward pass, so concurrency=8 / requests=32 exercises
    # batched prefill+decode rather than a single-sequence trickle.
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
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
    parser.add_argument(
        "--served-by",
        default="",
        help="Engine that serves the model, recorded in the benchmark identity "
        "(e.g. 'vllm' when benchmarking a vLLM server over --backend "
        "openai-compatible). Default: derived from the backend.",
    )
    parser.add_argument("--weight-source", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--out-file-base", default="llm_results")
    parser.add_argument(
        "--publish-each",
        action="store_true",
        help="Register and publish each workload immediately after it completes.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run publishing without interactive metadata prompts.",
    )
    parser.add_argument("-t", "--token", default=None)
    parser.add_argument(
        "--database-url",
        default="https://benchmarks.timoleitritz.dev",
        help="The URL of the AI Benchmark Database.",
    )
    return parser.parse_args()


def _default_base_url(backend: str) -> str:
    if backend == OLLAMA_BACKEND:
        return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if backend in LOCAL_PYTORCH_BACKENDS:
        return ""
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")


def build_generation_config_from_args(
    args: argparse.Namespace,
    backend: str = None,
) -> LLMGenerationConfig:
    # `backend` lets the caller build a config for a specific backend (used when
    # one invocation runs several backends); falls back to the parsed --backend.
    backend = backend or args.backend

    api_key = args.api_key
    if api_key is None and args.api_key_env:
        api_key = os.environ.get(args.api_key_env)

    # The simple-transformer backend takes the individual geometry flags; the HF
    # backend ignores these and builds from the model's own config.
    geometry = dict(
        embedding_dim=args.embedding_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
    )

    # Give the HF backend a real repo id when the user left --model at its
    # placeholder (otherwise it would inherit the simple transformer's label).
    model = args.model
    placeholder = ("", "SimpleTransformerLM")
    if backend in HF_BACKENDS and model in placeholder:
        model = DEFAULT_HF_MODEL

    # Default to the best local device (cuda/mps/cpu), matching how the CV
    # benchmarks pick their device, unless one was explicitly requested.
    device_name = args.device
    if device_name is None:
        if backend in LOCAL_PYTORCH_BACKENDS:
            from simple_ai_benchmarking.config_pt_tf import get_device_name_pytorch

            device_name = get_device_name_pytorch()
        else:
            device_name = "cpu"

    ai_framework_version = args.ai_framework_version
    ai_framework_extra_info = args.ai_framework_extra_info
    accelerator = args.accelerator
    weight_source = args.weight_source
    if backend in LOCAL_PYTORCH_BACKENDS:
        import torch

        ai_framework_version = ai_framework_version or torch.__version__
        ai_framework_extra_info = ai_framework_extra_info or device_name
        # The HF causal backend defaults to bf16; the lightweight reference
        # transformer keeps fp32 so its established results stay comparable.
        default_precision = "BF16" if backend == HF_CAUSAL_BACKEND else "FP32"
        compute_precision = args.compute_precision or default_precision
        quantization = args.quantization or "none"
        # Local backends apply no quantization (the torchao path was removed), so
        # accepting --quantization int8/int4 would silently record a precision the
        # run never used. Reject it instead of publishing mislabeled data.
        if quantization != "none":
            raise SystemExit(
                f"--quantization '{quantization}' is not supported for the local "
                f"backend '{backend}'. Low-bit quantization is only available via a "
                "serving engine (e.g. vLLM) over --backend openai-compatible."
            )
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
        backend=backend,
        device_name=device_name,
        model=model,
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
        served_by=args.served_by,
        ai_framework_version=ai_framework_version,
        ai_framework_extra_info=ai_framework_extra_info,
        model_params=args.model_params,
        base_url=args.base_url or _default_base_url(backend),
        api_key=api_key,
        timeout_s=args.timeout_s,
        model_cfg=GenerationModelConfig(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            embedding_dim=geometry["embedding_dim"],
            attention_heads=geometry["attention_heads"],
            transformer_layers=geometry["transformer_layers"],
            feedforward_dim=geometry["feedforward_dim"],
        ),
    )


def select_default_backends(selection):
    """Resolve the default backends to run from optional 0-based `-w` indices."""
    if selection is None:
        return list(DEFAULT_LLM_BACKENDS)
    try:
        return [DEFAULT_LLM_BACKENDS[i] for i in selection]
    except IndexError:
        raise SystemExit(
            f"--workloads indices {list(selection)} out of range; valid indices "
            f"are 0..{len(DEFAULT_LLM_BACKENDS) - 1}."
        )


def build_generation_configs(args: argparse.Namespace):
    """One config per backend to run: the explicit --backend, or the default local
    backends (optionally narrowed by `-w`), mirroring how the CV benchmark runs
    several models at once."""
    if args.backend is not None:
        if getattr(args, "workloads", None):
            raise SystemExit(
                "-w/--workloads selects among the default workloads and cannot be "
                "combined with an explicit --backend."
            )
        return [build_generation_config_from_args(args, args.backend)]
    backends = select_default_backends(getattr(args, "workloads", None))
    return [build_generation_config_from_args(args, backend) for backend in backends]


def _build_incremental_publisher(args):
    if not args.publish_each:
        return None
    token = args.token or os.environ.get("AI_BENCHMARK_DATABASE_TOKEN")
    if not token:
        raise SystemExit(
            "--publish-each requires -t/--token or AI_BENCHMARK_DATABASE_TOKEN."
        )
    from simple_ai_benchmarking.database import build_incremental_publisher
    from simple_ai_benchmarking.llm_database import (
        read_csv_and_create_llm_benchmark_dataset,
    )

    database_url = args.database_url.rstrip("/")
    return build_incremental_publisher(
        csv_path=args.out_file_base + ".csv",
        read_csv_fn=read_csv_and_create_llm_benchmark_dataset,
        submit_url=database_url + "/benchmarks/llm/submit/",
        database_url=database_url,
        token=token,
    )


def run_llm_generation_cli() -> None:
    from loguru import logger

    args = parse_arguments()
    publisher = _build_incremental_publisher(args)
    if args.backend is None:
        logger.info("Available default workloads (use -w to select a subset):")
        for i, backend in enumerate(DEFAULT_LLM_BACKENDS):
            logger.info(f"  [{i}] {backend}")
    configs = build_generation_configs(args)
    if args.backend is None:
        logger.warning(
            "Selected workloads: {}", [c.backend for c in configs]
        )
    workloads = [
        WorkloadFactory.create_workload(config, AIFramework.PYTORCH)
        for config in configs
    ]
    process_workloads(
        workloads,
        out_file_base=args.out_file_base,
        repetitions=args.repetitions,
        result_logger=LLMBenchmarkLogger(),
        on_workload_logged=publisher,
    )
