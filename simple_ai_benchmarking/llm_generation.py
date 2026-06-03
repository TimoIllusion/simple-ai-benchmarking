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
    HF_CAUSAL_FP4_BACKEND,
    HF_CAUSAL_FP8_BACKEND,
    OLLAMA_BACKEND,
    OPENAI_COMPATIBLE_BACKEND,
    PYTORCH_GENERATION_BACKEND,
    PYTORCH_KV_DECODER_BACKEND,
)

# The Hugging Face backends (full-precision plus the FP8/FP4 low-bit variants) all
# build a model from a Hugging Face repo id and substitute the default repo.
HF_BACKENDS = (
    HF_CAUSAL_BACKEND,
    HF_CAUSAL_FP8_BACKEND,
    HF_CAUSAL_FP4_BACKEND,
)

SUPPORTED_LLM_BACKENDS = (
    OPENAI_COMPATIBLE_BACKEND,
    OLLAMA_BACKEND,
    PYTORCH_GENERATION_BACKEND,
    PYTORCH_KV_DECODER_BACKEND,
    HF_CAUSAL_BACKEND,
    HF_CAUSAL_FP8_BACKEND,
    HF_CAUSAL_FP4_BACKEND,
)

# Local PyTorch backends that build and run a model in-process (vs HTTP serving
# backends). They share device auto-detection and framework metadata defaults.
LOCAL_PYTORCH_BACKENDS = (
    PYTORCH_GENERATION_BACKEND,
    PYTORCH_KV_DECODER_BACKEND,
    HF_CAUSAL_BACKEND,
    HF_CAUSAL_FP8_BACKEND,
    HF_CAUSAL_FP4_BACKEND,
)

# Backends `saib-llm` runs when no --backend is given: the lightweight reference
# transformer, the heavier custom KV-cache decoder, a real Hugging Face model, and
# its FP8/FP4 low-bit variants. Mirrors how the CV benchmark runs several models in
# a single invocation. The low-bit variants need torchao + a recent GPU (FP4 needs
# Blackwell); where unsupported they fail in isolation and the rest still run.
DEFAULT_LLM_BACKENDS = (
    PYTORCH_GENERATION_BACKEND,
    PYTORCH_KV_DECODER_BACKEND,
    HF_CAUSAL_BACKEND,
    HF_CAUSAL_FP8_BACKEND,
    HF_CAUSAL_FP4_BACKEND,
)

# Precision pinned per low-bit HF backend (precision is part of their identity).
HF_LOWBIT_PRECISION = {
    HF_CAUSAL_FP8_BACKEND: "FP8",
    HF_CAUSAL_FP4_BACKEND: "FP4",
}

# Fixed ~1B-parameter geometry for the custom KV-cache decoder. Intentionally not
# exposed as user-tunable presets: the custom LM simply defaults to ~1B. Mirrors
# the defaults baked into KVCacheDecoderLM.
KV_DECODER_DEFAULT_GEOMETRY = dict(
    embedding_dim=2048,
    transformer_layers=16,
    attention_heads=16,
    feedforward_dim=5632,
)

# Display/identity name for the custom KV-cache decoder, so its results are not
# mislabelled with the simple transformer's default --model placeholder.
KV_DECODER_DEFAULT_MODEL = "KVCacheDecoderLM-1B"

# Default Hugging Face causal LM for the huggingface-causal backend. Only the
# model config is fetched and the weights are randomly initialised (approach B),
# so no large checkpoint is downloaded. Qwen3 is open (Apache-2.0, ungated) and
# recent. Override with --model <hf_repo_id>.
DEFAULT_HF_MODEL = "Qwen/Qwen3-1.7B"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LLM generation benchmark.")
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_LLM_BACKENDS,
        default=None,
        help="Generation backend to run. Default: None, which runs all local "
        "workloads (simple transformer, KV-cache decoder, Hugging Face model, "
        "and its FP8/FP4 low-bit variants).",
    )
    parser.add_argument(
        "-w",
        "--workloads",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help="Indices (0-based) of the default workloads to run, e.g. `-w 1` for "
        "only the second or `-w 1 2` for the second and third. Default: None (run "
        "all). Cannot be combined with --backend.",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="SimpleTransformerLM")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
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
    parser.add_argument("--weight-source", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--out-file-base", default="llm_results")
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

    # The custom KV-cache decoder always uses its fixed ~1B geometry; other local
    # backends take the individual geometry flags (the HF backend ignores these
    # and builds from the model's own config).
    if backend == PYTORCH_KV_DECODER_BACKEND:
        geometry = dict(KV_DECODER_DEFAULT_GEOMETRY)
    else:
        geometry = dict(
            embedding_dim=args.embedding_dim,
            transformer_layers=args.transformer_layers,
            attention_heads=args.attention_heads,
            feedforward_dim=args.feedforward_dim,
        )

    # Give each local backend a meaningful model name when the user left --model at
    # its placeholder: HF backends take a real repo id, and the KV-cache decoder
    # gets its own name (otherwise it inherits the simple transformer's label).
    model = args.model
    placeholder = ("", "SimpleTransformerLM")
    if backend in HF_BACKENDS and model in placeholder:
        model = DEFAULT_HF_MODEL
    elif backend == PYTORCH_KV_DECODER_BACKEND and model in placeholder:
        model = KV_DECODER_DEFAULT_MODEL

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
        # The low-bit HF backends pin their precision (it is part of their
        # identity); the heavier backends default to bf16; the lightweight
        # reference transformer keeps fp32 so its established results stay
        # comparable.
        if backend in HF_LOWBIT_PRECISION:
            compute_precision = HF_LOWBIT_PRECISION[backend]
        else:
            default_precision = (
                "BF16"
                if backend in (PYTORCH_KV_DECODER_BACKEND, HF_CAUSAL_BACKEND)
                else "FP32"
            )
            compute_precision = args.compute_precision or default_precision
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


def run_llm_generation_cli() -> None:
    from loguru import logger

    args = parse_arguments()
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
    )
