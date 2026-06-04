import hashlib
import json
from typing import Any, Dict, Iterable, Mapping


CV_BENCHMARK_FAMILY = "cv"
LLM_BENCHMARK_FAMILY = "llm"

CV_SPEC_NAME = "saib_cv_classification"
LLM_SPEC_NAME = "saib_llm_generation"
# CV spec. Bumped 2.0 -> 2.1: default thread-pool caps (OMP/BLAS/MKL=8) change
# measured throughput, so 2.1 results carry fresh profile hashes and never mix
# with 2.0 rows.
SPEC_VERSION = "2.1"
# LLM generation moved to a v2 spec when real time-to-first-token replaced the
# v1 aggregate-only latency. v2 results carry their own profile/runner hashes and
# never mix with v1 rows. Bumped to 2.1 when local-backend concurrency became a
# batched forward pass (concurrency = batch size) instead of competing threads,
# changing throughput methodology for the pytorch backend. Newer backends (the
# KV-cache decoder and the Hugging Face causal-LM backend) keep this spec: they
# follow the same measurement contract and are distinguished by their own
# backend_protocol_class in the profile, so existing backends stay comparable.
# Bumped 2.1 -> 2.2 alongside the default thread-pool caps (OMP/BLAS/MKL=8), which
# change measured throughput; 2.2 results carry fresh profile hashes.
LLM_SPEC_VERSION = "2.2"

CV_RUNNER_ID = "saib.cv.classification.v2"
LLM_RUNNER_ID = "saib.llm.generation.v2"


def canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def _stable_shape(shape: Iterable[Any]) -> list:
    return [int(x) for x in shape]


def build_cv_profile(
    *,
    workload_type: str,
    model: str,
    compute_precision: str,
    batch_size: int,
    sample_shape: Iterable[Any],
    num_classes: int,
) -> Dict[str, Any]:
    return {
        "benchmark_family": CV_BENCHMARK_FAMILY,
        "benchmark_spec_name": CV_SPEC_NAME,
        "benchmark_spec_version": SPEC_VERSION,
        "model": model,
        "workload_type": workload_type,
        "batch_size": int(batch_size),
        "input_shape": _stable_shape(sample_shape),
        "num_classes": int(num_classes),
        "precision_policy": compute_precision,
        "iteration_semantics": "iterations_are_batches",
    }


def build_cv_profile_id(profile: Mapping[str, Any]) -> str:
    shape = "x".join(str(x) for x in profile["input_shape"])
    spec_major = str(profile["benchmark_spec_version"]).split(".", 1)[0]
    return (
        f"cv_{profile['model']}_{profile['workload_type']}_bs{profile['batch_size']}_"
        f"{shape}_c{profile['num_classes']}_v{spec_major}"
    ).lower().replace(" ", "_")


def build_llm_profile(
    *,
    backend: str,
    model: str,
    benchmark_type: str,
    compute_precision: str,
    quantization: str,
    context_length: int,
    prompt_tokens: int,
    generated_tokens: int,
    concurrency: int,
) -> Dict[str, Any]:
    return {
        "benchmark_family": LLM_BENCHMARK_FAMILY,
        "benchmark_spec_name": LLM_SPEC_NAME,
        "benchmark_spec_version": LLM_SPEC_VERSION,
        "backend_protocol_class": backend,
        "model": model,
        "benchmark_type": benchmark_type,
        "prompt_token_target": int(prompt_tokens),
        "generated_token_target": int(generated_tokens),
        "context_length": int(context_length),
        "concurrency": int(concurrency),
        "request_semantics": "requests_are_completed_generation_calls",
        "warmup_semantics": "not_encoded_in_result",
        "streaming_policy": "measure_first_token_latency",
        "ttft_semantics": "first_generated_token_latency_including_prefill",
        "counting_policy": "prompt_and_generated_tokens_counted_separately",
        "sampling_policy": "implementation_default",
        "precision_policy": compute_precision,
        "quantization": quantization,
    }


def build_llm_profile_id(profile: Mapping[str, Any]) -> str:
    return (
        f"llm_{profile['backend_protocol_class']}_{profile['model']}_"
        f"{profile['benchmark_type']}_p{profile['prompt_token_target']}_"
        f"g{profile['generated_token_target']}_ctx{profile['context_length']}_"
        f"c{profile['concurrency']}_v2"
    ).lower().replace(" ", "_")


def assign_benchmark_identity(
    obj: Any,
    *,
    family: str,
    spec_name: str,
    spec_version: str,
    runner_id: str,
    profile: Mapping[str, Any],
    profile_id: str,
) -> None:
    """Populate the shared benchmark_* identity fields on a BenchInfo dataclass.

    Both the CV and LLM BenchInfo variants carry the same eight benchmark_*
    columns derived the same way (profile/config hash of the profile, runner
    hash of the runner id). Centralizing it keeps the two __post_init__ hooks in
    sync instead of duplicating the hashing dance."""
    profile_hash = canonical_hash(profile)
    obj.benchmark_family = family
    obj.benchmark_spec_name = spec_name
    obj.benchmark_spec_version = spec_version
    obj.benchmark_profile_id = profile_id
    obj.benchmark_profile_hash = profile_hash
    obj.benchmark_config_hash = profile_hash
    obj.benchmark_runner_id = runner_id
    obj.benchmark_runner_hash = build_runner_hash(runner_id)


def build_runner_hash(runner_id: str) -> str:
    return canonical_hash(
        {
            "runner_id": runner_id,
            "runner_manifest_version": 1,
            "source_package": "simple-ai-benchmarking",
        }
    )


def build_payload_hash(payload: Mapping[str, Any]) -> str:
    excluded = {"benchmark_payload_hash", "submitted_by", "submitted_at"}
    canonical_payload = {
        key: value for key, value in payload.items() if key not in excluded
    }
    return canonical_hash(canonical_payload)
