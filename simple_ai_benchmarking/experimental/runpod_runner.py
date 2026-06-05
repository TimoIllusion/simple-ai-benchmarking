# Project Name: simple-ai-benchmarking
# File Name: runpod_runner.py
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

"""Experimental fire-and-forget benchmark runner on a throwaway RunPod GPU pod.

It creates an on-demand GPU pod whose **container start command** (passed to the
RunPod REST API v1 as a ``dockerStartCmd`` array) installs SAIB, runs the
benchmark(s), publishes the results to the database over HTTPS, and then
**self-terminates the pod** via a shell ``trap`` -- on success, crash, or a hung
step that hits its ``timeout``. There is no SSH: nothing connects back to the pod,
so it works on RunPod **secure cloud** (which usually exposes no public IP) just as
well as community cloud, and the local machine can disconnect immediately after the
create call returns.

Why no SSH (the previous design): driving the run over SSH required the pod to
expose a public IP (secure-cloud pods often don't), a registered SSH key, an
ssh-agent dance for passphrase keys, and a local process that stayed connected for
the whole run. Pushing the work into ``dockerStartCmd`` removes all of that: the
pod is autonomous and tears itself down.

Usage::

    export RUNPOD_API_KEY=...                 # from runpod.io account settings
    export AI_BENCHMARK_DATABASE_TOKEN=...     # database API token
    saib-runpod --gpu "NVIDIA GeForce RTX 4090"          # CV + LLM, auto image
    saib-runpod --gpu "NVIDIA B200" --workload llm        # Blackwell image, auto

Dry run (no API key needed, prints the plan and the exact container script)::

    saib-runpod --dry-run --gpu "NVIDIA B200"

Install the optional dependency (the RunPod SDK is NOT required -- this uses the
REST API over stdlib ``urllib`` -- but installing the extra is harmless)::

    pip install simple-ai-benchmarking[runpod]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git

To launch a whole fleet of GPUs in one go, see ``tools/run_fleet.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

REST_BASE = "https://rest.runpod.io/v1"
DEFAULT_DATABASE_URL = "https://timoillusion.pythonanywhere.com"

# Image per GPU generation.
#  - torch 2.4 / CUDA 12.4 works for everything up to Hopper (sm_90) and is widely
#    cached, so it is the safe default.
#  - Blackwell (sm_100 B200/B300, sm_120 RTX 50xx / RTX PRO Blackwell) is NOT
#    supported by torch 2.4 / CUDA 12.4 at all and needs torch >= 2.7 + CUDA 12.8.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
BLACKWELL_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"

PIP_BASE = (
    "simple-ai-benchmarking[pt]@git+"
    "https://github.com/TimoIllusion/simple-ai-benchmarking.git@main"
)

# RunPod gpuTypeId substrings that identify a Blackwell card. Matched case-folded.
BLACKWELL_MARKERS = (
    "B200",
    "B300",
    "RTX 5090",
    "RTX 5080",
    "RTX 5070",
    "RTX PRO 6000 BLACKWELL",
    "RTX PRO 5000 BLACKWELL",
    "RTX PRO 4500 BLACKWELL",
    "RTX PRO 4000 BLACKWELL",
)

# Default LLM workload index set (mirrors `saib-llm` default order):
#   0 simple-transformer, 1 huggingface-causal (random-init from config, BF16).
# Low-bit (FP8/FP4) benchmarking is handled by the separate `--workload vllm` path.
LLM_W_DEFAULT = "0 1"

# vLLM FP8/FP4 benchmark (`--workload vllm`). vLLM provides the production
# paged-attention KV cache and the low-bit kernels; SAIB only drives it over HTTP via
# the openai-compatible backend, so no torchao and no SAIB [pt] extra are involved.
# This is a separate, opt-in workload -- never part of the default `saib-llm` run.
# One pod serves the model and benchmarks several precisions in sequence (a fresh
# server per precision, since quantization is a launch-time flag) so bf16 vs fp8 vs
# fp4 are compared apples-to-apples on the same GPU.
#  - bf16: native half precision, the baseline.
#  - fp8: online dynamic FP8_E4M3 quant of the bf16 weights; Ada (SM 8.9+) or Blackwell.
#  - fp4 (NVFP4): needs a pre-quantized ModelOpt NVFP4 checkpoint AND a Blackwell GPU
#    (served with --quantization modelopt_fp4); gated to Blackwell + --vllm-nvfp4-model.
# The 7B Instruct model is the throughput-bound regime where quantization actually
# pays off (a 0.5B at low concurrency is overhead-bound and shows ~no fp8 gain).
VLLM_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
VLLM_DEFAULT_PRECISIONS = "bf16,fp8"
VLLM_PORT = 8000
VLLM_MAX_MODEL_LEN = 4096
VLLM_DEFAULT_DISK_GB = 60  # 7B weights + vLLM + HF cache; 40 GB hits Errno 28.

# Benchmark shape for the vLLM legs: a throughput-bound regime (high concurrency,
# short prompt/gen) where fp8/fp4 separate clearly from bf16.
VLLM_REQUESTS = 128
VLLM_CONCURRENCY = 64
VLLM_PROMPT_TOKENS = 256
VLLM_GENERATED_TOKENS = 256

# Pinned vLLM stack (validated on Blackwell sm_120). transformers has no upper bound
# in vLLM's own deps, so pip otherwise grabs 5.x and breaks the tokenizer; hf_transfer
# is required when the image sets HF_HUB_ENABLE_HF_TRANSFER=1. vLLM 0.11.0 has reliable
# SM120 fp8/bf16; NVFP4 dense SM120 kernels are only reliable from 0.13.0, so the fp4
# leg raises the floor.
VLLM_PINS = '"vllm==0.11.0" "transformers==4.57.1" hf_transfer'
VLLM_PINS_FP4 = '"vllm>=0.13.0" "transformers==4.57.1" hf_transfer'
# FlashInfer must be told the target is SM120 explicitly for the NVFP4 path.
VLLM_FP4_ENV = (
    "FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_FORCE_SM=120f "
    "FLASHINFER_DISABLE_VERSION_CHECK=1 "
)

# vLLM ships its own torch build, so install SAIB without the [pt] extra (the
# openai-compatible client is pure HTTP) to avoid fighting over torch/transformers.
VLLM_SAIB_SPEC = (
    "simple-ai-benchmarking@git+"
    "https://github.com/TimoIllusion/simple-ai-benchmarking.git@main"
)

DONE_MARKER = "SAIB_ALL_DONE"


def log(message: str) -> None:
    print(f"[saib-runpod {time.strftime('%H:%M:%S')}] {message}", flush=True)


def is_blackwell(gpu_id: str) -> bool:
    upper = gpu_id.upper()
    return any(marker in upper for marker in BLACKWELL_MARKERS)


@dataclass
class Config:
    api_key: Optional[str]
    db_token: Optional[str]
    gpus: List[str]
    image: str
    pip_spec: str
    workload: str  # "pt" | "llm" | "vllm" | "both"
    pt_args: str
    llm_args: str
    vllm_model: str
    vllm_precisions: str
    vllm_nvfp4_model: str
    vllm_max_model_len: int
    vllm_requests: int
    vllm_concurrency: int
    vllm_prompt_tokens: int
    vllm_generated_tokens: int
    database_url: str
    disk_gb: int
    cloud_type: str
    publish: bool
    keep: bool
    capacity_wait: int
    pt_timeout: int
    llm_timeout: int
    image_was_set: bool = False
    pip_was_set: bool = False
    llm_args_was_set: bool = False
    disk_was_set: bool = False


# --------------------------------------------------------------------------- #
# Generation-aware defaults
# --------------------------------------------------------------------------- #
def resolve_profile(cfg: Config) -> None:
    """Fill image / pip-spec / LLM workload set from the *first* requested GPU's
    generation, unless the user pinned them explicitly.

    Only Blackwell is special-cased automatically, because Blackwell *cannot* run
    on the default image at all -- so picking the torch 2.8 image for it is a
    correctness requirement, not a preference. Low-bit (FP8/FP4) benchmarking is
    done via the separate ``--workload vllm`` path, not the local pt/llm runs."""
    lead = cfg.gpus[0] if cfg.gpus else ""
    blackwell = is_blackwell(lead)

    if cfg.workload == "vllm":
        # vLLM needs a recent torch/CUDA, so default to the cu128 (torch 2.8) image
        # on every GPU (FP8 works on Ada too; that host just needs driver >= 12.8 --
        # prefer SECURE). SAIB is installed torch-free; vLLM is pip-installed in the
        # workload block and brings its own torch.
        if not cfg.image_was_set:
            cfg.image = BLACKWELL_IMAGE
        if not cfg.pip_was_set:
            cfg.pip_spec = VLLM_SAIB_SPEC
        if not cfg.disk_was_set:
            cfg.disk_gb = VLLM_DEFAULT_DISK_GB
        return

    if not cfg.image_was_set:
        cfg.image = BLACKWELL_IMAGE if blackwell else DEFAULT_IMAGE
    if not cfg.pip_was_set:
        cfg.pip_spec = PIP_BASE
    if not cfg.llm_args_was_set:
        cfg.llm_args = f"-w {LLM_W_DEFAULT}"


# --------------------------------------------------------------------------- #
# Container start script (runs ON the pod, exec'd by bash via dockerStartCmd)
# --------------------------------------------------------------------------- #
_THREAD_CAPS = r"""export DEBIAN_FRONTEND=noninteractive
# Cap BLAS/OMP threads: a pod is a container on a big host, so numpy/OpenBLAS/MKL/
# torch otherwise spawn one thread per HOST core (often 100+) on a few allocated
# vCPUs -> oversubscription that looks like a hang. Must be set before any import.
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
command -v git >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq git) || true
cd /workspace 2>/dev/null || cd /root 2>/dev/null || cd /"""

_SELF_TERMINATE = r"""SAIB_TERMINATED=0
cleanup() {
  [ "$SAIB_TERMINATED" = "1" ] && return
  SAIB_TERMINATED=1
  echo "== SAIB self-terminating pod ${RUNPOD_POD_ID} =="
  # python is present in all pytorch images, so teardown needs no curl.
  python -c "import urllib.request,os; r=urllib.request.Request('https://rest.runpod.io/v1/pods/'+os.environ['RUNPOD_POD_ID'],method='DELETE',headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']}); urllib.request.urlopen(r,timeout=30)" \
  || python3 -c "import urllib.request,os; r=urllib.request.Request('https://rest.runpod.io/v1/pods/'+os.environ['RUNPOD_POD_ID'],method='DELETE',headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']}); urllib.request.urlopen(r,timeout=30)" || true
  # Block so the container does NOT exit and get auto-restarted by RunPod before the
  # DELETE takes effect -- an exited container is restarted and the whole benchmark
  # re-runs in a loop (duplicate rows + runaway billing).
  sleep 900
}
trap cleanup EXIT"""


def _pt_block(cfg: Config) -> str:
    args = f" {cfg.pt_args}" if cfg.pt_args else ""
    publish_args = (
        ' --publish-each --non-interactive --database-url "${URL}"'
        if cfg.publish
        else ""
    )
    lines = [
        'echo "== [pt] running (timeout ${PT_TIMEOUT}s) =="',
        f'timeout -k 60 "${{PT_TIMEOUT}}" saib-pt{args}{publish_args} || echo "WARN saib-pt rc=$?"',
    ]
    if cfg.publish:
        lines += [
            'saib-register results_pt.csv -t "${AI_BENCHMARK_DATABASE_TOKEN}" '
            '--database-url "${URL}" --non-interactive || echo "WARN register pt"',
            'saib-pub results_pt.csv -t "${AI_BENCHMARK_DATABASE_TOKEN}" '
            '--database-url "${URL}" --non-interactive || echo "WARN pub pt"',
        ]
    return "\n".join(lines)


def _llm_block(cfg: Config) -> str:
    args = f" {cfg.llm_args}" if cfg.llm_args else ""
    publish_args = (
        ' --publish-each --non-interactive --database-url "${URL}"'
        if cfg.publish
        else ""
    )
    lines = [
        'echo "== [llm] running (timeout ${LLM_TIMEOUT}s) =="',
        f'timeout -k 60 "${{LLM_TIMEOUT}}" saib-llm{args}{publish_args} || echo "WARN saib-llm rc=$?"',
    ]
    if cfg.publish:
        lines += [
            'saib-register llm_results.csv -t "${AI_BENCHMARK_DATABASE_TOKEN}" '
            '--database-url "${URL}" --non-interactive || echo "WARN register llm"',
            'saib-pub-llm llm_results.csv -t "${AI_BENCHMARK_DATABASE_TOKEN}" '
            '--database-url "${URL}" --non-interactive || echo "WARN pub llm"',
        ]
    return "\n".join(lines)


@dataclass(frozen=True)
class _VllmPrecision:
    serve_quant_flag: str        # appended after `vllm serve "model"`
    env_prefix: str              # inline env for the serve command (FlashInfer fp4)
    compute_precision: str       # saib-llm --compute-precision value
    quantization: str            # saib-llm --quantization metadata
    uses_nvfp4_checkpoint: bool  # serve cfg.vllm_nvfp4_model instead of vllm_model
    requires_blackwell: bool     # leg is gated to Blackwell hosts
    pins: str                    # vLLM pip pins this precision needs


# Single source of truth for every vLLM precision leg: serve flags, gating, pins,
# and the recorded metadata. Adding a precision is one entry here (+ an alias).
_VLLM_PRECISIONS = {
    "bf16": _VllmPrecision("", "", "bf16", "none", False, False, VLLM_PINS),
    "fp8": _VllmPrecision(" --quantization fp8", "", "fp8", "fp8", False, False, VLLM_PINS),
    "fp4": _VllmPrecision(
        " --quantization modelopt_fp4", VLLM_FP4_ENV, "fp4", "nvfp4",
        True, True, VLLM_PINS_FP4,
    ),
}
# Accepted CLI spellings that map onto a canonical precision key.
_VLLM_PRECISION_ALIASES = {"nvfp4": "fp4"}


def _normalize_precision(p: str) -> str:
    p = p.strip().lower()
    return _VLLM_PRECISION_ALIASES.get(p, p)


def _vllm_served_model(cfg: Config, spec: _VllmPrecision) -> str:
    return cfg.vllm_nvfp4_model if spec.uses_nvfp4_checkpoint else cfg.vllm_model


def resolve_vllm_precisions(cfg: Config) -> Tuple[List[str], List[str]]:
    """Return (effective, skipped) precision legs for the vLLM workload.

    Unknown precisions raise (a typo like 'int8' must not silently fall through to
    a bf16 server with bf16 metadata). A precision is skipped (and reported) rather
    than launched into a guaranteed failure when its gating isn't met: fp4 (NVFP4)
    needs a Blackwell GPU *and* a pre-quantized checkpoint (--vllm-nvfp4-model)."""
    blackwell = is_blackwell(cfg.gpus[0] if cfg.gpus else "")
    effective: List[str] = []
    skipped: List[str] = []
    for raw in cfg.vllm_precisions.split(","):
        if not raw.strip():
            continue
        p = _normalize_precision(raw)
        spec = _VLLM_PRECISIONS.get(p)
        if spec is None:
            raise SystemExit(
                f"Unknown --vllm-precisions value '{raw.strip()}'. "
                f"Choose from: {', '.join(_VLLM_PRECISIONS)} "
                f"(aliases: {', '.join(_VLLM_PRECISION_ALIASES)})."
            )
        gated = (spec.requires_blackwell and not blackwell) or (
            spec.uses_nvfp4_checkpoint and not cfg.vllm_nvfp4_model
        )
        if gated:
            if p not in skipped:
                skipped.append(p)
            continue
        if p not in effective:
            effective.append(p)
    return effective, skipped


def _vllm_leg(cfg: Config, precision: str) -> List[str]:
    """Lines that serve one precision, benchmark it over openai-compatible, publish
    its own per-precision CSV, and tear the server down before the next leg."""
    spec = _VLLM_PRECISIONS[precision]
    model = _vllm_served_model(cfg, spec)
    quant_serve = spec.serve_quant_flag
    env = spec.env_prefix
    compute_precision = spec.compute_precision
    quant_meta = spec.quantization
    # Record the real GPU as the accelerator (the precision is already carried in
    # --compute-precision/--quantization); a "vllm-fp8" label would overwrite the
    # hardware field and lose which GPU produced the numbers.
    accel = cfg.gpus[0] if cfg.gpus else "vllm"
    base = f"http://localhost:{VLLM_PORT}"
    out_base = f"llm_results_vllm_{precision}"
    log_file = f"/workspace/vllm_{precision}.log"
    # Readiness probe via python (guaranteed present in every pytorch image; curl is
    # not), exit 0 only on HTTP 200 from /health.
    health = (
        "python -c \"import urllib.request,sys; "
        f"sys.exit(0 if urllib.request.urlopen('{base}/health',timeout=5).status==200 "
        'else 1)"'
    )
    lines = [
        f'echo "== [vllm:{precision}] serving {model} on port {VLLM_PORT} =="',
        # Background server; logs to a per-precision file. /health flips to 200 ready.
        f'{env}vllm serve "{model}"{quant_serve} --port {VLLM_PORT} '
        f"--max-model-len {cfg.vllm_max_model_len} --download-dir /workspace/hf "
        f"> {log_file} 2>&1 &",
        "VLLM_PID=$!",
        f'echo "== [vllm:{precision}] waiting for server (up to ~15 min: download + load) =="',
        f"for i in $(seq 1 180); do {health} >/dev/null 2>&1 && break; sleep 5; done",
        f'{health} >/dev/null 2>&1 '
        f'|| {{ echo "WARN vllm[{precision}] not ready"; tail -n 60 {log_file}; }}',
        f'echo "== [vllm:{precision}] running benchmark (timeout ${{LLM_TIMEOUT}}s) =="',
        f'timeout -k 60 "${{LLM_TIMEOUT}}" saib-llm --backend openai-compatible '
        f'--base-url "{base}" --model "{model}" --served-by vllm '
        f'--accelerator "{accel}" --compute-precision {compute_precision} '
        f"--quantization {quant_meta} --requests {cfg.vllm_requests} "
        f"--concurrency {cfg.vllm_concurrency} --prompt-tokens {cfg.vllm_prompt_tokens} "
        f"--generated-tokens {cfg.vllm_generated_tokens} --out-file-base {out_base} "
        f'|| echo "WARN saib-llm vllm[{precision}] rc=$?"',
        # Stop the server and wait for the port to free before the next precision.
        'kill "$VLLM_PID" >/dev/null 2>&1 || true',
        'wait "$VLLM_PID" 2>/dev/null || true',
    ]
    if cfg.publish:
        lines += [
            f'saib-register {out_base}.csv -t "${{AI_BENCHMARK_DATABASE_TOKEN}}" '
            f'--database-url "${{URL}}" --non-interactive || echo "WARN register vllm[{precision}]"',
            f'saib-pub-llm {out_base}.csv -t "${{AI_BENCHMARK_DATABASE_TOKEN}}" '
            f'--database-url "${{URL}}" --non-interactive || echo "WARN pub vllm[{precision}]"',
        ]
    return lines


def _vllm_block(cfg: Config) -> str:
    """Serve the model with vLLM and benchmark several precisions in sequence
    (bf16, fp8, fp4) through SAIB's openai-compatible backend. vLLM owns the KV
    cache and the quantization kernels; SAIB drives the HTTP API and publishes each
    precision as its own comparable result (served_by=vllm)."""
    effective, skipped = resolve_vllm_precisions(cfg)
    # Install the strictest pin set any effective leg needs (fp4 raises the floor).
    pins = VLLM_PINS
    if any(_VLLM_PRECISIONS[p].pins == VLLM_PINS_FP4 for p in effective):
        pins = VLLM_PINS_FP4
    lines = [
        'echo "== [vllm] installing vLLM =="',
        f"pip install {pins} || echo 'WARN vllm install failed'",
    ]
    for p in skipped:
        lines.append(
            f'echo "== [vllm] skipping {p}: needs a Blackwell GPU and --vllm-nvfp4-model =="'
        )
    if not effective:
        lines.append('echo "WARN no vllm precisions to run"')
        return "\n".join(lines)
    for precision in effective:
        lines += _vllm_leg(cfg, precision)
    return "\n".join(lines)


def build_container_script(cfg: Config) -> str:
    """The bash exec'd by the pod's dockerStartCmd. The DB token is read from the
    pod env (``AI_BENCHMARK_DATABASE_TOKEN``), never interpolated into the text, so
    it is not baked into the script string."""
    pip_requirements = [f'"{cfg.pip_spec}"']
    pip_index_args = ""

    blocks = [
        _THREAD_CAPS,
        _SELF_TERMINATE if not cfg.keep else 'echo "== --keep: pod will NOT self-terminate =="',
        f'URL="{cfg.database_url}"',
        f'PT_TIMEOUT="{cfg.pt_timeout}"',
        f'LLM_TIMEOUT="{cfg.llm_timeout}"',
        'echo "== installing SAIB =="',
        "python -m pip install --upgrade pip",
        f"pip install{pip_index_args} {' '.join(pip_requirements)}",
    ]
    if cfg.workload in ("pt", "both"):
        blocks.append(_pt_block(cfg))
    if cfg.workload in ("llm", "both"):
        blocks.append(_llm_block(cfg))
    if cfg.workload == "vllm":
        blocks.append(_vllm_block(cfg))
    blocks.append(f'echo "{DONE_MARKER}"')
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# RunPod REST API
# --------------------------------------------------------------------------- #
def rest_call(method: str, path: str, key: str, body: Optional[dict] = None) -> Tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        REST_BASE + path,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            txt = resp.read().decode()
            return resp.status, (json.loads(txt) if txt else {})
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:600]


def build_pod_body(cfg: Config, script: str) -> dict:
    env = {"RUNPOD_API_KEY": cfg.api_key}
    if cfg.publish:
        env["AI_BENCHMARK_DATABASE_TOKEN"] = cfg.db_token or ""
    return {
        "name": f"saib-{cfg.workload}-{int(time.time())}",
        "imageName": cfg.image,
        "computeType": "GPU",
        "gpuTypeIds": list(cfg.gpus),
        "gpuTypePriority": "availability",
        "cloudType": cfg.cloud_type,
        "gpuCount": 1,
        "containerDiskInGb": cfg.disk_gb,
        "volumeInGb": 0,
        "supportPublicIp": False,
        "dockerStartCmd": ["bash", "-lc", script],
        "env": env,
    }


def create_pod_with_retry(cfg: Config, script: str) -> str:
    """POST the pod, retrying on no-capacity while the ``--capacity-wait`` budget
    lasts. The REST gpuTypeIds list is itself a fallback set tried by RunPod by
    availability, so retrying re-attempts the whole list."""
    deadline = time.time() + max(0, cfg.capacity_wait)
    attempt = 0
    last = ""
    body = build_pod_body(cfg, script)
    while True:
        attempt += 1
        log(f"Requesting pod on {cfg.gpus} ({cfg.cloud_type})...")
        status, resp = rest_call("POST", "/pods", cfg.api_key, body)
        if status in (200, 201) and isinstance(resp, dict) and resp.get("id"):
            pod_id = resp["id"]
            log(f"Pod created: {pod_id}")
            return pod_id
        last = resp if isinstance(resp, str) else json.dumps(resp)
        log(f"Create failed (status {status}): {last[:200]}")
        if time.time() >= deadline:
            break
        backoff = min(30, 5 * attempt)
        remaining = int(deadline - time.time())
        log(f"Retrying in {backoff}s (~{remaining}s of wait budget left).")
        time.sleep(backoff)
    raise SystemExit(
        "Failed to create a pod. Try a longer --capacity-wait, a comma-separated "
        "--gpu fallback list, or --cloud-type COMMUNITY.\nLast error: " + last
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _prompt_secret(label: str) -> str:
    import getpass

    if not sys.stdin.isatty():
        return ""
    try:
        return getpass.getpass(f"{label}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental: run a SAIB benchmark on a self-terminating RunPod GPU pod (no SSH)."
    )
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--db-token", default=os.environ.get("AI_BENCHMARK_DATABASE_TOKEN"))
    parser.add_argument(
        "--gpu",
        default="NVIDIA GeForce RTX 4090",
        help="RunPod gpuTypeId, or a comma-separated fallback list (RunPod picks by availability).",
    )
    parser.add_argument(
        "--workload", choices=["pt", "llm", "vllm", "both"], default="both",
        help="Which benchmark(s) to run on the pod (default: both). 'vllm' serves "
        "the model with vLLM and benchmarks several precisions (bf16/fp8/fp4) in "
        "sequence via the openai-compatible backend; it is standalone (not part of "
        "'both').",
    )
    parser.add_argument(
        "--image", default=None,
        help="Container image. Default: auto by GPU generation (torch 2.4, or torch 2.8 for Blackwell).",
    )
    parser.add_argument(
        "--pip-spec", default=None,
        help="pip requirement installed on the pod. Default: auto ([pt], or a torch-free spec for --workload vllm).",
    )
    parser.add_argument("--pt-args", default="", help="Extra args for saib-pt, e.g. '-w 0'.")
    parser.add_argument(
        "--llm-args", default=None,
        help="Extra args for saib-llm, e.g. '-w 0 1'. Default: the default workload set (-w 0 1).",
    )
    parser.add_argument(
        "--vllm-model", default=VLLM_DEFAULT_MODEL,
        help=f"Model served by vLLM for the bf16/fp8 legs of --workload vllm. "
        f"Default: {VLLM_DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--vllm-precisions", default=VLLM_DEFAULT_PRECISIONS,
        help="Comma-separated precisions to benchmark in sequence on one pod: "
        "'bf16', 'fp8', 'fp4' (a fresh vLLM server per precision). fp4/NVFP4 is "
        "gated to Blackwell + --vllm-nvfp4-model and otherwise skipped. "
        f"Default: {VLLM_DEFAULT_PRECISIONS}.",
    )
    parser.add_argument(
        "--vllm-nvfp4-model", default="",
        help="Pre-quantized ModelOpt NVFP4 checkpoint repo id, served for the fp4 "
        "leg (Blackwell only). Required to include fp4 in --vllm-precisions.",
    )
    parser.add_argument(
        "--vllm-max-model-len", type=int, default=VLLM_MAX_MODEL_LEN,
        help=f"Max model/context length for the vLLM server (default: {VLLM_MAX_MODEL_LEN}).",
    )
    parser.add_argument("--vllm-requests", type=int, default=VLLM_REQUESTS,
                        help=f"Measured requests per vLLM leg (default: {VLLM_REQUESTS}).")
    parser.add_argument("--vllm-concurrency", type=int, default=VLLM_CONCURRENCY,
                        help=f"In-flight concurrent requests per vLLM leg (default: {VLLM_CONCURRENCY}).")
    parser.add_argument("--vllm-prompt-tokens", type=int, default=VLLM_PROMPT_TOKENS,
                        help=f"Prompt length per vLLM leg (default: {VLLM_PROMPT_TOKENS}).")
    parser.add_argument("--vllm-generated-tokens", type=int, default=VLLM_GENERATED_TOKENS,
                        help=f"Generation length per vLLM leg (default: {VLLM_GENERATED_TOKENS}).")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument(
        "--disk-gb", type=int, default=None,
        help="Container disk in GB. Default: 40, or 60 for --workload vllm (7B + vLLM).",
    )
    parser.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    parser.add_argument(
        "--capacity-wait", type=int, default=0,
        help="Seconds to keep retrying while RunPod has no capacity (with backoff). 0 = try once.",
    )
    parser.add_argument("--pt-timeout", type=int, default=3600, help="Per-step cap (s) for saib-pt.")
    parser.add_argument("--llm-timeout", type=int, default=5400, help="Per-step cap (s) for saib-llm.")
    parser.add_argument("--no-publish", action="store_true", help="Run but do not upload results.")
    parser.add_argument(
        "--keep", action="store_true",
        help="Do not self-terminate the pod (debugging). You must terminate it yourself!",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and container script, then exit (no API key needed).",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    gpus = [g.strip() for g in args.gpu.split(",") if g.strip()]
    cfg = Config(
        api_key=args.api_key,
        db_token=args.db_token,
        gpus=gpus,
        image=args.image or DEFAULT_IMAGE,
        pip_spec=args.pip_spec or PIP_BASE,
        workload=args.workload,
        pt_args=args.pt_args,
        llm_args=args.llm_args or "",
        vllm_model=args.vllm_model,
        vllm_precisions=args.vllm_precisions,
        vllm_nvfp4_model=args.vllm_nvfp4_model,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_requests=args.vllm_requests,
        vllm_concurrency=args.vllm_concurrency,
        vllm_prompt_tokens=args.vllm_prompt_tokens,
        vllm_generated_tokens=args.vllm_generated_tokens,
        database_url=args.database_url,
        disk_gb=args.disk_gb if args.disk_gb is not None else 40,
        cloud_type=args.cloud_type,
        publish=not args.no_publish,
        keep=args.keep,
        capacity_wait=args.capacity_wait,
        pt_timeout=args.pt_timeout,
        llm_timeout=args.llm_timeout,
        image_was_set=args.image is not None,
        pip_was_set=args.pip_spec is not None,
        llm_args_was_set=args.llm_args is not None,
        disk_was_set=args.disk_gb is not None,
    )
    resolve_profile(cfg)
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    script = build_container_script(cfg)

    if args.dry_run:
        log(f"Workload: {cfg.workload}  GPUs: {cfg.gpus}  image: {cfg.image}")
        log(f"pip-spec: {cfg.pip_spec}")
        log(f"pt-args: '{cfg.pt_args}'  llm-args: '{cfg.llm_args}'  publish: {cfg.publish}")
        if cfg.workload == "vllm":
            effective, skipped = resolve_vllm_precisions(cfg)
            log(f"vllm-model: {cfg.vllm_model}  precisions: {effective}"
                + (f"  skipped: {skipped}" if skipped else ""))
            log(f"max-model-len: {cfg.vllm_max_model_len}  disk-gb: {cfg.disk_gb}  "
                f"shape: r{cfg.vllm_requests}/c{cfg.vllm_concurrency}/"
                f"p{cfg.vllm_prompt_tokens}/g{cfg.vllm_generated_tokens}")
        print("\n--- container script (dockerStartCmd: bash -lc) ---")
        print(script)
        print("--- end container script ---")
        return 0

    if not cfg.api_key:
        cfg.api_key = _prompt_secret("Paste your RunPod API key (RUNPOD_API_KEY)")
    if not cfg.api_key:
        raise SystemExit("RUNPOD_API_KEY is required (env, --api-key, or interactive prompt).")
    if cfg.publish and not cfg.db_token:
        cfg.db_token = _prompt_secret(
            "Paste your AI Benchmark Database token (AI_BENCHMARK_DATABASE_TOKEN)"
        )
    if cfg.publish and not cfg.db_token:
        raise SystemExit(
            "AI_BENCHMARK_DATABASE_TOKEN is required to publish "
            "(env, --db-token, interactive prompt, or pass --no-publish)."
        )

    # The token lives in the pod env, so rebuild the body now that it is resolved.
    script = build_container_script(cfg)
    pod_id = create_pod_with_retry(cfg, script)
    log(
        f"Pod {pod_id} is installing SAIB and will run {cfg.workload}, "
        + ("publish, " if cfg.publish else "")
        + ("then self-terminate." if not cfg.keep else "and stay up (--keep).")
    )
    log("Fire-and-forget: nothing connects back to the pod. Monitor it in the RunPod console.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
