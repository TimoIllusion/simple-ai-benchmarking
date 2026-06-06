#!/usr/bin/env bash
# Fire a fleet of self-terminating RunPod pods that benchmark a vLLM-served LLM over
# its OpenAI-compatible endpoint -- one pod per GPU. This is the opt-in low-bit
# counterpart to tools/run_fleet.sh (which runs the default CV + local-LLM set).
#
# Each pod (launched with `saib-runpod --workload vllm`, REST dockerStartCmd, no SSH)
# installs SAIB torch-free, pip-installs the pinned vLLM stack, then for each requested
# precision: serves the model with vLLM, benchmarks it through SAIB's openai-compatible
# backend, publishes the result (served_by=vllm), and tears the server down before the
# next precision -- so bf16 vs fp8 are apples-to-apples on the same GPU. The pod then
# self-terminates (success, crash, or a hung step). Nothing connects back.
#
# vLLM brings its own torch, so every GPU uses the torch 2.8 / CUDA 12.8 image
# (auto-selected by saib-runpod for --workload vllm). fp8 + bf16 both run on Ada
# (SM 8.9+) and Blackwell, so the default spread covers one card of each generation.
# fp4/NVFP4 is intentionally left out for now: it needs a pre-quantized ModelOpt
# checkpoint (--vllm-nvfp4-model) AND a Blackwell GPU. To add it later, run a
# Blackwell card with e.g.:
#   VLLM_PRECISIONS=bf16,fp8,fp4 \
#   tools/run_fleet_vllm.sh --vllm-nvfp4-model <nvfp4-repo-id> \
#     --gpu "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
#
# Usage:
#   export RUNPOD_API_KEY=...
#   export AI_BENCHMARK_DATABASE_TOKEN=...
#   tools/run_fleet_vllm.sh                       # default 2-GPU spread below
#   VLLM_MODEL=... VLLM_PRECISIONS=bf16,fp8 tools/run_fleet_vllm.sh
#   tools/run_fleet_vllm.sh --no-publish          # any extra args pass through to each pod
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
: "${AI_BENCHMARK_DATABASE_TOKEN:?set AI_BENCHMARK_DATABASE_TOKEN}"

RUNPOD="${RUNPOD:-saib-runpod}"
CAP="${CAPACITY_WAIT:-180}"       # keep retrying thin-stock GPUs for a few minutes
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_PRECISIONS="${VLLM_PRECISIONS:-bf16,fp8}"   # fp4 is opt-in (Blackwell + checkpoint)

# Extra args ($@) pass through to every launch (e.g. --no-publish, --vllm-nvfp4-model).
EXTRA=("$@")

fire() {
  echo "=== launching: $* ==="
  "$RUNPOD" --capacity-wait "$CAP" \
    --workload vllm --vllm-model "$VLLM_MODEL" --vllm-precisions "$VLLM_PRECISIONS" \
    ${EXTRA[@]+"${EXTRA[@]}"} "$@" || echo "WARN: launch failed for: $*"
}

# --- One Blackwell + one Ada card; both run bf16 + fp8 --------------------------
fire --gpu "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" --cloud-type SECURE   # Blackwell
fire --gpu "NVIDIA RTX 6000 Ada Generation"                    --cloud-type SECURE   # Ada

echo "All launch calls issued. Pods self-terminate when done; watch the RunPod console."
