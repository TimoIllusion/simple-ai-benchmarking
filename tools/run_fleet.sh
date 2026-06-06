#!/usr/bin/env bash
# Fire a fleet of self-terminating RunPod benchmark pods (CV + LLM), one per GPU.
#
# Each pod is launched with `saib-runpod` (REST dockerStartCmd, no SSH): it installs
# SAIB, runs the benchmarks, publishes to the database, and self-terminates -- on
# success, crash, or a hung step. Nothing connects back, so this returns quickly and
# you monitor the pods in the RunPod console / the benchmark database.
#
# The container image is auto-selected per GPU generation by saib-runpod:
#   * Blackwell (B200 / RTX 5090 / RTX PRO Blackwell): torch 2.8 / CUDA 12.8 image.
#   * Everything else (Hopper / Ada / Ampere ...): torch 2.4 / CUDA 12.4 image.
# All pods run the full-precision default LLM set (-w 0 1). FP8/FP4 benchmarking via
# a vLLM OpenAI endpoint is a separate, opt-in fleet: see tools/run_fleet_vllm.sh.
#
# Usage:
#   export RUNPOD_API_KEY=...
#   export AI_BENCHMARK_DATABASE_TOKEN=...
#   tools/run_fleet.sh                 # the default 5-GPU spread below
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
: "${AI_BENCHMARK_DATABASE_TOKEN:?set AI_BENCHMARK_DATABASE_TOKEN}"

RUNPOD="${RUNPOD:-saib-runpod}"
CAP="${CAPACITY_WAIT:-180}"     # keep retrying thin-stock GPUs for a few minutes

fire() { echo "=== launching: $* ==="; "$RUNPOD" --capacity-wait "$CAP" "$@" || echo "WARN: launch failed for: $*"; }

# --- Blackwell: torch 2.8 image auto-selected by saib-runpod (default LLM set) --
fire --gpu "NVIDIA B200"               --cloud-type SECURE
fire --gpu "NVIDIA GeForce RTX 5090"   --cloud-type COMMUNITY

# --- Hopper / Ada: default torch 2.4 image, default LLM set ---------------------
fire --gpu "NVIDIA H100 80GB HBM3"   --cloud-type SECURE
fire --gpu "NVIDIA GeForce RTX 4090" --cloud-type SECURE

# --- Ampere baseline -----------------------------------------------------------
fire --gpu "NVIDIA RTX A6000"          --cloud-type SECURE

echo "All launch calls issued. Pods self-terminate when done; watch the RunPod console."
