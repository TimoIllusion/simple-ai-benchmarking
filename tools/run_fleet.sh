#!/usr/bin/env bash
# Fire a fleet of self-terminating RunPod benchmark pods (CV + LLM), one per GPU.
#
# Each pod is launched with `saib-runpod` (REST dockerStartCmd, no SSH): it installs
# SAIB, runs the benchmarks, publishes to the database, and self-terminates -- on
# success, crash, or a hung step. Nothing connects back, so this returns quickly and
# you monitor the pods in the RunPod console / the benchmark database.
#
# "lowbit only on respective GPUs and images" is enforced here per generation. The
# FP8/FP4 low-bit backends are excluded from the default LLM run everywhere (they are
# not yet producing correct results); they are only launched when opted into below.
#   * Blackwell (B200 / RTX 5090 / RTX PRO Blackwell): torch 2.8 image + [pt,lowbit],
#     default LLM set (-w 0 1 2)           -> handled automatically by saib-runpod.
#   * Hopper / Ada (H100 / RTX 4090 ...): FP8 needs the torch 2.8 image too, so when
#     opted in it is requested EXPLICITLY here (--image + --pip-spec + --llm-args
#     "--backend huggingface-causal-fp8"). Off by default; the torch 2.8 image needs
#     a host driver >= 12.8, so prefer SECURE (datacenter) hosts. Without it these
#     pods fall back to the safe torch 2.4 image and the default LLM set.
#   * Ampere & older (A6000 ...): no lowbit -> default torch 2.4 image + [pt],
#     LLM -w 0 1 2                          -> handled automatically by saib-runpod.
#
# Usage:
#   export RUNPOD_API_KEY=...
#   export AI_BENCHMARK_DATABASE_TOKEN=...
#   tools/run_fleet.sh                 # the default 5-GPU spread below
#   FP8_ON_HOPPER_ADA=1 tools/run_fleet.sh   # also launch the torch 2.8 FP8 opt-in
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
: "${AI_BENCHMARK_DATABASE_TOKEN:?set AI_BENCHMARK_DATABASE_TOKEN}"

RUNPOD="${RUNPOD:-saib-runpod}"
CAP="${CAPACITY_WAIT:-180}"     # keep retrying thin-stock GPUs for a few minutes
FP8="${FP8_ON_HOPPER_ADA:-0}"   # 1 = also launch the (experimental) FP8 opt-in run

BW="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
LB="simple-ai-benchmarking[pt,lowbit]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git@main"

fire() { echo "=== launching: $* ==="; "$RUNPOD" --capacity-wait "$CAP" "$@" || echo "WARN: launch failed for: $*"; }

# --- Blackwell: image + lowbit auto-selected by saib-runpod (default LLM set) --
fire --gpu "NVIDIA B200"               --cloud-type SECURE
fire --gpu "NVIDIA GeForce RTX 5090"   --cloud-type COMMUNITY

# --- Hopper / Ada: default LLM set, plus an optional experimental FP8 opt-in ---
fire --gpu "NVIDIA H100 80GB HBM3"   --cloud-type SECURE
fire --gpu "NVIDIA GeForce RTX 4090" --cloud-type SECURE
if [ "$FP8" = "1" ]; then
  fire --gpu "NVIDIA H100 80GB HBM3"   --cloud-type SECURE --image "$BW" --pip-spec "$LB" --llm-args "--backend huggingface-causal-fp8"
  fire --gpu "NVIDIA GeForce RTX 4090" --cloud-type SECURE --image "$BW" --pip-spec "$LB" --llm-args "--backend huggingface-causal-fp8"
fi

# --- Ampere baseline: no lowbit (auto) ----------------------------------------
fire --gpu "NVIDIA RTX A6000"          --cloud-type SECURE

echo "All launch calls issued. Pods self-terminate when done; watch the RunPod console."
