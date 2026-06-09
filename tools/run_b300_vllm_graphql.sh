#!/usr/bin/env bash
# Launch only the B300 vLLM benchmark through RunPod GraphQL.
#
# This is the B300-specific counterpart to tools/run_fleet_vllm.sh. It avoids
# REST /v1/pods because that schema can reject new GPU IDs before GraphQL does.
#
# Usage:
#   export RUNPOD_API_KEY=...
#   export AI_BENCHMARK_DATABASE_TOKEN=...
#   tools/run_b300_vllm_graphql.sh --debug
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/tools/run_b300_vllm_graphql.py" "$@"
