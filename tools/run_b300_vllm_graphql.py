#!/usr/bin/env python3
"""Launch a B300-only vLLM RunPod pod through GraphQL.

RunPod's REST v1 pod schema can lag newly listed GPU IDs. As of this writing,
the public GPU table lists ``NVIDIA B300 SXM6 AC`` while ``POST /v1/pods`` still
rejects it because the REST enum does not include B300. This script keeps the
same self-terminating vLLM workload used by ``saib-runpod --workload vllm`` but
submits the pod through the GraphQL ``podFindAndDeployOnDemand`` mutation, where
``gpuTypeId`` is a plain string.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simple_ai_benchmarking.experimental import runpod_runner

B300_GPU_ID = "NVIDIA B300 SXM6 AC"
GRAPHQL_URL = "https://api.runpod.io/graphql"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the B300 vLLM benchmark through RunPod GraphQL."
    )
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument(
        "--db-token", default=os.environ.get("AI_BENCHMARK_DATABASE_TOKEN")
    )
    parser.add_argument("--database-url", default=runpod_runner.DEFAULT_DATABASE_URL)
    parser.add_argument(
        "--cloud-type", choices=["SECURE", "COMMUNITY", "ALL"], default="SECURE"
    )
    parser.add_argument("--capacity-wait", type=int, default=180)
    parser.add_argument("--disk-gb", type=int, default=runpod_runner.VLLM_DEFAULT_DISK_GB)
    parser.add_argument("--vllm-model", default=runpod_runner.VLLM_DEFAULT_MODEL)
    parser.add_argument("--vllm-precisions", default=runpod_runner.VLLM_DEFAULT_PRECISIONS)
    parser.add_argument("--vllm-nvfp4-model", default="")
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=runpod_runner.VLLM_MAX_MODEL_LEN,
    )
    parser.add_argument("--vllm-requests", type=int, default=runpod_runner.VLLM_REQUESTS)
    parser.add_argument(
        "--vllm-concurrency", type=int, default=runpod_runner.VLLM_CONCURRENCY
    )
    parser.add_argument(
        "--vllm-prompt-tokens", type=int, default=runpod_runner.VLLM_PROMPT_TOKENS
    )
    parser.add_argument(
        "--vllm-generated-tokens",
        type=int,
        default=runpod_runner.VLLM_GENERATED_TOKENS,
    )
    parser.add_argument("--llm-timeout", type=int, default=5400)
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--pip-spec", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _runner_config(args: argparse.Namespace) -> runpod_runner.Config:
    runner_args = [
        "--gpu",
        B300_GPU_ID,
        "--workload",
        "vllm",
        "--cloud-type",
        "SECURE" if args.cloud_type == "ALL" else args.cloud_type,
        "--capacity-wait",
        str(args.capacity_wait),
        "--disk-gb",
        str(args.disk_gb),
        "--vllm-model",
        args.vllm_model,
        "--vllm-precisions",
        args.vllm_precisions,
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
        "--vllm-requests",
        str(args.vllm_requests),
        "--vllm-concurrency",
        str(args.vllm_concurrency),
        "--vllm-prompt-tokens",
        str(args.vllm_prompt_tokens),
        "--vllm-generated-tokens",
        str(args.vllm_generated_tokens),
        "--llm-timeout",
        str(args.llm_timeout),
        "--database-url",
        args.database_url,
    ]
    if args.vllm_nvfp4_model:
        runner_args += ["--vllm-nvfp4-model", args.vllm_nvfp4_model]
    if args.no_publish:
        runner_args.append("--no-publish")
    if args.debug:
        runner_args.append("--debug")
    if args.run_id:
        runner_args += ["--run-id", args.run_id]
    if args.pip_spec:
        runner_args += ["--pip-spec", args.pip_spec]
    parsed = runpod_runner.parse_args(runner_args)
    cfg = runpod_runner.build_config(parsed)
    cfg.api_key = args.api_key
    cfg.db_token = args.db_token
    return cfg


def _graphql_input(cfg: runpod_runner.Config, script: str, cloud_type: str) -> dict:
    env = [{"key": "SAIB_RUNPOD_API_KEY", "value": cfg.api_key or ""}]
    if cfg.publish or cfg.debug:
        env.append(
            {
                "key": "AI_BENCHMARK_DATABASE_TOKEN",
                "value": cfg.db_token or "",
            }
        )
    return {
        "name": f"saib-vllm-b300-{int(time.time())}",
        "imageName": cfg.image,
        "gpuTypeId": B300_GPU_ID,
        "gpuCount": 1,
        "cloudType": cloud_type,
        "containerDiskInGb": cfg.disk_gb,
        "volumeInGb": 0,
        "volumeMountPath": "/workspace",
        "supportPublicIp": False,
        "startJupyter": False,
        "startSsh": False,
        "minVcpuCount": 2,
        "minMemoryInGb": 15,
        "dockerArgs": "bash -lc " + shlex.quote(script),
        "env": env,
    }


def _graphql_call(api_key: str, payload: dict) -> dict:
    mutation = """
    mutation LaunchB300($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) {
        id
        imageName
        machineId
        desiredStatus
        costPerHr
        machine {
          podHostId
        }
      }
    }
    """
    body = json.dumps({"query": mutation, "variables": {"input": payload}}).encode()
    # The GraphQL docs support Authorization headers, but the query-parameter form
    # matches RunPod's pod examples and avoids some intermediary auth handling.
    url = GRAPHQL_URL + "?" + urllib.parse.urlencode({"api_key": api_key})
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "simple-ai-benchmarking/RunPodGraphQL",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode()[:1000]
        raise SystemExit(f"GraphQL request failed ({exc.code}): {msg}") from exc
    if data.get("errors"):
        raise SystemExit("GraphQL errors: " + json.dumps(data["errors"])[:1000])
    return data["data"]["podFindAndDeployOnDemand"]


def main() -> int:
    args = parse_args()
    if not args.dry_run and not args.api_key:
        raise SystemExit("RUNPOD_API_KEY is required.")
    if not args.dry_run and (not args.no_publish or args.debug) and not args.db_token:
        raise SystemExit(
            "AI_BENCHMARK_DATABASE_TOKEN is required to publish or stream debug logs."
        )

    cfg = _runner_config(args)
    script = runpod_runner.build_container_script(cfg)
    payload = _graphql_input(cfg, script, args.cloud_type)

    if args.dry_run:
        redacted = json.loads(json.dumps(payload))
        for item in redacted["env"]:
            item["value"] = "<redacted>" if item["value"] else ""
        print(json.dumps(redacted, indent=2, sort_keys=True))
        return 0

    pod = _graphql_call(args.api_key, payload)
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    print("GPU: " + B300_GPU_ID)
    print("Workload: vllm")
    if cfg.publish:
        print("Publishing: enabled")
    if cfg.debug:
        run_id = cfg.run_id or pod_id
        print(f"Live console: {cfg.database_url.rstrip('/')}/dashboard/runs/{run_id}/")
    print("The pod is self-terminating when the benchmark finishes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
