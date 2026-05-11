import argparse
import datetime
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cpuinfo
import psutil
import requests

from simple_ai_benchmarking.llm_results import (
    LLMBenchInfo,
    LLMBenchmarkLogger,
    LLMBenchmarkResult,
    LLMPerformanceResult,
)
from simple_ai_benchmarking.results import HWInfo, SWInfo


OPENAI_COMPATIBLE_BACKEND = "openai-compatible"
OLLAMA_BACKEND = "ollama"
PYTORCH_SIMPLE_TRANSFORMER_BACKEND = "pytorch-simple-transformer"
SUPPORTED_LLM_BACKENDS = (
    OPENAI_COMPATIBLE_BACKEND,
    OLLAMA_BACKEND,
    PYTORCH_SIMPLE_TRANSFORMER_BACKEND,
)


@dataclass
class LLMInferenceConfig:
    backend: str
    base_url: str
    model: str
    requests: int = 10
    warmup_requests: int = 1
    concurrency: int = 1
    prompt_tokens: int = 128
    generated_tokens: int = 256
    context_length: int = 4096
    timeout_s: float = 120.0
    api_key: Optional[str] = None
    compute_precision: str = ""
    quantization: str = ""
    model_params: int = 0
    ai_framework_version: str = ""
    ai_framework_extra_info: str = ""
    accelerator: str = "unknown"
    device: str = "cpu"
    vocab_size: int = 32000
    embedding_dim: int = 256
    transformer_layers: int = 4
    attention_heads: int = 4


@dataclass
class LLMRequestResult:
    duration_s: float
    prompt_tokens: int
    generated_tokens: int
    time_to_first_token_s: float


class LLMBackendClient:
    def __init__(
        self,
        config: LLMInferenceConfig,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self._local_lock = threading.Lock()
        self._local_model = None
        self._torch = None

        if self.config.backend == PYTORCH_SIMPLE_TRANSFORMER_BACKEND:
            self._setup_pytorch_simple_transformer()

    def generate(self, prompt: str) -> LLMRequestResult:
        if self.config.backend == OPENAI_COMPATIBLE_BACKEND:
            return self._generate_openai_compatible(prompt)
        if self.config.backend == OLLAMA_BACKEND:
            return self._generate_ollama(prompt)
        if self.config.backend == PYTORCH_SIMPLE_TRANSFORMER_BACKEND:
            return self._generate_pytorch_simple_transformer()
        raise ValueError(f"Unsupported LLM backend: {self.config.backend}")

    def _setup_pytorch_simple_transformer(self) -> None:
        import torch

        from simple_ai_benchmarking.models.pt.simple_transformer_lm import (
            SimpleTransformerLanguageModel,
        )

        self._torch = torch
        self._local_device = torch.device(self.config.device)
        self._local_model = SimpleTransformerLanguageModel(
            vocab_size=self.config.vocab_size,
            context_length=self.config.context_length,
            embedding_dim=self.config.embedding_dim,
            num_heads=self.config.attention_heads,
            num_layers=self.config.transformer_layers,
        ).to(self._local_device)
        self._local_model.eval()
        if self.config.model_params == 0:
            self.config.model_params = sum(p.numel() for p in self._local_model.parameters())

    def _generate_openai_compatible(self, prompt: str) -> LLMRequestResult:
        url = self.config.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.generated_tokens,
            "temperature": 0,
            "stream": False,
        }

        start = time.perf_counter()
        response = self.session.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.config.timeout_s,
        )
        duration_s = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})

        return LLMRequestResult(
            duration_s=duration_s,
            prompt_tokens=int(usage.get("prompt_tokens", self.config.prompt_tokens)),
            generated_tokens=int(
                usage.get("completion_tokens", self.config.generated_tokens)
            ),
            time_to_first_token_s=duration_s,
        )

    def _generate_ollama(self, prompt: str) -> LLMRequestResult:
        url = self.config.base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.config.generated_tokens,
                "num_ctx": self.config.context_length,
                "temperature": 0,
            },
        }

        start = time.perf_counter()
        response = self.session.post(url, json=payload, timeout=self.config.timeout_s)
        duration_s = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()

        return LLMRequestResult(
            duration_s=duration_s,
            prompt_tokens=int(data.get("prompt_eval_count", self.config.prompt_tokens)),
            generated_tokens=int(data.get("eval_count", self.config.generated_tokens)),
            time_to_first_token_s=duration_s,
        )

    def _generate_pytorch_simple_transformer(self) -> LLMRequestResult:
        assert self._torch is not None
        assert self._local_model is not None

        input_ids = self._torch.arange(
            self.config.prompt_tokens,
            device=self._local_device,
            dtype=self._torch.long,
        ).unsqueeze(0)
        input_ids = input_ids % self.config.vocab_size

        start = time.perf_counter()
        with self._local_lock:
            self._local_model.generate(input_ids, self.config.generated_tokens)
        duration_s = time.perf_counter() - start

        return LLMRequestResult(
            duration_s=duration_s,
            prompt_tokens=self.config.prompt_tokens,
            generated_tokens=self.config.generated_tokens,
            time_to_first_token_s=duration_s,
        )


class LLMInferenceBenchmark:
    def __init__(
        self,
        config: LLMInferenceConfig,
        client: Optional[LLMBackendClient] = None,
    ) -> None:
        self.config = config
        self.client = client or LLMBackendClient(config)

    def run(self) -> LLMBenchmarkResult:
        prompt = self._build_prompt()
        self._run_warmup(prompt)
        request_results = self._run_requests(prompt)
        return self._build_result(request_results)

    def _build_prompt(self) -> str:
        # Roughly one token per simple word for backend-independent prompt targets.
        words = ["benchmark"] * max(1, self.config.prompt_tokens)
        return " ".join(words)

    def _run_warmup(self, prompt: str) -> None:
        for _ in range(self.config.warmup_requests):
            self.client.generate(prompt)

    def _run_requests(self, prompt: str) -> List[LLMRequestResult]:
        request_results = []
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            futures = [
                executor.submit(self.client.generate, prompt)
                for _ in range(self.config.requests)
            ]
            for future in as_completed(futures):
                request_results.append(future.result())
        self.duration_s = time.perf_counter() - start
        return request_results

    def _build_result(self, request_results: List[LLMRequestResult]) -> LLMBenchmarkResult:
        completed_requests = len(request_results)
        prompt_tokens = self.config.prompt_tokens * completed_requests
        generated_tokens = self.config.generated_tokens * completed_requests
        total_tokens = prompt_tokens + generated_tokens
        duration_s = self.duration_s
        ttft_values = [result.time_to_first_token_s for result in request_results]

        sw_info = SWInfo(
            ai_framework_name=self.config.backend,
            ai_framework_version=self.config.ai_framework_version,
            ai_framework_extra_info=self.config.ai_framework_extra_info,
            python_version=platform.python_version(),
            os_version=platform.platform(aliased=False, terse=False),
        )
        hw_info = HWInfo(
            cpu=cpuinfo.get_cpu_info().get("brand_raw", "unknown"),
            num_cores=os.cpu_count() or 0,
            ram_gb=psutil.virtual_memory().total / 1e9,
            accelerator=self.config.accelerator,
        )
        bench_info = LLMBenchInfo(
            benchmark_type="inference",
            backend=self.config.backend,
            model=self.config.model,
            model_params=self.config.model_params,
            compute_precision=self.config.compute_precision,
            quantization=self.config.quantization,
            context_length=self.config.context_length,
            prompt_tokens=self.config.prompt_tokens,
            generated_tokens=self.config.generated_tokens,
            concurrency=self.config.concurrency,
            date=datetime.datetime.now().isoformat(),
        )
        performance = LLMPerformanceResult(
            requests=self.config.requests,
            duration_s=duration_s,
            prompt_tokens_per_second=prompt_tokens / duration_s if duration_s else 0.0,
            generated_tokens_per_second=generated_tokens / duration_s
            if duration_s
            else 0.0,
            total_tokens_per_second=total_tokens / duration_s if duration_s else 0.0,
            time_to_first_token_s=sum(ttft_values) / len(ttft_values)
            if ttft_values
            else 0.0,
        )
        return LLMBenchmarkResult(
            sw_info=sw_info,
            hw_info=hw_info,
            bench_info=bench_info,
            performance=performance,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LLM inference benchmark.")
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_LLM_BACKENDS,
        default=OPENAI_COMPATIBLE_BACKEND,
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=1)
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--out-file-base", default="llm_results")
    return parser.parse_args()


def _default_base_url(backend: str) -> str:
    if backend == OLLAMA_BACKEND:
        return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if backend == PYTORCH_SIMPLE_TRANSFORMER_BACKEND:
        return ""
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")


def build_config_from_args(args: argparse.Namespace) -> LLMInferenceConfig:
    api_key = args.api_key
    if api_key is None and args.api_key_env:
        api_key = os.environ.get(args.api_key_env)

    ai_framework_version = args.ai_framework_version
    ai_framework_extra_info = args.ai_framework_extra_info
    accelerator = args.accelerator
    if args.backend == PYTORCH_SIMPLE_TRANSFORMER_BACKEND:
        import torch

        ai_framework_version = ai_framework_version or torch.__version__
        ai_framework_extra_info = ai_framework_extra_info or args.device
        compute_precision = args.compute_precision or "FP32"
        quantization = args.quantization or "random_weights"
        if accelerator == "unknown":
            if args.device.startswith("cuda") and torch.cuda.is_available():
                accelerator = torch.cuda.get_device_name(None)
            elif args.device == "mps":
                accelerator = "Apple MPS"
            else:
                accelerator = "CPU"
    else:
        compute_precision = args.compute_precision
        quantization = args.quantization

    return LLMInferenceConfig(
        backend=args.backend,
        base_url=args.base_url or _default_base_url(args.backend),
        model=args.model,
        requests=args.requests,
        warmup_requests=args.warmup_requests,
        concurrency=args.concurrency,
        prompt_tokens=args.prompt_tokens,
        generated_tokens=args.generated_tokens,
        context_length=args.context_length,
        timeout_s=args.timeout_s,
        api_key=api_key,
        compute_precision=compute_precision,
        quantization=quantization,
        model_params=args.model_params,
        ai_framework_version=ai_framework_version,
        ai_framework_extra_info=ai_framework_extra_info,
        accelerator=accelerator,
        device=args.device,
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
    )


def run_llm_inference_cli() -> None:
    args = parse_arguments()
    config = build_config_from_args(args)
    benchmark = LLMInferenceBenchmark(config)
    result = benchmark.run()

    logger = LLMBenchmarkLogger()
    logger.add_result(result)
    logger.pretty_print_summary()
    logger.export_to_csv(args.out_file_base + ".csv")
    try:
        logger.export_to_excel(args.out_file_base + ".xlsx")
    except ModuleNotFoundError:
        pass
