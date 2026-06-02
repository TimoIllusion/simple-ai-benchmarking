# Project Name: simple-ai-benchmarking
# File Name: llm_workload.py
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


import datetime
import json
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import mean
from typing import List, Tuple

from simple_ai_benchmarking.config_structures import AIStage, LLMGenerationConfig
from simple_ai_benchmarking.llm_results import (
    LLMBenchInfo,
    LLMBenchmarkResult,
    LLMPerformanceResult,
)
from simple_ai_benchmarking.results import collect_hw_info, collect_sw_info
from simple_ai_benchmarking.workloads.ai_workload import AIWorkload


PYTORCH_GENERATION_BACKEND = "pytorch-simple-transformer"
PYTORCH_KV_DECODER_BACKEND = "pytorch-kv-decoder"
HF_CAUSAL_BACKEND = "huggingface-causal"
OPENAI_COMPATIBLE_BACKEND = "openai-compatible"
OLLAMA_BACKEND = "ollama"

# Floating-point precisions applied by a plain dtype cast of the random-weight
# model (no extra dependency). FP8 and int8/int4 are not dtype casts: they need
# real low-precision kernels (torchao) and a recent GPU, handled separately.
PRECISION_TO_DTYPE = {
    "": "float32",
    "FP32": "float32",
    "FP16": "float16",
    "BF16": "bfloat16",
}


@dataclass
class GenerationRequestResult:
    prompt_tokens: int
    generated_tokens: int
    time_to_first_token_s: float


class LLMGenerationWorkload(AIWorkload):
    """Base for token-generation workloads (sibling family to the CV workloads).

    Generation produces a token-oriented result, so build_result_log is overridden
    to emit LLMBenchmarkResult. Where the compute happens (local torch, or an HTTP
    server underneath) is an implementation detail of the subclass hooks."""

    def __init__(self, config: LLMGenerationConfig) -> None:
        super().__init__(config)
        self.cfg: LLMGenerationConfig  # for type hinting
        self._request_results: List[GenerationRequestResult] = []
        self._duration_s: float = 0.0

    def _prepare_synthetic_dataset(self):
        return None

    def prepare_execution(self) -> None:
        pass

    def _warmup(self) -> None:
        if self.cfg.warmup_requests <= 0:
            return
        self._run_warmup()

    def _execute(self) -> None:
        self._request_results, self._duration_s = self._run_measured()

    def _calculate_iterations(self) -> int:
        return self.cfg.requests

    def _get_ai_stage(self) -> AIStage:
        return AIStage.GENERATION

    @abstractmethod
    def _get_backend(self) -> str:
        pass

    @abstractmethod
    def _run_warmup(self) -> None:
        pass

    @abstractmethod
    def _run_measured(self) -> Tuple[List[GenerationRequestResult], float]:
        """Run the measured requests, returning per-request results and wall-clock.

        Generation workloads measure their own duration (including device sync),
        so it is returned here rather than taken from the outer benchmark timer."""

    def build_result_log(self) -> LLMBenchmarkResult:
        sw_info = collect_sw_info(
            self._get_ai_framework_name(),
            self._get_ai_framework_version(),
            self._get_ai_framework_extra_info(),
        )
        hw_info = collect_hw_info(self._get_accelerator_info())

        prompt_tokens = sum(r.prompt_tokens for r in self._request_results)
        generated_tokens = sum(r.generated_tokens for r in self._request_results)
        total_tokens = prompt_tokens + generated_tokens
        duration_s = self._duration_s
        ttft = (
            mean(r.time_to_first_token_s for r in self._request_results)
            if self._request_results
            else 0.0
        )

        bench_info = LLMBenchInfo(
            benchmark_type="inference",
            backend=self._get_backend(),
            model=self.cfg.model,
            model_params=self.cfg.model_params,
            compute_precision=self.cfg.compute_precision,
            quantization=self.cfg.quantization,
            context_length=self.cfg.context_length,
            prompt_tokens=self.cfg.prompt_tokens,
            generated_tokens=self.cfg.generated_tokens,
            concurrency=self.cfg.concurrency,
            date=datetime.datetime.now().isoformat(),
            weight_source=self.cfg.weight_source,
        )
        performance = LLMPerformanceResult(
            requests=self.cfg.requests,
            duration_s=duration_s,
            prompt_tokens_per_second=prompt_tokens / duration_s if duration_s else 0.0,
            generated_tokens_per_second=generated_tokens / duration_s
            if duration_s
            else 0.0,
            total_tokens_per_second=total_tokens / duration_s if duration_s else 0.0,
            time_to_first_token_s=ttft,
        )
        return LLMBenchmarkResult(
            sw_info=sw_info,
            hw_info=hw_info,
            bench_info=bench_info,
            performance=performance,
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__} | {self.cfg.model} | {self.cfg.device_name}"


class PyTorchLocalGeneration(LLMGenerationWorkload):
    """Local PyTorch generation: concurrency is the batch size of one forward pass."""

    def setup(self) -> None:
        import torch

        from simple_ai_benchmarking.models.generation_factory import (
            GenerationModelFactory,
        )

        self._torch = torch
        self._device = torch.device(self.cfg.device_name)
        # Keep the model context window consistent with the recorded one.
        self.cfg.model_cfg.context_length = self.cfg.context_length
        self._model = GenerationModelFactory.create_pytorch_model(
            self.cfg.model_cfg
        ).to(self._device)
        self._model.eval()
        if self.cfg.model_params == 0:
            self.cfg.model_params = sum(p.numel() for p in self._model.parameters())

    def _sync_device(self) -> None:
        """Block until queued accelerator work has finished so timing is real."""
        if self._device.type == "mps":
            self._torch.mps.synchronize()
        elif self._device.type == "cuda":
            self._torch.cuda.synchronize()

    def _generate_batch(self, batch_size: int) -> List[GenerationRequestResult]:
        prompt_tokens = self.cfg.prompt_tokens
        input_ids = (
            self._torch.arange(
                prompt_tokens, device=self._device, dtype=self._torch.long
            )
            .unsqueeze(0)
            .expand(batch_size, prompt_tokens)
            .contiguous()
        )
        input_ids = input_ids % self.cfg.model_cfg.vocab_size

        generated_tokens = max(1, self.cfg.generated_tokens)

        # Time the first generated token (prefill + one decode step) separately so
        # time-to-first-token is a real measurement, not full-generation latency.
        start = time.perf_counter()
        sequence = self._model.generate(input_ids, 1)
        self._sync_device()
        time_to_first_token_s = time.perf_counter() - start

        remaining_tokens = generated_tokens - 1
        if remaining_tokens > 0:
            self._model.generate(sequence, remaining_tokens)
            self._sync_device()

        return [
            GenerationRequestResult(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                time_to_first_token_s=time_to_first_token_s,
            )
            for _ in range(batch_size)
        ]

    def _run_warmup(self) -> None:
        for _ in range(self.cfg.warmup_requests):
            self._generate_batch(self.cfg.concurrency)

    def _run_measured(self) -> Tuple[List[GenerationRequestResult], float]:
        request_results: List[GenerationRequestResult] = []
        start = time.perf_counter()
        remaining = self.cfg.requests
        while remaining > 0:
            batch_size = min(self.cfg.concurrency, remaining)
            request_results.extend(self._generate_batch(batch_size))
            remaining -= batch_size
        duration_s = time.perf_counter() - start
        return request_results, duration_s

    def _get_backend(self) -> str:
        return PYTORCH_GENERATION_BACKEND

    def _get_ai_framework_name(self) -> str:
        return PYTORCH_GENERATION_BACKEND

    def _get_ai_framework_version(self) -> str:
        return self.cfg.ai_framework_version or self._torch.__version__

    def _get_ai_framework_extra_info(self) -> str:
        return self.cfg.ai_framework_extra_info or self.cfg.device_name

    def _get_accelerator_info(self) -> str:
        if self.cfg.accelerator != "unknown":
            return self.cfg.accelerator
        if self.cfg.device_name.startswith("cuda") and self._torch.cuda.is_available():
            return self._torch.cuda.get_device_name(None)
        if self.cfg.device_name == "mps":
            return "Apple MPS"
        return "CPU"

    def _get_model_parameters(self) -> int:
        return self.cfg.model_params or sum(
            p.numel() for p in self._model.parameters()
        )


class _DtypeQuantGeneration(PyTorchLocalGeneration):
    """Shared precision/quantization handling for the local model backends.

    FP32/FP16/BF16 are plain dtype casts of the random-weight model (no extra
    dependency). FP8 (compute_precision) and int8/int4 (quantization) use real
    low-precision kernels via torchao and a recent GPU; without torchao they fail
    with a clear message instead of silently running at a higher precision."""

    _SUPPORTED_PRECISIONS = ("FP32", "FP16", "BF16", "FP8")
    _SUPPORTED_QUANTIZATIONS = ("none", "int8", "int4")

    def _precision(self) -> str:
        return (self.cfg.compute_precision or "FP32").upper()

    def _quantization(self) -> str:
        return (self.cfg.quantization or "none").lower()

    def _resolve_base_dtype(self):
        precision = self._precision()
        if precision == "FP8":
            # The module is built in bf16; torchao casts the Linear layers to fp8.
            return self._torch.bfloat16
        dtype_name = PRECISION_TO_DTYPE.get(precision)
        if dtype_name is None:
            raise ValueError(
                f"Unsupported compute precision '{self.cfg.compute_precision}' for "
                f"{self._get_backend()}. Choose one of: "
                f"{', '.join(self._SUPPORTED_PRECISIONS)}."
            )
        return getattr(self._torch, dtype_name)

    def _apply_precision_quant(self, model):
        """Apply FP8 / int8 / int4 via torchao when requested; otherwise no-op.

        Call after the model is on its target device. Param counting should happen
        before this, since quantization swaps in tensor-subclass weights."""
        precision = self._precision()
        quantization = self._quantization()
        if quantization not in self._SUPPORTED_QUANTIZATIONS:
            raise ValueError(
                f"Unsupported quantization '{self.cfg.quantization}' for "
                f"{self._get_backend()}. Choose one of: "
                f"{', '.join(self._SUPPORTED_QUANTIZATIONS)}."
            )
        if precision != "FP8" and quantization == "none":
            return model
        try:
            from torchao.quantization import quantize_

            try:
                # torchao >= 0.x Config-class API.
                from torchao.quantization import (
                    Float8DynamicActivationFloat8WeightConfig as fp8_config,
                    Int4WeightOnlyConfig as int4_config,
                    Int8WeightOnlyConfig as int8_config,
                )
            except ImportError:
                # Older function-style API.
                from torchao.quantization import (
                    float8_dynamic_activation_float8_weight as fp8_config,
                    int4_weight_only as int4_config,
                    int8_weight_only as int8_config,
                )
        except ImportError as exc:
            raise NotImplementedError(
                f"compute_precision='{self.cfg.compute_precision}' / "
                f"quantization='{self.cfg.quantization}' on {self._get_backend()} "
                "needs torchao and a recent GPU: pip install torchao. "
                "FP32/FP16/BF16 work without it."
            ) from exc
        if precision == "FP8":
            quantize_(model, fp8_config())
        if quantization == "int8":
            quantize_(model, int8_config())
        elif quantization == "int4":
            quantize_(model, int4_config())
        return model


class PyTorchKVDecoderGeneration(_DtypeQuantGeneration):
    """Heavier local generation on a decoder-only transformer with a KV cache.

    Same measurement contract as PyTorchLocalGeneration, but the model keeps
    per-layer key/value caches: time-to-first-token is the prompt prefill, and the
    remaining tokens are produced one-at-a-time against the cache (O(1) per token)
    rather than re-running the whole sequence. Precision (FP32/FP16/BF16) is set by
    casting the random-weight model to the requested dtype."""

    def setup(self) -> None:
        import torch

        from simple_ai_benchmarking.models.generation_factory import (
            GenerationModelFactory,
        )

        self._torch = torch
        self._device = torch.device(self.cfg.device_name)
        self.cfg.model_cfg.context_length = self.cfg.context_length
        dtype = self._resolve_base_dtype()
        model = GenerationModelFactory.create_pytorch_kv_decoder_model(
            self.cfg.model_cfg
        )
        model = model.to(device=self._device, dtype=dtype)
        if self.cfg.model_params == 0:
            self.cfg.model_params = sum(p.numel() for p in model.parameters())
        model = self._apply_precision_quant(model)
        self._model = model.eval()

    def _generate_batch(self, batch_size: int) -> List[GenerationRequestResult]:
        prompt_tokens = self.cfg.prompt_tokens
        input_ids = (
            self._torch.arange(
                prompt_tokens, device=self._device, dtype=self._torch.long
            )
            .unsqueeze(0)
            .expand(batch_size, prompt_tokens)
            .contiguous()
        )
        input_ids = input_ids % self.cfg.model_cfg.vocab_size

        generated_tokens = max(1, self.cfg.generated_tokens)

        # Prefill (= time-to-first-token), then cached single-token decode steps.
        start = time.perf_counter()
        next_token, cache = self._model.prefill(input_ids)
        self._sync_device()
        time_to_first_token_s = time.perf_counter() - start

        for _ in range(generated_tokens - 1):
            next_token, cache = self._model.decode_step(next_token, cache)
        self._sync_device()

        return [
            GenerationRequestResult(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                time_to_first_token_s=time_to_first_token_s,
            )
            for _ in range(batch_size)
        ]

    def _get_backend(self) -> str:
        return PYTORCH_KV_DECODER_BACKEND

    def _get_ai_framework_name(self) -> str:
        return PYTORCH_KV_DECODER_BACKEND


class HuggingFaceCausalGeneration(_DtypeQuantGeneration):
    """Local generation on a real Hugging Face causal LM architecture.

    Builds the model from the repo's config and randomly initialises the weights
    (no checkpoint download), so it exercises a production architecture (e.g.
    Llama) at its true size. `cfg.model` is the Hugging Face repo id. Uses the
    model's built-in KV cache: prefill = time-to-first-token, then cached decode.
    Requires the optional `transformers` dependency; if it is missing the workload
    simply fails (and the surrounding run continues with the other workloads)."""

    def setup(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        self._torch = torch
        self._device = torch.device(self.cfg.device_name)
        dtype = self._resolve_base_dtype()

        config = AutoConfig.from_pretrained(self.cfg.model)
        self._vocab_size = config.vocab_size
        try:
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        except TypeError:
            # Older transformers without the torch_dtype kwarg on from_config.
            model = AutoModelForCausalLM.from_config(config).to(dtype=dtype)
        model = model.to(self._device)
        if self.cfg.model_params == 0:
            self.cfg.model_params = sum(p.numel() for p in model.parameters())
        model = self._apply_precision_quant(model)
        self._model = model.eval()

    def _generate_batch(self, batch_size: int) -> List[GenerationRequestResult]:
        torch = self._torch
        prompt_tokens = self.cfg.prompt_tokens
        input_ids = (
            torch.arange(prompt_tokens, device=self._device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, prompt_tokens)
            .contiguous()
        )
        input_ids = input_ids % self._vocab_size

        generated_tokens = max(1, self.cfg.generated_tokens)

        with torch.no_grad():
            # Prefill (= time-to-first-token), then cached single-token decode.
            start = time.perf_counter()
            outputs = self._model(input_ids=input_ids, use_cache=True)
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            self._sync_device()
            time_to_first_token_s = time.perf_counter() - start

            for _ in range(generated_tokens - 1):
                outputs = self._model(
                    input_ids=next_token, past_key_values=past, use_cache=True
                )
                past = outputs.past_key_values
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            self._sync_device()

        return [
            GenerationRequestResult(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                time_to_first_token_s=time_to_first_token_s,
            )
            for _ in range(batch_size)
        ]

    def _get_backend(self) -> str:
        return HF_CAUSAL_BACKEND

    def _get_ai_framework_name(self) -> str:
        return HF_CAUSAL_BACKEND

    def _get_ai_framework_version(self) -> str:
        if self.cfg.ai_framework_version:
            return self.cfg.ai_framework_version
        try:
            import transformers

            return transformers.__version__
        except Exception:
            return self._torch.__version__


class HTTPGenerationWorkload(LLMGenerationWorkload):
    """Generation against an HTTP serving backend (which may run locally).

    Concurrency here is the number of in-flight concurrent requests: the serving
    stack does its own (server-side) batching, so this load-tests the endpoint."""

    def setup(self) -> None:
        import requests

        if getattr(self, "_session", None) is None:
            self._session = requests.Session()

    def _build_prompt(self) -> str:
        # Roughly one token per simple word for backend-independent prompt targets.
        return " ".join(["benchmark"] * max(1, self.cfg.prompt_tokens))

    def _run_warmup(self) -> None:
        prompt = self._build_prompt()
        for _ in range(self.cfg.warmup_requests):
            self._generate_one(prompt)

    def _run_measured(self) -> Tuple[List[GenerationRequestResult], float]:
        prompt = self._build_prompt()
        request_results: List[GenerationRequestResult] = []
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as executor:
            futures = [
                executor.submit(self._generate_one, prompt)
                for _ in range(self.cfg.requests)
            ]
            for future in as_completed(futures):
                request_results.append(future.result())
        duration_s = time.perf_counter() - start
        return request_results, duration_s

    @abstractmethod
    def _generate_one(self, prompt: str) -> GenerationRequestResult:
        pass

    def _get_ai_framework_version(self) -> str:
        return self.cfg.ai_framework_version

    def _get_ai_framework_extra_info(self) -> str:
        return self.cfg.ai_framework_extra_info

    def _get_accelerator_info(self) -> str:
        return self.cfg.accelerator

    def _get_model_parameters(self) -> int:
        return self.cfg.model_params


class OpenAICompatibleGeneration(HTTPGenerationWorkload):

    def _get_backend(self) -> str:
        return OPENAI_COMPATIBLE_BACKEND

    def _get_ai_framework_name(self) -> str:
        return OPENAI_COMPATIBLE_BACKEND

    def _generate_one(self, prompt: str) -> GenerationRequestResult:
        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.cfg.generated_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        prompt_tokens = self.cfg.prompt_tokens
        generated_tokens = 0
        time_to_first_token_s = None

        start = time.perf_counter()
        with self._session.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.cfg.timeout_s,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        if time_to_first_token_s is None:
                            time_to_first_token_s = time.perf_counter() - start
                        generated_tokens += 1
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                    generated_tokens = int(
                        usage.get("completion_tokens", generated_tokens)
                    )
        duration_s = time.perf_counter() - start

        return GenerationRequestResult(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens or self.cfg.generated_tokens,
            time_to_first_token_s=time_to_first_token_s
            if time_to_first_token_s is not None
            else duration_s,
        )


class OllamaGeneration(HTTPGenerationWorkload):

    def _get_backend(self) -> str:
        return OLLAMA_BACKEND

    def _get_ai_framework_name(self) -> str:
        return OLLAMA_BACKEND

    def _generate_one(self, prompt: str) -> GenerationRequestResult:
        url = self.cfg.base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": self.cfg.generated_tokens,
                "num_ctx": self.cfg.context_length,
                "temperature": 0,
            },
        }

        prompt_tokens = self.cfg.prompt_tokens
        generated_tokens = 0
        time_to_first_token_s = None

        start = time.perf_counter()
        with self._session.post(
            url, json=payload, timeout=self.cfg.timeout_s, stream=True
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("response"):
                    if time_to_first_token_s is None:
                        time_to_first_token_s = time.perf_counter() - start
                    generated_tokens += 1
                if data.get("done"):
                    generated_tokens = int(data.get("eval_count", generated_tokens))
                    prompt_tokens = int(data.get("prompt_eval_count", prompt_tokens))
        duration_s = time.perf_counter() - start

        return GenerationRequestResult(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens or self.cfg.generated_tokens,
            time_to_first_token_s=time_to_first_token_s
            if time_to_first_token_s is not None
            else duration_s,
        )
