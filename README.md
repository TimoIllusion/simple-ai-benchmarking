# simple-ai-benchmarking (SAIB)

A simple application to quickly run tests on a variety of hardware and software for AI workloads, to get an intuition on the performance. No downloading of big datasets and only a few dependencies. For more sophisticated and complex benchmarking, I recommend to use [MLPerf Benchmarks](https://mlcommons.org/benchmarks/). 

Visit [timoillusion.pythonanywhere.com/benchmarks](https://timoillusion.pythonanywhere.com/benchmarks) to see current benchmark database.

I develop this application in my free time as a hobby.

## Quickstart

1. Install pytorch and/or tensorflow

2. Install SAIB and run pytorch benchmark:

   ```bash
   pip install git+https://github.com/TimoIllusion/simple-ai-benchmarking.git
   saib-pt
   ```  

**Notes**:
- When using a docker container, actual available cpu threads might differ from automatically detected thread counts from math libraries like BLAS, OpenMP, MKL etc. that are used in numpy and pytorch. This can especially happen when limiting available cpu resources with `--cpus` argument during docker container creation. This might cause performance issues and bad benchmark results. That's why it might be necessary to limit the thread counts used by the libraries via:
   ```bash
   export NUM_THREADS=16 # change depending on your system and environment
   ```
   ```bash
   OMP_NUM_THREADS=$NUM_THREADS \
   OPENBLAS_NUM_THREADS=$NUM_THREADS \
   MKL_NUM_THREADS=$NUM_THREADS \
   NUMEXPR_NUM_THREADS=$NUM_THREADS \
   saib-pt
   ```
- The model implementations for pytorch and tensorflow in this benchmark are very similar, but here might be small differences. 
- To install tensorflow and pytorch directly when installing SAIB, you can also install using the following commands:
  
  `pip install simple-ai-benchmarking[tf]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git` (installs tensorflow)
   
  `pip install simple-ai-benchmarking[pt]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git` (installs torch)
  
  Usually only CPU will be supported when installing with the two above options. It is recommended to setup pytorch or tensorflow prior.  

## Setup & Usage

1. Create a virtual python environment, e.g. with conda:  

   ```bash
   conda create -n saib python=3.10 -y
   conda activate saib
   ```

2. Install your prefered pytorch or tensorflow version and respective CUDA/ROCm/DML-Plugin/etc into the virtual environment.

3. Clone this repository and install it:

   ```bash
   git clone https://github.com/TimoIllusion/simple-ai-benchmarking.git
   cd simple-ai-benchmarking
   pip install .
   ```

6. Run pt or tf benchmarks in a console with activated environment:

   ```bash
   saib-pt
   ```
   ```bash
   saib-tf
   ```

   Note: Alternatively execute `python run_tf.py` for tf benchmark or `python run_pt.py` for pytorch benchmark.  

   For advanced users: Use `saib-pt -h` for advanced options like selecting specific benchmarks and batch sizes.
   

## LLM Inference Benchmarking

In addition to the vision/CNN workloads, SAIB can benchmark large language model (LLM) inference and report token throughput. Run it with the `saib-llm` entry point. With no arguments it runs the **two default local PyTorch backends** (the simple transformer and a Hugging Face causal LM — Qwen by default, with a real KV cache), requiring no server, just like `saib-pt` runs several models with sensible defaults. The custom ~1B KV-cache decoder and the FP8/FP4 low-bit variants are excluded from the default run but remain runnable via `--backend pytorch-kv-decoder` / `huggingface-causal-fp8` / `-fp4`:

```bash
saib-llm
```

### Install requirements

The local backends need PyTorch, and the `huggingface-causal` backend additionally needs `transformers`. The `pt` extra installs both (plus torchvision/torchaudio):

```bash
pip install simple-ai-benchmarking[pt]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git
```

To also run the low-bit workloads (FP8/FP4 and int8/int4 quantization, including the `huggingface-causal-fp8`/`-fp4` backends), add the `lowbit` extra to pull in `torchao`:

```bash
pip install simple-ai-benchmarking[pt,lowbit]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git
```

> **⚠️ `lowbit` is experimental and requires torch ≥ 2.6 (plus a recent GPU).** `torchao` is unpinned and tracks recent torch closely, so on older torch — e.g. the common CUDA 12.4 / torch 2.4 container images — `pip install …[pt,lowbit]` can **fail and abort the whole install**, leaving nothing runnable. On those images install plain `[pt]` instead (FP8/FP4 aren't usable on that hardware/torch anyway). This extra may break as `torchao` evolves.

- **`transformers`** is required for the `huggingface-causal` backends. If it is missing — or installed but incompatible with your torch — that workload fails with `Could not import module 'AutoModelForCausalLM'` while the other workloads still run (each workload runs in an isolated process). Note `transformers` 5.x imports `torch.distributed.tensor.device_mesh`, which only exists in **torch ≥2.5**, so on older torch you must use `transformers<5` (what the `pt` extra installs). Quick check:

  ```bash
  python -c "from transformers import AutoModelForCausalLM; import transformers; print('OK transformers', transformers.__version__)"
  ```

  Install it on its own with `pip install 'transformers<5'` (or upgrade to `torch>=2.5` for `transformers` 5.x).

- **`torchao`** is only needed for real low-precision kernels (`--compute-precision FP8`/`FP4` or `--quantization int8`/`int4`, including the `huggingface-causal-fp8`/`-fp4` backends) and a recent GPU — FP8 needs CUDA SM 8.9+ (Ada/Hopper), FP4/NVFP4 needs Blackwell SM100. Install via the `lowbit` extra (`pip install simple-ai-benchmarking[lowbit]@git+...`) or `pip install torchao`. Without it (or on unsupported hardware), those options raise `NotImplementedError`; plain `FP32`/`FP16`/`BF16` casts need no extra dependency. Quick check:

  ```bash
  python -c "import torchao; print('OK torchao', torchao.__version__)"
  ```

### Backends

Seven backends are supported — five self-contained local PyTorch backends and two HTTP backends:

- `pytorch-simple-transformer` — a small synthetic PyTorch transformer, requiring no external server or model weights (handy for quick hardware comparisons):

  ```bash
  saib-llm --backend pytorch-simple-transformer --device cuda
  ```

- `pytorch-kv-decoder` — a self-contained ~1B-parameter decoder with RoPE, SwiGLU and a real KV cache (so time-to-first-token reflects a meaningful prefill). Random weights, no server, defaults to BF16:

  ```bash
  saib-llm --backend pytorch-kv-decoder --device cuda
  ```

- `huggingface-causal` — a real model architecture built **random-initialized from its config** (only `config.json` is fetched, no weight download), default `Qwen/Qwen3-1.7B`, defaults to BF16. Requires `transformers`:

  ```bash
  saib-llm --backend huggingface-causal --model Qwen/Qwen3-1.7B --device cuda
  ```

- `huggingface-causal-fp8` — the same Hugging Face architecture quantized to **FP8** (weights + activations) via torchao. Requires `transformers` + `torchao` and a recent GPU (CUDA SM 8.9+, i.e. Ada/Hopper or newer):

  ```bash
  saib-llm --backend huggingface-causal-fp8 --model Qwen/Qwen3-1.7B --device cuda
  ```

- `huggingface-causal-fp4` — the same architecture quantized to **FP4** (NVFP4) via torchao. Requires `transformers` + `torchao` and an **NVIDIA Blackwell (SM100) GPU**:

  ```bash
  saib-llm --backend huggingface-causal-fp4 --model Qwen/Qwen3-1.7B --device cuda
  ```

- `openai-compatible` — benchmark any server exposing the OpenAI `/v1/chat/completions` API (e.g. vLLM, llama.cpp server, LM Studio, OpenAI itself):

  ```bash
  saib-llm --backend openai-compatible --base-url http://localhost:8000 --model my-model --api-key-env OPENAI_API_KEY
  ```

- `ollama` — benchmark a local [Ollama](https://ollama.com) server (defaults to `http://127.0.0.1:11434`, override with `--base-url` or `OLLAMA_BASE_URL`):

  ```bash
  saib-llm --backend ollama --model llama3
  ```

The benchmark performs configurable warmup and measured requests (optionally concurrent), repeats the measurement and averages the result, and reports prompt, generated, and total tokens per second, time to first token, and total duration. Each repetition runs in an isolated process. Results are written to `llm_results.csv` (and `.xlsx` if `openpyxl` is installed).

Common options (see `saib-llm -h` for the full list):

- `-w` / `--workloads` — when no `--backend` is given, 0-based indices selecting which of the default workloads to run, e.g. `-w 0` for only the first (simple transformer) or `-w 1` for only the second (Hugging Face causal LM). Default: run all (mirrors `saib-pt -w`)
- `--requests` / `--warmup-requests` — number of measured / warmup requests (default `10` / `1`)
- `--repetitions` — number of times the measurement is repeated and averaged (default `3`); for paid HTTP endpoints, lower this to reduce cost
- `--concurrency` — for HTTP backends (`openai-compatible`, `ollama`), the number of in-flight concurrent requests; for the local PyTorch backends, the batch size processed in a single batched forward pass (default `1`)
- `--prompt-tokens` / `--generated-tokens` — prompt and generation lengths (default `2048` / `256`)
- `--context-length` — model context window (default `4096`)
- `--device` — torch device for the local backends, e.g. `cpu`, `cuda`, `mps` (default `cpu`)
- `--compute-precision` — for `pytorch-kv-decoder` and `huggingface-causal`, applied to the model: `FP32`/`FP16`/`BF16` (plain casts), `FP8` (needs `torchao` + recent GPU), or `FP4`/NVFP4 (needs `torchao` + Blackwell SM100). The `huggingface-causal-fp8`/`-fp4` backends pin FP8/FP4. Recorded as metadata for other backends
- `--quantization` — for `pytorch-kv-decoder` and `huggingface-causal`: `none` (default), `int8`, or `int4` (the latter two need `torchao` + recent GPU). Recorded as metadata for other backends
- `--model-params`, `--weight-source`, `--accelerator` — metadata recorded with the result
- `--out-file-base` — output file name base (default `llm_results`)


## Publish to AI Benchmark Database

Currently results can only published by authenticated users, but user creation is manually handled currently. Contact me if you want to publish results.

To publish results to [timoillusion.pythonanywhere.com/benchmarks](https://timoillusion.pythonanywhere.com/benchmarks), execute these commands:

```bash
export AI_BENCHMARK_DATABASE_TOKEN=YOUR_TOKEN  
python publish.py benchmark_results_pt.csv
```

OR  
    
```bash
export AI_BENCHMARK_DATABASE_TOKEN=YOUR_TOKEN  
saib-pub benchmark_results_pt.csv
```

OR

```bash
saib-pub benchmark_results_pt.csv --user YOUR_USER --password YOUR_PASSWORD
```

Note: The arg --token can be used to pass the token directly to the script.

To publish LLM inference results, use `saib-pub-llm` with the CSV produced by `saib-llm`:

```bash
export AI_BENCHMARK_DATABASE_TOKEN=YOUR_TOKEN
saib-pub-llm llm_results.csv
```

Check [timoillusion.pythonanywhere.com/benchmarks](https://timoillusion.pythonanywhere.com/benchmarks) for the results.

### Registering benchmark profiles

Every result row carries a *benchmark profile* — the identity that makes runs comparable (`benchmark_family`, `benchmark_spec_name`/`_version`, `benchmark_profile_id`/`_hash`, `benchmark_runner_id`/`_hash`). Use `saib-register` to register the unique profiles found in a results CSV (CV or LLM) with the database, independently of submitting the measured numbers:

```bash
export AI_BENCHMARK_DATABASE_TOKEN=YOUR_TOKEN
saib-register llm_results.csv
```

OR

```bash
saib-register benchmark_results_pt.csv --user YOUR_USER --password YOUR_PASSWORD
```

It deduplicates by `benchmark_profile_hash` (each distinct profile is registered once), POSTs each to `/benchmarks/profiles/register/`, and uses the same token (`-t` / `AI_BENCHMARK_DATABASE_TOKEN`) or user/password authentication as the publish commands.

### Experimental: run on a throwaway RunPod GPU pod

`saib-runpod` runs the whole loop on a rented [RunPod](https://www.runpod.io/) GPU and the pod **self-terminates when it is done** (on success, crash, or a hung step that hits its timeout). It creates an on-demand pod whose **container start command** installs SAIB, runs the benchmark(s), registers + publishes the results, then deletes the pod. This is experimental and unsupported — capacity is best-effort and you are billed per second while a pod runs.

**No SSH.** The benchmark is driven entirely by the pod's `dockerStartCmd` (sent to the RunPod **REST API v1** as a `["bash","-lc", <script>]` array). Nothing connects back to the pod, so:

- It works on RunPod **secure cloud** (which usually exposes no public IP) just as well as community cloud — no public IP, no SSH key, no ssh-agent needed.
- The launch call returns immediately; your machine can disconnect while the pod runs and tears itself down on its own.
- The only dependency is Python stdlib (`urllib`) — the RunPod SDK is **not** required. (The `[runpod]` extra still exists for your own scripting, but `saib-runpod` doesn't need it.)

The pod self-terminates via a shell `trap` that issues `DELETE /v1/pods/$RUNPOD_POD_ID` (the API key is provided to the pod as an env var for exactly this). A per-step `timeout` guard means a hung `saib-pt`/`saib-llm` can't keep the pod alive forever, and thread caps (`OMP/BLAS=8`) are set before any import to avoid host-core oversubscription.

```bash
export RUNPOD_API_KEY=YOUR_RUNPOD_KEY
export AI_BENCHMARK_DATABASE_TOKEN=YOUR_DATABASE_TOKEN
saib-runpod --gpu "NVIDIA GeForce RTX 4090"          # CV + LLM, image auto-selected
saib-runpod --gpu "NVIDIA B200" --workload llm         # Blackwell image auto-selected
saib-runpod --dry-run --gpu "NVIDIA B200"              # print plan + container script, no API key
```

If `RUNPOD_API_KEY` or `AI_BENCHMARK_DATABASE_TOKEN` are not set (and not passed via `--api-key`/`--db-token`), you are prompted to paste them interactively (hidden input). In a non-interactive shell it errors instead of hanging, so supply them via env/flags there.

**Image and low-bit are auto-selected per GPU generation** (this is "lowbit only on respective GPUs and images"):

| GPU generation | Image | pip extra | Default LLM `-w` | Low-bit |
|---|---|---|---|---|
| **Blackwell** (B200, RTX 5090, RTX PRO Blackwell) | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (torch 2.8 / CUDA 12.8) | `[pt,lowbit]` + `torchao==0.13.0+cu128` | `0 1` | opt-in |
| **Everything else** (Hopper, Ada, Ampere, …) | `runpod/pytorch:2.4.0-...cuda12.4.1` (torch 2.4) | `[pt]` | `0 1` | opt-in |

Blackwell is special-cased automatically because it *cannot* run on the torch 2.4 image at all. **The FP8/FP4 low-bit backends are excluded from the default run everywhere** (currently not producing correct results); the Blackwell image still ships `[pt,lowbit]` so they can be opted into explicitly. FP8 on Hopper/Ada additionally needs the torch 2.8 image, which requires a host driver ≥ 12.8 and can fail to start on older-driver non-Blackwell hosts. Enable low-bit explicitly (prefer SECURE/datacenter hosts):

```bash
saib-runpod --gpu "NVIDIA H100 80GB HBM3" --cloud-type SECURE \
  --image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 \
  --pip-spec "simple-ai-benchmarking[pt,lowbit]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git@main" \
  --llm-args "--backend huggingface-causal-fp8"     # FP8 (Hopper/Ada have no FP4)
```

> ⚠️ Never pair `[pt,lowbit]` with the torch 2.4 image. The automatic RunPod profile pins the CUDA wheel `torchao==0.13.0+cu128`, the release built for its torch 2.8 image; custom image/pip overrides must select a torchao version compatible with their torch build.

#### FP8 / FP4 via vLLM (`--workload vllm`)

The in-process `torchao` low-bit backends above are experimental. For a reliable FP8/FP4 benchmark, `--workload vllm` instead serves a small model with [vLLM](https://docs.vllm.ai) (production paged-attention KV cache + mature quantization kernels) and benchmarks it through SAIB's `openai-compatible` backend. SAIB is installed torch-free (the HTTP client needs no torch); vLLM is pip-installed on the pod and brings its own torch. This workload is **standalone and never part of the default run** — you opt in explicitly.

```bash
# FP8 (online dynamic quant of bf16 weights) — works on Ada/Hopper/Blackwell:
saib-runpod --gpu "NVIDIA GeForce RTX 4090" --cloud-type SECURE --workload vllm

# FP4 (NVFP4) — Blackwell only, needs a pre-quantized ModelOpt NVFP4 checkpoint:
saib-runpod --gpu "NVIDIA B200" --workload vllm \
  --vllm-quant nvfp4 --vllm-model nvidia/<some-nvfp4-checkpoint>
```

The vLLM workload defaults to the torch 2.8 / CUDA 12.8 image on every GPU (so prefer SECURE/datacenter hosts, host driver ≥ 12.8). A small ungated model (`Qwen/Qwen2.5-0.5B-Instruct`, ~1 GB) is used so the checkpoint downloads in seconds. Flags: `--vllm-model`, `--vllm-quant fp8|nvfp4|none`, `--vllm-max-model-len`. FP8 needs no special checkpoint; NVFP4 currently requires a pre-quantized checkpoint passed via `--vllm-model`.

**GPU names** are the RunPod GPU *ids*, e.g. `NVIDIA GeForce RTX 4090`, `NVIDIA H100 80GB HBM3` (H100 SXM), `NVIDIA H200`, `NVIDIA B200` — not the short display names. List them with `python -c "import runpod,os; runpod.api_key=os.environ['RUNPOD_API_KEY']; print('\n'.join(g['id'] for g in runpod.get_gpus()))"`.

Useful flags: `--workload pt|llm|vllm|both` (default `both`; `vllm` is standalone, see above), `--gpu "A,B,C"` (comma-separated fallback list; RunPod picks by availability), `--cloud-type SECURE|COMMUNITY` (default `SECURE`), `--capacity-wait SECONDS` (keep retrying while RunPod has no capacity, with backoff), `--image`/`--pip-spec` (override the auto profile), `--pt-args`/`--llm-args "-w 0 1"` (passed to the benchmark commands), `--pt-timeout`/`--llm-timeout SECONDS` (per-step hang caps), `--no-publish`, `--keep` (do **not** self-terminate — you must delete the pod yourself), and `--dry-run` (print the plan and the exact container script without creating anything — no API key needed). The database token is provided to the pod as an env var, never baked into the image or the script string.

#### Launch a whole fleet

`tools/run_fleet.sh` fires a spread of GPUs in one go (each a self-terminating CV+LLM pod), with the low-bit matrix applied per generation — Blackwell auto, FP8 opt-in on Hopper/Ada, none on Ampere:

```bash
export RUNPOD_API_KEY=...  AI_BENCHMARK_DATABASE_TOKEN=...
tools/run_fleet.sh                       # default 5-GPU spread
FP8_ON_HOPPER_ADA=0 tools/run_fleet.sh   # skip the torch 2.8 FP8 opt-in (most reliable)
```

## Hardware Acceleration for PyTorch and TensorFlow

This section shows how to use various GPUs for training and inference benchmarking.

**PyTorch for NVIDIA GPUs**

1. Run `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` inside your environment (or check https://pytorch.org/get-started/locally/ for more instructions and options). This already comes with CUDA, only NVIDIA drivers are needed to run with gpu.

   Note: Newer versions of PyTorch on Linux automatically install CUDA packages as dependencies.

3. Run `python -c "import torch;print(torch.cuda.is_available())"`

**TensorFlow for NVIDIA GPUs**

1. Run `pip install tensorflow` (`tensorflow<=2.10` for windows native)

2. Run `pip list` and check https://www.tensorflow.org/install/source#gpu for the relevant CUDA/cudnn version for your tensorflow version

4. Install cuda and cudnn with `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0` (in the case of tensorflow 2.10, CUDA 11.2 and CUDNN 8.1 is needed)

   Note: Newer versions of TensorFlow on Linux automatically install CUDA packages as dependencies.

5. Run `python -c "import tensorflow;print(tensorflow.config.list_physical_devices())"` to check if GPU is available

   
**TensorFlow for AMD and Intel GPUs**

For all DirectX 12 capable GPUs, DirectML on Windows and WSL can be used. This is especially handy for AMD and Intel GPUs, since there support is not as widespread as CUDA for NVIDIA GPUs.

1. Install TensorFlow 2.10 (on Windows native) with tfdml plugin (see [tensorflow-directml-plugin](https://github.com/microsoft/tensorflow-directml-plugin) for more information):

   ```bash
   pip install tensorflow-cpu==2.10.0 tensorflow-directml-plugin
   ```

2. Run `python -c "import tensorflow;print(tensorflow.config.list_physical_devices())"` to check if GPU is available

## Upcoming Features

- [ ] Re-add PyTorch `torch.compile` in a robust, opt-in way (e.g. a `--compile` flag). It was removed from the default CV path because compiling at runtime pulls in TorchInductor/Triton and crashes on several setups (CPU/MPS, small GPUs reporting "Not enough SMs"), which silently failed the whole CV benchmark. When re-adding, guard it behind an explicit flag and fall back to the eager model on any compile error.
- [ ] Add config management with yaml file or python files to customize benchmarks more easily
- [ ] Expand testing suite
- [ ] Add more models (Language Models, Timeseries, Object Detection, Segmentation) and model registry
- [ ] Implement interfaces for TensorRT and ONNXRuntime
- [ ] Test Intel oneAPI support (see https://github.com/intel/intel-extension-for-tensorflow)
- [ ] Add plotting of results
- [ ] Add option to install package and put on pypi.org
- [ ] Validate, possibly fix MIXED_PRECISION options or remove them
- [ ] Add support for more datatypes (FP16, INT8, FP8, ...)
- [ ] Add user dialogue for device selection and/or optimize automatic device deduction
- [x] Add LLM inference benchmarking (Ollama, OpenAI-compatible, local PyTorch transformer)
- [x] Create website with database to publish results to
- [x] Refactor code structure with more object orientation and interfaces
- [x] Implement unified architecture for inference/train to use any tf/pytorch model with the same API
- [x] Refactor README and give more detailed installation procedures
- [x] Add very simple mode to very easily benchmark (only 1 simple model, default parameters etc.)
- [x] Test ROCm support
- [x] Add test pipeline
- [x] Refactor code into improved and refined structure
- [x] Remove dependencies on tf_addons and tf_datasets
- [x] Add warmup function
- [x] Implement simple baseline cnn with identical architecture for pytorch/tensorflow
- [x] Use synthetic data
- [x] Improve logging 
- [x] Save results to csv
- [x] Add models using pytorch
- [x] Add automatic mixed precision options

## License

Copyright (C) 2024 Timo Leitritz

This project (simple-ai-benchmarking) is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for the full license text.

## AI Assistance

Development of this project was supported by AI agents (Claude, ChatGPT).
