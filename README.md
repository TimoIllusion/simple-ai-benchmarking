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

In addition to the vision/CNN workloads, SAIB can benchmark large language model (LLM) inference and report token throughput. Run it with the `saib-llm` entry point. With no arguments it runs **all five local PyTorch backends** (simple transformer, ~1B KV-cache decoder, a Hugging Face architecture, and that architecture in FP8 and FP4), requiring no server, just like `saib-pt` runs several models with sensible defaults:

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

- `-w` / `--workloads` — when no `--backend` is given, 0-based indices selecting which of the default workloads to run, e.g. `-w 1` for only the second (KV-cache decoder) or `-w 1 2` for the second and third. Default: run all (mirrors `saib-pt -w`)
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

`saib-runpod` runs the whole loop on a rented [RunPod](https://www.runpod.io/) GPU and **always terminates the pod afterwards** (even on error or Ctrl-C): it creates an on-demand pod, connects over SSH, installs SAIB, runs the benchmark, registers + publishes the results, then tears the pod down. This is experimental and unsupported — capacity is best-effort and you are billed per second while a pod runs.

It uses your system OpenSSH client (no extra SSH library), so the only optional dependency is the RunPod SDK:

```bash
pip install simple-ai-benchmarking[runpod]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git
```

Prerequisites:

- A **RunPod API key** (`RUNPOD_API_KEY`).
- An **SSH key registered in your RunPod account** (Account → Settings → SSH Public Keys). RunPod injects that key into the pod. Point `--ssh-key` at the matching private key.
- If that private key is **passphrase-protected, load it into ssh-agent first** (`ssh-add ~/.ssh/your_key`) — otherwise SSH can't authenticate non-interactively and the run aborts (the tool preflight-checks this *before* creating a pod, so it won't waste money).

Then:

```bash
export RUNPOD_API_KEY=YOUR_RUNPOD_KEY
export AI_BENCHMARK_DATABASE_TOKEN=YOUR_DATABASE_TOKEN
ssh-add ~/.ssh/id_ed25519              # if your key has a passphrase
saib-runpod --ssh-key ~/.ssh/id_ed25519 --gpu "NVIDIA GeForce RTX 4090" --cloud-type COMMUNITY
```

If `RUNPOD_API_KEY` or `AI_BENCHMARK_DATABASE_TOKEN` are not set (and not passed via `--api-key`/`--db-token`), you are prompted to paste them interactively (hidden input). In a non-interactive shell (e.g. CI or a background job) it errors instead of hanging, so supply them via env/flags there.

**GPU names** are the RunPod GPU *ids*, e.g. `NVIDIA GeForce RTX 4090`, `NVIDIA H100 80GB HBM3` (H100 SXM), `NVIDIA H200` (H200 SXM) — not the short display names. The runner connects over the pod's direct public IP, which **community cloud** provides reliably (secure cloud often has no public IP); prefer `--cloud-type COMMUNITY`.

Useful flags: `--workload pt|tf|llm`, `--gpu "A,B,C"` (comma-separated fallback list tried in order), `--cloud-type COMMUNITY|SECURE|ALL`, `--capacity-wait SECONDS` (keep retrying the requested GPU(s) with backoff while RunPod has no capacity), `--ssh-timeout`/`--run-timeout SECONDS`, `--extra-args "-w 0 1"` (passed to the benchmark command), `--no-publish`, `--keep` (leave the pod running for debugging — you must then terminate it yourself), and `--dry-run` (print the plan and the exact remote script without creating anything — no API key needed). The database token is sent to the pod over the encrypted SSH channel, not baked into any image or argument string.

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

