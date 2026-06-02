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

In addition to the vision/CNN workloads, SAIB can benchmark large language model (LLM) inference and report token throughput. Run it with the `saib-llm` entry point:

```bash
saib-llm --backend ollama --model llama3
```

Three backends are supported:

- `openai-compatible` (default) — benchmark any server exposing the OpenAI `/v1/chat/completions` API (e.g. vLLM, llama.cpp server, LM Studio, OpenAI itself):

  ```bash
  saib-llm --backend openai-compatible --base-url http://localhost:8000 --model my-model --api-key-env OPENAI_API_KEY
  ```

- `ollama` — benchmark a local [Ollama](https://ollama.com) server (defaults to `http://127.0.0.1:11434`, override with `--base-url` or `OLLAMA_BASE_URL`):

  ```bash
  saib-llm --backend ollama --model llama3
  ```

- `pytorch-simple-transformer` — run a self-contained synthetic PyTorch transformer locally, requiring no external server or model weights (handy for quick hardware comparisons):

  ```bash
  saib-llm --backend pytorch-simple-transformer --model simple-transformer --device cuda
  ```

The benchmark performs configurable warmup and measured requests (optionally concurrent), repeats the measurement and averages the result, and reports prompt, generated, and total tokens per second, time to first token, and total duration. Each repetition runs in an isolated process. Results are written to `llm_results.csv` (and `.xlsx` if `openpyxl` is installed).

Common options (see `saib-llm -h` for the full list):

- `--requests` / `--warmup-requests` — number of measured / warmup requests (default `10` / `1`)
- `--repetitions` — number of times the measurement is repeated and averaged (default `3`); for paid HTTP endpoints, lower this to reduce cost
- `--concurrency` — for HTTP backends (`openai-compatible`, `ollama`), the number of in-flight concurrent requests; for the local `pytorch-simple-transformer` backend, the batch size processed in a single batched forward pass (default `1`)
- `--prompt-tokens` / `--generated-tokens` — prompt and generation lengths (default `128` / `256`)
- `--context-length` — model context window (default `4096`)
- `--device` — torch device for the local backend, e.g. `cpu`, `cuda`, `mps` (default `cpu`)
- `--compute-precision`, `--quantization`, `--model-params`, `--weight-source`, `--accelerator` — metadata recorded with the result
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

