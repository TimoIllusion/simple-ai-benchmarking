# simple-ai-benchmarking (SAIB)

A simple application to quickly run tests on a variety of hardware and software for AI workloads, to get an intuition on the performance. No downloading of big datasets and only a few dependencies. For more sophisticated and complex benchmarking, I recommend to use [MLPerf Benchmarks](https://mlcommons.org/benchmarks/). 

Visit [timoillusion.pythonanywhere.com/benchmarks](https://timoillusion.pythonanywhere.com/benchmarks) to see current benchmark database.

I develop this application in my free time as a hobby.

## Quickstart

Assuming tensorflow or pytorch is already installed in your environment:

1. Install SAIB directly using pip: `pip install git+https://github.com/TimoIllusion/simple-ai-benchmarking.git`

2. Run benchmark with command `saib-tf` or `saib-pt` using tensorflow or pytorch respectively.  


NOTE: The results **are** comparable, since the same model architectures are used per default.

NOTE: To publish the results to the [AI Benchmark Database](https://timoillusion.pythonanywhere.com/benchmarks), cloning of the repo is required, see [Setup & Usage](https://github.com/TimoIllusion/simple-ai-benchmarking/tree/main#setup--usage).

NOTE: To install tensorflow and pytorch directly when installing SAIB, you can also install using the following commands: 

`pip install simple-ai-benchmarking[tf]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git` (installs tensorflow)

OR

`pip install simple-ai-benchmarking[pt]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git` (installs torch)

NOTE: Usually only CPU will be supported when installing with the two above options. It is recommended to setup pytorch or tensorflow prior.

## Setup & Usage

1. Create a conda environment via `conda create -n saib python=3.10 -y` and activate it `conda activate saib`.

2. Install your prefered pytorch or tensorflow version and respective CUDA/ROCm/DML-Plugin/etc.

3. Clone this repository and install it:

   ```bash
   git clone https://github.com/TimoIllusion/simple-ai-benchmarking.git
   cd simple-ai-benchmarking
   pip install .
   ```

5. Run `saib-pt` or `saib-tf` in a console with activated environment for tensorflow or pytorch benchmark respectively.

   NOTE: Alternatively execute `python run_tf.py` for tf benchmark or `python run_pt.py` for pytorch benchmark.

## Hardware Acceleration for PyTorch and TensorFlow

This section shows how to use GPUs for training and inference benchmarking.

### Setup TensorFlow for NVIDIA GPUs

1. Create and activate a virtual environment

2. Run `pip install tensorflow` (`tensorflow<2.11` for windows native) and run `pip list` and check https://www.tensorflow.org/install/source#gpu for the relevant CUDA/cudnn version for your tensorflow version

3. Install cuda and cudnn with `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0` (in the case of tensorflow 2.11, CUDA 11.2 and CUDNN 8.1 is needed)

4. Run `python -c "import tensorflow;print(tensorflow.config.list_physical_devices())"` to check if GPU is available
   
### Setup TensorFlow for AMD and Intel GPUs

For AMD and Intel GPUs, DirectML on Windows and WSL can be used. 

To setup everything, run steps 1-2 from the NVIDIA GPU setup and install the directml requirements with:

`pip install tensorflow-cpu==2.10.0 tensorflow-directml-plugin`

See https://github.com/microsoft/tensorflow-directml-plugin for more information.

Clone repo and run benchmark with `python run_pt.py` or `python run_tf.py`

### Setup PyTorch for NVIDIA GPUs

1. Run `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` inside your environment (or check https://pytorch.org/get-started/locally/ for more instructions and options). This already comes with CUDA, only NVIDIA drivers are needed to run with gpu.

2. Run `python -c "import torch;print(torch.cuda.is_available())"`

## Publish to AI Benchmark Database

Currently results can only published by authenticated users, but user creation is not possible currently. 

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

Check [timoillusion.pythonanywhere.com/benchmarks](https://timoillusion.pythonanywhere.com/benchmarks) for the results.

## Upcoming

- [ ] Add very simple mode to very easily benchmark (only 1 simple model, default parameters etc.)
- [ ] Add config management with yaml file or python files to customize benchmarks more easily
- [ ] Expand testing suite
- [ ] Refactor README and give more detailed installation procedures
- [ ] Refactor code structure with more object orientation and interfaces
- [ ] Add more models (Language Models, Timeseries, Object Detection, Segmentation) and model registry
- [ ] Implement unified architecture for inference/train to use any tf/pytorch model with the same API
- [ ] Implement interfaces for TensorRT and ONNXRuntime
- [ ] Test ROCm support
- [ ] Test Intel oneAPI support (see https://github.com/intel/intel-extension-for-tensorflow)
- [ ] Add plotting of results
- [ ] Add option to install package and put on pypi.org
- [ ] Add support for multiple datatypes (FP32, FP16, INT8...)
- [ ] Add user dialogue for device selection and/or optimize automatic device deduction
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
