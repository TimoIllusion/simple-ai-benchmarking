# simple-ai-benchmarking (SAIB)

A simple application to quickly run tests on a variety of hardware and software for AI workloads, to get an intuition on the performance. No downloading of big datasets and only a few dependencies. For more sophisticated and complex benchmarking, I recommend to use [MLPerf Benchmarks](https://mlcommons.org/benchmarks/). 

Visit [timoillusion.pythonanywhere.com/benchmarks](https://timoillusion.pythonanywhere.com/benchmarks) to see current benchmark database.

I develop this application in my free time as a hobby.

## Quickstart

1. Install pytorch and/or tensorflow

2. Install SAIB and run pt and/or tf benchmark:

   ```bash
   pip install git+https://github.com/TimoIllusion/simple-ai-benchmarking.git
   saib-pt
   saib-tf
   ```  

**Notes**:
- The results **are** comparable, since the same model architectures are used per default.
- To publish the results to the [AI Benchmark Database](https://timoillusion.pythonanywhere.com/benchmarks), cloning of the repo is required, see [Setup & Usage](https://github.com/TimoIllusion/simple-ai-benchmarking/tree/main#setup--usage).
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

## Hardware Acceleration for PyTorch and TensorFlow

This section shows how to use various GPUs for training and inference benchmarking.

**PyTorch for NVIDIA GPUs**

1. Run `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` inside your environment (or check https://pytorch.org/get-started/locally/ for more instructions and options). This already comes with CUDA, only NVIDIA drivers are needed to run with gpu.

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

## License

Copyright (C) [2024] [Timo Leitritz]

This project (simple-ai-benchmarking) is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for the full license text.

