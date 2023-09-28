# simple-ai-benchmarking (SAI)

A collection of simple scripts and notebooks to quickly run reproducible tests on a variety of hardware and software for AI workloads.

## Quickstart

Assuming tensorflow or pytorch is already installed in your environment:

1. Install SAI directly using pip: `pip install git+https://github.com/TimoIllusion/simple-ai-benchmarking.git`

2. Run benchmark: `sai-tf` or `sai-pt` to benchmark tensorflow or pytorch respectively. 

NOTE: The results are not comparable, since different model architectures are used per default.

To install tensorflow and pytorch when installing SAI, you can also install using the following commands: 

`pip install git+https://github.com/TimoIllusion/simple-ai-benchmarking.git[tf]` (installs tensorflow)

OR

`pip install git+https://github.com/TimoIllusion/simple-ai-benchmarking.git[pt]` (installs torch)

NOTE: Usually only CPU will be supported when installing with the two above options. It is recommended to setup pytorch or tensorflow prior.

## Setup & Usage

1. Create a conda environment via `conda create -n sai python=3.9 -y` and activate it `conda activate sai`.

2. [OPTIONAL] Install your prefered pytorch or tensorflow version and respective CUDA version.

3. Either clone this repo and run `pip install .` OR just run `pip install git+https://github.com/TimoIllusion/simple-ai-benchmarking.git` (use extras [tf,pt,...] for direct install of cpu tensorflow, cpu pytorch or directml tensorflow)

4. Run `sai-tf` or `sai-pt` in a console with activated environment for tensorflow or pytorch benchmark respectively. Alternatively execute `python run.py` for tf benchmark.

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

Clone repo and run benchmark with `python run.py`.

### Setup PyTorch for NVIDIA GPUs

1. Run `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` inside your environment (or check https://pytorch.org/get-started/locally/ for more instructions and options). This already comes with CUDA, only NVIDIA drivers are needed to run with gpu.

2. Run `python -c "import torch;print(torch.cuda.is_available())"`

## Example results [LATEST]

WIP

## Example results [LEGACY]

| Model             | Batch size | Software Framework | GPU                    | CPU                 | Inference Speed (it/s) | Training Speed (it/s) |
|-------------------|------------|---------------------|-----------------------|---------------------|------------------------|-----------------------|
| MLPMixer           | 128        | tensorflow2.9.0+cuda11.2    | NVIDIA RTX 4090        | AMD Ryzen 7 7800X3D    | 18743.99                | 760.49                |
| MLPMixer           | 128        | tensorflow2.10.0+directml0.4 | NVIDIA RTX 4090        | AMD Ryzen 7 7800X3D    | 7979.99                 | 75.98                 |
| MLPMixer           | 128        | tensorflow2.10.0+cuda11.2    | NVIDIA RTX 2060 Mobile | AMD Ryzen 7 4800H    | 5354.33                 | 39.34                 |
| MLPMixer           | 128        | tensorflow2.10.0+directml0.4 | AMD RX 6600            | Intel Core i5 12600K | 2699.31                 | 68.92                 |
| MLPMixer           | 128        | tensorflow2.10.0   | -      | AMD Ryzen 7 7800X3D                   | 1965.07                 | 207.56                |
| EfficientNet       | 64         | tensorflow2.9.0+cuda11.2     | NVIDIA RTX 4090        | AMD Ryzen 7 7800X3D    | 2190.57                 | 64.62                 |
| EfficientNet       | 64         | tensorflow2.10.0+directml0.4 | NVIDIA RTX 4090        | AMD Ryzen 7 7800X3D    | 1775.09                 | 39.14                 |
| EfficientNet       | 64         | tensorflow2.10.0+directml0.4 | AMD RX 6600            | Intel Core i5 12600K | 238.92                  | 27.54                 |
| EfficientNet       | 64         | tensorflow2.10.0   |   -    | AMD Ryzen 7 7800X3D                   | 108.16                  | 18.47                 |
| EfficientNet       | 32         | tensorflow2.10.0+cuda11.2    | NVIDIA RTX 2060 Mobile | AMD Ryzen 7 4800H    | 487.68                  | 22.39                 |


## Upcoming

- [x] Remove dependencies on tf_addons and tf_datasets
- [ ] Add warmup function
- [x] Use synthetic data
- [ ] Add support for multiple datatypes (FP32, FP16, INT8...)
- [ ] Implement unified architecture for inference/train to use any tf/pytorch model with the same API
- [ ] Model registry
- [ ] Add config file to customize benchmarks
- [ ] Add unit tests
- [ ] Add more models (Language Models, Timeseries, Object Detection, Segmentation)
- [x] Add models using pytorch
- [ ] Refactor code into improved and refined structure
- [x] Improve logging 
- [ ] Save results to csv
- [ ] Implement interfaces for TensorRT and ONNXRuntime
- [ ] Add plotting
- [ ] ROCm support
- [ ] Intel oneAPI support (see https://github.com/intel/intel-extension-for-tensorflow)
- [ ] Add option to install package and put on pypi.org