# simple-ai-benchmarking (SAIB)

A simple application to quickly run tests on a variety of hardware and software for AI workloads, to get an intuition on the performance. No downloading of big datasets and only a few dependencies. For more sophisticated and complex benchmarking, I recommend to use [MLPerf Benchmarks](https://mlcommons.org/benchmarks/). 

I develop this application in my free time as a hobby.

## Quickstart

Assuming tensorflow or pytorch is already installed in your environment:

1. Install SAIB directly using pip: `pip install git+https://github.com/TimoIllusion/simple-ai-benchmarking.git`

2. Run benchmark with command `saib-tf` or `saib-pt` using tensorflow or pytorch respectively. 

NOTE: The results **are** comparable, since the same model architectures are used per default.

To install tensorflow and pytorch when installing SAIB, you can also install using the following commands: 

`pip install simple-ai-benchmarking[tf]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git` (installs tensorflow)

OR

`pip install simple-ai-benchmarking[pt]@git+https://github.com/TimoIllusion/simple-ai-benchmarking.git` (installs torch)

NOTE: Usually only CPU will be supported when installing with the two above options. It is recommended to setup pytorch or tensorflow prior.

## Setup & Usage

1. Create a conda environment via `conda create -n saib python=3.9 -y` and activate it `conda activate saib`.

2. [OPTIONAL] Install your prefered pytorch or tensorflow version and respective CUDA version.

3. Either clone this repo and run `pip install .` OR just run `pip install git+https://github.com/TimoIllusion/simple-ai-benchmarking.git` (use extras [tf,pt,...] for direct install of cpu tensorflow, cpu pytorch or directml tensorflow)

4. Run `saib-tf` or `saib-pt` in a console with activated environment for tensorflow or pytorch benchmark respectively. Alternatively execute `python run_tf.py` for tf benchmark or `python run_pt.py` for pytorch benchmark.

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

## Example results [v0.3.2 - LATEST]

Results on NVIDIA RTX 4090 with Ryzen 7800X3D 8-Core CPU on Windows 11, PyTorch 2.1.2+cu121, Python 3.10.13:

| #RUN |  Lib  |           Model           |       Accelerator       |     Precision     | BS | it/s train | it/s infer |
|------|-------|---------------------------|-------------------------|-------------------|----|------------|------------|
|  0   | torch | PTSimpleClassificationCNN | NVIDIA GeForce RTX 4090 | DEFAULT_PRECISION | 8  |  3022.56   |  6243.25   |
|  1   | torch | PTSimpleClassificationCNN | NVIDIA GeForce RTX 4090 |    MIXED_FP16     | 8  |  2924.48   |  6416.01   |
|  2   | torch |          ResNet           | NVIDIA GeForce RTX 4090 | DEFAULT_PRECISION | 1  |   62.87    |   152.81   |

Results on NVIDIA RTX 4090 with Ryzen 7800X3D 8-Core CPU on Windows 11, PyTorch 1.12.0+cu116, Python 3.9.18:

| #RUN |  Lib  |           Model           |       Accelerator       |     Precision     | BS | it/s train | it/s infer |
|------|-------|---------------------------|-------------------------|-------------------|----|------------|------------|
|  0   | torch | PTSimpleClassificationCNN | NVIDIA GeForce RTX 4090 | DEFAULT_PRECISION | 8  |  1819.13   |  3423.22   |
|  1   | torch | PTSimpleClassificationCNN | NVIDIA GeForce RTX 4090 |    MIXED_FP16     | 8  |  1445.27   |  2969.95   |
|  2   | torch |          ResNet           | NVIDIA GeForce RTX 4090 | DEFAULT_PRECISION | 1  |   34.52    |   115.38   |

Results on NVIDIA RTX 4090 with Ryzen 7800X3D 8-Core CPU on Windows 11, TensorFlow 2.10 with CUDA 11.2 and CUDNN 8.8, Python 3.10.13:

| #RUN |  Lib  |           Model           |       Accelerator       |     Precision     | BS | it/s train | it/s infer |
|------|-------|---------------------------|-------------------------|-------------------|----|------------|------------|
|  0   | tensorflow | TFSimpleClassificationCNN | NVIDIA GeForce RTX 4090 | DEFAULT_PRECISION | 8  |  1633.47   |  3926.38   |
|  1   | tensorflow | TFSimpleClassificationCNN | NVIDIA GeForce RTX 4090 |    MIXED_FP16     | 8  |  1505.08   |  4003.79   |
|  2   | tensorflow |          ResNet           | NVIDIA GeForce RTX 4090 | DEFAULT_PRECISION | 1  |   53.57    |   194.31   |

Results on M1 Max 10C/24GPU 32 GB, MacOS Sonoma, PyTorch 2.2.1, Python 3.10.14:

| #RUN |  Lib  |           Model           |       Accelerator       |     Precision     | BS | it/s train | it/s infer |
|------|-------|---------------------------|-------------------------|-------------------|----|------------|------------|
|  0   | torch | PTSimpleClassificationCNN |      M1 MAX      | DEFAULT_PRECISION | 8  |   728.66   |  2443.32   |
|  1   | torch | PTSimpleClassificationCNN |      M1 MAX       |    MIXED_FP16     | 8  |   692.36   |  2338.85   |
|  2   | torch |          ResNet           |      M1 MAX        | DEFAULT_PRECISION | 1  |   22.28    |   93.61    |

Note: In these results, TensorFlow Benchmarks are using a much older version for TF and CUDA in comparison to PyTorch Benchmarks (due to missing support for native Windows). In the future, experiments will be conducted in WSL2, to use more similar versions.

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
