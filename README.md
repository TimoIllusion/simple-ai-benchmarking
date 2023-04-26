# simple-ai-benchmarking

A collection of simple scripts and notebooks to quickly run reproducible tests on a variety of hardware and software for AI workloads.

## Simple setup: Install as package

1. Clone this repository.

2. Create a conda environment via `conda create -n saibench python=3.8 -y` and activate it `conda activate saibench`.

3. Go into the directory of this cloned repo and run `pip install .`. Optionally install CUDA in your system or environment afterwards.

4. Run `sai_benchmark` in a console with activated environment

## Setup for NVIDIA GPUs

Run `SETUP.bat` (only for windows) or follow these steps:

1. Clone this repository.

2. Create a conda environment via `conda create -n saibench python=3.8 -y` and activate it `conda activate saibench`.

3. Install cuda and cudnn with `conda install -c conda-forge cudatoolkit=11.2 cudnn`.

4. Install remaining packages using `pip install -r requirements.txt` from root of this repository.

5. Run benchmark with `python run.py`

## Setup for AMD and Intel GPUs

For AMD and Intel GPUs, DirectML on Windows and WSL can be used. 

To setup everything, run steps 1-2 from the NVIDIA GPU setup and install the directml requirements with:

`pip install -r directml_requirements.txt`

See https://github.com/microsoft/tensorflow-directml-plugin for more information.

Run benchmark with `python run.py`.

## Example results

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

## Utilized and modified open source code

- https://github.com/keras-team/keras-io/blob/master/examples/vision/mlp_image_classification.py (Apache License 2.0)
- https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_efficientnet_fine_tuning.py (Apache License 2.0)

## Upcoming

- [ ] Add warmup function
- [ ] Use synthetic data
- [ ] Implement unified architecture for inference/train to use any tf/pytorch model with the same API
- [ ] Add unit tests
- [ ] Add more models (Language Models, Timeseries, Object Detection, Segmentation)
- [ ] Add models using pytorch
- [ ] Refactor code into improved and refined structure
- [x] Improve logging 
- [ ] Add plotting
- [ ] ROCm support
- [ ] Intel oneAPI support (see https://github.com/intel/intel-extension-for-tensorflow)
- [ ] Add option to install package and put on pypi.org