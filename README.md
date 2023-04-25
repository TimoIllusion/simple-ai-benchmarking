# simple-ai-benchmarking

A collection of simple scripts and notebooks to quickly run reproducible tests on a variety of hardware and software for AI workloads.

## Setup for NVIDIA GPUs

1. Clone this repository.

2. Create a conda environment via `conda create -n saibench python=3.8 -y` and activate it `conda activate saibench`.

3. Install cuda and cudnn with `conda install -c conda-forge cudatoolkit=11.2 cudnn`.

4. Install remaining packages using `pip install -r requirements.txt` from root of this repository.

## Setup for AMD and Intel GPUs

For AMD and Intel GPUs, DirectML on Windows and WSL can be used. 

To setup everything, run steps 1-2 from the NVIDIA GPU setup and install the directml requirements with:

`pip install -r directml_requirements.txt`

See https://github.com/microsoft/tensorflow-directml-plugin for more information.

## Run Benchmark

Run benchmark with in activated conda environment:

`python benchmark.py`

## Example results

MLPMixer Benchmark (batch_size 128, input shape 32x32x3):

- NVIDIA RTX4090 [TF2.9.0+cuda11.2] + AMD Ryzen 7 7800X3D: 18743.99 it/s (inference), 760.49 it/s (training)
- NVIDIA RTX4090 [TF2.10.0+directml0.4] + AMD Ryzen 7 7800X3D: 7979.99 it/s (inference), 75.98 it/s (training)
- AMD RX6600 [TF2.10.0+directml0.4] + Intel Core i5 12600K: 2699.31 it/s (inference), 68.92 it/s (training)

EfficientNet Benchmark (batch_size 64, input shape 224x224x3):

- NVIDIA RTX4090 [TF2.9.0+cuda11.2] + AMD Ryzen 7 7800X3D: 2190.57 it/s (inference), 64.62 it/s (training)
- NVIDIA RTX4090 [TF2.10.0+directml0.4] + AMD Ryzen 7 7800X3D: 1775.09 it/s (inference), 39.14 it/s (training)
- AMD RX6600 [TF2.10.0+directml0.4] + Intel Core i5 12600K: 238.92 it/s (inference), 27.54 it/s (training)

## Utilized and modified open source code

- https://github.com/keras-team/keras-io/blob/master/examples/vision/mlp_image_classification.py (Apache License 2.0)
- https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_efficientnet_fine_tuning.py (Apache License 2.0)

## Upcoming

- [ ] Add warmup function
- [ ] Implement unified architecture for inference/train to use any tf/pytorch model using synthetic data
- [ ] Add more models (Language Models, Timeseries, Object Detection, Segmentation)
- [ ] Add models using pytorch
- [ ] Refactor code into improved and refined structure
- [ ] Improve logging
- [ ] Add plotting
- [ ] ROCm support
- [ ] Intel oneAPI support (see https://github.com/intel/intel-extension-for-tensorflow)