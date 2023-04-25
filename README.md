# simple-ai-benchmarking

A collection of simple scripts and notebooks to quickly run reproducible tests on a variety of hardware and software for AI workloads.

## Setup for NVIDIA GPUs

1. Clone this repository.

2. Create a conda environment via `conda create -n saibench python=3.8 -y` and activate it `conda activate saibench`.

3. Install cuda and cudnn with `conda install -c conda-forge cudatoolkit=11.2 cudnn`.

4. Install remaining packages using `pip install -r requirements.txt` from root of this repository.

## Setup for AMD and Intel GPUs

For AMD and Intel GPUs, DirectML on Windows and WSL can be used. Install the directml plugin after step 4 (with from setup for NVIDIA above):

`pip install tensorflow-directml-plugin`

See https://github.com/microsoft/tensorflow-directml-plugin for more information.

## Example results

MLPMixer Benchmark (batch_size 128, input shape 32x32x3):

- NVIDIA RTX4090 [TF2.9.0+cuda11.2]: 18743.99 it/s (inference), 760.49 it/s (training)
- AMD RX6600 [TF2.10.0+directml0.4]: 2699.31 it/s (inference), 68.92 it/s (training)

EfficientNet Benchmark (batch_size 64, input shape 224x224x3):

- NVIDIA RTX4090 [TF2.9.0+cuda11.2]: 2190.57 it/s (inference), 64.62 it/s (training)
- AMD RX6600 [TF2.10.0+directml0.4]: 238.92 it/s (inference), 27.54 it/s (training)

## Utilized and modified open source code

- https://github.com/keras-team/keras-io/blob/master/examples/vision/mlp_image_classification.py (Apache License 2.0)

## Upcoming

- [ ] Add warmup function
- [ ] Implement unified architecture for inference/train to use any tf/pytorch model using synthetic data
- [ ] Add more models (Language Models, Timeseries, Object Detection, Segmentation)
- [ ] Add models using pytorch
- [ ] Refactor code into improved and refined structure
- [ ] Improve logging
- [ ] Add plotting