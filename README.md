# simple-ai-benchmarking

A collection of simple scripts and notebooks to quickly run reproducible tests on a variety of hardware and software for AI workloads.

## Setup

1. Clone this repository.

2. Create a conda environment via `conda create -n saibench python=3.8 -y` and activate it `conda activate saibench`.

3. Install cuda and cudnn with `conda install -c conda-forge cudatoolkit=11.2 cudnn`.

4. Install remaining packages using `pip install -r requirements.txt` from root of this repository.

## Example results

MLPMixer Benchmark (batch_size 128, input shape 32x32x3):

- RTX4090[TF2.8+cu11.2]: 18743.99 it/s (inference), 760.49 it/s (training)
- RX6600[TF2.10+directml0.4]: 2699.31 it/s (inference), 68.92 it/s (training)

## Utilized and modified open source code

- https://github.com/keras-team/keras-io/blob/master/examples/vision/mlp_image_classification.py (Apache License 2.0)