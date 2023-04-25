# simple-ai-benchmarking

A collection of simple scripts and notebooks to quickly run reproducible tests on a variety of hardware and software for AI workloads.

## Setup

1. Clone this repository.

2. Create a conda environment via `conda create -n saibench python=3.8 -y` and activate it `conda activate saibench`.

3. Install cuda and cudnn with `conda install -c conda-forge cudatoolkit=11.2 cudnn`.

4. Install remaining packages using `pip install -r requirements.txt` from root of this repository.

## Utilized and modified open source code

- https://github.com/keras-team/keras-io/blob/master/examples/vision/mlp_image_classification.py (Apache License 2.0)