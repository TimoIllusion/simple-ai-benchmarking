# Project Name: simple-ai-benchmarking
# File Name: export_onnx.py
# Author: Timo Leitritz
# Copyright (C) 2024 Timo Leitritz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os

from simple_ai_benchmarking.models.factory import ClassificationModelFactory
from simple_ai_benchmarking.models.factory import (
    AIFramework,
    ClassificationModelConfig,
)
from simple_ai_benchmarking.models.factory import ModelIdentifier
from simple_ai_benchmarking.config_structures import ImageShape


MODEL_TO_EXPORT = ModelIdentifier.RESNET50

img_shape = ImageShape(224, 224, 3)

model_cfg = ClassificationModelConfig(
    model_identifier=MODEL_TO_EXPORT,
    num_classes=100,
    model_shape=img_shape,
)

pt_model = ClassificationModelFactory.create_model(model_cfg, AIFramework.PYTORCH)
tf_model = ClassificationModelFactory.create_model(model_cfg, AIFramework.TENSORFLOW)


# export models to onnx

import torch

shape = (1,) + img_shape.to_tuple_chw()
dummy_input = torch.randn(shape)
pt_model.eval()
pt_model(dummy_input)

# export

torch.onnx.export(
    pt_model, dummy_input, f"{MODEL_TO_EXPORT.value}_pt.onnx", verbose=True
)

import tensorflow as tf

shape = (1,) + img_shape.to_tuple_hwc()
dummy_input = tf.random.normal(shape)
tf_model(dummy_input)

model_name = f"{MODEL_TO_EXPORT.value}_tf"
tf.saved_model.save(tf_model, model_name)

os.system(
    f"python -m tf2onnx.convert --saved-model {model_name} --output {model_name}.onnx"
)
