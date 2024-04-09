from simple_ai_benchmarking.models.factory import ClassificationModelFactory
from simple_ai_benchmarking.config_structures import (
    AIFramework,
    ClassificiationModelConfig,
    ImageShape,
)
from simple_ai_benchmarking.models.factory import ModelIdentifier

import json


MODEL_TO_COMPARE = ModelIdentifier.SIMPLE_CLASSIFICATION_CNN


img_shape = ImageShape(224, 224, 3)
model_cfg = ClassificiationModelConfig(
    model_identifier=MODEL_TO_COMPARE,
    num_classes=100,
    model_shape=img_shape,
)

pt_model = ClassificationModelFactory.create_model(model_cfg, AIFramework.PYTORCH)
tf_model = ClassificationModelFactory.create_model(model_cfg, AIFramework.TENSORFLOW)


import tensorflow as tf

shape = (1, ) + img_shape.to_tuple_hwc()
dummy_input = tf.random.normal(shape)
tf_model(dummy_input)
tf_model.summary()

import torch

shape = (1, ) + img_shape.to_tuple_chw()
dummy_input = torch.randn(shape)
pt_model.eval()
pt_model(dummy_input)
print(pt_model)

# count parameters per layer and show including layer name
print("Pytorch")
pt_layer_info = []

pt_temp_param_sum = 0
for i, (name, param) in enumerate(pt_model.named_parameters()):
    num_params = param.numel()
    pt_temp_param_sum += num_params
    print(name, num_params)
    print("- Accumulated params:", pt_temp_param_sum)

    pt_layer_info.append(
        {
            "layer_name": name,
            "weight_shape": list(param.shape),
            "weight_params": num_params,
            "accumulated_params": pt_temp_param_sum,
        }
    )

with open("pt_model.json", "w") as f:
    json.dump(pt_layer_info, f, indent=4)


# dump layer info to a json (in a list)
print("Tensorflow")

tf_layer_info = []
tf_temp_param_sum = 0
for i, layer in enumerate(tf_model.layers):
    print("Layer Name:", layer.name)
    print("Parameters:")
    for weight in layer.weights:
        num_params = weight.shape.num_elements()
        print(f"\tName: {weight.name}, Shape: {weight.shape}, Params: {num_params}")
        tf_temp_param_sum += num_params
        print("- Accumulated params:", tf_temp_param_sum)

        tf_layer_info.append(
            {
                "layer_name": layer.name,
                "weight_name": weight.name,
                "weight_shape": weight.shape.as_list(),
                "weight_params": num_params,
                "accumulated_params": tf_temp_param_sum,
            }
        )

with open("tf_model.json", "w") as f:
    json.dump(tf_layer_info, f, indent=4)


print("Parameters Pytorch Model:", pt_temp_param_sum)
print("Parameters Tensorflow Model:", tf_temp_param_sum)

# calc absolute and relative difference of tensorflow model to pytorch model. If difference is <1 %, print "Model similarity: OK"

diff_params = abs(pt_temp_param_sum - tf_temp_param_sum)
rel_diff = diff_params / pt_temp_param_sum

print("Difference in parameters (abs):", diff_params)
print("Difference in parameters (rel, %):", rel_diff * 100)

if rel_diff < 0.01:
    print("Model similarity: OK")
else:
    print("Model similarity: NOT OK")
