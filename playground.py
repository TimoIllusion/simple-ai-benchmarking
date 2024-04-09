from simple_ai_benchmarking.models.pt.vit import VisionTransformer as ViTPT
from simple_ai_benchmarking.models.tf.vit_torchvision_tf import create_vit_b_16

import json



pt_model = ViTPT(100)


tf_model = create_vit_b_16()

import tensorflow as tf
dummy_input = tf.random.normal((1, 224, 224, 3))

tf_model(dummy_input)
# tf_model.summary()

import torch
dummy_input = torch.randn(1, 3, 224, 224)
pt_model.eval()
pt_model(dummy_input)
# print(pt_model)

#count parameters per layer and show including layer name

pt_layer_info = []

temp_param_sum  = -768
for name, param in pt_model.named_parameters():
    num_params = param.numel()
    temp_param_sum += num_params
    print(name, num_params)
    print("- Accumulated params:", temp_param_sum)
    
    pt_layer_info.append({
        "layer_name": name,
        "weight_shape": list(param.shape),
        "weight_params": num_params,
        "accumulated_params": temp_param_sum,
    })
    
with open("pt_model.json", "w") as f:
    json.dump(pt_layer_info, f, indent=4)


# dump layer info to a json (in a list)
print("Tensorflow")

layer_info = []

temp_param_sum  = 0
for layer in tf_model.layers:
    print("Layer Name:", layer.name)
    print("Parameters:")
    for weight in layer.weights:
        num_params = weight.shape.num_elements()
        print(f"\tName: {weight.name}, Shape: {weight.shape}, Params: {num_params}")
        temp_param_sum += num_params
        print("- Accumulated params:", temp_param_sum)
        
        layer_info.append({
            "layer_name": layer.name,
            "weight_name": weight.name,
            "weight_shape": weight.shape.as_list(),
            "weight_params": num_params,
            "accumulated_params": temp_param_sum,
        })
        
with open("tf_model.json", "w") as f:
    json.dump(layer_info, f, indent=4)