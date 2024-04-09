from simple_ai_benchmarking.models.factory import ClassificationModelFactory
from simple_ai_benchmarking.models.factory import AIFramework, ClassificiationModelConfig
from simple_ai_benchmarking.models.factory import ModelIdentifier

import json


MODEL_TO_COMPARE = ModelIdentifier.RESNET50

model_cfg = ClassificiationModelConfig(
    model_identifier=MODEL_TO_COMPARE,
    num_classes=100,
    model_shape=(224, 224, 3),
)

pt_model = ClassificationModelFactory.create_model(model_cfg, AIFramework.PYTORCH)
tf_model = ClassificationModelFactory.create_model(model_cfg, AIFramework.TENSORFLOW)



import tensorflow as tf
dummy_input = tf.random.normal((1, 224, 224, 3))
tf_model(dummy_input)
tf_model.summary()

import torch
dummy_input = torch.randn(1, 3, 224, 224)
pt_model.eval()
pt_model(dummy_input)
print(pt_model)

#count parameters per layer and show including layer name
print("Pytorch")
pt_layer_info = []

pt_temp_param_sum  = 0
for i, (name, param) in enumerate(pt_model.named_parameters()):
    num_params = param.numel()
    pt_temp_param_sum += num_params
    print(name, num_params)
    print("- Accumulated params:", pt_temp_param_sum)
    
    pt_layer_info.append({
        "layer_name": name,
        "weight_shape": list(param.shape),
        "weight_params": num_params,
        "accumulated_params": pt_temp_param_sum,
    })
    
with open("pt_model.json", "w") as f:
    json.dump(pt_layer_info, f, indent=4)



# dump layer info to a json (in a list)
print("Tensorflow")

tf_layer_info = []
tf_temp_param_sum  = 0
for i, layer in enumerate(tf_model.layers):
    print("Layer Name:", layer.name)
    print("Parameters:")
    for weight in layer.weights:
        num_params = weight.shape.num_elements()
        print(f"\tName: {weight.name}, Shape: {weight.shape}, Params: {num_params}")
        tf_temp_param_sum += num_params
        print("- Accumulated params:", tf_temp_param_sum)
        
        tf_layer_info.append({
            "layer_name": layer.name,
            "weight_name": weight.name,
            "weight_shape": weight.shape.as_list(),
            "weight_params": num_params,
            "accumulated_params": tf_temp_param_sum,
        })
        
with open("tf_model.json", "w") as f:
    json.dump(tf_layer_info, f, indent=4)
    
    
print("Parameters Pytorch Model:", pt_temp_param_sum)
print("Parameters Tensorflow Model:", tf_temp_param_sum)