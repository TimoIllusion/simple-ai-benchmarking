from simple_ai_benchmarking.models.pt.vit import VisionTransformer as ViTPT
from simple_ai_benchmarking.models.tf.vit_torchvision_tf import create_vit_b_16





pt_model = ViTPT(100)


tf_model = create_vit_b_16()

import tensorflow as tf
dummy_input = tf.random.normal((1, 224, 224, 3))

tf_model(dummy_input)
tf_model.summary()

import torch
dummy_input = torch.randn(1, 3, 224, 224)
pt_model.eval()
pt_model(dummy_input)
print(pt_model)


