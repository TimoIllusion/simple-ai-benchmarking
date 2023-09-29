import torch
import torch.nn as nn
import numpy as np

class PTSimpleClassificationCNN:

    @staticmethod
    def find_flatten_dim(model, input_shape):
        dummy_input = torch.randn(1, *input_shape)
        output_feat = model(dummy_input)
        return int(np.prod(output_feat.size()))

    @staticmethod
    def build_model(num_classes, input_shape):
        
        model_conv_part = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),  # Output: 222x222x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 111x111x32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),  # Output: 109x109x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 54x54x64

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # Output: 52x52x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 26x26x64

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # Output: 24x24x64
            nn.ReLU()
        )

        input_dim = PTSimpleClassificationCNN.find_flatten_dim(model_conv_part, input_shape)
        
        model = nn.Sequential(
            model_conv_part,
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )
        
        return model