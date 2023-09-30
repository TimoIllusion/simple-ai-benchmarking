import torch
import torch.nn as nn
import numpy as np

class PTSimpleClassificationCNN(nn.Module):
    
    def __init__(self, num_classes, input_shape):
        super(PTSimpleClassificationCNN, self).__init__()
        
        self.model_conv_part = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        
        input_dim = self.find_flatten_dim(input_shape)
        
        self.model = nn.Sequential(
            self.model_conv_part,
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )
    
    def find_flatten_dim(self, input_shape):
        dummy_input = torch.randn(1, *input_shape)
        output_feat = self.model_conv_part(dummy_input)
        return int(np.prod(output_feat.size()))
    
    def forward(self, x):
        return self.model(x)