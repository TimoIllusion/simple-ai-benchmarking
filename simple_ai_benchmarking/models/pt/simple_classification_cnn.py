# Project Name: simple-ai-benchmarking
# File Name: simple_classification_cnn.py
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


import torch
import torch.nn as nn
import numpy as np


class SimpleClassificationCNN(nn.Module):

    def __init__(self, num_classes, input_shape):

        super(SimpleClassificationCNN, self).__init__()

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
            nn.ReLU(),
        )

        input_dim = self.find_flatten_dim(input_shape)

        self.model = nn.Sequential(
            self.model_conv_part,
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1),
        )

    def find_flatten_dim(self, input_shape):

        dummy_input = torch.randn(1, *input_shape)
        output_feat = self.model_conv_part(dummy_input)

        return int(np.prod(output_feat.size()))

    def forward(self, x):
        return self.model(x)
