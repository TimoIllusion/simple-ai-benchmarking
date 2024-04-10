# Project Name: simple-ai-benchmarking
# File Name: vit.py
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


import torch.nn as nn
import torchvision


class VisionTransformer(nn.Module):

    def __init__(self, num_classes):
        super(VisionTransformer, self).__init__()
        self.vit_base_16 = torchvision.models.vit_b_16(num_classes=num_classes)

    def forward(self, x):
        return self.vit_base_16(x)

if __name__ == '__main__':

    # convert to onnx
    model = VisionTransformer(num_classes=100)

    import torch
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "vit_b_16.onnx", verbose=True)