import torch.nn as nn
import torchvision


class ResNet50(nn.Module):

    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(num_classes=num_classes)

    def forward(self, x):
        return self.resnet50(x)
