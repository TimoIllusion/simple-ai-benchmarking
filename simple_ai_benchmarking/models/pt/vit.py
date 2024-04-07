import torch.nn as nn
import torchvision


class VisionTransformer(nn.Module):

    def __init__(self, num_classes):
        super(VisionTransformer, self).__init__()
        self.vit_base_16 = torchvision.models.vit_b_16(num_classes=num_classes)
        
        print(self.vit_base_16)

    def forward(self, x):
        return self.vit_base_16(x)

if __name__ == '__main__':
    model = VisionTransformer(num_classes=1000)
    # convert to onnx
    
    import torch
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "vit_b_16.onnx", verbose=True)
    