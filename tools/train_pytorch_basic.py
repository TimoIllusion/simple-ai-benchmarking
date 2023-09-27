# Simple training script to train torchvision models and simple cnn, based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import os

import torch
import torch.nn as nn
import torchvision

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.dropout_low = nn.Dropout(p=0.2)
        self.dropout_high = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(256, 10)
        self.smax = nn.Softmax(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.bn1_2(x)
        x = self.pool(x)
        x = self.dropout_low(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.bn2_2(x)
        x = self.pool(x)
        x = self.dropout_low(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.bn3_2(x)
        x = self.pool(x)
        x = self.dropout_low(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.dropout_high(x)
        
        x = self.fc2(x)
        x = self.smax(x)
        
        return x

def main():
    
    epochs = 200
        
    # Define transforms for the CIFAR10 dataset
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(degrees=10),
        # torchvision.transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        # torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.1),
        # torchvision.transforms.RandomPerspective(distortion_scale=0.1),
        # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
        # torchvision.transforms.Resize(224),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
        # torchvision.transforms.Resize(224),
    ])

    # Load the CIFAR10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # create untrained model (see https://paperswithcode.com/sota/image-classification-on-cifar-10 for SOTA accuracy overview and https://pytorch.org/vision/0.15/models.html#classification for available models)
    # model = torchvision.models.resnet50(pretrained=False, num_classes=10) # ~95 % accuracy on cifar10, good accuracy, widely used
    # model = torchvision.models.resnet101(pretrained=False, num_classes=10)
    # model = torchvision.models.resnet152(pretrained=False, num_classes=10)
    # model = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=10)
    #model = torchvision.models.mobilenet_v3_small(pretrained=False, num_classes=10) # common, very lightweight and quite accurate 
    # model = torchvision.models.mobilenet_v3_large(pretrained=False, num_classes=10) 
    # model = torchvision.models.convnext_small(pretrained=False, num_classes=10)
    # model = torchvision.models.efficientnet_b0(pretrained=True, num_classes=10)
    # model = torchvision.models.efficientnet_b7(pretrained=False, num_classes=10) # Nr. 24 SOTA @ 98.9 %
    
    # alternative: use model pretrained on imagenet, freeze backbone and then train linear layer on cifar10 from that checkpoint
    # model = torchvision.models.resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc = torch.nn.Linear(2048, 10)

    model = BasicCNN()
    
    print(model)
    
    # move to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.half()
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.half()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters())
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Save the model checkpoint
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Train the model
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.half()
            # labels = labels.half()
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print('Epoch %d - Iteration %5d - Loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Test the model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                images = images.half()
                
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test dataset: %d %%' % (100 * correct / total))

    checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print('Finished training')

if __name__ == '__main__':
    main()

