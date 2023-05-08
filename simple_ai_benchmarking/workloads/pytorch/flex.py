import os

import torch
import torchvision


def main():
    
    epochs = 200
        
    # Define transforms for the CIFAR10 dataset
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
    ])

    # Load the CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # create untrained model (see https://paperswithcode.com/sota/image-classification-on-cifar-10 for SOTA accuracy overview and https://pytorch.org/vision/0.15/models.html#classification for available models)
    # model = torchvision.models.resnet50(pretrained=False, num_classes=10) # ~95 % accuracy on cifar10, good accuracy, widely used
    # model = torchvision.models.resnet101(pretrained=False, num_classes=10)
    # model = torchvision.models.resnet152(pretrained=False, num_classes=10)
    # model = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=10)
    model = torchvision.models.mobilenet_v3_small(pretrained=False, num_classes=10) # common, very lightweight and quite accurate 
    # model = torchvision.models.mobilenet_v3_large(pretrained=False, num_classes=10) 
    # model = torchvision.models.convnext_small(pretrained=False, num_classes=10)
    # model = torchvision.models.efficientnet_b0(pretrained=True, num_classes=10)
    # model = torchvision.models.efficientnet_b7(pretrained=False, num_classes=10) # Nr. 24 SOTA @ 98.9 %
    
    # Alternative: use model pretrained on imagenet, freeze backbone and then train linear layer on cifar10 from that checkpoint
    # model = torchvision.models.resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc = torch.nn.Linear(2048, 10)
    
    
    print(model)
    
    # move to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Save the model checkpoint
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Train the model
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print('Epoch %d - Iteration %5d - Loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Test the model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
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

