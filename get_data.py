import torch
from torchvision import datasets, transforms

def data(batch_size, transform):
    traindata = datasets.ImageFolder(root="train/", transform=transform)
    testdata = datasets.ImageFolder(root="test/", transform=transform)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)

    return traindata, trainloader, testdata, testloader


