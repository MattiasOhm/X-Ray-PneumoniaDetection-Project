import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchmetrics.classification import MulticlassConfusionMatrix
from get_data import data

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            nn.Flatten(),
            nn.Linear(in_features=350464, out_features=1024),
            nn.LeakyReLU(0.02),
            nn.BatchNorm1d(num_features=1024),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.02),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=3),
            #nn.Softmax()
        )

    def forward(self, x):
        return self.convlayer(x)

if __name__ == "__main__":  
    # Hyperparameters
    batch_size = 32
    epochs = 50
    lr = 1e-3
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _, trainloader, _, testloader = data(batch_size=batch_size, transform=transform)


    training_writer = SummaryWriter(log_dir=f"runs/DeepCNN_lr_{lr}")

    # Training

    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    step = 0
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for early stopping
    early_stop_counter = 0  # Counter to track the number of epochs without improvement


    for epoch in range(epochs):
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = train_loss / len(trainloader)
        train_accuracy = 100 * (correct_train / total_train)

        val_loss = 0.0
        correct_val = 0
        total_val = 0
        model.eval()
        with torch.no_grad():
            for batch, (test_images, test_labels) in enumerate(testloader):
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                output = model(test_images)
                loss = criterion(output, test_labels)
                val_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total_val += test_labels.size(0)
                correct_val += (predicted == test_labels).sum().item()
        
        val_loss = val_loss / len(testloader)
        val_accuracy = 100 * (correct_val / total_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_DeepCNN.pth")
            early_stop_counter = 0  # Reset early stop counter
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Log to tensorboard
        training_writer.add_scalar("Train loss", train_loss, step)
        training_writer.add_scalar("Train accuracy", train_accuracy, step)
        training_writer.add_scalar("Validation loss", val_loss, step)
        training_writer.add_scalar("Validation accuracy", val_accuracy, step)
        step += 1

        print(f'Epoch: {epoch+1} / {epochs}  | Training loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.4f} | Validation loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}')