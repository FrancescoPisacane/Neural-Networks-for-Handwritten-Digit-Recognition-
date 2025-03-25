import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import MNIST
import torchvision

from tqdm.notebook import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

# Define batch size for training
batch_size = 256

# Set the dataset root directory
data_set_root = "./datasets"

# Load MNIST dataset with transformations
train = MNIST(data_set_root, train=True,  download=True, transform=transforms.ToTensor())
test  = MNIST(data_set_root, train=False, download=True, transform=transforms.ToTensor())

# Create DataLoader objects for batching and shuffling
train_loader = dataloader.DataLoader(train, shuffle=True, batch_size=batch_size)
test_loader = dataloader.DataLoader(test, shuffle=False, batch_size=batch_size)

# Fetch a batch of images and labels
images, labels = next(iter(train_loader))
print("The input data shape is :\n", images.shape)
print("The target output data shape is :\n", labels.shape)

# Display a batch of images
plt.figure(figsize=(20,10))
out = torchvision.utils.make_grid(images, 32)
plt.imshow(out.numpy().transpose((1, 2, 0)))

# Define a simple Multi-Layer Perceptron model
class Simple_MLP(nn.Module):
    def __init__(self, num_classes):
        super(Simple_MLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)  # Apply softmax activation for classification
        return x

# Instantiate the model
model = Simple_MLP(10)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define the number of training epochs
n_epochs = 10

# Display the model architecture
print(model)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, loss_logger):
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        outputs = model(data)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_logger.append(loss.item())
    return model, optimizer, loss_logger

# Testing function
def test_model(model, test_loader, criterion, loss_logger):
    with torch.no_grad():
        correct_predictions = 0
        total_predictions = 0
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == target).sum().item()
            total_predictions += target.shape[0]

            loss = criterion(outputs, target)
            loss_logger.append(loss.item())
        
        acc = (correct_predictions / total_predictions) * 100.0
        return loss_logger, acc

# Initialize lists to store loss and accuracy metrics
train_loss = []
test_loss = []
test_acc = []

# Training loop
for i in trange(n_epochs, desc="Epoch", leave=False):
    model, optimizer, train_loss = train_epoch(model, train_loader, criterion, optimizer, train_loss)
    test_loss, acc = test_model(model, test_loader, criterion, test_loss)
    test_acc.append(acc)

print("Final Accuracy: %.2f%%" % acc)

# Fetch test images and make predictions
test_images, labels = next(iter(test_loader))
y = model(test_images)

# Display a test image
plt.imshow(test_images[9].detach().numpy().squeeze(), cmap='Greys_r')
plt.show()
plt.scatter(np.arange(10), y[9].detach().cpu().numpy())
plt.show()

# Print probability distribution of predicted classes
torch.set_printoptions(precision=8, sci_mode=False)
print(y[9])

# Save the trained model
torch.save(model.state_dict(), 'modello_digit_recognition.pth')

# Plot training loss
plt.title('Loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(train_loss)

# Plot test loss
plt.title('Test Loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(test_loss)

# Plot test accuracy
plt.title('Test Accuracy')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy (%)')
plt.plot(test_acc)
