import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the model architecture (should match the trained model)
class DigitRecognitionModel(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

# Create an instance of the model
model = DigitRecognitionModel()

# Load saved model weights
model.load_state_dict(torch.load('modello_digit_recognition.pth'))

# Set the model to evaluation mode
model.eval()

# Function to load and preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Load and preprocess the image
image_path = 'path/to/your/image.jpg'  # Replace with the actual image path
image = preprocess_image(image_path)

# Pass the image through the model
output = model(image)

# Interpret the output
probabilities = torch.nn.functional.softmax(output, dim=1)
predicted_class = torch.argmax(probabilities, dim=1)

# Print the result
print(f'Predicted digit: {predicted_class.item()}')
print(f'Probabilities: {probabilities.detach().numpy()}')
