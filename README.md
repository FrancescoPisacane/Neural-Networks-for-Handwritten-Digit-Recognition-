# Handwritten Digit Recognition with PyTorch

## Project Description

This project implements an automatic handwritten digit recognition system using a neural network built with PyTorch. The model is designed to classify images of digits from 0 to 9, based on grayscale images of size 28x28 pixels, similar to those in the MNIST dataset.  

The provided code allows you to train the model, evaluate its performance, and use the trained model to classify custom images containing handwritten digits.  

## Project Structure

Model Training: The model is trained using the MNIST dataset, optimized with the Adam algorithm, and evaluated using the CrossEntropyLoss function.  
Neural Network Architecture: A fully connected feedforward neural network is implemented with four dense layers, ReLU activation functions, and a softmax output for  classification.  
Performance Evaluation: The model is tested on validation data to measure accuracy and loss, allowing performance monitoring.  
Prediction on Custom Images: A script allows loading external images, preprocessing them, and classifying them using the trained model.  

## How to Use the Project

Train the Model
Run the training script (Digit_recognition.ipynb or train_model.py).
The model will be trained using the MNIST dataset.
The trained model weights will be saved as a .pth file.
Use the Model for Predictions
Run the predict_digit.py script to classify custom images.
Provide the path to an image containing a handwritten digit.
The model will process the image and return the predicted digit along with classification probabilities.

## Requirements

To run the code, you need to install the following Python libraries:

torch
torchvision
PIL (Pillow)
numpy
You can install them by running: pip install torch torchvision pillow numpy

## Conclusion

This project is an excellent starting point for exploring deep learning applied to image recognition. It can be extended and improved with more advanced techniques, such as convolutional neural networks (CNNs), to achieve higher accuracy.
