# MNIST Digit Classification using PyTorch

This project implements a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset.

## Model Architecture

Conv2D → ReLU → MaxPool  
Conv2D → ReLU → MaxPool  
Fully Connected Layer  
Output Layer (10 classes)

## Project Structure

model.py → CNN architecture
train.py → training pipeline
test.py → inference
requirements.txt → dependencies

## Installation

pip install -r requirements.txt

## Training

python train.py

## Testing

python test.py

## Dataset

MNIST dataset automatically downloads during training.

## Framework

PyTorch
