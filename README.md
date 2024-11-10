# Neural Network from Scratch in Python

This project is an implementation of a neural network from scratch using only `numpy`. 
The goal is to classify handwritten digits from the MNIST dataset, an introductory machine 
learning problem that involves recognizing digits from 28x28 pixel grayscale images.

This implementation does not use high-level machine learning libraries like TensorFlow or Keras, 
focusing instead on understanding the foundations of neural networks by implementing the necessary 
math and processes from the ground up.

## Overview

The neural network has:
- **Three Layers**:
  - Input layer with 784 nodes (one for each pixel in the 28x28 image)
  - Hidden layer with 10 nodes
  - Output layer with 10 nodes, each representing a digit from 0 to 9
- **Activation Functions**:
  - ReLU (Rectified Linear Unit) for the hidden layer
  - Softmax for the output layer to interpret predictions as probabilities

This neural network implementation includes:
1. **Forward Propagation**: To compute predictions
2. **Backpropagation**: To calculate gradients and optimize weights and biases
3. **Gradient Descent**: To iteratively minimize the loss function by adjusting weights and biases

## Key Concepts

- **Activation Functions**: ReLU and Softmax help the network learn complex patterns beyond simple linear transformations.
- **Cost Function**: Measures the difference between the network’s prediction and the actual label.
- **Learning Rate**: Controls the size of updates to the weights and biases during gradient descent.

## Dataset

The MNIST dataset provides tens of thousands of labeled examples of handwritten digits in low-resolution grayscale format. 
Each image is represented as a 784-dimensional vector (28x28 pixels), with pixel values between 0 (black) and 255 (white).

## Code Structure

- `initialize_parameters`: Initializes weights and biases for the layers.
- `forward_propagation`: Computes the output predictions based on the current weights and biases.
- `back_propagation`: Calculates gradients for updating weights and biases to reduce error.
- `gradient_descent`: Orchestrates the learning process by iterating through forward and backward passes, updating parameters on each iteration.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nishat-Ahmad/Digit-Classification.git
   cd neural-network-from-scratch

2. Install dependencies:
    ```bash 
    pip install numpy pandas matplotlib

3. Obtain the MNIST dataset and place it in the project directory.
   
Running the Project
To train and test the neural network:
Preprocess the dataset (e.g., shuffle and split into training and validation sets).
Run the script:
  ```bash
  python neural_network.py
  ```
Credits
This project was inspired by Samson Zhang's tutorial on YouTube (https://www.youtube.com/watch?v=w8yWXqWQYmU&t=541s),
which provided the framework for implementing a neural network without machine learning libraries.
