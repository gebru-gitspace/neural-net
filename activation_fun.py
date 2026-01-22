"""
Activation functions for neural networks
=======================================

This module implements common activation functions used in neural networks.

Features:
---------
- ReLU (Rectified Linear Unit)
- Softmax
- Sigmoid
- Tanh

Gebru
Jan 2026
"""

import numpy as np


# Activation Functions

class ReLU:
    """
    Rectified Linear Unit (ReLU) activation.
    ReLU(x) = max(0, x)
    """

    def forward(self, inputs):
        """ Forward pass."""
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """ Backward pass."""
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    """
    Softmax activation for multi-class classification output.
    Converts raw scores into probabilities that sum to 1.
    """

    def forward(self, inputs):
        """ Forward pass with numerical stability."""
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        """ Backward pass using the Jacobian matrix."""
        self.dinputs = np.empty_like(dvalues)

        for i, (output, dvalue) in enumerate(zip(self.output, dvalues)):
            output = output.reshape(-1, 1)
            jacobian = np.diagflat(output) - np.dot(output, output.T)
            self.dinputs[i] = np.dot(jacobian, dvalue)


class Sigmoid:
    """
    Sigmoid activation function.
    Sigmoid(x) = 1 / (1 + e^(-x))
    Commonly used for binary classification.
    """

    def forward(self, inputs):
        """ Forward pass."""
        self.output = 1 / (1 + np.exp(-inputs))
        self.inputs = inputs

    def backward(self, dvalues):
        """ Backward pass."""
        self.dinputs = dvalues * (self.output * (1 - self.output))


class Tanh:
    """
    Hyperbolic Tangent (Tanh) activation function.

    Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Output range: (-1, 1)
    """

    def forward(self, inputs):
        """ Forward pass."""
        self.output = np.tanh(inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        """ Backward pass."""
        self.dinputs = dvalues * (1 - self.output ** 2)
