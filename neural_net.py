"""
simple neural network implementation
============

This is a minimal neural network framework implemented from scratch using NumPy.

Features:
---------
- Dense (fully connected) layers
- ReLU and Softmax activation functions
- Forward and backward propagation
- Categorical Cross-Entropy loss
- Batch-based Gradient Descent optimization
- Simple 3-layer neural network (1 neuron per layer for clarity)

Gebru
Jan 2026
"""

import numpy as np
from activation_fun import ReLU, Softmax
from loss_fun import CategoricalCrossEntropy
from optimizer import SGD

# ============================================================
# Dense Layer
# ============================================================
class Dense:
    """
    Fully connected neural network layer.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    n_neurons : int
        Number of neurons in the layer.
    """

    def __init__(self, n_inputs, n_neurons):
        # self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """ Forward pass. """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        """ Backward pass. """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


# ============================================================
# Training Example
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # Dummy batch input
    # # --------------------------------------------------------
    # X = np.array([[1.0],
    #               [2.0],
    #               [3.0],
    #               [4.0]])

    # y = np.array([0, 0, 0, 0])  # single-class labels

    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [2.5, 2.0],
        [3.0, 3.5],
        [3.5, 0.5],
        [4.0, 1.0]
    ])

    # Two classes (0 or 1)
    y = np.array([0, 0, 1, 1, 0, 1])

    # --------------------------------------------------------
    # Network Architecture (3 layers, 1 neuron each)
    # --------------------------------------------------------
    dense1 = Dense(2, 4)
    relu1 = ReLU()

    dense2 = Dense(4, 3)
    relu2 = ReLU()

    dense3 = Dense(3, 2)
    softmax = Softmax()

    loss_fn = CategoricalCrossEntropy()
    optimizer = SGD(learning_rate=0.1)

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    for epoch in range(1000):

        # Forward pass
        dense1.forward(X)
        relu1.forward(dense1.output)

        dense2.forward(relu1.output)
        relu2.forward(dense2.output)

        dense3.forward(relu2.output)
        softmax.forward(dense3.output)

        loss = loss_fn.forward(softmax.output, y)

        # Backward pass
        loss_fn.backward(softmax.output, y)
        softmax.backward(loss_fn.dinputs)
        dense3.backward(softmax.dinputs)

        relu2.backward(dense3.dinputs)
        dense2.backward(relu2.dinputs)

        relu1.backward(dense2.dinputs)
        dense1.backward(relu1.dinputs)

        # Update parameters
        optimizer.update(dense1)
        optimizer.update(dense2)
        optimizer.update(dense3)

        #show loss and accuracy
        if epoch % 200 == 0:
            predictions = np.argmax(softmax.output, axis=1)
            accuracy = np.mean(predictions == y)
            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {loss:.6f} | "
                f"Accuracy: {accuracy:.2f}"
            )

