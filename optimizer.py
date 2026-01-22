"""
Optimization algorithms for training neural networks using gradient descent.

This module implements several common optimizers used to update neural network
parameters based on computed gradients during backpropagation.

Implemented Optimizers:
-----------------------
1. SGD (Stochastic Gradient Descent)
2. SGD with Momentum
3. Adam (Adaptive Moment Estimation)

All optimizers are designed to work with layers that have:
- weights, biases
- dweights, dbiases

Gebru
Jan 2026
"""

import numpy as np


# ============================================================
# 1. Stochastic Gradient Descent (SGD)
# ============================================================
class SGD:
    """
    Standard Stochastic Gradient Descent optimizer.

    Update rule:
        W = W - learning_rate * dW
        b = b - learning_rate * db

    This optimizer uses only the current gradient.
    """

    def __init__(self, learning_rate=0.01):
        """
        Parameters
        ----------
        learning_rate : float
            Step size used to update parameters.
        """
        self.learning_rate = learning_rate

    def update(self, layer):
        """
        Update weights and biases of a layer.
        """
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


# ============================================================
# 2. SGD with Momentum
# ============================================================
class SGD_Momentum:
    """
    Stochastic Gradient Descent with Momentum.

    Momentum accelerates learning by accumulating a velocity vector
    in directions of persistent gradient descent.

    Update rule:
        v = momentum * v - learning_rate * dW
        W = W + v
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Parameters
        ----------
        learning_rate : float
            Step size for updates.
        momentum : float
            Controls how much past gradients influence current updates.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, layer):
        """
        Update weights and biases using momentum.
        """
        # Initialize velocity terms if they do not exist
        if not hasattr(layer, "weight_velocity"):
            layer.weight_velocity = np.zeros_like(layer.weights)
            layer.bias_velocity = np.zeros_like(layer.biases)

        # Update velocities
        layer.weight_velocity = (
            self.momentum * layer.weight_velocity
            - self.learning_rate * layer.dweights
        )

        layer.bias_velocity = (
            self.momentum * layer.bias_velocity
            - self.learning_rate * layer.dbiases
        )

        # Update parameters
        layer.weights += layer.weight_velocity
        layer.biases += layer.bias_velocity


# ============================================================
# 3. Adam Optimizer
# ============================================================
class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Adam combines:
    - Momentum (first moment estimate)
    - Adaptive learning rates (second moment estimate)

    It is widely used due to fast convergence and stability.
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7
    ):
        """
        Parameters
        ----------
        learning_rate : float
            Base learning rate.
        beta1 : float
            Decay rate for first moment (mean of gradients).
        beta2 : float
            Decay rate for second moment (variance of gradients).
        epsilon : float
            Small constant to avoid division by zero.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

    def update(self, layer):
        """
        Update weights and biases using Adam optimization.
        """

        # Initialize moment estimates if they do not exist
        if not hasattr(layer, "m_w"):
            layer.m_w = np.zeros_like(layer.weights)
            layer.v_w = np.zeros_like(layer.weights)
            layer.m_b = np.zeros_like(layer.biases)
            layer.v_b = np.zeros_like(layer.biases)

        self.iterations += 1

        # First moment (mean of gradients)
        layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * layer.dweights
        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.dbiases

        # Second moment (variance of gradients)
        layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * (layer.dweights ** 2)
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.dbiases ** 2)

        # Bias correction
        m_w_corr = layer.m_w / (1 - self.beta1 ** self.iterations)
        v_w_corr = layer.v_w / (1 - self.beta2 ** self.iterations)

        m_b_corr = layer.m_b / (1 - self.beta1 ** self.iterations)
        v_b_corr = layer.v_b / (1 - self.beta2 ** self.iterations)

        # Update parameters
        layer.weights -= (
            self.learning_rate * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
        )
        layer.biases -= (
            self.learning_rate * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)
        )

