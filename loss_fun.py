"""
Loss functions for training neural networks using gradient-based optimization.

This module implements several commonly used loss functions, each with
both forward and backward passes to support backpropagation.

Implemented Loss Functions:
---------------------------
1. Mean Squared Error (MSE)
2. Mean Absolute Error (MAE)
3. Huber Loss
4. Binary Cross-Entropy
5. Categorical Cross-Entropy

All loss classes follow a consistent interface:
- forward(y_pred, y_true)
- backward(y_pred, y_true)

Gebru
Jan 2026
"""

import numpy as np


# ============================================================
# 1. Mean Squared Error (MSE)
# ============================================================
class MeanSquaredError:
    """
    Mean Squared Error (MSE) loss.

    Used for regression problems.

    Formula:
        L = mean((y_true - y_pred)^2)
    """

    def forward(self, y_pred, y_true):
        """
        Compute the MSE loss.
        """
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_pred, y_true):
        """
        Compute gradient of the loss w.r.t. predictions.
        """
        samples = len(y_pred)
        self.dinputs = -2 * (y_true - y_pred) / samples


# ============================================================
# 2. Mean Absolute Error (MAE)
# ============================================================
class MeanAbsoluteError:
    """
    Mean Absolute Error (MAE) loss.

    Less sensitive to outliers than MSE.

    Formula:
        L = mean(|y_true - y_pred|)
    """

    def forward(self, y_pred, y_true):
        """
        Compute the MAE loss.
        """
        return np.mean(np.abs(y_true - y_pred))

    def backward(self, y_pred, y_true):
        """
        Compute gradient of the loss w.r.t. predictions.
        """
        samples = len(y_pred)
        self.dinputs = np.sign(y_pred - y_true) / samples


# ============================================================
# 3. Huber Loss
# ============================================================
class HuberLoss:
    """
    Huber loss.

    Combines MSE and MAE for robust regression.

    Formula:
        if |error| <= delta:
            0.5 * error^2
        else:
            delta * (|error| - 0.5 * delta)
    """

    def __init__(self, delta=1.0):
        """
        Parameters
        ----------
        delta : float
            Threshold at which loss transitions from quadratic to linear.
        """
        self.delta = delta

    def forward(self, y_pred, y_true):
        """
        Compute the Huber loss.
        """
        error = y_pred - y_true
        condition = np.abs(error) <= self.delta

        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)

        return np.mean(np.where(condition, squared_loss, linear_loss))

    def backward(self, y_pred, y_true):
        """
        Compute gradient of the loss w.r.t. predictions.
        """
        error = y_pred - y_true
        self.dinputs = np.where(
            np.abs(error) <= self.delta,
            error,
            self.delta * np.sign(error)
        ) / len(y_pred)


# ============================================================
# 4. Binary Cross-Entropy
# ============================================================
class BinaryCrossEntropy:
    """
    Binary Cross-Entropy loss.

    Used for binary classification with sigmoid output.

    Formula:
        L = -[y*log(p) + (1-y)*log(1-p)]
    """

    def forward(self, y_pred, y_true):
        """
        Compute binary cross-entropy loss.
        """
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return np.mean(
            -(y_true * np.log(y_pred) +
              (1 - y_true) * np.log(1 - y_pred))
        )

    def backward(self, y_pred, y_true):
        """
        Compute gradient of the loss w.r.t. predictions.
        """
        samples = len(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        self.dinputs = (
            -(y_true / y_pred -
              (1 - y_true) / (1 - y_pred))
        ) / samples


# ============================================================
# 5. Categorical Cross-Entropy
# ============================================================
class CategoricalCrossEntropy:
    """
    Categorical Cross-Entropy loss.

    Used for multi-class classification with softmax output.
    """

    def forward(self, y_pred, y_true):
        """
        Compute categorical cross-entropy loss.
        """
        samples = len(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred * y_true, axis=1)

        return np.mean(-np.log(correct_confidences))

    def backward(self, y_pred, y_true):
        """
        Compute gradient of the loss w.r.t. predictions.
        """
        samples = len(y_pred)
        labels = y_pred.shape[1]

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / y_pred
        self.dinputs /= samples

