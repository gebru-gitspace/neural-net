# Simple Neural Network From Scratch (NumPy)

This project is a **minimal yet realistic neural network framework implemented entirely from scratch using NumPy**.
It is designed for **learning, demonstration, and evaluation purposes**, focusing on **clarity, correctness, and conceptual understanding** rather than performance or large-scale datasets.

---

## 📌 Project Objectives

The goal of this project is to demonstrate a clear understanding of:

* How neural networks work internally
* Forward propagation and backpropagation
* Gradient-based optimization
* Activation functions and loss functions
* Batch-based training using vectorized operations

All components are implemented **without using deep learning libraries** such as PyTorch or TensorFlow.

---

## 🧠 Network Architecture

The implemented neural network follows this structure:

```
Input Layer (2 features)
      ↓
Dense Layer (4 neurons)
      ↓
ReLU Activation
      ↓
Dense Layer (3 neurons)
      ↓
ReLU Activation
      ↓
Dense Layer (2 neurons)
      ↓
Softmax Activation
      ↓
Categorical Cross-Entropy Loss
```

### Why this architecture?

* **2 input features**: realistic tabular input
* **Multiple hidden layers**: demonstrates feature learning
* **ReLU**: introduces non-linearity
* **Softmax + Cross-Entropy**: standard multi-class classification setup
* **Small size**: easy to trace mathematically and explain

---

## 📁 Project Structure

```
.
├── neural_net.py        # Main training script
├── README.md            # Project documentation
├── loss_fun.py          # CategoricalCrossEnthropy, MAE, MSE
├── activation_fun.py    # ReLu, Sigmoid, Softmax, Tanh
├── optimizer.py         # SDG, SDG Momentum, and Adam
```

All components (layers, activations, loss, optimizer) are implemented **separatelly within the folder** for clarity.

---

## ⚙️ Implemented Components

### 1️⃣ Dense (Fully Connected) Layer

* Performs linear transformation: `output = X · W + b`
* Supports batch processing
* Computes gradients for weights, biases, and inputs

### 2️⃣ Activation Functions

#### ReLU (Rectified Linear Unit)

* Forward: `max(0, x)`
* Backward: blocks gradients for negative inputs

#### Softmax

* Converts logits into probability distributions
* Used in the output layer

---

### 3️⃣ Loss Function

#### Categorical Cross-Entropy

* Measures how well predicted probabilities match true class labels
* Suitable for multi-class classification
* Implemented with numerical stability (clipping)

---

### 4️⃣ Optimizer

#### Stochastic Gradient Descent (SGD)

* Updates weights using gradients
* Learning rate controls step size

```python
W = W - learning_rate * dW
```

---

## 📦 Batch Processing

* Inputs are processed in **batches**, not one sample at a time
* Each row of the input matrix represents one sample

Example:

```python
X.shape == (batch_size, num_features)
```

Batch processing:

* Improves computational efficiency
* Stabilizes gradient updates
* Matches real-world training methods

---

## 🔁 Forward and Backward Propagation

### Forward Pass

1. Inputs pass through dense layers
2. Activations apply non-linear transformations
3. Output probabilities are produced by Softmax
4. Loss is computed

### Backward Pass

1. Loss gradient is computed
2. Gradients flow backward through Softmax and Dense layers
3. Weight and bias gradients are accumulated over the batch
4. Optimizer updates parameters

---

### Run the Training Script

```bash
python neural_net.py
```

---

## 📊 Example Output

```
Epoch    0 | Loss: 0.6931 | Acc: 0.50
Epoch  200 | Loss: 0.51   | Acc: 0.67
Epoch  400 | Loss: 0.31   | Acc: 0.83
Epoch  800 | Loss: 0.18   | Acc: 1.00
```

This shows:

* Initial random guessing (`loss ≈ ln(2)`)
* Gradual loss reduction
* Accuracy improvement
* Correct learning behavior

---

## 🎓 Educational Value

This project demonstrates:

* Manual implementation of neural network math
* Correct gradient flow and parameter updates
* Importance of proper activation–loss pairing
* The role of batch processing
* Why weight initialization matters

---

**Gebru**
January 2026

---

## 🏁 Final Note

This project prioritizes **understanding over abstraction**.
Every line of code is meant to be explainable, traceable, and educational.

If you can explain this project clearly, you understand neural networks at a fundamental level.
