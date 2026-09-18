"""
model.py

A small feedforward neural network: 5 -> 64 -> 64 -> 1, matching the
architecture in the project README. Written directly in NumPy (forward
pass, ReLU, backward pass, gradient descent update) instead of pulling in
a full deep learning framework, since the network is tiny.

This also makes the IMC mapping step easier later: the weight matrices
and layer sizes are plain NumPy arrays that Raya can inspect directly to
model matrix-vector operations for the in-memory computing analysis.
"""

import numpy as np


class OptionPriceNN:
    def __init__(self, input_size=5, hidden_size=64, seed=42):
        rng = np.random.default_rng(seed)

        # He initialization, reasonable default for ReLU networks
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_size), size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_size), size=(hidden_size, hidden_size))
        self.b2 = np.zeros((1, hidden_size))

        self.W3 = rng.normal(0, np.sqrt(2.0 / hidden_size), size=(hidden_size, 1))
        self.b3 = np.zeros((1, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(np.float64)

    def forward(self, X):
        """
        Runs a forward pass and caches the intermediate values needed
        for backpropagation.
        """
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)

        z3 = a2 @ self.W3 + self.b3  # linear output, price is a raw number

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}
        return z3, cache

    def backward(self, cache, y_true, learning_rate):
        """
        Computes gradients for MSE loss and updates weights in place with
        plain gradient descent.
        """
        m = y_true.shape[0]
        y_pred = cache["z3"]

        # dL/dz3 for MSE loss = (2/m) * (y_pred - y_true)
        dz3 = (2.0 / m) * (y_pred - y_true)
        dW3 = cache["a2"].T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_derivative(cache["z2"])
        dW2 = cache["a1"].T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(cache["z1"])
        dW1 = cache["X"].T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

    def param_count(self):
        total = 0
        for param in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]:
            total += param.size
        return total

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load(self, path):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W3, self.b3 = data["W3"], data["b3"]
