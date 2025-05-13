import os
import pickle
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pennylane as qml
from pennylane.optimize import AdamOptimizer

MODEL_PATH = "saved_models/qml_model.pkl"

class QuantumClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_qubits=4, n_layers=2, epochs=30, lr=0.1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self.params = None

        # Create quantum device (simulator)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        # Bind the circuit to a QNode
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, inputs, weights):
        """Define the quantum circuit."""
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)

        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RZ(weights[layer, i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return qml.expval(qml.PauliZ(0))

    def quantum_model(self, X, weights):
        """Evaluate the quantum circuit for multiple inputs."""
        return np.array([self.qnode(x, weights) for x in X])

    def cost(self, weights, X, y):
        """Compute the mean squared error cost function."""
        preds = self.quantum_model(X, weights)
        return np.mean((preds - y) ** 2)

    def fit(self, X, y):
        """Train the quantum model using gradient descent."""
        X = self.scaler.fit_transform(X)
        X = self._pad_features(X)

        self.weights = 0.01 * np.random.randn(self.n_layers, self.n_qubits)
        optimizer = AdamOptimizer(self.lr)

        for epoch in range(self.epochs):
            self.weights, cost = optimizer.step_and_cost(lambda w: self.cost(w, X, y), self.weights)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: cost = {cost:.4f}")
        return self

    def predict(self, X):
        """Make binary predictions."""
        X = self.scaler.transform(X)
        X = self._pad_features(X)
        preds = self.quantum_model(X, self.weights)
        return (preds > 0).astype(int)

    def predict_proba(self, X):
        """Return probabilities."""
        X = self.scaler.transform(X)
        X = self._pad_features(X)
        preds = self.quantum_model(X, self.weights)
        return np.vstack((1 - preds, preds)).T

    def _pad_features(self, X):
        """Pad or truncate feature vectors to match qubit count."""
        if X.shape[1] > self.n_qubits:
            return X[:, :self.n_qubits]
        elif X.shape[1] < self.n_qubits:
            padding = np.zeros((X.shape[0], self.n_qubits - X.shape[1]))
            return np.hstack((X, padding))
        return X

# Train and save the quantum model
def train_and_save_qml_model():
    import pandas as pd
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns="Outcome")
    y = df["Outcome"].values

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    model = QuantumClassifier(n_qubits=8, n_layers=2, epochs=50)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": model.scaler
        }, f)
    print("Quantum model trained and saved.")

# Load the saved quantum model
def load_qml_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    model.scaler = data["scaler"]
    return model

# Make predictions with the loaded quantum model
def predict_qml(model, X_test):
    return model.predict(X_test)
