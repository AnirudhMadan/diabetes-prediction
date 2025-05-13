import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pennylane as qml
from pennylane.optimize import AdamOptimizer

MODEL_PATH = "saved_models/qml_model.pkl"

class QuantumClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_qubits=8, n_layers=4, epochs=100, lr=0.05):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self.params = None
        self.weights = None

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, inputs, weights):
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)

        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RZ(weights[layer, i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])  # Add final ring connection

        return qml.expval(qml.PauliZ(0))

    def quantum_model(self, X, weights):
        return np.array([(self.qnode(x, weights) + 1) / 2 for x in X])

    def cost(self, weights, X, y):
        preds = self.quantum_model(X, weights)
        preds = np.clip(preds, 1e-5, 1 - 1e-5)
        return -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        X = self._pad_features(X)

        self.weights = 0.01 * np.random.randn(self.n_layers, self.n_qubits)
        optimizer = AdamOptimizer(self.lr)

        for epoch in range(self.epochs):
            self.weights, cost = optimizer.step_and_cost(lambda w: self.cost(w, X, y), self.weights)
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}: cost = {cost:.4f}")
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        X = self._pad_features(X)
        preds = self.quantum_model(X, self.weights)
        return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        X = self._pad_features(X)
        preds = self.quantum_model(X, self.weights)
        return np.vstack((1 - preds, preds)).T

    def _pad_features(self, X):
        if X.shape[1] > self.n_qubits:
            return X[:, :self.n_qubits]
        elif X.shape[1] < self.n_qubits:
            padding = np.zeros((X.shape[0], self.n_qubits - X.shape[1]))
            return np.hstack((X, padding))
        return X

# Train and save the quantum model
def train_and_save_qml_model():
    df = pd.read_csv("diabetes.csv")

    # Replace zeros with median values (only for selected columns)
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_to_replace:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].median(), inplace=True)

    X = df.drop(columns="Outcome")
    y = df["Outcome"].values

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    model = QuantumClassifier(n_qubits=8, n_layers=4, epochs=100, lr=0.05)
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

# Predict using the loaded model
def predict_qml(model, X_test):
    return model.predict(X_test)
