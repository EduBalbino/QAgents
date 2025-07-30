# %%
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler

import math

# %%
print("Loading and preparing data...")
# Load the dataset
df = pd.read_csv('data/PCA_CICIDS2017.csv')
print("Dataset loaded.")

# Prepare data
X = df[['PC_1', 'PC_2', 'PC_3', 'PC_4', 'PC_5', 'PC_6', 'PC_7', 'PC_8']]
y = df['Label']
print("Features and labels extracted.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
print("Data split into training and testing sets.")

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled.")

# Transform labels to {-1, 1}
Y_train = np.array(y_train.values * 2 - 1, requires_grad=False)
Y_test = np.array(y_test.values * 2 - 1, requires_grad=False)
print("Labels transformed.")

X_train_scaled = np.array(X_train_scaled, requires_grad=False)
print("Data preparation complete.")

# %%
"""
Classificadores híbridos (computador clássico + circuito quântico)
"""

# %%
num_qubits = X_train_scaled.shape[1]
num_layers = 3

print(f"Initializing quantum device with {num_qubits} qubits.")
dev = qml.device("default.qubit", wires=num_qubits)
print("Quantum device initialized.")

# quantum circuit functions
def statepreparation(x):
    qml.AngleEmbedding(x, wires=range(num_qubits), rotation='Y')

def layer(W):
    for i in range(num_qubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[num_qubits - 1, 0])

@qml.qnode(dev, interface="autograd")
def circuit(weights, x):
    statepreparation(x)
    for W in weights:
        layer(W)
    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, X):
    return circuit(weights, X) + bias

def square_loss(labels, predictions):
    return np.mean((labels - predictions) ** 2)

def accuracy(labels, predictions):
    return np.sum(np.sign(predictions) == labels) / len(labels)


def cost(weights, bias, X, Y):
    predictions = variational_classifier(weights, bias, X)
    return square_loss(Y, predictions)

# %%
print("Initializing training parameters...")
# setting init params
np.random.seed(42)
weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

opt = AdamOptimizer(0.1)
num_it = 20
batch_size = 16

weights = weights_init
bias = bias_init
print("Starting training...")
for it in range(num_it):
    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, len(X_train_scaled), (batch_size,))
    X_batch = X_train_scaled[batch_index]
    Y_batch = Y_train[batch_index]
    
    weights, bias = opt.step(lambda w, b: cost(w, b, X_batch, Y_batch), weights, bias)

    # Compute accuracy on the batch
    predictions_batch = np.sign(variational_classifier(weights, bias, X_batch))
    acc_batch = accuracy(Y_batch, predictions_batch)

    # Print the batch cost and batch accuracy
    print(
        "Iter: {:5d} | Batch Cost: {:0.7f} | Batch Accuracy: {:0.7f} ".format(
            it + 1, cost(weights, bias, X_batch, Y_batch), acc_batch
        )
    )

# Final accuracy on the entire training set
print("\nCalculating final metrics on the training set...")
predictions_train_list = []
for i in range(0, len(X_train_scaled), batch_size):
    X_batch = X_train_scaled[i:i + batch_size]
    predictions_batch = variational_classifier(weights, bias, X_batch)
    predictions_train_list.append(predictions_batch)

predictions_train = np.concatenate(predictions_train_list)
final_acc = accuracy(Y_train, np.sign(predictions_train))
final_cost = square_loss(Y_train, predictions_train)

print("Training complete.")
print(f"Final Training Cost: {final_cost:0.7f} | Final Training Accuracy: {final_acc:0.7f}")


# %%
print("\nEvaluating model on test data...")
X_test_scaled = np.array(X_test_scaled, requires_grad=False)

predictions_test_list = []
for i in range(0, len(X_test_scaled), batch_size):
    X_batch = X_test_scaled[i:i + batch_size]
    predictions_batch = variational_classifier(weights, bias, X_batch)
    predictions_test_list.append(predictions_batch)

predictions = np.concatenate(predictions_test_list)
predictions_signed = np.sign(predictions)


print("--- Test Results ---")
print(f"Acurácia: {accuracy_score(Y_test, predictions_signed)}")
print(f"Precisão: {precision_score(Y_test, predictions_signed, average='macro')}")
print(f"Recall: {recall_score(Y_test, predictions_signed, average='macro')}")
print(f"F1-Score: {f1_score(Y_test, predictions_signed, average='macro')}")
print("--------------------")
