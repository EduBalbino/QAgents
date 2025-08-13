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

import time
import sys
import os
import datetime

# --- Start of logging snippet ---
# Create logs directory if it doesn't exist
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Get current timestamp for a unique log file name
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_filename = os.path.join(log_dir, f"QML_CICI-2019_{timestamp}.log")

# Redirect stdout to a logger object that writes to both console and file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_filename)
# --- End of logging snippet ---
import random # Needed for the new sampling method

# ==============================================================================
# --- MASTER CONFIGURATION ---
# Set a number for a fast dev run (e.g., 5000).
# Set to None to use the full dataset for a production run.
DEV_RUN_SAMPLE_SIZE = 500000
# ==============================================================================

# %%
# --- NEW: EFFICIENT DATA LOADING FUNCTION ---
def load_and_sample_csv(filepath, sample_size):
    """
    Efficiently loads a CSV, sampling it *before* loading into memory.
    
    Args:
        filepath (str): Path to the CSV file.
        sample_size (int or None): The number of rows to randomly sample.
                                   If None, loads the entire file.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the full or sampled data.
    """
    if sample_size is None:
        print("Loading full dataset into memory...")
        return pd.read_csv(filepath)

    # --- Two-pass method for memory-efficient sampling ---
    # Pass 1: Get the total number of lines without loading data
    print("Quickly scanning file to count rows...")
    with open(filepath, 'r') as f:
        # Count lines, subtracting 1 for the header
        num_lines = sum(1 for line in f) - 1

    # Ensure we don't try to sample more than what's available
    sample_size = min(sample_size, num_lines)
    
    # Pass 2: Select random rows to skip and load only the desired ones
    print(f"File has {num_lines} rows. Loading a random sample of {sample_size} rows.")
    
    # Create a list of rows to skip. We add 1 because skiprows is 0-indexed and we want to skip lines after the header.
    rows_to_skip = sorted(random.sample(range(1, num_lines + 1), num_lines - sample_size))
    
    # Pandas can take a list of row numbers to skip, which is very efficient
    df = pd.read_csv(filepath, skiprows=rows_to_skip)
    return df
# --- END NEW FUNCTION ---

# %%
print("--- Starting Data Preparation ---")
start_time = time.time()

# --- FIX: Use the new efficient loading function ---
df = load_and_sample_csv('data/PCA_CIC-DDoS2019.csv', DEV_RUN_SAMPLE_SIZE)
print(f"Dataset loaded. Final shape: {df.shape}")
# --- END FIX ---

# Prepare data
# Define features for the CICI-2019 dataset
features = [f'PC_{i+1}' for i in range(8)]
X = df[features]
y = df['Label']
print("Features and labels extracted.")

# Split data
# --- FIX: Use stratified splitting to maintain label distribution in train/test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# --- END FIX ---
print("Data split into training and testing sets.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled.")

# Transform labels to {-1, 1}
Y_train = np.array(y_train.values * 2 - 1, requires_grad=False)
Y_test = np.array(y_test.values * 2 - 1, requires_grad=False)
print("Labels transformed.")
print(f"Data preparation complete. Time elapsed: {time.time() - start_time:.2f} seconds.")

# %%
"""
Classificadores híbridos (computador clássico + circuito quântico)
"""

# %%
print("\n--- Starting Quantum Setup ---")
num_qubits = X_train_scaled.shape[1]
num_layers = 3

print(f"Initializing lightning.qubit device with {num_qubits} qubits for multi-core performance.")
dev = qml.device("lightning.qubit", wires=num_qubits)
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

def square_loss(labels, predictions, class_weights=None):
    loss = (labels - predictions) ** 2
    if class_weights is not None:
        loss = class_weights * loss
    return np.mean(loss)

def accuracy(labels, predictions):
    return np.sum(np.sign(predictions) == labels) / len(labels)

# --- FIX: Modify cost function to include class weights ---
def cost(weights, bias, X, Y, class_weights_map=None):
    predictions = variational_classifier(weights, bias, X)
    
    # Apply class weights if a map is provided
    weights_tensor = None
    if class_weights_map:
        weights_tensor = np.array([class_weights_map[label] for label in Y])

    return square_loss(Y, predictions, class_weights=weights_tensor)
# --- END FIX ---

# %%
print("\n--- Initializing Training ---")
np.random.seed(42)
weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

opt = AdamOptimizer(0.1)
batch_size = 100
# --- FIX: Ensure batch size is not larger than the training set ---
batch_size = min(batch_size, len(X_train_scaled))
# --- END FIX ---

# --- FIX: Calculate iterations for one epoch ---
# An epoch is one full pass over the training data.
num_it = len(X_train_scaled) // batch_size
print(f"Training for 1 epoch ({num_it} iterations for a training set of size {len(X_train_scaled)}).")
# --- END FIX ---

validation_size = min(5 * batch_size, len(X_train_scaled) - 1)
val_indices = np.random.randint(0, len(X_train_scaled), validation_size)
X_val = X_train_scaled[val_indices]
Y_val = Y_train[val_indices]
print(f"Validation set created with {validation_size} data points.")

weights = weights_init
bias = bias_init

# --- FIX: Calculate class weights to handle imbalance ---
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights from the training data
class_labels = np.unique(Y_train)
class_weights_array = compute_class_weight('balanced', classes=class_labels, y=Y_train)
class_weights_map = {label: weight for label, weight in zip(class_labels, class_weights_array)}

print(f"\nClass weights computed for handling imbalance: {class_weights_map}")
# --- END FIX ---

print(f"Parameters initialized. Starting training for {num_it} iterations with batch size {batch_size}.")
start_time = time.time()

for it in range(num_it):
    batch_index = np.random.randint(0, len(X_train_scaled), (batch_size,))
    X_batch = X_train_scaled[batch_index]
    Y_batch = Y_train[batch_index]
    
    # Perform the optimization step, passing the class weights to the cost function
    weights, bias = opt.step(lambda w, b: cost(w, b, X_batch, Y_batch, class_weights_map), weights, bias)

    # --- FIX: Print progress every 10 iterations or on the last iteration ---
    if (it + 1) % 10 == 0 or (it + 1) == num_it:
        # Calculate metrics for the current batch
        predictions_batch = variational_classifier(weights, bias, X_batch)
        cost_batch = square_loss(Y_batch, predictions_batch)
        acc_batch = accuracy(Y_batch, np.sign(predictions_batch))
        
        print(
            f"Iter: {it + 1:5d} | Batch Cost: {cost_batch:0.7f} | Batch Acc: {acc_batch:0.7f}"
        )
    # --- END FIX ---

print(f"\nTraining loop finished. Total training time: {time.time() - start_time:.2f} seconds.")

# Quick validation check
print("\n--- Running Quick Validation Check ---")
start_time = time.time()
predictions_val = variational_classifier(weights, bias, X_val)
val_acc = accuracy(Y_val, np.sign(predictions_val))
val_cost = square_loss(Y_val, predictions_val)
print(f"Validation Cost: {val_cost:0.7f} | Validation Accuracy: {val_acc:0.7f}")
print(f"Validation check completed in: {time.time() - start_time:.2f} seconds.")


# %%
print("\n--- Evaluating model on test data ---")
start_time = time.time()

if len(X_test_scaled) < 1000:
    print("Test set is small, processing in a single batch.")
    predictions = variational_classifier(weights, bias, X_test_scaled)
else:
    print("Test set is large, processing in mini-batches.")
    predictions_test_list = []
    for i in range(0, len(X_test_scaled), batch_size):
        X_batch = X_test_scaled[i:i + batch_size]
        predictions_batch = variational_classifier(weights, bias, X_batch)
        predictions_test_list.append(predictions_batch)
    predictions = np.concatenate(predictions_test_list)

predictions_signed = np.sign(predictions)

# --- FIX: Calculate and print accuracy per label ---
print("--- Test Results ---")
# Overall accuracy
print(f"Acurácia (Overall): {accuracy_score(Y_test, predictions_signed):.4f}")

# Accuracy for label -1 (Benign)
mask_benign = Y_test == -1
acc_benign = accuracy_score(Y_test[mask_benign], predictions_signed[mask_benign])
print(f"Acurácia (Benign):  {acc_benign:.4f} (from {np.sum(mask_benign)} samples)")

# Accuracy for label 1 (Attack)
mask_attack = Y_test == 1
acc_attack = accuracy_score(Y_test[mask_attack], predictions_signed[mask_attack])
print(f"Acurácia (Attack):  {acc_attack:.4f} (from {np.sum(mask_attack)} samples)")

# Other metrics
print(f"Precisão: {precision_score(Y_test, predictions_signed, average='macro', zero_division=0):.4f}")
print(f"Recall:   {recall_score(Y_test, predictions_signed, average='macro', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(Y_test, predictions_signed, average='macro', zero_division=0):.4f}")
# --- END FIX ---
print("--------------------")
print(f"Evaluation finished in: {time.time() - start_time:.2f} seconds.")