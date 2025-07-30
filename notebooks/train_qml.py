import time
print(f"[{time.time()}] Script starting...")

print(f"[{time.time()}] Importing pandas...")
import pandas as pd
print(f"[{time.time()}] pandas imported.")

print(f"[{time.time()}] Importing pennylane...")
import pennylane as qml
print(f"[{time.time()}] pennylane imported.")

print(f"[{time.time()}] Importing numpy from pennylane...")
from pennylane import numpy as np
print(f"[{time.time()}] numpy imported.")

print(f"[{time.time()}] Importing AdamOptimizer...")
from pennylane.optimize import AdamOptimizer
print(f"[{time.time()}] AdamOptimizer imported.")

print(f"[{time.time()}] Importing train_test_split...")
from sklearn.model_selection import train_test_split
print(f"[{time.time()}] train_test_split imported.")

print(f"[{time.time()}] Importing MinMaxScaler...")
from sklearn.preprocessing import MinMaxScaler
print(f"[{time.time()}] MinMaxScaler imported.")

print(f"[{time.time()}] Importing accuracy_score and classification_report...")
from sklearn.metrics import accuracy_score, classification_report
print(f"[{time.time()}] sklearn.metrics imported.")

print(f"[{time.time()}] All imports successful.")

# # Load and preprocess data
# try:
#     print(f"[{time.time()}] Loading data...")
#     df = pd.read_csv('data/PCA_CICIDS2017.csv', usecols=['PC_1', 'PC_2', 'PC_3', 'PC_4', 'PC_5', 'PC_6', 'PC_7', 'PC_8', 'Label'])
#     print(f"[{time.time()}] Data loaded successfully. Shape: {df.shape}")
    
#     print(f"[{time.time()}] Sampling 1% of the data...")
#     df_sample = df.sample(frac=0.01, random_state=42)
#     print(f"[{time.time()}] Sampled data shape: {df_sample.shape}")

#     X = df_sample.drop('Label', axis=1).values
#     y = df_sample['Label'].values

#     print(f"[{time.time()}] Splitting data into training and testing sets...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     print(f"[{time.time()}] Data split. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

#     print(f"[{time.time()}] Scaling data...")
#     scaler = MinMaxScaler(feature_range=(0, np.pi))
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     print(f"[{time.time()}] Data scaled.")

#     Y_train = y_train * 2 - 1
#     print(f"[{time.time()}] Labels remapped.")

# except Exception as e:
#     print(f"[{time.time()}] An error occurred during data preprocessing: {e}")
#     exit()

# # Quantum Circuit Definition
# print(f"[{time.time()}] Defining quantum circuit...")
# num_qubits = X_train.shape[1]
# num_layers = 3
# dev = qml.device("default.qubit", wires=num_qubits)

# def statepreparation(x):
#     qml.AngleEmbedding(x, wires=range(num_qubits))

# def layer(W):
#     for i in range(num_qubits):
#         qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
#     for i in range(num_qubits - 1):
#         qml.CNOT(wires=[i, i + 1])
#     qml.CNOT(wires=[num_qubits - 1, 0])

# @qml.qnode(dev, interface="autograd")
# def circuit(weights, x):
#     statepreparation(x)
#     for W in weights:
#         layer(W)
#     return qml.expval(qml.PauliZ(0))

# def variational_classifier(weights, bias, x):
#     return circuit(weights, x) + bias

# def square_loss(labels, predictions):
#     return np.mean((labels - predictions) ** 2)

# def cost(weights, bias, X, Y):
#     predictions = [variational_classifier(weights, bias, x) for x in X]
#     return square_loss(Y, predictions)

# print(f"[{time.time()}] Quantum circuit defined.")

# # Training the model
# print(f"[{time.time()}] Starting model training...")
# np.random.seed(42)
# weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
# bias_init = np.array(0.0, requires_grad=True)

# opt = AdamOptimizer(0.1)
# batch_size = 5
# num_epochs = 10

# weights = weights_init
# bias = bias_init

# try:
#     for epoch in range(num_epochs):
#         epoch_start_time = time.time()
#         print(f"[{epoch_start_time}] Starting Epoch {epoch+1}/{num_epochs}...")
        
#         batch_index = np.random.randint(0, len(X_train_scaled), (batch_size,))
#         X_batch = X_train_scaled[batch_index]
#         Y_batch = Y_train[batch_index]
        
#         weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

#         if (epoch + 1) % 2 == 0:
#             predictions_train = [np.sign(variational_classifier(weights, bias, x)) for x in X_train_scaled]
#             acc_train = accuracy_score(Y_train, predictions_train)
#             current_cost = cost(weights, bias, X_train_scaled, Y_train)
#             print(f"[{time.time()}] Epoch {epoch+1:2d} | Cost: {current_cost:0.7f} | Accuracy: {acc_train:0.7f}")

#     print(f"[{time.time()}] Training finished.")

# except Exception as e:
#     print(f"[{time.time()}] An error occurred during training: {e}")
#     exit()

# # Evaluate the model
# print(f"[{time.time()}] Evaluating model...")
# try:
#     predictions_test = [np.sign(variational_classifier(weights, bias, x)) for x in X_test_scaled]
#     Y_test = y_test * 2 - 1

#     total_accuracy = accuracy_score(Y_test, predictions_test)
#     print(f"\n[{time.time()}] Total Accuracy on Test Set: {total_accuracy:.4f}")

#     print(f"\n[{time.time()}] Classification Report (Per-class accuracy):")
#     print(classification_report(Y_test, predictions_test))

# except Exception as e:
#     print(f"[{time.time()}] An error occurred during evaluation: {e}")

# print(f"[{time.time()}] Script finished.")

