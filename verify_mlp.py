import numpy as np
import tensorflow as tf
import os

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("--- Testing NumPy Implementation ---")

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialization
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 10000

np.random.seed(42)
W1 = np.random.uniform(size=(input_size, hidden_size))
b1 = np.random.uniform(size=(1, hidden_size))
W2 = np.random.uniform(size=(hidden_size, output_size))
b2 = np.random.uniform(size=(1, output_size))

# Training Loop
for epoch in range(epochs):
    # Forward Pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    
    # Loss
    error = y - final_output
    loss = np.mean(np.square(error))
    
    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden_layer = d_output.dot(W2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)
    
    # Update Weights
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden_layer) * learning_rate
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

print(f"NumPy Final Loss: {loss:.4f}")
print("NumPy Predictions:", np.round(final_output).flatten())

print("\n--- Testing TensorFlow Implementation ---")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_dim=2, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mean_squared_error',
              metrics=['accuracy'])

history = model.fit(X, y, epochs=100, verbose=0) # Reduced epochs for speed check
print("TF Final Loss:", history.history['loss'][-1])
print("TF Final Accuracy:", history.history['accuracy'][-1])
print("TF Predictions:", np.round(model.predict(X)).flatten())
