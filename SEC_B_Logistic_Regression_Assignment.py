
# ===============================================
# SEC-B Assignment - Machine Learning
# Name: Aaryat Khatri
# Course: BCA (AI & Data Science)
# Topic: Logistic Regression using TensorFlow
# ===============================================

# ------------------------------
# 1. Role of Weights in a Neuron
# ------------------------------
'''
Weights in a neuron determine the importance of input features.
Each input is multiplied by a weight, and the weighted sum is passed
through an activation function.

Formula:
z = w1x1 + w2x2 + ... + b

Weights are updated during training using backpropagation.
'''

# ------------------------------
# 2. Activation Function
# ------------------------------
'''
An activation function introduces non-linearity into the model.
Common activation functions:
- Sigmoid
- ReLU
- Tanh

In logistic regression, we use the Sigmoid activation function.
'''

# ------------------------------
# 3. Probability Distribution in ML
# ------------------------------
'''
A probability distribution describes how values of a random variable
are distributed.

In logistic regression, output follows Bernoulli distribution (0 or 1).
'''

# ------------------------------
# 4. Gradient in Optimization
# ------------------------------
'''
Gradient is the vector of partial derivatives of the loss function.
It helps in updating weights using Gradient Descent.

Update Rule:
w = w - learning_rate * gradient
'''

# ------------------------------
# 5. Logistic Regression Project using TensorFlow
# ------------------------------

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Create dataset
X, y = make_classification(n_samples=1000, 
                           n_features=2, 
                           n_classes=2, 
                           random_state=42)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Build Logistic Regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Make predictions
predictions = model.predict(X_test[:5])
print("Sample Predictions:", predictions)
