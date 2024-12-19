import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from data_preprocessing import DataPrepare


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_loss_and_gradient_with_regularization(X, y_onehot, weights, mu):
    # Compute logits
    logits = X @ weights

    # Compute probabilities
    probs = softmax(logits)

    # Cross-entropy loss
    n_samples = X.shape[0]
    cross_entropy_loss = -np.sum(y_onehot * np.log(probs + 1e-15)) / n_samples

    # Add regularization term to loss
    reg_loss = (mu / 2) * np.sum(weights ** 2)
    total_loss = cross_entropy_loss + reg_loss

    # Compute gradient with regularization
    gradient = (X.T @ (probs - y_onehot)) / n_samples + mu * weights

    return total_loss, gradient


def train_softmax_with_regularization(X, y, num_classes, learning_rate=0.01, num_iterations=1000, mu=0.1):
    n_samples, n_features = X.shape

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y)

    y_onehot = np.eye(num_classes)[y_train_encoded]  # Convert y to one-hot encoding

    # Initialize weights (random initialization)
    weights = np.zeros((n_features, num_classes))

    # Gradient Descent optimization
    for i in range(num_iterations):
        # Compute loss and gradient
        loss, gradient = compute_loss_and_gradient_with_regularization(X, y_onehot, weights, mu)

        # Update weights using Gradient Descent
        weights -= learning_rate * gradient

        # Print loss every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    return weights


def predict(X, weights):
    logits = X @ weights
    probs = softmax(logits)
    return np.argmax(probs, axis=1)


def softmax_regression():
    dl = DataPrepare()
    x_train = dl.X_train_class
    y_train = dl.Y_train_class

    x_test = dl.X_test_class
    y_test = dl.Y_test_class

    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    num_classes = len(np.unique(y_train))
    weights = train_softmax_with_regularization(
        x_train, y_train, num_classes,
        learning_rate=0.01, num_iterations=5000, mu=0.01
    )
    # Make predictions
    y_pred = predict(x_test, weights)

    print(f"Custom Implementation Accuracy: {accuracy_score(y_test_encoded, y_pred) * 100:.2f}%")
