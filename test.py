import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("dataset/creditcard.csv")

# Separate the dataset into fraud and non-fraud datasets
df_fraud = df[df["Class"] == 1]
df_non_fraud = df[df["Class"] == 0]

# Under-sample the non-fraud transactions to match the number of fraud transactions
df_non_fraud_under = df_non_fraud.sample(len(df_fraud), random_state=42)

# Combine the under-sampled non-fraud transactions with the fraud transactions
df_balanced = pd.concat([df_fraud, df_non_fraud_under])

# Shuffle the combined dataset
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Proceed with preprocessing on the balanced dataset
X = df_balanced.drop("Class", axis=1).values
y = df_balanced["Class"].values


# Data Preprocessing Functions
def split(X, y, test_size=1 / 3):
    np.random.seed(42)  # For reproducibility
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def manual_scale(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


# Evaluation Metrics
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return precision, recall, f1


# Preprocess the data
X_scaled = manual_scale(X)
X_train, X_test, y_train, y_test = split(X_scaled, y)


# SVM Implementation
class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)


# Logistic Regression Implementation
class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, lambda_=0.1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_ = lambda_
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        for _ in range(self.iterations):
            h = self.sigmoid(X.dot(self.theta))
            gradient = np.dot(X.T, (h - y)) / m
            gradient[1:] += (self.lambda_ / m) * self.theta[
                1:
            ]  # Regularization for j >= 1
            self.theta -= self.learning_rate * gradient

    def predict_prob(self, X):
        return self.sigmoid(X.dot(self.theta))

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)


# Model Training and Evaluation
model_lr = LogisticRegression(learning_rate=0.1, iterations=3000, lambda_=0.1)
model_lr.fit(X_train, y_train)
predictions_lr = model_lr.predict(X_test)
accuracy_lr = accuracy(y_test, predictions_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%")

svm = LinearSVM()
svm.fit(X_train, y_train)
predictions_svm = svm.predict(X_test)
predictions_svm = np.where(predictions_svm < 0, 0, 1)
accuracy_svm = accuracy(y_test, predictions_svm)
precision_svm, recall_svm, f1_svm = precision_recall_f1(y_test, predictions_svm)
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")
print(f"SVM Precision: {precision_svm}, Recall: {recall_svm}, F1-Score: {f1_svm}")
