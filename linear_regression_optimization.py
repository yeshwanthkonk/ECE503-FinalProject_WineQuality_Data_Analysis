# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPrepare



# Custom Optimization using MSE as Loss
def mse_loss(w, X, y):
    y_pred = X @ w
    return np.mean((y - y_pred) ** 2)

def run():
    dl = DataPrepare()
    # Optimization
    initial_w = np.zeros(dl.X_train.shape[1])
    result = minimize(mse_loss, initial_w, args=(dl.X_train, dl.Y_train), method='BFGS')
    optimized_w = result.x

    # Predictions using Optimized Weights
    y_pred_opt = dl.X_test @ optimized_w
    mse_opt = mean_squared_error(dl.Y_test, y_pred_opt)
    print(f"Optimized Regression - MSE: {mse_opt:.4f}")


def predict_actual_plot(y_test, y_pred):
    # Scatter plot of actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.title("Actual vs Predicted Quality")
    plt.show()

def line_plot(y_test, y_pred):
    limit = 200
    plt.figure(figsize=(8, 6))
    plt.plot(range(y_test.shape[0])[:limit], y_test[:limit], label='true lables', color='blue', linestyle='--')  # Solid line
    plt.plot(range(y_pred.shape[0])[:limit], y_pred[:limit], label='predicted', color='red', linestyle='-')
    plt.xlabel('Samples')
    plt.ylabel('Quality Value')
    plt.title('Actual vs Predicted Quality')
    plt.legend(loc='upper left')  # Position the legend

    # Show the plot
    plt.show()


def residual_plot(y_test, y_pred):
    # Residuals
    residuals = y_test - y_pred

    # Residual plot
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=20, color='green')
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()


def coff_visualization(coef_, columns):
    # Coefficients bar plot
    coefficients = pd.Series(coef_, index=columns)
    coefficients.sort_values().plot(kind='bar', figsize=(10, 6), color='purple')
    plt.title("Feature Coefficients")
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.show()


def linear_model():
    dl = DataPrepare()
    x_train = np.hstack([np.ones((dl.X_train.shape[0], 1)), dl.X_train])
    y_train = dl.Y_train
    x_test = np.hstack([np.ones((dl.X_test.shape[0], 1)), dl.X_test])
    y_test = dl.Y_test
    optimized_w = np.linalg.pinv(x_train) @ y_train

    y_pred = x_test @ optimized_w

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2):", r2)

    line_plot(y_test, y_pred)
    predict_actual_plot(y_test, y_pred)
    residual_plot(y_test, y_pred)
    coff_visualization(optimized_w[1:], dl.columns)
