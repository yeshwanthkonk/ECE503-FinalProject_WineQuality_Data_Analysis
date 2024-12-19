from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from data_preprocessing import DataPrepare
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
import numpy as np


def confusion_mat(y_test, y_pred):
    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Bad Quality', 'Good Quality']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_curve(y_test, probs):
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def logistic_regression():
    dl = DataPrepare()
    x_train = dl.X_train
    y_train = (dl.Y_train >= 7).astype(int)

    x_test = dl.X_test
    y_test = (dl.Y_test >= 7).astype(int)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(dl.X_test)

    probs = model.predict_proba(x_test)[:, 1]

    # Evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    confusion_mat(y_test, y_pred)
    plot_roc_curve(y_test, probs)
