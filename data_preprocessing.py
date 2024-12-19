import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ucimlrepo import fetch_ucirepo


class DataPrepare:

    def __init__(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = [None] * 4
        self.X_train_class, self.X_test_class, self.Y_train_class, self.Y_test_class = [None] * 4
        self.columns = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol']
        self.load()

    def load(self):
        # fetch dataset
        wine_quality = fetch_ucirepo(id=186)
        self.preprocessing(wine_quality.data.features, wine_quality.data.targets["quality"])

    def preprocessing(self, features, true_pred):
        # Standardize Features
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(features)

        # Classification: Binarize quality into 3 classes: Low, Medium, High
        true_class = pd.cut(true_pred, bins=[0, 5, 7, 10], labels=["Low", "Medium", "High"])

        # Train-Test Split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(x_scaled, true_pred, test_size=0.2, random_state=42)
        self.X_train_class, self.X_test_class, self.Y_train_class, self.Y_test_class = train_test_split(x_scaled, true_class, test_size=0.2, random_state=42)
