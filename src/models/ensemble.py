from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        # Majority voting
        final_predictions = np.round(np.mean(predictions, axis=0))
        return final_predictions

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy

    def save_models(self, file_paths):
        for model, file_path in zip(self.models, file_paths):
            joblib.dump(model, file_path)

    def load_models(self, file_paths):
        self.models = [joblib.load(file_path) for file_path in file_paths]