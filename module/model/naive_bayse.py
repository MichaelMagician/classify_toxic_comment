import numpy as np
from sklearn.naive_bayes import MultinomialNB

class NaiveBayes(object):
    def __init__(self, classes, logger) -> None:
        self.models = {}
        self.classes = classes
        self.logger = logger
        for cls in classes:
            self.models[cls] = MultinomialNB()

    def fit(self, train_x, train_y):
        for index, cls in enumerate(self.classes):
            self.models[cls].fit(train_x, train_y[:,index])

    def predict(self, train_x) -> np.ndarray:
        predictions = np.zeros((train_x.shape[0], len(self.classes)))
        for index, cls in enumerate(self.classes):
            predictions[:, index] = self.models[cls].predict(train_x)
        return predictions

    def predict_proba(self, train_x) -> np.ndarray:
        predictions = np.zeros((train_x.shape[0], len(self.classes)))
        for index, cls in enumerate(self.classes):
            predictions[:, index] = self.models[cls].predict_proba(train_x)
        return predictions
