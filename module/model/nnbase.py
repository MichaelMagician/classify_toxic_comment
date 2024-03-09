import numpy as np

class NnBase(object):
    def __init__(self, config, logger, params=None) -> None:
        self.models = {}
        self.config = config
        self.logger = logger
        self.params = params
        self.model = self._build_model()

    def _build_model(self):
        pass

    def fit(self, train_x, train_y):        
        self.model.fit(train_x, train_y, epochs=self.config['epochs'], batch_size=self.config['batch_size'])

    def predict(self, train_x) -> np.ndarray:
        predictions = self.model.predict(train_x)
        return (predictions >= 0.5).astype(int)

    def predict_proba(self, train_x) -> np.ndarray:
        predictions = self.model.predict(train_x)
        return predictions
