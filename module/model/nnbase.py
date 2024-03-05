class NnBase(object):
    def __init__(self, classes, logger, params={}) -> None:
        self.models = {}
        self.classes = classes
        self.logger = logger
        self.params = params
        self.model = self._build_model()

    def _build_model(self):
        pass

    def fit(self, train_x, train_y):        
        self.models.fit(train_x, train_y, epochs=self.config['epochs'], verbose=True,batch_size=self.config['batch_size'])

    def predict(self, train_x) -> np.ndarray:
        predictions = self.model.predict(train_x)
        return predictions >= 0.5       

    def predict_proba(self, train_x) -> np.ndarray:
        predictions = self.model.predict(train_x)
        return predictions
