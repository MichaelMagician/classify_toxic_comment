from  module.model.naive_bayse import NaiveBayes
from sklearn.metrics import accuracy_score

class Trainer:
    def __init__(self,config, logger):
        self.config = config
        self.logger = logger
        self.classes = config['classes']
        self.select_model()

    def select_model(self):
        if self.config['model_name'] == 'naivebayse':
            self.model = NaiveBayes(self.classes, self.logger)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self.model

    def validate(self, validate_x, validate_y):
        pred_y = self.model.predict(validate_x)
        return self.get_metrics(pred_y, validate_y)

    def get_metrics(self, pred_y, y):
        return accuracy_score(pred_y, y)
