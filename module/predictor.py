import pandas as pd

class Predictor(object):
    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.model = model

    def predict(self,X):
        return self.model.predict(X)

    def predict_proba(self,X):
        return self.model.predict_proba(X)
    
    def save_csv(self, test_ids, probs):
        columns = ['id'] + self.config['classes']
        try:
            r = pd.concat([test_ids, probs], axis=1)
            r.columns = columns
            r.to_csv('.\data\output\result.csv', index=False)
        except Exception as e:
            self.logger.exception(e)
