import numpy as np
import pandas as pd

class Predictor(object):
    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.model = model

    def predict(self,X) -> np.array:
        return self.model.predict(X)

    def predict_proba(self,X) -> np.array:
        return self.model.predict_proba(X)
    
    def save_csv(self, test_ids, probs):
        columns = ['id'] + self.config['classes']
        output_path = self.config['output_path']
        try:
            r = np.concatenate((test_ids[:,np.newaxis],probs), axis=1)
            r = pd.DataFrame(r, columns =columns)
            r.to_csv(output_path, index=False)            
        except Exception as e:
            self.logger.exception(e)
