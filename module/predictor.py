import numpy as np
import pandas as pd

from module.calibrator import Calibrator

class Predictor(object):
    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.model = model        
        self.calibrator = Calibrator(config, logger) if self.config['enable_calibration'] else None

    def predict(self,X) -> np.array:
        if self.calibrator is not None :
            return self.predict_proba(X) >= 0.5
        else:
            return self.model.predict(X)

    def predict_proba(self,X) -> np.array:
        prob = self.model.predict_proba(X)
        return prob if self.calibrator is None else self.calibrator.calibrate(prob)        
    
    def train_validator(self, validate_x, validate_y):
        self.calibrators.fit(validate_y, self.model.predict_proba(validate_x))

    def save_csv(self, test_ids, probs):
        columns = ['id'] + self.config['classes']
        output_path = self.config['output_path']
        try:
            r = np.concatenate((test_ids[:,np.newaxis],probs), axis=1)
            r = pd.DataFrame(r, columns =columns)
            r.to_csv(output_path, index=False)            
        except Exception as e:
            self.logger.exception(e)
