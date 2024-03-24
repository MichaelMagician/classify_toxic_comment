import numpy as np
import pandas as pd
from module.calibrator import Calibrator

class Predictor(object):
    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.model = model        
        self.calibrators = [Calibrator(config, logger) for i in range(len(self.config['classes'])) ] if self.config['enable_calibration'] else None

    def predict(self,X) -> np.array:
        if self.calibrators is not None :
            return 1 if self.predict_proba(X) >= 0.5 else 0
        else:
            return self.model.predict(X)

    def predict_proba(self,X) -> np.array:
        prob = self.model.predict_proba(X)
        return prob if self.calibrators is None else self._calibrate(prob)        
    
    def _calibrate(self, prob):
        r = [ self.calibrators[i].calibrate(prob[:,i]) for i in range(len(self.calibrators))]
        r = np.stack(r, axis=1)
        return r
    
    def train_calibrator(self, validate_x, validate_y):
        pred_probs = self.model.predict_proba(validate_x)
        for i in range(len(self.calibrators)):
            category = self.config['classes'][i]
            self.calibrators[i].fit(validate_y[:,i], pred_probs[:,i] )
            self.calibrators[i].plot_reliability_diagrams(validate_y[:,i], pred_probs[:,i],category)

    def save_csv(self, test_ids, probs):
        columns = ['id'] + self.config['classes']
        output_path = self.config['output_path']
        try:
            r = np.concatenate((test_ids[:,np.newaxis],probs), axis=1)
            r = pd.DataFrame(r, columns =columns)
            r.to_csv(output_path, index=False)            
        except Exception as e:
            self.logger.exception(e)
