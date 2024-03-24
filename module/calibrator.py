import os
from numpy import ndarray
import numpy as np
from sklearn import calibration
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class Calibrator:
    def __init__(self, config, logger):   
        classes = config['classes']
        self.config = config
        # self.model = calibration.CalibratedClassifierCV(model, cv='prefit')
        if(config['calibrator_type'] == 'platt_scaling'):
            self.model = LogisticRegression()
        elif(config['calibrator_type'] == 'isotonic'):
            self.model = IsotonicRegression()

    def fit(self, true_label, proba_predict):
        try:
            probs = np.expand_dims(proba_predict, axis=1)
            self.model.fit(probs, true_label)   
        # calculate ece before and after  
        except Exception as e:
            print("Error " , e)
    
    def calculate_ece(self, validate_x, validate_y):
        pass

    def calibrate(self, prob: ndarray) -> ndarray:         
        if(self.config['calibrator_type'] == 'platt_scaling'):            
            return self.model.predict_proba(prob.reshape(-1,1))[:,1]
        elif(self.config['calibrator_type'] == 'isotonic'):            
            return self.model.predict(prob)
        else:
            return prob

    def plot_reliability_diagrams(self, y_valid, proba_valid, category):
        prob_true, prob_pred =calibration.calibration_curve(y_valid,proba_valid, n_bins=10, strategy='uniform')        # plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Perfect calibration')                
        disp = calibration.CalibrationDisplay(prob_true, prob_pred, proba_valid)
        disp.plot()
        save_path = os.path.join(self.config['calibrators_output_path'], f"{category}.png")
        plt.savefig(fname=save_path, format="png")
