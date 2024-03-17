from sklearn import calibration
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class Calibrator:
    def __init__(self, config, logger):   
        classes = config['classes']
        self.config = config
        if(config['calibrator_type'] == 'platt_scaling'):
            self.model = [LogisticRegression() for _ in range(len(classes)) ]
        elif(config['calibrator_type'] == 'isotonic'):
            self.model = [IsotonicRegression() for _ in range(len(classes)) ]

    def fit(self, y_valid, proba_valid):
        self.model.fit(y_valid, proba_valid)     
    
    def calculate_ece(self, validate_x, validate_y):
        pass

    def calibrate(self, prob):
        return prob

    def draw_reliability_chart(self, y_valid, proba_valid):
        y_means, proba_means =calibration.calibration_curve(y_valid,proba_valid, normalize=False, n_bins=5, strategy='uniform')
        plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Perfect calibration')
        plt.plot(proba_means, y_means)
        plt.show()
