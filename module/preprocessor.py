from sklearn.model_selection import train_test_split
import pandas as pd

class Preprocessor:
    def __init__(self):
        pass

    def _get_train_data(self):
        return pd.read_csv('../data/train.csv')
    
    def _get_test_data(self):
        return pd.read_csv('../data/test.csv')

    def process(self):
        test_data = self._get_test_data()
        train_data = self._get_train_data()

        
        

    