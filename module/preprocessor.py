from sklearn.model_selection import train_test_split
import pandas as pd
 
class preprocessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def process(self):
        # split x and y for train data
        # tokenrize train/test data
        # train_test split train_data
        # vectorize train_x, validate_x and test_x
        return train_x, train_y, validate_x, validate_y, test_x
    
    def _parse(self, df: DataFrame, is_test=False):
        '''
        split x and y
        returns:
            tokenrized_input(np.array)  #[I, love, nyc]
            n_hot_labels (np.array) #[1,0,0,0,1,0]
            or
            test_id for test dataset
        '''
    
    # load data
    def load_data(self):
        # get x and lables
        pass
    # tokenrization
    
    
    # vectorization 
    def count_vectorization(self, train_x, validate_x, test_x):
        pass