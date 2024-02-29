import string
from typing import Iterable
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
import nltk
from sklearn.feature_extraction.text import CountVectorizer

class Preprocessor:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def _get_train_data(self):
        return pd.read_csv(self.config['input_trainset'])
    
    def _get_test_data(self):
        return pd.read_csv(self.config['input_testset'])

    def process(self):
        test_data = self._get_test_data()
        train_data = self._get_train_data()

        # split x and y from train data; tokenize X
        X, y = self._parse(train_data, False)
        test_X, id = self._parse(test_data, True)

        train_x, validate_x, train_y, validate_y = train_test_split(X, y, test_size=self.config['split_ratio'], random_state=self.config['random_state'])

        # vectorize
        input_convertor = self.config['input_convertor']
        if(input_convertor == 'count_vectorization'):
            train_x, validate_x, test_X = self.count_vectorization(train_x, validate_x, test_X)         
        else:
            raise Exception('not supported convertor')    
        return train_x, train_y, validate_x, validate_y, test_X
    
    def _parse(self, df: DataFrame, is_test=False):
        '''
        split x and y
        returns:
            tokenrized_input(np.array)  #[I, love, nyc]
            n_hot_labels (np.array) #[1,0,0,0,1,0]
            or
            test_id for test dataset
        '''
        data_x = df[self.config['input_text_column']].fillna('unknown')
        if self.config['skip_tokenization']:
            X = df[self.config['input_text_column']].values
        else:
            X = data_x.apply(Preprocessor._tokenize).values
            
        if is_test:
            y = df[self.config['input_id_column']].values
        else:
            y = df.drop([self.config['input_text_column'],self.config['input_id_column']], axis=1).values

        return X,y
    
    @staticmethod
    def _tokenize(s : str):
        s = s.strip().lower()
        translator = str.maketrans('','', string.punctuation) 
        words = nltk.word_tokenize(s)

        tokenized_list = [w.translate(translator) for w in words if len(w.translate(translator)) > 0]    
        return tokenized_list  
    
    
    # vectorization 
    def count_vectorization(self, train_x, validate_x, test_x):
        vector = CountVectorizer()
        train_x_vectorized = vector.fit_transform(train_x)
        validate_x_vectorrized = vector.transform(validate_x)
        test_x_vectorized = vector.transform(test_x)
        return train_x_vectorized, validate_x_vectorrized, test_x_vectorized

