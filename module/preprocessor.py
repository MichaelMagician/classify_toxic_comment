import string
from typing import Iterable
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import tensorflow as tf
import keras 

class Preprocessor:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.additional_data = {}
    
    def _get_train_data(self):
        return pd.read_csv(self.config['input_trainset'])
    
    def _get_test_data(self):
        return pd.read_csv(self.config['input_testset'])

    def process(self) ->tuple[np.ndarray, np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        '''
        return train_x, train_y, validate_x, validate_y, test_X, ids
        '''
        test_data = self._get_test_data()
        train_data = self._get_train_data()

        # split x and y from train data; tokenize X
        X, y = self._parse(train_data, False)
        test_X, ids = self._parse(test_data, True)

        train_x, validate_x, train_y, validate_y = train_test_split(X, y, test_size=self.config['split_ratio'], random_state=self.config['random_seed'])

        # vectorize
        input_convertor = self.config['input_convertor']
        if(input_convertor == 'count_vectorization'):
            train_x, validate_x, test_X = self.count_vectorization(train_x, validate_x, test_X)         
        if(input_convertor == 'nn_vectorization'):
            train_x, validate_x, test_X = self.nn_vectorization(train_x, validate_x, test_X)         
        else:
            raise Exception('not supported convertor')    
        return train_x, train_y, validate_x, validate_y, test_X, ids
    
    def _parse(self, df: DataFrame, is_test=False) -> tuple[np.ndarray, np.ndarray]:
        '''
        split x and y. In addition, tokenize X
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
        vector = CountVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x, stop_words='english')
        train_x_vectorized = vector.fit_transform(train_x)
        validate_x_vectorrized = vector.transform(validate_x)
        test_x_vectorized = vector.transform(test_x)
        return train_x_vectorized, validate_x_vectorrized, test_x_vectorized

    def tfidf_vectorization(self, train_x, validate_x, test_x):
        vector = TfidfVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x, stop_words='english')
        train_x_vectorized = vector.fit_transform(train_x)
        validate_x_vectorrized = vector.transform(validate_x)
        test_x_vectorized = vector.transform(test_x)
        return train_x_vectorized, validate_x_vectorrized, test_x_vectorized

    def nn_vectorization(self, train_x, validate_x, test_x): 
        '''
        turn words into id vectors
        '''
        #sentence to word ids
        self.word2id ={}
        self.id2word ={}
        special_tokens = ['<pad>','<unk>']

        for token in special_tokens:
            Preprocessor.addchar(self.word2id, self.id2word, token)
        
        for sentence in train_x:
            for word in sentence:
                Preprocessor.addchar(self.word2id, self.id2word, word)
            
        # word ids
        train_x_vectorized = self.cnn_get_word_id_list(train_x, self.word2id)
        validate_x_vectorrized = self.cnn_get_word_id_list(validate_x, self.word2id)
        test_x_vectorized = self.cnn_get_word_id_list(test_x, self.word2id)

        self.additional_data['vocab_size'] = len(self.word2id.keys())
        return train_x_vectorized, validate_x_vectorrized, test_x_vectorized
    
    def cnn_get_word_id_list(self, x, word2id):
        r = []
        for sentence in x:
            ids = [word2id.get(w, word2id['<unk>']) for w in sentence]
            r.append(ids)
        ids = keras.utils.pad_sequences(r, maxlen=self.config['maxlen'], padding='post', value=word2id['<pad>'])
        return ids

    @staticmethod
    def addchar(word2id, id2word, word):
        id = len(word2id)
        word2id[word] = id
        id2word[id] = word
