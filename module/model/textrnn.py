import numpy as np
from keras import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Embedding, Dropout, SimpleRNN, Input
from keras.optimizers import Adam
from module.model.nnbase import NnBase
import tensorflow as tf

class TextRnn(NnBase):    
    def _build_model(self):
        maxlen = self.config['maxlen']
        model = Sequential()
        inputs = Input(shape=(maxlen,))
        model.add(Embedding(self.params['vocab_size'], self.config['embedding_dim'], trainable=True, embeddings_initializer='uniform'))
        model.add(SimpleRNN(128))
        model.add(Dense(len(self.config['classes']), activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        
        return model