import numpy as np
from keras import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Embedding, Dropout, Input
from keras.optimizers import Adam
from module.model.nnbase import NnBase
import tensorflow as tf

class TextCnn(NnBase):    
    def _build_model(self):
        model = Sequential()
        maxlen = self.config['maxlen']
        model = Sequential()
        inputs = Input(shape=(maxlen,))
        model.add(Embedding(self.params['vocab_size'], self.config['embedding_dim']))
        model.add(Conv1D(128, 7,activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(256, 7,activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(512, 7,activation='relu', padding='same'))
        model.add(MaxPooling1D(padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(len(self.config['classes'])))
        model.add(Dense(len(self.config['classes']), activation='sigmoid'))

        model(inputs=inputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        
        return model