import numpy as np
from keras import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Embedding, Dropout, LSTM, Bidirectional
from keras.optimizers import Adam
from module.model.nnbase import NnBase
import tensorflow as tf

class TextBiLSTM(NnBase):    
    def _build_model(self):
        model = Sequential()

        if self.params.get('embedding_matrix', None) is not None:
            o_dim = self.params['embedding_matrix'].shape[1]
            embedding_layer = Embedding(input_dim=self.params['vocab_size'], output_dim=o_dim, trainable=False)
            embedding_layer.build((1,))
            embedding_layer.set_weights([self.params['embedding_matrix']])
            model.add(embedding_layer)
        else:
            model.add(Embedding(self.params['vocab_size'], self.config['embedding_dim'], trainable=True, embeddings_initializer='uniform'))
        
        model.add(Bidirectional(LSTM(80)))
        model.add(Dense(len(self.config['classes']), activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        
        return model