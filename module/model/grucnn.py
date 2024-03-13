import numpy as np
from keras import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Embedding, Dropout, LSTM, Bidirectional, GRU,Input
from keras.optimizers import Adam
from module.model.nnbase import NnBase
from keras.models import Model
import tensorflow as tf

class TextGruCnn(NnBase):    
    def _build_model(self):
        maxlen = self.config['maxlen']        
        inputs = Input(shape=(maxlen,))

        if self.params.get('embedding_matrix', None) is not None:            
            o_dim = self.params['embedding_matrix'].shape[1]            
            embedding_layer = Embedding(input_dim=self.params['vocab_size'], output_dim=o_dim, trainable=False)
            embedding_layer.build((1,))
            embedding_layer.set_weights([self.params['embedding_matrix']])
            x = embedding_layer(inputs)
        else:            
            x = Embedding(self.params['vocab_size'], self.config['embedding_dim'], trainable=True, embeddings_initializer='uniform')(inputs, output)

        x = (Bidirectional(GRU(128, return_sequences=True, dropout=self.config['dropout_rate'], recurrent_dropout=0.1)))(x)
        x = Conv1D(64, 7,activation='relu', padding='same')(x)
        x = MaxPooling1D()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(len(self.config['classes']), activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)   
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        
        return model