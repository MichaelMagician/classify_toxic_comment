import numpy as np
from keras import Sequential
from keras import Model
from keras.layers import Dense, MaxPooling1D, Flatten, Embedding, Dropout, Layer, MultiHeadAttention,LayerNormalization, Input, GlobalAveragePooling1D
from keras.optimizers import Adam
from module.model.nnbase import NnBase
import tensorflow as tf
import keras.ops as ops

class TransformerClassifier(NnBase):    
    def _build_model(self):
        embed_dim  = self.config['embedding_dim']
        num_heads = self.config['num_heads']
        ff_dim = self.config['ff_dim']
        maxlen = self.config['maxlen']
        vocab_size = self.config['vocab_size']
        learning_rate = self.config['learning_rate']        
        drop_out_rate = self.config['dropout_rate']

        inputs = Input(shape=(maxlen,))
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        x = embedding_layer(input)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, learning_rate)
        x = transformer_block(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(drop_out_rate=drop_out_rate)(x)
        x = Dense(20, activation='relu')(x)
        x = Dropout(drop_out_rate=drop_out_rate)(x)
        outputs = Dense(len(self.config['classes']), activation='sigmoid')(x)
      
        model = Model(inputs=inputs, outputs=outputs)      
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        
        return model
    

class TransformerBlock(Layer):        
    def __init__(self,embed_dim, num_heads, ff_dim, rate=0.1):
        self.att = MultiHeadAttention(num_heads, key_dim=ff_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation='relu'), Dense(embed_dim)]
        )
        # self.ffn.add(Dense(ff_dim, activation='relu'))
        # self.ffn.add(Dense(embed_dim))
        self.layernorm1=LayerNormalization(epsilon=1e-6)
        self.layernorm2=LayerNormalization(epsilon=1e-6)
        self.dropout1=Dropout(rate)
        self.dropout2=Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        vocab_size=self.params['vocab_size']
        embed_dim=self.config['embedding_dim']
        embedding_layer = None

        if self.params.get('embedding_matrix', None) is not None:
            o_dim = self.params['embedding_matrix'].shape[1]
            embedding_layer = Embedding(input_dim=self.params['vocab_size'], output_dim=o_dim, trainable=False)
            embedding_layer.build((1,))
            embedding_layer.set_weights([self.params['embedding_matrix']])            
        else:
            embedding_layer = Embedding(vocab_size, embed_dim)
        
        self.token_emb = embedding_layer
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        return x + positions