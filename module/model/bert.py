
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Embedding, Dropout, LSTM, Bidirectional, GRU,Input
from keras.optimizers import Adam
from module.model.nnbase import NnBase
from keras.models import Model
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization


class BertClassifier(NnBase):    
    def _build_model(self, train_ds):
        maxlen = self.config['maxlen']   
        tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'     
        tfhub_handle_encoder = ''

        inputs = Input(shape=(maxlen,),dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(inputs)
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='bert_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = Dropout(self.config['dropout_rate'])(net)        
        outputs = Dense(len(self.config['classes']), activation='sigmoid')(net)

        model = Model(inputs=inputs, outputs=outputs)   

        optimizer = self._create_optimizer(train_ds)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        
        return model

    def _create_optimizer(self, train_ds):
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = steps_per_epoch * self.config['epochs']
        num_warmup_steps = int(0.1*num_train_steps)
        optimizer = optimization.create_optimizer(init_lr=self.config['learning_rate'],num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps,optimizer_type='adaw' )
        return optimizer