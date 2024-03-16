
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from module.model.nnbase import NnBase
from keras.models import Model
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

class SmallBertClassifier(NnBase):   

    def _build_model(self):        
        tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
        tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'             

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        #net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net)
        net = tf.keras.layers.Dense(len(self.config['classes']), activation=None)(net)
        net = tf.keras.layers.Dense(len(self.config['classes']), activation='sigmoid', name='classifier')(net)
        model = tf.keras.Model(text_input, net)
        optimizer = self._create_optimizer(self.params['train_ds'] )
        model.compile(optimizer=optimizer,
                                  loss='binary_crossentropy')
        
        model.summary()        
        return model

    def _create_optimizer(self, train_ds):
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = steps_per_epoch * self.config['epochs']
        num_warmup_steps = int(0.1*num_train_steps)
        optimizer = optimization.create_optimizer(init_lr=self.config['learning_rate'],
                                                  num_train_steps=num_train_steps, 
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')
        return optimizer
    
    def fit_with_tf_ds(self, train_ds,validate_ds):
        self.model.fit(x=train_ds, validation_data=validate_ds,epochs=self.config['epochs'],batch_size=1)
        return self.model
