from module.model.grucnn import TextGruCnn
from  module.model.naive_bayse import NaiveBayes
from module.model.small_bert import SmallBertClassifier
from  module.model.textcnn import TextCnn
from  module.model.textrnn import TextRnn
from  module.model.textbilstm import TextBiLSTM
from sklearn.metrics import accuracy_score
from module.model.transformer import TransformerClassifier

class Trainer:
    def __init__(self,config, logger, params=None):
        self.config = config
        self.logger = logger                
        self.params = params        
        self.select_model()
        
    def select_model(self):
        if self.config['model_name'] == 'naivebayse':
            self.model = NaiveBayes(self.config['classes'], self.logger)
        elif self.config['model_name'] == 'textcnn':
            self.model = TextCnn(self.config, self.logger ,self.params)
        elif self.config['model_name'] == 'textrnn':
            self.model = TextRnn(self.config, self.logger ,self.params)
        elif self.config['model_name'] == 'textBiLSTM':
            self.model = TextBiLSTM(self.config, self.logger,self.params)
        elif self.config['model_name'] == 'transformer':
            self.model = TransformerClassifier(self.config, self.logger,self.params)
        elif self.config['model_name'] == 'textgrucnn':
            self.model = TextGruCnn(self.config, self.logger,self.params)            
        elif self.config['model_name'] == 'smallbert':
            self.model = SmallBertClassifier(self.config, self.logger,self.params)            
        else:
            raise Exception("model not supported")

    def fit(self, X, y):
        
        self.model.fit(X, y)
        return self.model

    def fit_with_tf_ds(self, train_ds, validate_ds):
        return self.model.fit_with_tf_ds(train_ds, validate_ds)        

    def validate(self, validate_x, validate_y):
        pred_y = self.model.predict(validate_x)
        return self.get_metrics(pred_y, validate_y)

    def get_metrics(self, pred_y, y):
        return accuracy_score(pred_y, y)
