from  module.model.naive_bayse import NaiveBayes
from  module.model.textcnn import TextCnn
from  module.model.textrnn import TextRnn
from  module.model.textbilstm import TextBiLSTM
from sklearn.metrics import accuracy_score

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

    def fit(self, X, y):
        self.model.fit(X, y)
        return self.model

    def validate(self, validate_x, validate_y):
        pred_y = self.model.predict(validate_x)
        return self.get_metrics(pred_y, validate_y)

    def get_metrics(self, pred_y, y):
        return accuracy_score(pred_y, y)
