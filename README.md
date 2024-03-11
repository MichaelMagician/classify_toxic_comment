# classify_toxic_comment

### data source: 
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


### Config generation
```
himl hiera/model=naivebayse/ --output-file config/nb_config.yaml
himl hiera/model=textcnn/ --output-file config/cnn_config.yaml
himl hiera/model=textrnn/ --output-file config/rnn_config.yaml
himl hiera/model=textBiLSTM/ --output-file config/lstm_config.yaml
himl hiera/model=transformer/ --output-file config/transformer_config.yaml
```

### run
python run.py --config config/nb_config.yaml
python run.py --config config/cnn_config.yaml
python run.py --config config/rnn_config.yaml
python run.py --config config/lstm_config.yaml
python run.py --config config/transformer_config.yaml

### scores
naive bayse: 0.68
cnn: 0.62
rnn:0.50
lstm:0.60
transformer:0.61

