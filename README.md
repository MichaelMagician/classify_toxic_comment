# classify_toxic_comment

### data source: 
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


### Config generation
```
himl hiera/model=naivebayse/ --output-file config/nb_config.yaml
himl hiera/model=textcnn/ --output-file config/cnn_config.yaml
```

### run
python run.py --config config/config.yaml

### scores
naive bayse: 0.68

