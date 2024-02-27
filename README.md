# classify_toxic_comment

### data source: 
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


###Config generation
`himl hiera/naivebayse --output-file config/config.yaml`

### run
python run.py --config config/config.yaml