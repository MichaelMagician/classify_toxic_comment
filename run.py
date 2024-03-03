import logging
import yaml
import argparse
from module import preprocessor, predictor, trainer

def parse_args_and_set_up_logger():
    parser = argparse.ArgumentParser(description='parse arguments')
    parser.add_argument('--config', type=str, required=True)    
    args = parser.parse_args()

    logging.basicConfig()
    logger = logging.getLogger()

    config = None
    with open(args.config, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)            
        except Exception as e:
            logger.error(e)
    return logger,config

if __name__ == '__main__':
    
    logger, config = parse_args_and_set_up_logger()
                
    preprocessor = preprocessor.Preprocessor(config['preprocessing'], logger)
    train_x, train_y, validate_x, validate_y, test_X, ids = preprocessor.process()
    trainer = trainer.Trainer(config['training'], logger)
    model = trainer.fit(train_x, train_y)
    metrics = trainer.validate(validate_x, validate_y)
    print( f'metrics: {metrics}'  )

    predictor = predictor.Predictor(config['predict'], logger, model)
    probs = predictor.predict(test_X)
    predictors = predictor.save_csv(ids, probs)

