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
    pre_processor = preprocessor.Preprocessor(config['preprocessing'], logger)
    
    model = None
    ids = None
    try:        
        if config['training']['model_name'] =='smallbert':
            train_ds, val_ds, validate_x, validate_y, test_X, ids = pre_processor.process()
            additional_data = pre_processor.additional_data
            additional_data['train_ds'] = train_ds
            trainer = trainer.Trainer(config['training'], logger, additional_data)
            model = trainer.fit_with_tf_ds(train_ds, val_ds)
        else:        
            train_x, train_y, validate_x, validate_y, test_X, ids = pre_processor.process()            
            trainer = trainer.Trainer(config['training'], logger, pre_processor.additional_data)
            model = trainer.fit(train_x, train_y)
        
        metrics = trainer.validate(validate_x, validate_y)
        print( f'metrics: {metrics}'  )

        #predict
        predictor = predictor.Predictor(config['predict'], logger, model)
        if config['predict']['enable_calibration']:
            predictor.train_calibrator(validate_x, validate_y)
        probs = predictor.predict(test_X)
        predictors = predictor.save_csv(ids, probs)
    except Exception as e:
        print(f'{e}')
