import logging
import yaml
import argparse
from module import preprocessor, predictor

if __name__ == 'main':
    # set up logging
    parser = argparse.ArgumentParser(description='parse arguments')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger()

    config = None
    with open(args.config, 'r') as config_file:
        try:
            config = yaml.load(config_file)            
        except Exception as e:
            logger.error(e)
                
    preprocessor = preprocessor.Preprocessor(config['preprocessing'], logger)
    train_x, train_y, validate_x, validate_y, test_X = preprocessor.process()