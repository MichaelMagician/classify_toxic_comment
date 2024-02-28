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

    with open(args.config, 'r') as config_file:
        try:
            config = yaml.load(config_file)
            preprocessor = preprocessor.Preprocessor()
        except Exception as e:
            logger.error(e)