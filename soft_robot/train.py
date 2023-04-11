import argparse
import logging
import os
from config import cfg
from soft_robot.engine import Engine
from soft_robot.sw_engine import SwEngine

# $python train.py --config tensegrity_1.0.yaml
logging_kwargs = dict(
    level="INFO",
    format="%(asctime)s %(threadName)s %(levelname)s %(name)s - %(message)s",
    style='%',
)
logging.basicConfig(**logging_kwargs)
logger = logging.getLogger('DEnKF')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file path', required=True)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    args = parser.parse_args()
    config_file = args.config
    if config_file and os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    if bool(args.batch_size):
        cfg.train.batch_size = args.batch_size
    if bool(args.num_epochs):
        cfg.train.num_epochs = args.num_epochs
    return cfg, config_file


def main():
    cfg, config_file = parse_args()
    cfg.freeze()
    ####### check all the parameter settings #######
    logger.info("{}".format(cfg))
    logger.info("check mode - {}".format(cfg.mode.mode))
    # Create directory for logs and experiment name
    if not os.path.exists(cfg.train.log_directory):
        os.mkdir(cfg.train.log_directory)
    if not os.path.exists(os.path.join(cfg.train.log_directory, cfg.train.model_name)):
        os.mkdir(os.path.join(cfg.train.log_directory, cfg.train.model_name))
        os.mkdir(os.path.join(cfg.train.log_directory, cfg.train.model_name, 'summaries'))
    else:
        logger.warning('This logging directory already exists: {}. Over-writing current files'
                       .format(os.path.join(cfg.train.log_directory, cfg.train.model_name)))

    if "smartwatch" in cfg.train.model_name:
        # the smartwatch data set requires slightly adapted procedures
        train_engine = SwEngine(args=cfg, logger=logger)
    else:
        train_engine = Engine(args=cfg, logger=logger)

    ####### start the training #######
    if cfg.mode.mode == 'train':
        train_engine.train()
    if cfg.mode.mode == 'test':
        train_engine.online_test()


if __name__ == "__main__":
    main()
