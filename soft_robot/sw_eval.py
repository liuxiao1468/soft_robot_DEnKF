import argparse
import logging
from pathlib import Path

from matplotlib import pyplot as plt

from config import cfg
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
    args = parser.parse_args()
    config_file = Path(args.config)
    if not config_file.exists():
        raise UserWarning("file {} does not exist".format(args.config))
    cfg.merge_from_file(config_file)
    cfg.freeze()


if __name__ == "__main__":
    parse_args()

    checkpoint_path = Path(cfg.train.log_directory) / cfg.train.model_name / "checkpoint-980000"

    # the smartwatch data set requires a different engine
    train_engine = SwEngine(args=cfg, logger=logger)
    gts, preds = train_engine.eval_model(checkpoint_path=checkpoint_path)

    fig, ax = plt.subplots(nrows=3, ncols=1)

    ax[0].plot(gts[:, 0])
    ax[0].plot(preds[:, 0])
    ax[1].plot(gts[:, 1])
    ax[1].plot(preds[:, 1])
    ax[2].plot(gts[:, 2])
    ax[2].plot(preds[:, 2])

    plt.tight_layout()
    plt.show()
