"""Main script."""
import argparse

from utils.data_utils import get_cfg
from utils.experiment_utils import ExperimentLoop
from utils.constants import TRAIN, VAL


def main(cfg, mode):
    """Run an experiment in train/val mode.

    Arguments:
        cfg {[dict]} -- [configuration settings for the experiment]
        mode {[str]} -- [train/val]
    """
    experiment = ExperimentLoop(cfg, mode)
    if mode == TRAIN:
        experiment.train()
    elif mode == VAL:
        experiment.validation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", nargs="?",
                        type=str, default="configs/seg_instance.yml",
                        help="Configuration to use")
    parser.add_argument("--mode", nargs="?", choices=['train', 'val'],
                        type=str, default='train',
                        help="train/val mode")
    args = parser.parse_args()
    cfg = get_cfg(args.config)
    mode = args.mode
    main(cfg, mode)
