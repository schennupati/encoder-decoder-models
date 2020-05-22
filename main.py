"""Main script."""
import argparse
import logging
import os

import numpy as np
import torch

from utils.constants import TRAIN, VAL
from utils.data_utils import get_cfg
from utils.experiment_utils import ExperimentLoop


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
                        type=str, default="configs/panoptic_seg_contour_binary.yml",
                        help="Configuration to use")
    parser.add_argument("--mode", nargs="?", choices=['train', 'val'],
                        type=str, default='val',
                        help="train/val mode")
    parser.add_argument("--debug", choices=['True', 'False'], type=str,
                        default=False, help='Debug Mode toggle on/off.')
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                            level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                            level=logging.INFO)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg = get_cfg(dir_path, args.config)
    mode = args.mode
    main(cfg, mode)
