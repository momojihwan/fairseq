# -*- coding: utf-8 -*-

"""Network related utility tools."""

import logging
from typing import Dict

import numpy as np
import torch


def get_subsample(train_args, mode, arch):
    """Parse the subsampling factors from the args for the specified `mode` and `arch`.
    Args:
        train_args: argument Namespace containing options.
        mode: one of ('asr', 'mt', 'st')
        arch: one of ('rnn', 'rnn-t', 'rnn_mix', 'rnn_mulenc', 'transformer')
    Returns:
        np.ndarray / List[np.ndarray]: subsampling factors.
    """
    if arch == "transformer":
        return np.array([1])

    elif mode == "mt" and arch == "rnn":
        # +1 means input (+1) and layers outputs (train_args.elayer)
        subsample = np.ones(train_args.encoder_layers + 1, dtype=np.int)
        logging.warning("Subsampling is not performed for machine translation.")
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif (
        (mode == "asr" and arch in ("rnn", "rnn-t"))
        or (mode == "mt" and arch == "rnn")
        or (mode == "st" and arch == "rnn")
    ):
        subsample = np.ones(train_args.encoder_layers + 1, dtype=np.int)
        print("sub 1 : ", subsample)
        ss = train_args.subsample.split("_")
        for j in range(min(train_args.encoder_layers + 1, len(ss))):
            subsample[j] = int(ss[j])
        print("sub 2 : ", subsample)
        exit()
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    else:
        raise ValueError("Invalid options: mode={}, arch={}".format(mode, arch))
