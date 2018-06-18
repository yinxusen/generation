"""General utility functions"""
from __future__ import print_function

import json
import logging
import sys
from itertools import chain, imap

import tensorflow as tf

def flatmap(f, items):
    return list(chain.from_iterable(imap(f, items)))


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def transpose_batch_time(x):
    x_rank = tf.rank(x)
    x_t = tf.transpose(x,
                       perm=tf.concat(([1, 0], tf.range(2, x_rank)), axis=0))
    return x_t


def input_batch_size(x):
    shape = x.get_shape()
    batch_size = shape[1].value
    if batch_size is not None:
        return batch_size
    else:
        return tf.shape(x)[1]


def input_len_sentence(x):
    shape = x.get_shape()
    len_sentence = shape[2].value
    if len_sentence is not None:
        return len_sentence
    else:
        return tf.shape(x)[2]
