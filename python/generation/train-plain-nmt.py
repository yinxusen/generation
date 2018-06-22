import argparse
import logging
import os

import tensorflow as tf

from generation.model.utils import Params, set_logger
from generation.model.training import train_and_evaluate
from generation.model.plain_nmt_input_fn import dialogue_input_fn
from generation.model.plain_nmt_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('-d', '--data_dir', default='data/small',
                    help="Directory containing the dataset")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    params = Params(json_path)

    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    params.update(json_path)
    num_oov_buckets = params.num_oov_buckets

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'training.log'))

    # Get paths for vocabularies and dataset
    path_words = os.path.join(args.data_dir, 'words.txt')
    path_tags = os.path.join(args.data_dir, 'tags.txt')
    path_train_sentences = os.path.join(args.data_dir, 'train/sentences.txt')
    path_train_labels = os.path.join(args.data_dir, 'train/labels.txt')
    path_eval_sentences = os.path.join(args.data_dir, 'dev/sentences.txt')
    path_eval_labels = os.path.join(args.data_dir, 'dev/labels.txt')

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    # Specify other parameters for the dataset and the model
    params.eval_size = params.dev_size
    params.buffer_size = params.batch_size * 10

    # Create the two iterators over the two datasets
    train_inputs = dialogue_input_fn(
        'train', path_train_sentences, path_train_labels, path_words, path_tags, params)
    eval_inputs = dialogue_input_fn('eval', path_eval_sentences, path_eval_labels, path_words, path_tags, params)
    logging.info("done.")

    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)
    logging.info("done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(
        train_model_spec, eval_model_spec, args.model_dir, params)
