import argparse
import logging
import os

import tensorflow as tf

from generation.model.utils import Params, set_logger, steps_per_epoch
from generation.model.input_fn import load_dialogue_from_text, dialogue_input_fn
from generation.model.nmt_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/small',
                    help="Directory containing the dataset")

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert(os.path.isfile(json_path),
           "No json configuration file found at {}".format(json_path))
    params = Params(json_path)

    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert(os.path.isfile(json_path),
           "No json file found at {}, run build.py".format(json_path))
    params.update(json_path)
    num_oov_buckets = params.num_oov_buckets

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'inferring.log'))

    # Get paths for vocabularies and dataset
    path_words = os.path.join(args.data_dir, 'words.txt')
    path_tags = os.path.join(args.data_dir, 'tags.txt')
    path_eval_sentences = os.path.join(args.data_dir, 'test/sentences.txt')
    path_eval_labels = os.path.join(args.data_dir, 'test/labels.txt')

    # Load Vocabularies
    words = tf.contrib.lookup.index_table_from_file(
        path_words,
        num_oov_buckets=num_oov_buckets)
    tags = tf.contrib.lookup.index_table_from_file(path_tags)

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_sentences = load_dialogue_from_text(path_eval_sentences, words)
    test_labels = load_dialogue_from_text(path_eval_labels, tags)

    # Specify other parameters for the dataset and the model
    params.infer_size = params.test_size
    params.id_pad_word = words.lookup(tf.constant(params.pad_word))
    params.id_pad_tag = tags.lookup(tf.constant(params.pad_tag))

    # Create iterator over the test set
    inputs = dialogue_input_fn('infer', test_sentences, test_labels, params)
    logging.info("done.")

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('infer', inputs, params, reuse=False)
    logging.info("done.")

    idx2tags = tf.contrib.lookup.index_to_string_table_from_file(
        vocabulary_file=path_tags,
        default_value='UNK',
    )

    logging.info("Starting inference")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(model_spec['variable_init_op'])
        sess.run(model_spec['iterator_init_op'])
        sess.run(tf.tables_initializer())

        # Reload weights from the weights subdirectory
        save_path = '{}/best_weights'.format(args.model_dir)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        num_steps = steps_per_epoch(params.infer_size, params.batch_size)
        with open('{}/infer-of-test.txt'.format(args.data_dir), 'w') as f_infer:
            for i in range(num_steps):
                print('process batch - {}'.format(i))
                pred, t_len, s_len = sess.run(
                    [idx2tags.lookup(model_spec['predictions']),
                     inputs['num_tgt_tokens'],
                     inputs['num_tgt_sentences']])
                for raw_sentences, n_tokens, n_sentences in zip(pred, t_len, s_len):
                    f_infer.write('\n')
                    for x, y in zip(raw_sentences[:n_sentences], n_tokens[:n_sentences]):
                        f_infer.write('{}\n'.format(' '.join(x[:y])))
