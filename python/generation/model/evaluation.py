import logging
import os

from tqdm import trange
import tensorflow as tf

from generation.model.utils import save_dict_to_json, steps_per_epoch


def evaluate_sess(sess, model_spec, num_steps, writer=None, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries.
        params: (Params) hyperparameters
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()

    # Load the evaluation dataset into the pipeline
    #  and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # compute metrics over the dataset
    for _ in trange(num_steps):
        sess.run(update_metrics)

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("Eval metrics: " + metrics_string)

    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)
    return metrics_val


def evaluate(model_spec, model_dir, params, restore_from):
    """
    Evaluate a model
    :param model_spec:
    :param model_dir:
    :param params:
    :param restore_from:
    :return:
    """
    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # Evaluate
        num_steps = steps_per_epoch(params.eval_size, params.batch_size)
        metrics = evaluate_sess(sess, model_spec, num_steps)
        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir,
                                 "metrics_test_{}.json".format(metrics_name))
        save_dict_to_json(metrics, save_path)
