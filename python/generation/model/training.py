import logging
import os

from tqdm import trange
import tensorflow as tf

from generation.model.utils import steps_per_epoch, save_dict_to_json
from generation.model.evaluation import evaluate_sess


def train_sess(sess, model_spec, num_steps, writer, params):
    """Train the model on `num_steps` batches

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries
        params: (Params) hyperparameters
    """
    # Get relevant graph operations or nodes needed for training
    loss = model_spec['loss']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        # Evaluate summaries for tensorboard only once in a while
        if i % params.save_summary_steps == 0:
            # Perform a mini-batch update
            _, _, loss_val, summ, global_step_val = sess.run(
                [train_op, update_metrics, loss, summary_op, global_step])
            # Write summaries for tensorboard
            writer.add_summary(summ, global_step_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])
        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss_val))

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("Train metrics: " + metrics_string)


def train_and_evaluate(train_model_spec,
                       eval_model_spec,
                       model_dir, params):
    """
    Train and evaluate model.
    :param train_model_spec:
    :param eval_model_spec:
    :param model_dir:
    :param params:
    :return:
    """

    last_saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)
    best_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

    with tf.Session() as sess:
        # Initialize model variables
        sess.run(train_model_spec['variable_init_op'])

        begin_at_epoch = 0
        restore_from = tf.train.latest_checkpoint(
            '{}/last_weights'.format(model_dir))
        if restore_from is not None:
            # Reload weights from directory if specified
            logging.info("Try to restore parameters")
            begin_at_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        # for tensorboard
        train_writer = tf.summary.FileWriter(
            os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(
            os.path.join(model_dir, 'eval_summaries'), sess.graph)

        early_stop_cnt = 0
        early_stop_epochs = 10
        best_eval_acc = 0.0

        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            if early_stop_cnt >= early_stop_epochs:
                logging.info(
                    'stop training by early_stop: {}'.format(early_stop_epochs))
                break

            logging.info("Epoch {}/{}".format(
                epoch + 1, begin_at_epoch + params.num_epochs))

            num_steps = steps_per_epoch(params.train_size, params.batch_size)
            train_sess(sess, train_model_spec, num_steps, train_writer, params)

            # Save weights
            last_save_path = os.path.join(model_dir,
                                          'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch+1)

            # Evaluate for one epoch on validation set
            num_steps = steps_per_epoch(params.eval_size, params.batch_size)
            metrics = evaluate_sess(sess, eval_model_spec, num_steps,
                                    eval_writer)

            # If best_eval, best_save_path
            eval_acc = metrics['accuracy']
            if eval_acc > best_eval_acc:
                early_stop_cnt = 0  # reset early stop
                # Store new best accuracy
                best_eval_acc = eval_acc
                # Save weights
                best_save_path = os.path.join(model_dir,
                                              'best_weights', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path,
                                                 global_step=epoch+1)
                logging.info(
                    "Found new best accuracy, saving in {}".format(
                        best_save_path))
                # Save best eval metrics in a json file in the model directory
                best_json_path = os.path.join(model_dir,
                                              "metrics_eval_best_weights.json")
                save_dict_to_json(metrics, best_json_path)
            else:
                early_stop_cnt += 1
                logging.info('no improvement {} epochs'.format(early_stop_cnt))

            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir,
                                          "metrics_eval_last_weights.json")
            save_dict_to_json(metrics, last_json_path)
