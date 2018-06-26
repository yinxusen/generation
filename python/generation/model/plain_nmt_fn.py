"""
Nested NMT model for context-sensitive dialogue generation.
"""

import tensorflow as tf


def build_model(mode, inputs, params, sentence_max_len=None):
    """
    Build recurring NMT model for dialogue generation.
    :param mode:
    :param inputs:
    :param params:
    :param sentence_max_len:
    :return:
    """
    src = inputs['src']
    tgt_in = inputs['tgt_in']

    num_tgt_tokens = inputs['num_tgt_tokens']
    num_src_tokens = inputs['num_src_tokens']

    if params.model_version == 'lstm':
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
    elif params.model_version == 'stack_lstm':
        def new_cell(state_size):
            cell = tf.nn.rnn_cell.LSTMCell(state_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                output_keep_prob=1.-params.dropout_rate)
            return cell
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [new_cell(params.lstm_num_units)
             for _ in range(params.lstm_num_layers)])
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [new_cell(params.lstm_num_units)
             for _ in range(params.lstm_num_layers)])
    else:
        raise ValueError(
            'Unknown model_version: {}'.format(params.model_version))

    projection_layer = tf.layers.Dense(units=params.number_of_tags,
                                       use_bias=True)

    src_embeddings = tf.get_variable(
        name="src_embeddings", dtype=tf.float32,
        shape=[params.vocab_size, params.embedding_size])
    tgt_embeddings = tf.get_variable(
        name="tgt_embeddings", dtype=tf.float32,
        shape=[params.number_of_tags, params.embedding_size])

    src_emb = tf.nn.embedding_lookup(src_embeddings, src)
    tgt_in_emb = tf.nn.embedding_lookup(tgt_embeddings, tgt_in)

    batch_size = tf.shape(src_emb)[0]

    _, inner_state = tf.nn.dynamic_rnn(
        encoder_cell, src_emb, sequence_length=num_src_tokens,
        initial_state=None,
        dtype=tf.float32)
    if mode == 'train' or mode == 'eval':
        helper = tf.contrib.seq2seq.TrainingHelper(
            tgt_in_emb,
            num_tgt_tokens,
            time_major=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, inner_state,
            output_layer=projection_layer)
        outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, output_time_major=False, impute_finished=True,
            swap_memory=True)
        rnn_output = outputs.rnn_output
    elif mode == 'infer':
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            tgt_embeddings,
            tf.fill([batch_size], 0), 1)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, inner_state,
            output_layer=projection_layer)
        outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, output_time_major=False,
            swap_memory=True, maximum_iterations=5*2)
        rnn_output = outputs.rnn_output
    else:
        raise ValueError('unknown mode {}'.format(mode))
    return rnn_output


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    num_tgt_tokens = inputs['num_tgt_tokens']

    mask = tf.sequence_mask(num_tgt_tokens)
    labels = inputs['tgt_out']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, inputs, params)
        predictions = tf.argmax(logits, -1)

    # Define loss and accuracy (we need to apply a mask to account for padding)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=labels)
    losses = tf.boolean_mask(losses, mask)
    loss = tf.reduce_mean(losses)
    # the computing of accuracy forget about masking
    accuracies = tf.boolean_mask(tf.equal(labels, predictions), mask)
    accuracy = tf.reduce_mean(tf.cast(accuracies, tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(
                labels=tf.boolean_mask(labels, mask),
                predictions=tf.boolean_mask(predictions, mask)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                         scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(),
                                  tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['predictions'] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
