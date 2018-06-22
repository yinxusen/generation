"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


def load_src_dialogue(path_txt, vocab):
    dataset = (tf.data.TextLineDataset(path_txt)
               .map(lambda s: tf.string_split([s]).values)
               .map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens))))
    return dataset


def load_tgt_dialogue(path_txt, vocab):
    dataset = tf.data.TextLineDataset(path_txt)
    tgt_input = (dataset
                 .map(lambda s: '<s> ' + s)
                 .map(lambda s: tf.string_split([s]).values)
                 .map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens))))
    tgt_output = (dataset
                  .map(lambda s: s + ' </s>')
                  .map(lambda s: tf.string_split([s]).values)
                  .map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens))))
    return tgt_input, tgt_output


def dialogue_input_fn(mode,
                      path_src, path_tgt,
                      path_src_vocab, path_tgt_vocab,
                      params):
    """
    Read dialogue as input for NMT.
    :param mode: "train" or "infer"
    :param src_set: source texts
    :param tgt_set: target texts
    :param params:
    :return:
    """
    is_training = (mode == 'train')
    is_inferring = (mode == 'infer')
    buffer_size = params.buffer_size if is_training else 1

    src_vocab = tf.contrib.lookup.index_table_from_file(
        path_src_vocab, num_oov_buckets=params.num_oov_buckets)
    tgt_vocab = tf.contrib.lookup.index_table_from_file(path_tgt_vocab)

    src_set = load_src_dialogue(path_src, src_vocab)
    tgt_in_set, tgt_out_set = load_tgt_dialogue(path_tgt, tgt_vocab)
    dataset = tf.data.Dataset.zip((src_set, tgt_in_set, tgt_out_set))

    padded_shapes = ((tf.TensorShape([None]),  # src
                      tf.TensorShape([])),  # size(src)
                     (tf.TensorShape([None]),  # tgt input
                      tf.TensorShape([])),  # size(tgt)
                     (tf.TensorShape([None]),  # tgt output
                      tf.TensorShape([])))  # size(tgt)

    padding_values = ((tf.constant(0, dtype='int64'),
                       tf.constant(0, dtype='int32')),
                      (tf.constant(0, dtype='int64'),
                       tf.constant(0, dtype='int32')),
                      (tf.constant(0, dtype='int64'),
                       tf.constant(0, dtype='int32')))

    if is_inferring:
        dataset = (dataset
                   .padded_batch(params.batch_size,
                                 padded_shapes=padded_shapes,
                                 padding_values=padding_values)
                   .prefetch(1))
    else:
        dataset = (dataset
                   .shuffle(buffer_size=buffer_size)
                   .padded_batch(params.batch_size,
                                 padded_shapes=padded_shapes,
                                 padding_values=padding_values)
                   .prefetch(1))

    iterator = dataset.make_initializable_iterator()
    ((src, num_src_tokens),
     (tgt_in, num_tgt_tokens),
     (tgt_out, _)) = iterator.get_next()
    init_op = iterator.initializer

    inputs = {
        'src': src,
        'num_src_tokens': num_src_tokens,
        'tgt_in': tgt_in,
        'tgt_out': tgt_out,
        'num_tgt_tokens': num_tgt_tokens,
        'iterator_init_op': init_op
    }

    return inputs


if __name__ == '__main__':
    from utils import Params, transpose_batch_time, input_batch_size, steps_per_epoch
    from nmt_fn import build_model

    data_dir = '/Users/xusenyin/git-store/dnd/dataset-for-dialogue'
    model_dir = '/Users/xusenyin/experiments/dialogue-model'
    path_json = model_dir + '/params.json'
    path_src_vocab = data_dir + '/words.txt'
    path_tgt_vocab = data_dir + '/tags.txt'
    path_src = data_dir + '/dev/sentences.txt'
    path_tgt = data_dir + '/dev/labels.txt'

    params = Params(path_json)
    params.update(data_dir + '/dataset_params.json')
    params.eval_size = params.dev_size
    params.buffer_size = params.train_size  # buffer size for shuffling

    inputs = dialogue_input_fn(
        'train', path_src, path_tgt, path_src_vocab, path_tgt_vocab, params)

    src = inputs['src']
    tgt_input = inputs['tgt_in']
    tgt_output = inputs['tgt_out']
    num_tgt_tokens = inputs['num_tgt_tokens']
    res = tf.reduce_max(num_tgt_tokens, axis=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(inputs['iterator_init_op'])

        for i in range(steps_per_epoch(2000, 128)):
            print(sess.run([tf.shape(res), res]))
