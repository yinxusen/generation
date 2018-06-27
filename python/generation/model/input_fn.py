"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


def load_dataset_from_text(path_txt, vocab):
    """Create tf.data Instance from txt file

    Args:
        path_txt: (string) path containing one example per line
        vocab: (tf.lookuptable)

    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens for each example
    """
    # Load txt file, one example per line
    dataset = tf.data.TextLineDataset(path_txt)

    # Convert line into list of tokens, splitting by white space
    dataset = dataset.map(lambda string: tf.string_split([string]).values)

    # Lookup tokens to return their ids
    dataset = dataset.map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens)))

    return dataset


def input_fn(mode, sentences, labels, params):
    """Input function for NER

    Args:
        mode: (string) 'train', 'eval' or any other mode you can think of
                     At training, we shuffle the data and have multiple epochs
        sentences: (tf.Dataset) yielding list of ids of words
        datasets: (tf.Dataset) yielding list of ids of tags
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train')
    is_infering = (mode == 'infer')
    buffer_size = params.buffer_size if is_training else 1

    # Zip the sentence and the labels together
    dataset = tf.data.Dataset.zip((sentences, labels))

    # Create batches and pad the sentences of different length
    padded_shapes = ((tf.TensorShape([None]),  # sentence of unknown size
                      tf.TensorShape([])),     # size(words)
                     (tf.TensorShape([None]),  # labels of unknown size
                      tf.TensorShape([])))     # size(tags)

    padding_values = ((params.id_pad_word,   # sentence padded on the right with id_pad_word
                       0),                   # size(words) -- unused
                      (params.id_pad_tag,    # labels padded on the right with id_pad_tag
                       0))                   # size(tags) -- unused

    if is_infering:
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
                   .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    ((sentence, sentence_lengths), (labels, _)) = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'sentence': sentence,
        'labels': labels,
        'sentence_lengths': sentence_lengths,
        'iterator_init_op': init_op
    }

    return inputs


def sparse_index_to_count(indices):
    ones_indices = tf.ones_like(indices, dtype=tf.int32)
    return tf.segment_sum(ones_indices, indices[:, 0])[:, 0]


def string_tensor_to_dense_mat(string_tensor, lookup_tbl):
    return (string_tensor
            .map(lambda chunks: (tf.string_split(chunks, delimiter='-'),
                                 tf.size(chunks)))
            .map(lambda chunks, size: (
                tf.sparse_to_dense(chunks.indices,
                                   chunks.dense_shape,
                                   tf.cast(lookup_tbl.lookup(chunks.values),
                                           tf.int32)),
                sparse_index_to_count(chunks.indices),
                size)))


def load_src_dialogue(path_txt, vocab):
    dataset = (tf.data.TextLineDataset(path_txt)
               .map(lambda s: tf.string_split([s]).values))
    return string_tensor_to_dense_mat(dataset, vocab)


def load_tgt_dialogue(path_txt, vocab):
    dataset = (tf.data.TextLineDataset(path_txt)
               .map(lambda s: tf.string_split([s]).values))
    tgt_input = dataset.map(
        lambda substrs: tf.map_fn(lambda s: '<s>-' + s, substrs))
    tgt_output = dataset.map(
        lambda substrs: tf.map_fn(lambda s: s + '-</s>', substrs))
    return (string_tensor_to_dense_mat(tgt_input, vocab),
            string_tensor_to_dense_mat(tgt_output, vocab))


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

    padded_shapes = ((tf.TensorShape([None, None]),  # src
                      tf.TensorShape([None]),  # size(indices)
                      tf.TensorShape([])),  # size(src)
                     (tf.TensorShape([None, None]),  # tgt input
                      tf.TensorShape([None]),  # size(indices)
                      tf.TensorShape([])),  # size(tgt)
                     (tf.TensorShape([None, None]),  # tgt output
                      tf.TensorShape([None]),  # size(indices)
                      tf.TensorShape([])))  # size(tgt)

    src_eos = tf.cast(src_vocab.lookup(tf.constant('</s>')), tf.int32)
    tgt_eos = tf.cast(tgt_vocab.lookup(tf.constant('</s>')), tf.int32)
    tgt_sos = tf.cast(tgt_vocab.lookup(tf.constant('<s>')), tf.int32)

    padding_values = ((src_eos,
                       tf.constant(0, dtype='int32'),
                       tf.constant(0, dtype='int32')),
                      (tgt_eos,
                       tf.constant(0, dtype='int32'),
                       tf.constant(0, dtype='int32')),
                      (tgt_eos,
                       tf.constant(0, dtype='int32'),
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
    ((src, num_src_tokens, num_src_sentences),
     (tgt_in, num_tgt_tokens, num_tgt_sentences),
     (tgt_out, _, _)) = iterator.get_next()
    init_op = iterator.initializer

    inputs = {
        'src': src,
        'num_src_tokens': num_src_tokens,
        'num_src_sentences': num_src_sentences,
        'tgt_in': tgt_in,
        'tgt_out': tgt_out,
        'num_tgt_tokens': num_tgt_tokens,
        'num_tgt_sentences': num_tgt_sentences,
        'iterator_init_op': init_op,
        'tgt_sos_id': tgt_sos,
        'tgt_eos_id': tgt_eos
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
    path_src = data_dir + '/test/sentences.txt'
    path_tgt = data_dir + '/test/labels.txt'

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
