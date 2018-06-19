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


def load_dialogue_from_text(path_txt, vocab):
    """
    Load dialogue from text.
    Either read player's input or master's input.
    Input should be in the format:
      command-sentence-1 command-sentence-2 ...

    :param path_txt:
    :param vocab:
    :return:
    """
    dataset = tf.data.TextLineDataset(path_txt)
    dataset = (dataset
               .map(lambda string: tf.string_split([string]).values)
               .map(lambda chunks: (tf.string_split(chunks, delimiter='-'),
                                    tf.size(chunks)))
               .map(lambda chunks, size: (tf.sparse_to_dense(chunks.indices,
                                                             chunks.dense_shape,
                                                             vocab.lookup(chunks.values) + 1),
                                          size))
               .map(lambda dat, num_sentences: (dat,
                                                tf.count_nonzero(dat, axis=1, dtype=tf.int32),
                                                num_sentences))
               )

    return dataset


def dialogue_input_fn(mode, src_set, tgt_set, params):
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

    dataset = tf.data.Dataset.zip((src_set, tgt_set))

    padded_shapes = ((tf.TensorShape([None, None]),  # src
                      tf.TensorShape([None]),  # size(indices)
                      tf.TensorShape([])),  # size(src)
                     (tf.TensorShape([None, None]),  # tgt
                      tf.TensorShape([None]),  # size(indices)
                      tf.TensorShape([])))  # size(tgt)

    padding_values = ((tf.constant(0, dtype='int64'),
                       tf.constant(0, dtype='int32'),
                       tf.constant(0, dtype='int32')),
                      (tf.constant(0, dtype='int64'),
                       tf.constant(0, dtype='int32'),
                       tf.constant(0, dtype='int32')))

    if is_inferring:
        dataset = (dataset
                   .padded_batch(params.batch_size,
                                 padded_shapes=padded_shapes,
                                 padding_values=padding_values)
                   .prefetch(1))
    else:
        dataset = (dataset  # .shuffle(buffer_size=buffer_size)
                   .padded_batch(params.batch_size,
                                 padded_shapes=padded_shapes,
                                 padding_values=padding_values)
                   .prefetch(1))

    iterator = dataset.make_initializable_iterator()
    ((src, num_src_tokens, num_src_sentences),
     (tgt, num_tgt_tokens, num_tgt_sentences)) = iterator.get_next()
    init_op = iterator.initializer

    inputs = {
        'src': src,
        'num_src_tokens': num_src_tokens,
        'num_src_sentences': num_src_sentences,
        'tgt': tgt,
        'num_tgt_tokens': num_tgt_tokens,
        'num_tgt_sentences': num_tgt_sentences,
        'iterator_init_op': init_op
    }

    return inputs


if __name__ == '__main__':
    from utils import Params, transpose_batch_time, input_batch_size
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

    words = tf.contrib.lookup.index_table_from_file(path_src_vocab)
    tags = tf.contrib.lookup.index_table_from_file(path_tgt_vocab)

    params.id_pad_word = words.lookup(tf.constant(params.pad_word))
    params.id_pad_tag = tags.lookup(tf.constant(params.pad_tag))

    src = load_dialogue_from_text(path_src, words)
    tgt = load_dialogue_from_text(path_tgt, tags)

    inputs = dialogue_input_fn('train', src, tgt, params)

    src = inputs['src']
    tgt = inputs['tgt']

    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
    projection_layer = tf.layers.Dense(units=params.number_of_tags,
                                       use_bias=True)
    zero_padding = tf.constant([[0] * params.embedding_size],
                               dtype=tf.float32)
    src_embeddings_ = tf.get_variable(
        name="src_embeddings", dtype=tf.float32,
        shape=[params.vocab_size, params.embedding_size])
    src_embeddings = tf.concat([zero_padding, src_embeddings_], axis=0)
    tgt_embeddings_ = tf.get_variable(
        name="tgt_embeddings", dtype=tf.float32,
        shape=[params.number_of_tags, params.embedding_size])
    tgt_embeddings = tf.concat([zero_padding, tgt_embeddings_], axis=0)

    src_emb = tf.nn.embedding_lookup(src_embeddings, src)
    tgt_emb = tf.nn.embedding_lookup(tgt_embeddings, tgt)

    src_emb_time_major = transpose_batch_time(src_emb)
    tgt_emb_time_major = transpose_batch_time(tgt_emb)

    time_steps = tf.shape(src_emb_time_major)[0]
    batch_size = input_batch_size(src_emb_time_major)

    init_time = tf.constant(0, dtype=tf.int32)
    init_state = encoder_cell.zero_state(batch_size, tf.float32)
    init_ta = tf.TensorArray(
       dtype=tf.float32, size=time_steps,
       element_shape=tgt_emb_time_major.shape[1:-1].concatenate(
           tf.TensorShape([params.number_of_tags])))

    src_ta = tf.TensorArray(
        dtype=src_emb_time_major.dtype, size=time_steps,
        element_shape=src_emb_time_major.shape[1:])
    src_ta = src_ta.unstack(src_emb_time_major)

    tgt_ta = tf.TensorArray(
        dtype=tgt_emb_time_major.dtype, size=time_steps,
        element_shape=tgt_emb_time_major.shape[1:])
    tgt_ta = tgt_ta.unstack(tgt_emb_time_major)

    with tf.Session() as sess:
        sess.run(inputs['iterator_init_op'])
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        i = 0
        while i < 15:
            print(sess.run([tf.shape(tgt_ta.read(0))]))
            i += 1
