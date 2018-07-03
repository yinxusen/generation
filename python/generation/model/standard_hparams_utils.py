# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""standard hparams utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def create_standard_hparams():
    return tf.contrib.training.HParams(
        # Data
        player_vocab_size=0,
        master_vocab_size=0,
        sos='<s>',
        eos='</s>',
        num_oov_buckets=0,
        train_size=0,
        test_size=0,
        dev_size=0,
        buffer_size=0,
        eval_size=0,
        infer_size=0,

        # Networks
        model_version='lstm',
        lstm_num_layers=0,
        lstm_num_units=0,
        embedding_size=0,
        learning_rate=.0,
        batch_size=0,
        num_epochs=0,
        dropout_rate=.0,
        save_summary_steps=0,
    )


def load_from_json(params, j_file):
    with open(j_file, 'r') as p:
        params.parse_json(p.read())
    return params
