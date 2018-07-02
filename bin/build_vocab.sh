#!/bin/bash

set -e -x

DHOME=$1

cat $DHOME/train/player.txt $DHOME/dev/player.txt $DHOME/test/player.txt | tr ' -' '\n' | awk '{key[$0];} END { for( word in key ) { print word }}' | sort > $DHOME/player-tokens.txt
echo '<s>' >> $DHOME/player-tokens.txt
echo '</s>' >> $DHOME/player-tokens.txt

cat $DHOME/train/master.txt $DHOME/dev/master.txt $DHOME/test/master.txt | tr ' -' '\n' | awk '{key[$0];} END { for( word in key ) { print word }}' | sort > $DHOME/master-tokens.txt
echo '<s>' >> $DHOME/master-tokens.txt
echo '</s>' >> $DHOME/master-tokens.txt

NUM_PTOKENS=`wc -l $DHOME/player-tokens.txt | awk '{print $1}'`
NUM_MTOKENS=`wc -l $DHOME/master-tokens.txt | awk '{print $1}'`

echo '' > $DHOME/dataset_params.json

cat > $DHOME/dataset_params.json <<EOL
{
    "player_vocab_size": $NUM_PTOKENS,
    "master_vocab_size": $NUM_MTOKENS,
    "sos": "<s>",
    "eos": "</s>",
    "num_oov_buckets": 0,
    "train_size": `wc -l $DHOME/train/player.txt | awk '{print $1}'`,
    "test_size": `wc -l $DHOME/test/player.txt | awk '{print $1}'`,
    "dev_size": `wc -l $DHOME/dev/player.txt | awk '{print $1}'`
}
EOL
