#!/bin/bash
set -eu -o pipefail
if [ $# -lt 3 ]
then
    echo "Usage: $0 model_dir target_prefix [data_splits path]"
    exit 1
fi

MODEL_DIR=$1
TARGET_PREFIX=$2
DATA_SPLITS=$3

#declare -a LIST=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 )

declare -a LIST=( 0 1 2 3 4 )

# Sanity check:
#declare -a LIST=( 0 )

for i in "${LIST[@]}"; do
    echo "Getting predictions for ${TARGET_PREFIX} using Dropout seed ${i}"
    python3 ../src/bert_squad.py --bert_model bert-base-uncased --do_predict --predict_file=${DATA_SPLITS}/${TARGET_PREFIX}_test_split_0.jsonl --output_dir=${MODEL_DIR} --predict_batch_size=16 --loss_scale 128 --do_lower_case --use_pretrained --predict_mrqa --output_name=${TARGET_PREFIX}_dropout --test_time_dropout --dropout_seed=${i}
    echo
done

