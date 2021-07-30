#!/bin/bash
set -eu -o pipefail
if [ $# -lt 3 ]
then
    echo "Usage: $0 model_dir dataset_prefix [data_splits path]"
    exit 1
fi

MODEL_DIR=$1
DATASET_PREFIX=$2
DATA_SPLITS=$3

declare -a LIST=( 0 1 2 3 4 )
# Sanity check: 
#declare -a LIST=( 0 )

if [ ${DATASET_PREFIX} == "triviaqa" ]
then
    DATASET_NAME="TriviaQA"
fi
if [ ${DATASET_PREFIX} == "hotpotqa" ]
then
    DATASET_NAME="HotpotQA"
fi
if [ ${DATASET_PREFIX} == "newsqa" ]
then
    DATASET_NAME="NewsQA"
fi
if [ ${DATASET_PREFIX} == "nq" ]
then
    DATASET_NAME="NaturalQuestions"
fi
if [ ${DATASET_PREFIX} == "searchqa" ]
then
    DATASET_NAME="SearchQA"
fi

for i in "${LIST[@]}"; do
    echo "Getting predictions for ${DATASET_NAME} using Dropout seed ${i}"
    python3 ../src/bert_squad.py --bert_model bert-base-uncased --do_predict --predict_file=../data/mrqa/train/${DATASET_NAME}.jsonl --predict_on_train=${DATA_SPLITS}/${DATASET_PREFIX}_train_dev_qids.txt --output_dir=${MODEL_DIR} --predict_batch_size=16 --fp16 --loss_scale 128 --do_lower_case --use_pretrained --predict_mrqa --output_name=${DATASET_PREFIX}_train_dropout --test_time_dropout --dropout_seed=${i}
    echo
done
echo
echo

