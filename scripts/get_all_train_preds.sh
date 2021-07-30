#!/bin/bash
set -eu -o pipefail

if [ $# == 0 ]
then
    echo "Usage: $0 model_dir"
    exit 1
fi

MODEL_DIR=$1

declare -a DATASET_PREFIXES=( "triviaqa" "hotpotqa" "newsqa" "nq" "searchqa" )
declare -a DATASET_NAMES=( "TriviaQA" "HotpotQA" "NewsQA" "NaturalQuestions" "SearchQA")

for i in 0 1 2 3 4; do
    DATASET_PREFIX="${DATASET_PREFIXES[i]}"
    DATASET_NAME="${DATASET_NAMES[i]}"
    # Predict on train
    echo "${MODEL_DIR} on ${DATASET_NAME} train"
    python3 ../src/bert_squad.py --bert_model bert-base-uncased --do_predict --predict_file=../data/mrqa/train/${DATASET_NAME}.jsonl --output_dir=${MODEL_DIR} --predict_batch_size=4 --fp16 --loss_scale 128 --do_lower_case --use_pretrained --predict_mrqa --output_name=${DATASET_PREFIX}_train --predict_on_train=data_splits/${DATASET_PREFIX}_train_dev_qids.txt
done

