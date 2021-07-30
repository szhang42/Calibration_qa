#!/bin/bash
set -eu -o pipefail

if [ $# == 0 ]
then
    echo "Usage: $0 model_dir"
    exit 1
fi

MODEL_DIR=$1

# Time estimate: 1 hr 30 min

declare -a DATASET_PREFIXES=( "squad1.1" "triviaqa" "hotpotqa" "newsqa" "nq" "searchqa" )
declare -a DATASET_NAMES=( "SQuAD" "TriviaQA" "HotpotQA" "NewsQA" "NaturalQuestions" "SearchQA")

for i in 0 1 2 3 4 5; do
    DATASET_PREFIX="${DATASET_PREFIXES[i]}"
    DATASET_NAME="${DATASET_NAMES[i]}"
    #Predict on dev
    echo "${MODEL_DIR} on ${DATASET_NAME} dev"
    python3 ../src/bert_squad.py --bert_model bert-base-uncased --do_predict --predict_file=../data/mrqa/dev/${DATASET_NAME}.jsonl --output_dir=${MODEL_DIR} --predict_batch_size=16 --fp16 --loss_scale 128 --do_lower_case --use_pretrained --predict_mrqa --output_name=${DATASET_PREFIX}

done

echo "${MODEL_DIR} on SQuAD2.0 dev"
python3 ../src/bert_squad.py --bert_model bert-base-uncased --do_predict --predict_file=../data/squad/dev-v2.0.json --output_dir=${MODEL_DIR} --predict_batch_size=16 --fp16 --loss_scale 128 --do_lower_case --use_pretrained --output_name=squad2.0

