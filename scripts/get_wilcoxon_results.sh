#!/bin/bash
set -eu -o pipefail

if [ $# -lt 2 ]
then
    echo "Usage: $0 model_dir target_prefix"
    exit 1
fi

MODEL_DIR=$1
INPUT_DATASET_PREFIX=$2

echo
echo "Evaluating model ${MODEL_DIR} on ${INPUT_DATASET_PREFIX}"
echo

declare -a EXPOSE_PREFIXES=( "triviaqa" "hotpotqa" "newsqa" "nq" "searchqa" )
for EXPOSE_PREFIX in "${EXPOSE_PREFIXES[@]}"; do
        echo "Exposed to: ${EXPOSE_PREFIX}, tested on ${INPUT_DATASET_PREFIX}"
        python3 run_experiment.py ${MODEL_DIR} qa wilcoxon --target_prefix=${INPUT_DATASET_PREFIX} --expose_prefix=${EXPOSE_PREFIX}
    done
echo




