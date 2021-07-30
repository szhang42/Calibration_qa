#!/bin/bash
set -eu -o pipefail

if [ $# -lt 2 ]
then
    echo "Usage: $0 model_dir output_dir"
    exit 1
fi

MODEL_DIR=$1
OUTPUT_DIR=$2

declare -a TARGET_PREFIXES=( "triviaqa" "hotpotqa" "newsqa" "nq" "searchqa" )

echo "Strict maxprob evaluation:"

for TARGET_PREFIX in "${TARGET_PREFIXES[@]}"; do
    echo "Evaluating model ${MODEL_DIR} on ${TARGET_PREFIX}"
    python3 run_experiment.py ${MODEL_DIR} qa maxprob --target_prefix=${TARGET_PREFIX} --strict_eval --output_dir=${OUTPUT_DIR}
    echo
    done
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo


