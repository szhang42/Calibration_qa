#!/bin/bash
set -eu -o pipefail

if [ $# -lt 3 ]
then
    echo "Usage: $0 model_dir target_prefix output_dir"
    exit 1
fi

MODEL_DIR=$1
INPUT_DATASET_PREFIX=$2
OUTPUT_DIR=$3

echo
echo "Evaluating model ${MODEL_DIR} on ${INPUT_DATASET_PREFIX}"
echo

# Sanity check:
# declare -a PERCENTAGES=( 0.1 1.0 )
declare -a PERCENTAGES=( 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 )

for PERCENTAGE in "${PERCENTAGES[@]}"; do
    echo "Percentage = ${PERCENTAGE}"
    echo "Maxprob:"
    python3 run_experiment.py ${MODEL_DIR} qa maxprob --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR} --fraction_id=${PERCENTAGE}
    echo
    echo "Wrote results to ${OUTPUT_DIR}"
    echo

    # Sanity check:
    # declare -a EXPOSE_PREFIXES=( "triviaqa" )
    declare -a EXPOSE_PREFIXES=( "triviaqa" "hotpotqa" "newsqa" "nq" "searchqa" )
    for EXPOSE_PREFIX in "${EXPOSE_PREFIXES[@]}"; do
            echo "Exposed to: ${EXPOSE_PREFIX}, tested on ${INPUT_DATASET_PREFIX}"
            python3 run_experiment.py ${MODEL_DIR} qa extrapolate --target_prefix=${INPUT_DATASET_PREFIX} --expose_prefix=${EXPOSE_PREFIX} --output_dir=${OUTPUT_DIR} --fraction_id=${PERCENTAGE}
        done
    echo
    echo "Wrote results to ${OUTPUT_DIR}"
    echo
done

