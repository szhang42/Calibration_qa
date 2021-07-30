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

echo "Minimum:"
#python3 run_experiment.py ${MODEL_DIR} qa minimum --target_prefix=${INPUT_DATASET_PREFIX}
echo

echo "Maxprob:"
#python3 run_experiment.py ${MODEL_DIR} qa maxprob --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo

echo "Test-time dropout (mean):"
#python3 run_experiment.py ${MODEL_DIR} qa ttdo --ttdo_type=mean --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo

echo "Test-time dropout (-var):"
#python3 run_experiment.py ${MODEL_DIR} qa ttdo --ttdo_type=neg_var --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo

echo "Calibrator without exposure (without TTDO features):"
#python3 run_experiment.py ${MODEL_DIR} qa extrapolate --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo

echo "Calibrator without TTDO features:"
declare -a EXPOSE_PREFIXES=( "triviaqa" "hotpotqa" "newsqa" "nq" "searchqa" )
for EXPOSE_PREFIX in "${EXPOSE_PREFIXES[@]}"; do
        echo "Exposed to: ${EXPOSE_PREFIX}, tested on ${INPUT_DATASET_PREFIX}"
        python3 run_experiment.py ${MODEL_DIR} qa extrapolate --target_prefix=${INPUT_DATASET_PREFIX} --expose_prefix=${EXPOSE_PREFIX} --output_dir=${OUTPUT_DIR} --strict_eval
    done
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo

echo "Calibrator with TTDO features:"
for EXPOSE_PREFIX in "${EXPOSE_PREFIXES[@]}"; do
        echo "Exposed to: ${EXPOSE_PREFIX}, tested on ${INPUT_DATASET_PREFIX}"
        #python3 run_experiment.py ${MODEL_DIR} qa extrapolate --ttdo_calibrator --target_prefix=${INPUT_DATASET_PREFIX} --expose_prefix=${EXPOSE_PREFIX} --output_dir=${OUTPUT_DIR}
    done
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo

