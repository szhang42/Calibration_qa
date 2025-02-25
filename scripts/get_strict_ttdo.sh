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

declare -a TARGET_PREFIXES=( "triviaqa" "hotpotqa" "newsqa" "nq" "searchqa" )


echo "Test-time dropout (mean):"
python3 run_experiment.py ${MODEL_DIR} qa ttdo --ttdo_type=mean --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo

echo "Test-time dropout (-var):"
python3 run_experiment.py ${MODEL_DIR} qa ttdo --ttdo_type=neg_var --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo



===============================================================


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

echo "Test-time dropout (mean):"
python3 run_experiment.py ${MODEL_DIR} qa ttdo --ttdo_type=mean --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo

echo "Test-time dropout (-var):"
python3 run_experiment.py ${MODEL_DIR} qa ttdo --ttdo_type=neg_var --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo












#echo "Strict ttdo mean:"
#
#for TARGET_PREFIX in "${TARGET_PREFIXES[@]}"; do
#    echo "Evaluating model ${MODEL_DIR} on ${TARGET_PREFIX}"
##    python3 run_experiment.py ${MODEL_DIR} qa maxprob --target_prefix=${TARGET_PREFIX} --strict_eval --output_dir=${OUTPUT_DIR}
#    python3 run_experiment.py ${MODEL_DIR} qa ttdo --ttdo_type=mean --strict_eval --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
#
#    echo
#    done
#echo
#echo "Wrote results to ${OUTPUT_DIR}"
#echo
#
#
#echo "Strict ttdo variance:"
#for EXPOSE_PREFIX in "${EXPOSE_PREFIXES[@]}"; do
#        echo "Exposed to: ${EXPOSE_PREFIX}, tested on ${INPUT_DATASET_PREFIX}"
#        #python3 run_experiment.py ${MODEL_DIR} qa extrapolate --ttdo_calibrator --target_prefix=${INPUT_DATASET_PREFIX} --expose_prefix=${EXPOSE_PREFIX} --output_dir=${OUTPUT_DIR}
#        python3 run_experiment.py ${MODEL_DIR} qa ttdo --ttdo_type=neg_var --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
#    done
#echo
#echo "Wrote results to ${OUTPUT_DIR}"
#echo