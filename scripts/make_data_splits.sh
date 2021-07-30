#!/bin/bash
set -eu -o pipefail

if [ $# -ge 1 ]
then
    echo "Usage: $0"
    exit 1
fi

declare -a DATASETS=( "triviaqa" "hotpotqa" "newsqa" "nq" "searchqa" "squad1.1" )

echo "Generating splits and storing in ./data_splits/"
echo

for DATASET in "${DATASETS[@]}"; do
    python3 store_splits.py ${DATASET}
    echo
done

echo "Done!"

