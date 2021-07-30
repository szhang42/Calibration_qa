#!/bin/bash
set -eu -o pipefail

if [ $# -gt 0 ]
then
    echo "Usage: $0"
    exit 1
fi

cp filter_hard.py ../data/mrqa/train
cd ../data/mrqa/train
echo "Downloading original (non-MRQA) HotpotQA train data to data/mrqa/train/"
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
echo "Cross-referencing to filter not-hard questions out of MRQA HotpotQA"
python3 filter_hard.py
echo "Moving old MRQA data to HotpotQA_mixed.jsonl"
mv HotpotQA.jsonl HotpotQA_mixed.jsonl
echo "Renaming HotpotQA_hard.jsonl to HotpotQA.jsonl"
mv HotpotQA_hard.jsonl HotpotQA.jsonl
echo "Done"


