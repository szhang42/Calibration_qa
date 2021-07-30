"""

Create and store train and dev splits of the same length
for all datasets (1600, 400).
And test 4K
Splits on contexts, not questions.
Writes data in ./data_splits/ in MRQA jsonl format.

Usage: python store_splits.py dataset_prefix

"""

import json
import pdb
import math
import numpy as np
import os
import argparse
from tqdm import tqdm
from utils import read_gold_data, DATA_DIR, greedy_subsample


def main():
    datasets = { 'squad1.1': 'SQuAD',
                 'triviaqa': 'TriviaQA', 
                 'hotpotqa': 'HotpotQA',
                 'newsqa': 'NewsQA', 
                 'nq': 'NaturalQuestions',
                 'searchqa': 'SearchQA'
                }
    parser = argparse.ArgumentParser(description='Store splits')
    parser.add_argument('dataset_prefix', type=str, default=None,
                        help='Dataset prefix to split',
                        choices=['squad1.1', 'triviaqa', 'hotpotqa',\
                                 'newsqa', 'nq', 'searchqa'])
    args = parser.parse_args()
    dataset_prefix = args.dataset_prefix
    dataset_name = datasets[dataset_prefix]

    if os.path.exists('data_splits'):
        print("Note that ./data_splits already exists.")
    else:
        os.makedirs('data_splits')
    
    print("Dataset: {}".format(dataset_name))
    train_len = 1600
    dev_len = 400
    test_len = 4000
    greedy_subsample(dataset_prefix, dataset_name, 'train', train_len)
    greedy_subsample(dataset_prefix, dataset_name, 'dev', dev_len)
    greedy_subsample(dataset_prefix, dataset_name, 'test', test_len)
    print()


if __name__ == '__main__':
    main()

