
# Read data_splits/triviaqa_{train/dev}_split_*, compile QIDs
import json
import pdb
from tqdm import tqdm


def main():
    dataset_prefixes = ['triviaqa', 'hotpotqa', 'newsqa', 
                            'nq', 'searchqa']
    for dataset_prefix in dataset_prefixes:
        print("Compiling QIDs from {}".format(dataset_prefix))
        all_qid_dict = {}
        for split_type in ['train', 'dev']:
            for split_no in tqdm(range(10)):
                f = open('./data_splits/{}_{}_split_{}.jsonl'
                            .format(
                            dataset_prefix, split_type, split_no))
                for line in f:
                    data = json.loads(line)
                    if 'header' in data.keys():
                        continue
                    for qa in data['qas']:
                        all_qid_dict[qa['qid']] = 1
        f = open('./data_splits/{}_train_dev_qids.txt'
                        .format(dataset_prefix), 'w')
        for qid in all_qid_dict.keys():
            f.write(qid + '\n')
        print("Written to {}".format('./data_splits/{}_train_dev_qids.txt'
                        .format(dataset_prefix)))

if __name__ == '__main__':
    main()

