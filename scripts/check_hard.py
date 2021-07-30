import json
import ujson
from tqdm import tqdm

def read_qa_data(path_to_jsonl, hard_ids):
    """
    Read gold data into a dictionary, given the JSONL path.
    Returns: {GUID: [list_of_answers]}
    """
    qa_data = {}
    count = 0
    with open(path_to_jsonl, 'rb') as f:
        for i, line in enumerate(tqdm(f)):
            example = json.loads(line)
            if 'header' in example:
                continue
            if example['id'] not in hard_ids:
                count += 1
    print("{} examples NOT HARD".format(count))

def main():

    filename = 'hotpot_train_v1.1.json'
    data = json.load(open(filename, 'r'))
    hard_ids = [x['_id'] for x in data if x['level']=='hard']
    read_qa_data('hotpotqa_train_split_0.jsonl', hard_ids)

    """
    for s in ['train', 'dev']:
        for split in range(10):
            filename2 = 'hotpotqa_{}_split_{}.jsonl'.format(s, split)
            print(filename2)
            read_qa_data(filename2, hard_ids)
    """

if __name__ == '__main__':
    main()

