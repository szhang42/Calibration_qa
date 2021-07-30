"""
Opens 

"""

import json
from tqdm import tqdm

def filter_data(path_to_jsonl, hard_ids):
    qa_data = {}
    writer = open('HotpotQA_hard.jsonl', 'wb')
    count = 0
    with open(path_to_jsonl, 'rb') as f:
        for i, line in enumerate(tqdm(f)):
            example = json.loads(line)
            if 'header' in example or example['id'] in hard_ids:
                count += 1
                writer.write((json.dumps(example) + '\n').encode())
    print("{} examples written".format(count))

def main():
    original_hotpotqa = 'hotpot_train_v1.1.json'
    data = json.load(open(original_hotpotqa, 'r'))
    hard_ids = [x['_id'] for x in data if x['level']=='hard']
    filter_data('HotpotQA.jsonl', hard_ids)


if __name__ == '__main__':
    main()

