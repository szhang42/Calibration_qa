

import json
import pdb


path_to_squad_json = '../data/squad/dev-v2.0.json'

string_data = ""
with open(path_to_squad_json, 'r') as f:
    for line in f:
        string_data += line
data = json.loads(string_data)['data']


path_to_mrqa_jsonl = 'data_splits/squad2.0_test_split_0.jsonl'

fp = open(path_to_mrqa_jsonl, 'wb')
header_dict = {"header": {"dataset": "SQuAD2.0", "split": "test"}}
fp.write((json.dumps(header_dict) + '\n').encode())

for article_no, article in enumerate(data):
    for paragraph_no, paragraph in enumerate(article['paragraphs']):
        example_dict = {}
        example_dict['context'] = paragraph['context']
        example_dict['qas'] = []
        for qa_no, qa in enumerate(paragraph['qas']):
            qa_dict = {}
            qa_dict['qid'] = qa['id']
            qa_dict['question'] = qa['question']
            if len(qa['answers']) > 0:
                qa_dict['answers'] = [a['text'] for a in qa['answers']]
            else:
                qa_dict['answers'] = [""]
            example_dict['qas'].append(qa_dict)
        fp.write((json.dumps(example_dict) + '\n').encode())

print()

