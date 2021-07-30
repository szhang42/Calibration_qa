"""
Utils


"""

import json
import os
import string
import re
import matplotlib.pyplot as plt
import copy
from sklearn import metrics
import termcolor
import pdb
import numpy as np
from math import ceil
from tqdm import tqdm
from scipy.stats import wilcoxon
from collections import OrderedDict


MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, 'data')

"""
Read QA and NLI files
"""

def read_nbest_data(path_to_json, guid_list):
    """Construct JSON path and read nbest data into a dictionary, given args."""
    nbest_data = {}
    string_data = ""
    with open(path_to_json, 'r') as f:
        for line in f:
            string_data += line
    data = json.loads(string_data)
    nbest_data = {k: v for k, v in data.items() if k in guid_list}
    return nbest_data


def read_nbest_data_from_long(path, guid_list):
    nbest_data = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            guid_dict = json.loads(line)
            guid = list(guid_dict.keys())[0]
            if guid in guid_list:
                nbest_data[guid] = guid_dict[guid]
    return nbest_data
    

def read_gold_data(task, path_to_jsonl, dataset_prefix, dataset_subset=None):
    """
    Calls helper functions to read gold data, depending on 
    task (qa/nli) and dataset_subset (antonym, ..., all).
    Read gold data into a dictionary, given the JSONL path.
    Returns: {QID: [list_of_answers]}
    """
    if task == 'qa' and dataset_prefix !='squad2.0':
        return read_qa_data(path_to_jsonl)
    if task == 'qa' and dataset_prefix =='squad2.0':
        return read_squad2_data(path_to_jsonl)
    if dataset_prefix == 'mnli' or dataset_subset == 'all':
        return read_all_nli_data(path_to_jsonl, dataset_prefix)
    else:
        return read_typed_nli_data(path_to_jsonl, dataset_prefix, dataset_subset)
        

def read_qa_data(path_to_jsonl):
    """
    Read gold data into a dictionary, given the JSONL path.
    Returns: {GUID: [list_of_answers]}
    """
    qa_data = {}
    if not os.path.exists(path_to_jsonl):
        print('Warning: File {} does not exist. Returning empty dictionary.'.format(path_to_jsonl))
        return {}
    with open(path_to_jsonl, 'rb') as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            if i == 0 and 'header' in example:
                continue
            for qa in example['qas']:
                qa_data[qa['qid']] = {}
                qa_data[qa['qid']]['context'] = example['context']
                qa_data[qa['qid']]['question'] = qa['question']
                qa_data[qa['qid']]['answers'] = qa['answers']
    return qa_data


def read_squad2_data(path_to_jsonl):
    """
    Read gold data into a dictionary, given the JSONL path.
    Returns: {GUID: [list_of_answers]}
    """
    qa_data = {}
    if not os.path.exists(path_to_jsonl):
        print('Warning: File {} does not exist. Returning empty dictionary.'.format(path_to_jsonl))
        return {}
    with open(path_to_jsonl, 'rb') as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            if i == 0 and 'header' in example:
                continue
            # print('example', example['data'])
            for qa in example['data']:
                for qa_new in qa['paragraphs']:
                    for a in qa_new['qas']:
                        # print('example', a);input()
                        qa_data[a['id']] = {}
                        qa_data[a['id']]['context'] = qa_new['context']
                        qa_data[a['id']]['question'] = a['question']
                        qa_data[a['id']]['answers'] = a['answers']
    return qa_data


"""
EM calculation
"""

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """Calculate the EM score for a given prediction."""
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


"""
Get risk coverage data
"""

def get_risk_coverage_info(prob_list, em_list):
    num = int(len(prob_list)/2)
    sources = [0 for i in range(num)]
    sources.extend([1 for i in range(num)])
    assert len(sources) == len(prob_list)
    tuples = [(x,y,z) for x,y,z in zip(prob_list, em_list, sources)]
    sorted_tuples = sorted(tuples, key=lambda x: -x[0])
    sorted_probs = [x[0] for x in sorted_tuples]
    sorted_em = [x[1] for x in sorted_tuples]
    sorted_sources = [x[2] for x in sorted_tuples]
    total_questions = len(sorted_em)
    total_correct = 0
    covered = 0
    risks = []
    coverages = []

    for em, prob in zip(sorted_em, sorted_probs):
        covered += 1
        if em:
            total_correct += 1
        risks.append(1 - (total_correct/covered))
        coverages.append(covered/total_questions)        
    auc = round(metrics.auc(coverages, risks), 4)

    
    return risks, coverages, auc, sorted_sources, sorted_em


def get_coverage_cutoff(risks, accuracy_cutoff):
    index = len(risks)
    while risks[index-1] >= (1.0-accuracy_cutoff) and index > 0:
        index -= 1
    return index


def get_per_domain_info(all_sorted_sources, all_sorted_em, all_risks):
    assert len(all_sorted_sources) == len(all_sorted_em)
    assert len(all_sorted_sources) == len(all_risks)
    accuracies = {  0.8: { 'id': [], 'ood': []}, 
                    0.9: { 'id': [], 'ood': []} }
    coverages = {   0.8: { 'id': [], 'ood': []},
                    0.9: { 'id': [], 'ood': []} }

    for sorted_sources, sorted_em, risks in zip(all_sorted_sources, all_sorted_em, all_risks):
        for acc_cutoff in [0.8, 0.9]:
            index = get_coverage_cutoff(risks, acc_cutoff)
            num_ood = sum(sorted_sources[:index])
            num_id = index - num_ood
            correct_ood = sum([1 for i in range(index) if sorted_sources[i] == 1 and sorted_em[i] == 1])
            acc_ood = correct_ood/num_ood
            accuracies[acc_cutoff]['ood'].append(acc_ood)
            cov_ood = num_ood / index
            coverages[acc_cutoff]['ood'].append(cov_ood)
            correct_id = sum([1 for i in range(index) if sorted_sources[i] == 0 and sorted_em[i] == 1])
            acc_id = correct_id/num_id
            accuracies[acc_cutoff]['id'].append(acc_id)
            cov_id = num_id / index
            coverages[acc_cutoff]['id'].append(cov_id)

    for acc_cutoff in [0.8, 0.9]:
        for domain in ['id', 'ood']:
            accuracies[acc_cutoff][domain] = round(100*np.mean(accuracies[acc_cutoff][domain]), 2)
            coverages[acc_cutoff][domain] = round(100*np.mean(coverages[acc_cutoff][domain]), 2)

    print()
    print("At 80% Accuracy:")
    print("Accuracy in-domain: {}, Accuracy OOD: {}".format(accuracies[0.8]['id'], accuracies[0.8]['ood']))
    print("Percentage of the answered questions that were in-domain: {}, OOD: {}".format(coverages[0.8]['id'], coverages[0.8]['ood']))
    print()
    print("At 90% Accuracy:")
    print("Accuracy in-domain: {}, Accuracy OOD: {}".format(accuracies[0.9]['id'], accuracies[0.9]['ood']))
    print("Percentage of the answered questions that were in-domain: {}, OOD: {}".format(coverages[0.9]['id'], coverages[0.9]['ood']))
    print()
    return

"""
Graphing scripts
"""

def pyx_graph(dev_guid_list, maxprobs, em_dict, name):
    # Sort maxprobs belonging to dev
    # Bin, calculate EM of each, graph.
    #pdb.set_trace()
    maxprobs_dev = {k: v for k, v in maxprobs.items() if k in dev_guid_list}
    sorted_maxprobs = OrderedDict(sorted(maxprobs_dev.items(), key=lambda x: x[1]))
    # Bins should be maxprob ranges, not frequency
    # i.e. equal width, not equal depth.
    num_bins = 20
    bin_limits = [(i+1)/num_bins for i in range(num_bins)]
    bin_correct = [0 for i in range(num_bins)]
    bin_totals = [0 for i in range(num_bins)]
    i = 0
    current_bin = 0
    while i < len(maxprobs_dev):
        guid, maxprob = list(sorted_maxprobs.items())[i]
        if maxprob > bin_limits[current_bin]:
            current_bin += 1
        bin_totals[current_bin] += 1
        if em_dict[guid]:
            bin_correct[current_bin] += 1
        i += 1
    #pdb.set_trace()
    for i in range(num_bins):
        if bin_totals[i] == 0:
            bin_totals[i] = 1
    probs = [bin_correct[i]/bin_totals[i] for i in range(num_bins)]
    plt.plot([i/num_bins for i in range(num_bins)], probs, label=name)
    print(probs)
    leg = plt.legend(loc=0, prop={'size': 8}, title='Dataset')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot([0,1],[0,1], color='k', linestyle='dashed')
    plt.title('Maxprob (x) vs P(EM=1|x) for {}'.format(name))
    plt.xlabel('x')
    plt.ylabel('p(EM=1|x)')
    plt.savefig('pyx_{}.png'.format(name))
    return
   

def pyx_graph_all(dev_guid_list, maxprobs, em_dict, name):
    # Sort maxprobs belonging to dev
    # Bin, calculate EM of each, graph.
    maxprobs_dev = {k: v for k, v in maxprobs.items() if k in dev_guid_list}
    # Bins should be maxprob ranges, not frequency
    # i.e. equal width, not equal depth.
    num_bins = 20
    bin_correct = [0 for i in range(num_bins)]
    bin_totals = [0 for i in range(num_bins)]
    i = 0
    #pdb.set_trace()
    for guid, prob_list in maxprobs_dev.items():
        em_value = em_dict[guid]
        for prob in prob_list:
            bin_index = ceil(prob * num_bins) - 1
            bin_totals[bin_index] += 1
            if em_value:
                bin_correct[bin_index] += 1
    # Make sure there are no empty bins (prevent /0)
    #pdb.set_trace()
    for i in range(num_bins):
        if bin_totals[i] == 0:
            bin_totals[i] = 1
    #pdb.set_trace()
    probs = [bin_correct[i]/bin_totals[i] for i in range(num_bins)]
    target_name = name.split('_')[-1]
    plt.plot([(i/num_bins)+0.025 for i in range(num_bins)], probs, label=target_name)
    print(probs)
    leg = plt.legend(loc=0, prop={'size': 10}, title='Dataset')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.plot([0,1],[0,1], color='k', linestyle='dashed')
    plt.title('Calibration plot for trained Calibrator', fontsize=18)
    plt.xlabel('Calibrator probability', fontsize=14)
    plt.ylabel('Probability of correctness', fontsize=14)
    plt.savefig('{}_cal_pyx.png'.format(name), dpi=400)
    return
 

def pyx_graph_mp(dev_guid_list, maxprobs, em_dict, name):
    # Sort maxprobs belonging to dev
    # Bin, calculate EM of each, graph.
    maxprobs_dev = {k: v for k, v in maxprobs.items() if k in dev_guid_list}
    # Bins should be maxprob ranges, not frequency
    # i.e. equal width, not equal depth.
    num_bins = 20
    bin_correct = [0 for i in range(num_bins)]
    bin_totals = [0 for i in range(num_bins)]
    i = 0
    #pdb.set_trace()
    for guid, prob in maxprobs_dev.items():
        em_value = em_dict[guid]
        bin_index = ceil(prob * num_bins) - 1
        bin_totals[bin_index] += 1
        if em_value:
            bin_correct[bin_index] += 1
    # Make sure there are no empty bins (prevent /0)
    #pdb.set_trace()
    for i in range(num_bins):
        if bin_totals[i] == 0:
            bin_totals[i] = 1
    #pdb.set_trace()
    probs = [bin_correct[i]/bin_totals[i] for i in range(num_bins)]
    target_name = name.split('_')[-1]
    plt.plot([(i/num_bins)+0.025 for i in range(num_bins)], probs, label=target_name)
    leg = plt.legend(loc=0, prop={'size': 10}, title='Dataset')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.plot([0,1],[0,1], color='k', linestyle='dashed')
    plt.title('Calibration plot for MaxProb', fontsize=18)
    plt.xlabel('MaxProb', fontsize=14)
    plt.ylabel('Probability of correctness', fontsize=14)
    plt.savefig('{}_mp_pyx.png'.format(name), dpi=400)
    return




def ungrouped_histogram(source_dev_guid_list, target_dev_guid_list,\
                    source_maxprobs, target_maxprobs, \
                    args):
    colors = ['C0', 'C1']
    maxprobs = {'in-domain': [], 'OOD': []}
    maxprobs['in-domain'] = [source_maxprobs[guid] for guid in source_dev_guid_list]
    maxprobs['OOD'] = [target_maxprobs[guid] for guid in target_dev_guid_list]

    i = 0
    for dataset in ['in-domain', 'OOD']:
        #maxprobs[dataset].append(0.0001)
        T, x, _ = plt.hist(maxprobs[dataset], bins=10, histtype='step', density=False, linewidth=0.00001)
        bin_centers = np.array(0.5*(x[1:]+x[:-1]))
        bin_centers = [a+0.05 for a in bin_centers]
        T = [100*T[i]/sum(T) for i in range(len(T))]
        plt.plot(bin_centers, T, color=colors[i], label='{}'.format(dataset))
        print(bin_centers, T)
        plt.fill_between(bin_centers, T, facecolor=colors[i], alpha=0.2)
        i += 1

    leg = plt.legend(loc=1, prop={'size': 10}, title='Dataset')
    plt.xlabel('MaxProb', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    if args.task == 'qa':
        plt.ylim([0.0, 40.0])
        plt.title('Histogram of MaxProb \n for in-domain and OOD data', fontsize=18)
        plt.savefig('hist-test-mp.png'.format(args.target_prefix), dpi=400)


def ungrouped_histogram_cal(source_dev_guid_list, target_dev_guid_list,\
                    source_maxprobs, target_maxprobs, \
                    args):
    colors = ['C0', 'C1']
    maxprobs = {'in-domain': [], 'OOD': []}
    for guid in source_dev_guid_list:
        maxprobs['in-domain'].extend(source_maxprobs[guid])
    for guid in target_dev_guid_list:
        maxprobs['OOD'].extend(target_maxprobs[guid])

    i = 0
    for dataset in ['in-domain', 'OOD']:
        maxprobs[dataset].append(0.0001)
        T, x, _ = plt.hist(maxprobs[dataset], bins=10, histtype='step', density=False, linewidth=0.001)
        bin_centers = np.array(0.5*(x[1:]+x[:-1]))
        bin_centers = [a+0.05 for a in bin_centers]
        T = [100*T[i]/sum(T) for i in range(len(T))]
        print(bin_centers, T)
        plt.plot(bin_centers, T, color=colors[i], label='{}'.format(dataset))
        plt.fill_between(bin_centers, T, facecolor=colors[i], alpha=0.2)
        i += 1

    leg = plt.legend(loc=1, prop={'size': 10}, title='Dataset')
    plt.xlabel('Calibrator probability', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    if args.task == 'qa':
        plt.ylim([0.0, 40.0])
        plt.title('Histogram of Calibrator Probabilities \n for in-domain and OOD data', fontsize=18)
        plt.savefig('hist-test-all.png'.format(args.target_prefix), dpi=400)
        plt.clf()

def new_histogram(source_dev_guid_list, target_dev_guid_list,\
                    source_maxprobs, target_maxprobs, \
                    source_em_dict, target_em_dict, \
                    args):
    colors = ['C0', 'C1', 'C2', 'C3']
    maxprobs = {'source': {0: [], 1: []}, 'target': {0: [], 1: []}}
    for guid in source_dev_guid_list:
        maxprobs['source'][source_em_dict[guid]].append(source_maxprobs[guid])
    for guid in target_dev_guid_list:
        maxprobs['target'][target_em_dict[guid]].append(target_maxprobs[guid])
    for em in [0, 1]:
        maxprobs['source'][em] = sorted(maxprobs['source'][em])
        maxprobs['target'][em] = sorted(maxprobs['target'][em])

    i = 0
    for dataset in ['source', 'target']:
        for em in [0, 1]:
            T, x, _ = plt.hist(maxprobs[dataset][em], bins=10, histtype='step', density=False, linewidth=0.001)
            bin_centers = np.array(0.5*(x[1:]+x[:-1]))
            T = [100*T[i]/sum(T) for i in range(len(T))]
            plt.plot(bin_centers, T, color=colors[i], label='{}: EM={}'.format(dataset, em), linewidth=0.8)
            i += 1

    leg = plt.legend(loc=0, prop={'size': 8}, title='Dataset')
    plt.xlabel('Maxprob')
    plt.ylabel('Frequency')
    if args.task == 'qa':
        plt.ylim([0.0, 40.0])
        plt.title('Histogram of Maxprobs per Group for SQuAD+{}'.format(args.target_prefix))
        plt.savefig('histogram-{}.png'.format(args.target_prefix), dpi=400)


def histogram(sourceDataset, targetDataset):
    # You have QID: EM and Maxprob: EM for both source and target
    # You also have Dev GUID lists for both
    # Sort into the four different groups, bucket, graph
    # {source/target}_maxprobs = {em: []}
    maxprobs = {'source': {0: [], 1: []}, 'target': {0: [], 1: []}}
    for guid in sourceDataset.dev_guid_list:
        maxprobs['source'][sourceDataset.em_dict[guid]].append(sourceDataset.maxprobs[guid])
    for guid in targetDataset.dev_guid_list:
        maxprobs['target'][targetDataset.em_dict[guid]].append(targetDataset.maxprobs[guid])
    for em in [0, 1]:
        maxprobs['source'][em] = sorted(maxprobs['source'][em])
        maxprobs['target'][em] = sorted(maxprobs['target'][em])

    for dataset in ['source', 'target']:
        for em in [0, 1]:
            T, x, _ = plt.hist(maxprobs[dataset][em], bins=10, histtype='step', density=False, linewidth=0.01)
            bin_centers = np.array(0.5*(x[1:]+x[:-1]))
            T = [100*T[i]/sum(T) for i in range(len(T))]
            plt.plot(bin_centers, T, label='{}: EM={}'.format(dataset, em), linewidth=0.8)

    leg = plt.legend(loc=0, prop={'size': 8}, title='Dataset')
    plt.xlabel('Maxprob')
    plt.ylabel('Frequency')
    plt.ylim([0.0, 100.0])
    plt.title('Histogram of Maxprobs per Group')
    plt.savefig('histogram.png', dpi=400)


"""
Creating splits of MRQA data
"""

# Picking contexts whose questions add up to the requested number.
def greedy_subsample(dataset_prefix, dataset_name, dest_split, num_qids):
    
    if dataset_name == 'SQuAD' or dest_split == 'test':
        source_split = 'dev'
    else:
        source_split = 'train'

    # Read data, generate [(context_ID, num_Qs)]
    path = os.path.join(DATA_DIR, 'mrqa/{}/{}.jsonl'.format(source_split, dataset_name))
    context_list = []
    print("Reading data from {}...".format(path))
    with open(path, 'rb') as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            if i == 0 and 'header' in example:
                continue
            example['context_id'] = i-1
            context_list.append((i-1, len(example['qas'])))

    num_contexts = len(context_list)
    # If SQuAD, reserve the last half deterministically RANDOMLY for 4K test.
    if dataset_prefix == 'squad1.1':
        np.random.seed(24)
        np.random.shuffle(context_list)
        if dest_split != 'test':
            print("Selecting first half for non-test")
            context_list = context_list[:int(0.5*num_contexts)]
        else:
            print("Selecting second half for test")
            context_list = context_list[int(0.5*num_contexts):]

    num_contexts = len(context_list)
    # Make sure you're picking from disjoint
    # Sometimes performance on first half << second
    # So randomize it in a fixed way beforehand.
    np.random.seed(24)
    np.random.shuffle(context_list)
    # 80-20 split because same as actual sample ratio
    if dest_split == 'train':
        print("Selecting first 80% for train")
        context_list = context_list[:int(0.8*num_contexts)]
    elif dest_split == 'dev':
        print("Selecting last 20% for dev")
        context_list = context_list[int(0.8*num_contexts):]
    # else: split is test, and we don't want to partition.
    print("Number of contexts: {}".format(len(context_list)))
    fixed_context_list = copy.deepcopy(context_list)
    seed = -1
    if dest_split == 'test':
        max_seed = 0
    else:
        max_seed = 9
    name_seed = 0

    while seed < max_seed:
        seed += 1
        print("Seed: {}".format(seed))
        context_list = copy.deepcopy(fixed_context_list)
        np.random.seed(seed)
        # This is very expensive:
        np.random.shuffle(context_list)
        # Pick context IDs (greedily) that add up to num_qids
        total_qids = 0
        i = 0
        while total_qids + context_list[i][1] < num_qids:
            total_qids += context_list[i][1]
            i += 1
        selected_context_ids = [x[0] for x in context_list[:i]]
        # Now we're at position i in the context_list
        remaining_qids = num_qids - total_qids
        # Now we want to pick the contexts that will add up
        # to remaining_qids
        truth_table = [1 if x[1]==remaining_qids
                        else 0 for x in context_list[i:]]
        if sum(truth_table) > 0:
            selected_context_ids.append(
                        context_list[i + truth_table.index(1)][0])
        else:
            # It's okay if this happens - will keep going till 
            # it makes 10 splits
            print("List generated is shorter"
                    " than expected by {} questions".format(remaining_qids))
            max_seed += 1
            continue
        # Save corresponding data in output_path
        output_path = 'data_splits/{}_{}_split_{}.jsonl'.format(dataset_prefix, dest_split, name_seed)
        writer = open(output_path, 'wb')
        with open(path, 'rb') as f:
            for i, line in enumerate(f):
                context_id = i-1
                example = json.loads(line)
                new_example = copy.deepcopy(example)
                if 'header' in example:
                    new_example['header']['split'] = dest_split
                if not i == 0 or not 'header' in example:
                    if context_id not in selected_context_ids:
                        continue
                writer.write((json.dumps(new_example) + '\n').encode())
        name_seed += 1

    return



"""
Test time dropout utils
"""

def get_average_probs(args, eitherDataset, dataset_prefix):
    probs = {k: [] for k in eitherDataset.dev_guid_list}
    for i in tqdm(range(5)):
        nbest_path = os.path.join(args.model_dir, 'dropout_real', \
                        '{}_dropout{}-nbest_predictions.json'.format\
                        (dataset_prefix, i))
        if os.path.exists(nbest_path):
            nbest_data = read_nbest_data(nbest_path, \
                                        eitherDataset.dev_guid_list)
        else:
            nbest_path += 'l'
            nbest_data = read_nbest_data_from_long(nbest_path, \
                                        eitherDataset.dev_guid_list)
        """
        if 'squad1.1' in args.model_dir:
            nbest_path = os.path.join(args.model_dir, 'dropout_real', \
                        '{}_dropout{}-nbest_predictions.json'.format\
                        (dataset_prefix, i))
            nbest_data = read_nbest_data(nbest_path, \
                                        eitherDataset.dev_guid_list)
        else:
            nbest_path = os.path.join(args.model_dir, 'dropout_real', \
                        '{}_dropout{}-nbest_predictions.jsonl'.format\
                        (dataset_prefix, i))
            nbest_data = read_nbest_data_from_long(nbest_path, \
                                        eitherDataset.dev_guid_list)
        """
        for guid in eitherDataset.dev_guid_list:
            orig_pred = eitherDataset.preds[guid]
            flag = 0
            for nbest_item in nbest_data[guid]:
                if nbest_item['text'] == orig_pred:
                    probs[guid].append(nbest_item['probability'])
                    flag = 1
                    break
            if flag == 0:
                probs[guid].append(0)
    
    for guid, probs_list in probs.items():
        if args.ttdo_type == 'mean':
            probs[guid] = np.mean(probs_list)
        else:
            probs[guid] = -np.var(probs_list)
    return probs

def get_more_prob_stats(args, guid_list, preds_list, dataset_prefix, train_or_dev):
    if train_or_dev == 'train':
        dropout_dir = 'dropout_real_train'
        nbest_prefix = '{}_train'.format(dataset_prefix)
    else:
        dropout_dir = 'dropout_real'
        nbest_prefix = dataset_prefix
    probs = [{k: [] for k in guid_list} for i in range(4)]
    means = [{k: [] for k in guid_list} for i in range(4)]
    variances = [{k: [] for k in guid_list} for i in range(4)]

    for i in tqdm(range(30)):
        if train_or_dev == 'train':
            nbest_path = os.path.join(args.model_dir, dropout_dir, \
                        '{}_dropout{}-nbest_predictions.jsonl'.format\
                        (nbest_prefix, i))
            nbest_data = read_nbest_data_from_long(nbest_path, \
                                                guid_list)
        else:
            nbest_path = os.path.join(args.model_dir, dropout_dir, \
                        '{}_dropout{}-nbest_predictions.json'.format\
                            (nbest_prefix, i))
            nbest_data = read_nbest_data(nbest_path, \
                                                guid_list)
        for guid in guid_list:
            for j in range(4):
                preds = preds_list[j]
                orig_pred = preds[guid]
                flag = 0
                for nbest_item in nbest_data[guid]:
                    if nbest_item['text'] == orig_pred:
                        probs[j][guid].append(nbest_item['probability'])
                        flag = 1
                        break
                if flag == 0:
                    probs[j][guid].append(0)

    for j in range(4):
        for guid, probs_list in probs[j].items():
            means[j][guid] = np.mean(probs_list)
            variances[j][guid] = np.var(probs_list)
    return means, variances


def get_more_prob_stats_dev(args, guid_list, preds_list, dataset_prefix, train_or_dev):
    if True: #train_or_dev == 'train':
        dropout_dir = 'dropout_real_train'
        nbest_prefix = '{}_train'.format(dataset_prefix)
    else:
        dropout_dir = 'dropout_real'
        nbest_prefix = dataset_prefix
    probs = [{k: [] for k in guid_list} for i in range(5)]
    means = [{k: [] for k in guid_list} for i in range(5)]
    variances = [{k: [] for k in guid_list} for i in range(5)]

    for i in tqdm(range(30)):
        if True: #train_or_dev == 'train':
            nbest_path = os.path.join(args.model_dir, dropout_dir, \
                        '{}_dropout{}-nbest_predictions.jsonl'.format\
                        (nbest_prefix, i))
            nbest_data = read_nbest_data_from_long(nbest_path, \
                                                guid_list)
        else:
            nbest_path = os.path.join(args.model_dir, dropout_dir, \
                        '{}_dropout{}-nbest_predictions.json'.format\
                            (nbest_prefix, i))
            nbest_data = read_nbest_data(nbest_path, \
                                                guid_list)
        for guid in guid_list:
            for j in range(5):
                preds = preds_list[j]
                orig_pred = preds[guid]
                flag = 0
                for nbest_item in nbest_data[guid]:
                    if nbest_item['text'] == orig_pred:
                        probs[j][guid].append(nbest_item['probability'])
                        flag = 1
                        break
                if flag == 0:
                    probs[j][guid].append(0)

    for j in range(5):
        for guid, probs_list in probs[j].items():
            means[j][guid] = np.mean(probs_list)
            variances[j][guid] = np.var(probs_list)
    return means, variances


def get_more_prob_stats_test(args, guid_list, preds_list, dataset_prefix, train_or_dev):
    if dataset_prefix != 'squad1.1' and train_or_dev == 'train':
        dropout_dir = 'dropout_real'
        nbest_prefix = '{}_train'.format(dataset_prefix)
    else:
        dropout_dir = 'dropout_real'
        nbest_prefix = dataset_prefix
    probs = [{k: [] for k in guid_list} for i in range(5)]
    means = [{k: [] for k in guid_list} for i in range(5)]
    variances = [{k: [] for k in guid_list} for i in range(5)]

    for i in tqdm(range(5)):
        nbest_path = os.path.join(args.model_dir, dropout_dir, \
                        '{}_dropout{}-nbest_predictions.jsonl'.format\
                        (nbest_prefix, i))
        if os.path.exists(nbest_path):
            nbest_data = read_nbest_data_from_long(nbest_path, \
                                                guid_list)
        else:
            nbest_path = nbest_path[:-1]
            nbest_data = read_nbest_data(nbest_path, \
                                                guid_list)
        """
        if dataset_prefix != 'squad1.1' and train_or_dev == 'train':
            nbest_path = os.path.join(args.model_dir, dropout_dir, \
                        '{}_dropout{}-nbest_predictions.jsonl'.format\
                        (nbest_prefix, i))
            nbest_data = read_nbest_data_from_long(nbest_path, \
                                                guid_list)
        else:
            nbest_path = os.path.join(args.model_dir, dropout_dir, \
                        '{}_dropout{}-nbest_predictions.json'.format\
                            (nbest_prefix, i))
            nbest_data = read_nbest_data(nbest_path, \
                                                guid_list)
        """
        for guid in guid_list:
            for j in range(5):
                preds = preds_list[j]
                orig_pred = preds[guid]
                flag = 0
                for nbest_item in nbest_data[guid]:
                    if nbest_item['text'] == orig_pred:
                        probs[j][guid].append(nbest_item['probability'])
                        flag = 1
                        break
                if flag == 0:
                    probs[j][guid].append(0)

    for j in range(5):
        for guid, probs_list in probs[j].items():
            means[j][guid] = np.mean(probs_list)
            variances[j][guid] = np.var(probs_list)
    return means, variances

"""
Statistical significance

"""

def get_wilcoxon(maxprob_source_dict, maxprob_target_dict, cal_source_dict, cal_target_dict):
    # Remember multiple splits
    cal_source_probs = []
    cal_target_probs = []
    maxprob_source_probs = []
    maxprob_target_probs = []

    for k in cal_source_dict.keys():
        cal_source_probs.extend([float(x) for x in cal_source_dict[k]])
        maxprob_source_probs.extend([float(x) for x in maxprob_source_dict[k]])
    
    for k in cal_target_dict.keys():
        cal_target_probs.extend([float(x) for x in cal_target_dict[k]])
        maxprob_target_probs.extend([float(x) for x in maxprob_target_dict[k]])

    print("Wilcoxon of source: {}".format(wilcoxon(cal_source_probs, maxprob_source_probs)))
    
    print("Wilcoxon of target: {}".format(wilcoxon(cal_target_probs, maxprob_target_probs)))

    cal_source_probs.extend(cal_target_probs)
    maxprob_source_probs.extend(maxprob_target_probs)

    print("Wilcoxon of both: {}".format(wilcoxon(cal_source_probs, maxprob_source_probs)))

    return

"""
Error analysis

"""

def render_example(context, question, answers, pred, prob, guid):
    context = context.lower()
    answers = [normalize_answer(a) for a in answers]
    ans_keywords = []
    for a in answers:
        ans_keywords.extend(a.split())
    pred = normalize_answer(pred)
    pred_keywords = pred.split()
    question = normalize_answer(question)
    q_keywords = question.split()
    print()
    print("Context of {}".format(guid))
    for word in context.split():
        if normalize_answer(word) in ans_keywords:
            print(termcolor.colored(word, 'green'), end=" ")
        elif normalize_answer(word) in pred_keywords:
            print(termcolor.colored(word, 'yellow'), end=" ")
        elif normalize_answer(word) in q_keywords:
            print(termcolor.colored(word, 'blue'), end=" ")
        else:
            print(word, end=" ")
    print()
    print("Question: {}".format(termcolor.colored(question, 'blue')))
    print("Answers: {}".format(termcolor.colored(answers, 'green')))
    print("Prediction: {}".format(termcolor.colored(pred, 'yellow')))
    print()
    return

"""
Write output 

"""

def write_output(args, source_probs, target_probs, source_em, target_em, auc, cov80, cov90):
    model_name = args.model_dir.split('/')[-1]
    if model_name == "":
        model_name = args.model_dir.split('/')[-2]
    if args.mode == 'extrapolate':
        mode = 'calibrator'
        if args.ttdo_calibrator:
            mode += '_with_ttdo'
        if args.expose_prefix == None:
            path_prefix = '{}_{}_{}_{}'.format(model_name, mode,
                                    'None', args.target_prefix)
        else:
            path_prefix = '{}_{}_{}_{}'.format(model_name, mode,
                                    args.expose_prefix, args.target_prefix)
    elif args.mode == 'maxprob':
        path_prefix = '{}_{}_{}'.format(model_name, args.mode,
                                        args.target_prefix)
    else: # ttdo
        path_prefix = '{}_{}_{}_{}'.format(model_name, args.mode,
                                args.ttdo_type, args.target_prefix)

    if args.fraction_id != None:
        path_prefix += '_{}'.format(int(args.fraction_id*100))

    if args.ablate and args.mode == 'extrapolate':
        path_prefix += '_no_{}'.format(args.ablate)

    if args.strict_eval:
        path_prefix += '_strict'

    new_source_probs = {}
    new_target_probs = {}
    for k in source_probs.keys():
        new_source_probs[k] = source_probs[k] #np.mean(source_probs[k])
    for k in target_probs.keys():
        new_target_probs[k] = target_probs[k] #np.mean(target_probs[k])
    
    auc = round(auc*100, 2)
    results_dict = {'auc': auc, 'cov_at_acc_80': cov80,
                    'cov_at_acc_90': cov90}

    source_probs_path = os.path.join(args.output_dir, path_prefix+'_source_probs.json')
    target_probs_path = os.path.join(args.output_dir, path_prefix+'_target_probs.json')
    source_em_path = os.path.join(args.output_dir, path_prefix+'_source_em.json')
    target_em_path = os.path.join(args.output_dir, path_prefix+'_target_em.json')

    auc_path = os.path.join(args.output_dir, path_prefix+'_auc_cov.json')

    json.dump(new_source_probs, open(source_probs_path, 'w'))
    json.dump(new_target_probs, open(target_probs_path, 'w'))
    json.dump(source_em, open(source_em_path, 'w'))
    json.dump(target_em, open(target_em_path, 'w'))
    json.dump(results_dict, open(auc_path, 'w'))


def write_features_output(args, output_features):
    model_name = args.model_dir.split('/')[-1]
    if model_name == "":
        model_name = args.model_dir.split('/')[-2]
    if args.mode == 'extrapolate':
        mode = 'calibrator'
        if args.ttdo_calibrator:
            mode += '_with_ttdo'
        if args.expose_prefix == None:
            path_prefix = '{}_{}_{}_{}'.format(model_name, mode,
                                               'None', args.target_prefix)
        else:
            path_prefix = '{}_{}_{}_{}'.format(model_name, mode,
                                               args.expose_prefix, args.target_prefix)
    elif args.mode == 'maxprob':
        path_prefix = '{}_{}_{}'.format(model_name, args.mode,
                                        args.target_prefix)
    else:  # ttdo
        path_prefix = '{}_{}_{}_{}'.format(model_name, args.mode,
                                           args.ttdo_type, args.target_prefix)

    if args.fraction_id != None:
        path_prefix += '_{}'.format(int(args.fraction_id * 100))

    if args.ablate and args.mode == 'extrapolate':
        path_prefix += '_no_{}'.format(args.ablate)

    if args.strict_eval:
        path_prefix += '_strict'

    # new_source_probs = {}
    # new_target_probs = {}
    # for k in source_probs.keys():
    #     new_source_probs[k] = source_probs[k]  # np.mean(source_probs[k])
    # for k in target_probs.keys():
    #     new_target_probs[k] = target_probs[k]  # np.mean(target_probs[k])
    #
    # auc = round(auc * 100, 2)
    # results_dict = {'auc': auc, 'cov_at_acc_80': cov80,
    #                 'cov_at_acc_90': cov90}
    #
    # source_probs_path = os.path.join(args.output_dir, path_prefix + '_source_probs.json')
    # target_probs_path = os.path.join(args.output_dir, path_prefix + '_target_probs.json')
    # source_em_path = os.path.join(args.output_dir, path_prefix + '_source_em.json')
    # target_em_path = os.path.join(args.output_dir, path_prefix + '_target_em.json')
    # auc_path = os.path.join(args.output_dir, path_prefix + '_auc_cov.json')

    context_len_features_path = os.path.join(args.output_dir, path_prefix + '_context_len.json')
    answer_entropy_features_path = os.path.join(args.output_dir, path_prefix + '_answer_entropy.json')
    softmax_temp_features_path = os.path.join(args.output_dir, path_prefix + '_softmax_temp.json')
    max_features_path = os.path.join(args.output_dir, path_prefix + '_max_features.json')
    other_features_path = os.path.join(args.output_dir, path_prefix + '_other_features.json')
    pred_len_features_path = os.path.join(args.output_dir, path_prefix + '_pred_len_features.json')
    ttdo_mean_features_path = os.path.join(args.output_dir, path_prefix + '_ttdo_mean_features.json')
    ttdo_var_features_path = os.path.join(args.output_dir, path_prefix + '_ttdo_var_features.json')





    # json.dump(new_source_probs, open(source_probs_path, 'w'))
    # json.dump(new_target_probs, open(target_probs_path, 'w'))
    # json.dump(results_dict, open(auc_path, 'w'))
    # print('features', output_features)
    # print('len', len(output_features))
    json.dump(output_features[0], open(context_len_features_path, 'w'))
    json.dump(output_features[1], open(answer_entropy_features_path, 'w'))
    json.dump(output_features[2], open(softmax_temp_features_path, 'w'))
    json.dump(output_features[3], open(max_features_path, 'w'))
    json.dump(output_features[4], open(other_features_path, 'w'))
    json.dump(output_features[5], open(pred_len_features_path, 'w'))
    json.dump(output_features[6], open(ttdo_mean_features_path, 'w'))
    json.dump(output_features[7], open(ttdo_var_features_path, 'w'))