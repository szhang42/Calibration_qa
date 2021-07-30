"""
Get confidence intervals of Calibrator versus MaxProb
using paired bootstrap test.

"""

import argparse
import pdb
import utils
import json
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from datasets import QaDataset

CAL_SOURCE_PROBS = {}
CAL_TARGET_PROBS = {}


def initialize_args():
    parser = argparse.ArgumentParser(description='Statistical significance')
    parser.add_argument('squad_model_dir', type=str, default=None,
                        help='Path to SQuAD model')
    parser.add_argument('aug_model_dir', type=str, default=None,
                        help='Path to Aug models')
    """
    parser.add_argument('squad_model_dir', type=str, default=None,
                        help='Path to SQuAD model')
    parser.add_argument('triviaqa_model_dir', type=str, default=None,
                        help='Path to SQuAD+TriviaQA model')
    parser.add_argument('hotpotqa_model_dir', type=str, default=None,
                        help='Path to SQuAD+HotpotQA model')
    parser.add_argument('newsqa_model_dir', type=str, default=None,
                        help='Path to SQuAD+NewsQA model')
    parser.add_argument('nq_model_dir', type=str, default=None,
                        help='Path to SQuAD+NaturalQuestions model')
    parser.add_argument('searchqa_model_dir', type=str, default=None,
                        help='Path to SQuAD+SearchQA model')
    """
    parser.add_argument('--num_trials', type=int, default=1000,
                        help='Number of trials of paired bootstrap test')
    args = parser.parse_args()
    return args



def get_calibrator_results(args, expose_dataset, target_dataset):
    results = {}
    results['source_probs'] = json.load(open('{}/{}_calibrator_{}_{}_source_probs.json'.format(args.squad_model_dir, args.model_name, expose_dataset, target_dataset)))
    for k, v in results['source_probs'].items():
        if type(v) == list:
            results['source_probs'][k] = np.mean(v)
    results['target_probs'] = json.load(open('{}/{}_calibrator_{}_{}_target_probs.json'.format(args.squad_model_dir, args.model_name, expose_dataset, target_dataset)))
    for k, v in results['target_probs'].items():
        if type(v) == list:
            results['target_probs'][k] = np.mean(v)
    results['source_em'] = json.load(open('{}/{}_calibrator_{}_{}_source_em.json'.format(args.squad_model_dir, args.model_name, expose_dataset, target_dataset)))
    results['target_em'] = json.load(open('{}/{}_calibrator_{}_{}_target_em.json'.format(args.squad_model_dir, args.model_name, expose_dataset, target_dataset)))
    return results


def get_maxprob_results(args, target_dataset):
    results = {}
    results['source_probs'] = json.load(open('{}/{}_maxprob_{}_source_probs.json'.format(args.aug_model_dir, args.model_name, target_dataset)))
    results['target_probs'] = json.load(open('{}/{}_maxprob_{}_target_probs.json'.format(args.aug_model_dir, args.model_name, target_dataset)))
    for k, v in results['source_probs'].items():
        if type(v) == list:
            results['source_probs'][k] = np.mean(v)
    for k, v in results['target_probs'].items():
        if type(v) == list:
            results['target_probs'][k] = np.mean(v)

    results['source_em'] = json.load(open('{}/{}_maxprob_{}_source_em.json'.format(args.aug_model_dir, args.model_name, target_dataset)))
    results['target_em'] = json.load(open('{}/{}_maxprob_{}_target_em.json'.format(args.aug_model_dir, args.model_name, target_dataset)))
    return results


def main():
    args = initialize_args()
    datasets = ['triviaqa', 'hotpotqa', 'newsqa', 'nq', 'searchqa']
    calibrator_wins = 0

    grand_calibrator_dict = {d: {d2: {'source_probs': {}, 
                                      'target_probs': {}, 
                                      'source_em': {},
                                      'target_em': {} } 
                                         for d2 in datasets} 
                                            for d in datasets}


    grand_maxprob_dict = {d: {d2: {'source_probs': {},
                                      'target_probs': {},
                                      'source_em': {},
                                      'target_em': {} }
                                         for d2 in datasets}
                                            for d in datasets}
    # Get test sample indices
    for expose_index, expose_prefix in enumerate(tqdm(datasets)):
        for target_index, target_prefix in enumerate(datasets):
            if target_index == expose_index:
                continue
            # Get calibrator results
            args.model_name = 'squad1.1'
            grand_calibrator_dict[expose_prefix][target_prefix] = \
                get_calibrator_results(args, expose_prefix, target_prefix)
            args.model_name = 'squad1.1_{}_2K'.format(expose_prefix)
            # Get maxprob results
            grand_maxprob_dict[expose_prefix][target_prefix] = \
                get_maxprob_results(args, target_prefix)


    sum_calibrator_auc = 0
    sum_maxprob_auc = 0

    all_avg_cal = []
    all_avg_mp = []

    for trial_no in tqdm(range(args.num_trials)):
        np.random.seed(trial_no)
        chosen_indices = np.random.choice(list(range(4000)), 4000, replace=True)
        calibrator_aucs = []
        maxprob_aucs = []
        for expose_index, expose_prefix in enumerate(datasets):
            for target_index, target_prefix in enumerate(datasets):
                if target_index == expose_index:
                    continue
                calibrator_dict = grand_calibrator_dict[expose_prefix][target_prefix]
                maxprob_dict = grand_maxprob_dict[expose_prefix][target_prefix]
                chosen_source_indices = [list(calibrator_dict['source_probs'].keys())[i] for i in chosen_indices]
                chosen_target_indices = [list(calibrator_dict['target_probs'].keys())[i] for i in chosen_indices]             
                # Check if it's the same as maxprob_dict equivalent
                # Note that EMs aren't the same
                calibrator_em_list = [calibrator_dict['source_em'][g] for g in chosen_source_indices]
                calibrator_em_list.extend([calibrator_dict['target_em'][g] for g in chosen_target_indices])
                calibrator_prob_list = [calibrator_dict['source_probs'][g] for g in chosen_source_indices]
                calibrator_prob_list.extend([calibrator_dict['target_probs'][g] for g in chosen_target_indices])
                _, _, calibrator_auc, _, _ = utils.get_risk_coverage_info(calibrator_prob_list, calibrator_em_list)
                calibrator_aucs.append(calibrator_auc)

                maxprob_em_list = [maxprob_dict['source_em'][g] for g in chosen_source_indices]
                maxprob_em_list.extend([maxprob_dict['target_em'][g] for g in chosen_target_indices])
                maxprob_prob_list = [maxprob_dict['source_probs'][g] for g in chosen_source_indices]
                maxprob_prob_list.extend([maxprob_dict['target_probs'][g] for g in chosen_target_indices])
                _, _, maxprob_auc, _, _ = utils.get_risk_coverage_info(maxprob_prob_list, maxprob_em_list)
                maxprob_aucs.append(maxprob_auc)

        # Take averages across all combinations
        calibrator_avg_auc = round(np.mean(calibrator_aucs)*100, 2)
        maxprob_avg_auc = round(np.mean(maxprob_aucs)*100, 2)
        if calibrator_avg_auc <= maxprob_avg_auc:
            calibrator_wins += 1
        sum_calibrator_auc += calibrator_avg_auc
        sum_maxprob_auc += maxprob_avg_auc
        all_avg_cal.append(calibrator_avg_auc)
        all_avg_mp.append(maxprob_avg_auc)

    pdb.set_trace()
    shave = int(0.025*args.num_trials)
    
    cal_range_95 = sorted(all_avg_cal)[shave:(args.num_trials-shave)]
    mp_range_95 = sorted(all_avg_mp)[shave:(args.num_trials-shave)]
    diffs = [all_avg_mp[x]-all_avg_cal[x] for x in range(args.num_trials)]
    diffs_range_95 = sorted(diffs)[shave:(args.num_trials-shave)]
    print(diffs_range_95[0])
    print(diffs_range_95[-1])
    
    p_value = (args.num_trials - calibrator_wins) / args.num_trials
    print()
    print("The calibrator wins {} of {} times.".format(calibrator_wins, args.num_trials))
    print("p-value = {}".format(p_value))
    print("Average calibrator AUC is {}".format(sum_calibrator_auc/args.num_trials))
    print("Average maxprob AUC is {}".format(sum_maxprob_auc/args.num_trials))
    print()

if __name__ == '__main__':
    main()


