"""
Get p-value of Calibrator versus MaxProb
using paired bootstrap test.

"""

import argparse
import pdb
import utils
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
    parser.add_argument('task', type=str, default=None,
                        choices=['nli', 'qa'], help='Task type')
    parser.add_argument('mode', type=str, default=None,
                        choices=['statistical'],
                        help='Mode of execution')
    parser.add_argument('--num_trials', type=int, default=1000,
                        help='Number of trials of paired bootstrap test')
    parser.add_argument('--classifier', type=str, default='random_forest',
                        choices = ['random_forest'],
                        help='Classifier type for calibrator')
    parser.add_argument('--reg', type=float, default=6.0,
                        help='Regularization value for classifier')
    parser.add_argument('--reg2', type=int, default=175,
                        help='num estimators')
    parser.add_argument('--ttdo_calibrator', action='store_true',
                        help='Use TTDO features in calibrator')
    parser.add_argument('--ttdo_type', type=str, default='mean',
                        choices=['mean', 'neg_var'], help='TTDO mean or -var?')
    parser.add_argument('--fraction_id', type=float, default=None,
                        help='fraction of D_cal/test that is in-domain')
    parser.add_argument('--ablate', type=str, default=None,
                            choices=['maxprob', 'other_prob', 'all_prob',
                                        'context_len', 'pred_len'],
                            help='Which feature(s) to ablate')
    parser.add_argument('--strict_eval', action='store_true',
                        help='Only consider one ans option correct')
    args = parser.parse_args()
    return args


def maxprob(args, sourceDataset, targetDataset):
    maxprobs_list = np.asarray([sourceDataset.maxprobs[guid] \
                                for guid in sourceDataset.dev_guid_list] + \
                                [targetDataset.maxprobs[guid] \
                                for guid in targetDataset.dev_guid_list])
    em_list = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                        [targetDataset.em_dict[guid] \
                            for guid in targetDataset.dev_guid_list])
    risks, coverages, auc, sorted_sources, sorted_em = utils.get_risk_coverage_info(maxprobs_list, em_list)
    return risks, coverages, auc, sorted_sources, sorted_em


def extrapolate(args, sourceDataset, exposeDataset, targetDataset):
    source_features = sourceDataset.generate_features(args, 'train')
    source_features.update(sourceDataset.generate_features(args, 'dev'))
    if exposeDataset:
        expose_features = exposeDataset.generate_features(args, 'train')
    target_features = targetDataset.generate_features(args, 'dev')

    # Train on source + expose
    x_train = np.asarray([source_features[guid] \
                            for guid in sourceDataset.train_guid_list])
    y_train = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.train_guid_list])
    if exposeDataset:
        x_train = np.concatenate((x_train, np.asarray([expose_features[guid] \
                                for guid in exposeDataset.train_guid_list])))
        y_train = np.concatenate((y_train,
                                np.asarray([exposeDataset.em_dict[guid] \
                                for guid in exposeDataset.train_guid_list])))

    # Test on source + target
    x_dev = np.asarray([source_features[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                        [target_features[guid] \
                            for guid in targetDataset.dev_guid_list])
    y_dev = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                         [targetDataset.em_dict[guid] \
                            for guid in targetDataset.dev_guid_list])
    risks, coverages, auc, sorted_sources, sorted_em = classifier(args, x_train, y_train, x_dev, y_dev, len(sourceDataset.dev_guid_list), list(sourceDataset.dev_guid_list), list(targetDataset.dev_guid_list))
    return risks, coverages, auc, sorted_sources, sorted_em


def classifier(args, x_train, y_train, x_dev, y_dev, len_source=0, source_dev_guid_list=None, target_dev_guid_list=None):
    clf = RandomForestClassifier(n_estimators=args.reg2, \
                                     max_depth=args.reg, random_state=0)
    clf.fit(x_train, y_train)
    train_score = clf.score(x_train, y_train)
    dev_score = clf.score(x_dev, y_dev)
    probs = [p[1] for p in clf.predict_proba(x_dev)]
    risks, coverages, auc, sorted_sources, sorted_em = utils.get_risk_coverage_info(probs, y_dev)
    global CAL_SOURCE_PROBS, CAL_TARGET_PROBS
    i, j = 0, 0
    while i < len(source_dev_guid_list):
        if source_dev_guid_list[i] in CAL_SOURCE_PROBS:
            CAL_SOURCE_PROBS[source_dev_guid_list[i]].append(probs[i])
        else:
            CAL_SOURCE_PROBS[source_dev_guid_list[i]] = [probs[i]]
        i += 1
    while j < len(target_dev_guid_list):
        if target_dev_guid_list[j] in CAL_TARGET_PROBS:
            CAL_TARGET_PROBS[target_dev_guid_list[j]].append(probs[i])
        else:
            CAL_TARGET_PROBS[target_dev_guid_list[j]] = [probs[i]]
        i += 1
        j += 1
    return risks, coverages, auc, sorted_sources, sorted_em


def get_maxprob_results(args):
    # Do the things
    results_dict = {}
    sourceDataset = QaDataset(args, 'squad1.1', 0)
    targetDataset = QaDataset(args, args.target_prefix, 0)
    source_probs = { g: sourceDataset.maxprobs[g] \
                        for g in sourceDataset.dev_guid_list }
    results_dict['source_probs'] = source_probs
    target_probs = { g: targetDataset.maxprobs[g] \
                        for g in targetDataset.dev_guid_list }
    results_dict['target_probs'] = target_probs
    results_dict['source_em'] = sourceDataset.em_dict
    results_dict['target_em'] = targetDataset.em_dict
    return results_dict


def get_calibrator_results(args):
    # Do the things
    results_dict = {}
    global CAL_SOURCE_PROBS, CAL_TARGET_PROBS
    CAL_SOURCE_PROBS = {}
    CAL_TARGET_PROBS = {}
    source_em = {}
    target_em = {}
    for split_no in range(10):
        sourceDataset = QaDataset(args, 'squad1.1', split_no)
        if args.expose_prefix:
            exposeDataset = QaDataset(args, args.expose_prefix, split_no)
        else:
            exposeDataset = None
        targetDataset = QaDataset(args, args.target_prefix, split_no)
        source_em.update(sourceDataset.em_dict)
        target_em.update(targetDataset.em_dict)
        _, _, _, _, _ = extrapolate(args, sourceDataset, \
                                        exposeDataset, targetDataset)

    source_probs = { g: np.mean(CAL_SOURCE_PROBS[g]) \
                        for g in CAL_SOURCE_PROBS.keys() }
    target_probs = { g: np.mean(CAL_TARGET_PROBS[g]) \
                        for g in CAL_TARGET_PROBS.keys() }
    
    results_dict['source_probs'] = source_probs
    results_dict['target_probs'] = target_probs
    results_dict['source_em'] = sourceDataset.em_dict
    results_dict['target_em'] = targetDataset.em_dict
    return results_dict



def main():
    args = initialize_args()
    datasets = ['triviaqa', 'hotpotqa', 'newsqa', 'nq', 'searchqa']
    calibrator_wins = 0
    mrqa_model_names = [ args.triviaqa_model_dir, args.hotpotqa_model_dir,
                         args.newsqa_model_dir, args.nq_model_dir,
                         args.searchqa_model_dir]

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
            args.target_prefix = target_prefix
            args.expose_prefix = expose_prefix
            args.model_dir = args.squad_model_dir
            args.mode = 'extrapolate'
            # Get calibrator results
            grand_calibrator_dict[expose_prefix][target_prefix].update(
                    get_calibrator_results(args))
            args.expose_prefix = None
            args.model_dir = mrqa_model_names[expose_index]
            args.mode = 'maxprob'
            # Get maxprob results
            grand_maxprob_dict[expose_prefix][target_prefix].update(
                    get_maxprob_results(args))

    sum_calibrator_auc = 0
    sum_maxprob_auc = 0

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

    p_value = (args.num_trials - calibrator_wins) / args.num_trials
    print()
    print("The calibrator wins {} of {} times.".format(calibrator_wins, args.num_trials))
    print("p-value = {}".format(p_value))
    print("Average calibrator AUC is {}".format(sum_calibrator_auc/args.num_trials))
    print("Average maxprob AUC is {}".format(sum_maxprob_auc/args.num_trials))
    print()

if __name__ == '__main__':
    main()


