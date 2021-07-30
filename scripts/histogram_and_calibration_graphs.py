"""
Get Histogram and Calibration graphs for MaxProb.

"""

import argparse
import pdb
import os
import utils
import numpy as np
import json
import matplotlib.pyplot as plt
from math import ceil


def initialize_args():
    parser = argparse.ArgumentParser(description='MaxProb graphing')
    parser.add_argument('mode', type=str, default=None,
                        choices=['calibrator', 'maxprob'],
                        help='What to graph')
    parser.add_argument('results_dir', type=str, default=None,
                        help='Path to results directory')
    parser.add_argument('model_name', type=str, default=None,
                        help='Model name that generated results, '
                                'i.e. first segment of output names')
    parser.add_argument('output_filename_hist', type=str, default=None,
                        help='Output filename for histogram graph')
    parser.add_argument('output_filename_cal', type=str, default=None,
                        help='Output filename for calibration graph')
    parser.add_argument('--strict', action='store_true',
                        help='Strict evaluation?')
    args = parser.parse_args()
    return args


def get_histogram_graph(args, datasets, probabilities, name):
    colors = ['C0', 'C1']
    i = 0
    for dataset in ['in-domain', 'OOD']:
        if type(list(probabilities[dataset].values())[0]) == list:
            probs = []
            for p_list in list(probabilities[dataset].values()):
                probs.extend(p_list)
        else:
            probs = list(probabilities[dataset].values())
        
        probs.extend([0.00001]) # to align the bins, TODO
        T, x, _ = plt.hist(probs, bins=10, histtype='step', density=False, linewidth=0.00001)
        bin_centers = np.array(0.5*(x[1:]+x[:-1]))
        #bin_centers = [a+0.05 for a in bin_centers]
        T = [100*T[i]/sum(T) for i in range(len(T))]
        new_bin_centers = [0]
        new_bin_centers.extend(bin_centers)
        new_bin_centers.append(1)
        bin_centers = new_bin_centers
        new_T = [T[0]]
        new_T.extend(T)
        new_T.append(T[-1])
        T = new_T
        
        plt.plot(bin_centers, T, color=colors[i], label='{}'.format(dataset))
        plt.fill_between(bin_centers, T, facecolor=colors[i], alpha=0.2)
        i += 1

    leg = plt.legend(loc=2, prop={'size': 14})#, title='Dataset')
    plt.xlabel(name, fontsize=17)
    plt.ylabel('Frequency', fontsize=18)
    plt.ylim([0.0, 45.0])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Histogram of {} \n for in-domain and OOD data'.format(name), fontsize=18)
    plt.savefig(args.output_filename_hist, dpi=400)


def get_calibration_graph(args, datasets, probabilities, correctness, name):
    # Bins should be maxprob ranges, not frequency
    # i.e. equal width, not equal depth.
    num_bins = 20
    bin_correct = [0 for i in range(num_bins)]
    bin_totals = [0 for i in range(num_bins)]
    colors = ['C0', 'C1']
    i = 0
    for dataset in ['in-domain', 'OOD']:
        for guid, prob in probabilities[dataset].items():
            em = correctness[dataset][guid]
            if type(prob) == list:
                for p in prob:
                    bin_index = ceil(p * num_bins) - 1
                    bin_totals[bin_index] += 1
                    if em:
                        bin_correct[bin_index] += 1
            else:
                bin_index = ceil(prob * num_bins) - 1
                bin_totals[bin_index] += 1
                if em:
                    bin_correct[bin_index] += 1
        # Make sure there are no empty bins (prevent div-by-0)
        for i in range(num_bins):
            if bin_totals[i] == 0:
                bin_totals[i] = 1
        probs = [bin_correct[i]/bin_totals[i] for i in range(num_bins)]
        plt.plot([(i/num_bins)+0.025 for i in range(num_bins)], probs, label=dataset)
    
    leg = plt.legend(loc=2, prop={'size': 14})#, title='Dataset')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.plot([0,1],[0,1], color='k', linestyle='dashed')
    plt.title('Calibration plot for {}'.format(name), fontsize=18)
    plt.xlabel(name, fontsize=17)
    plt.ylabel('Probability of correctness', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(args.output_filename_cal, dpi=400)


def main():
    args = initialize_args()
    datasets = ['triviaqa', 'hotpotqa', 'newsqa', 'nq', 'searchqa']

    probabilities = {'in-domain': {}, 'OOD': {}}
    correctness = {'in-domain': {}, 'OOD': {}}

    if args.mode == 'calibrator':
        name = 'Calibrator Probability'
        for expose_dataset in datasets:
            for target_dataset in datasets:
                if expose_dataset == target_dataset:
                    continue
                if args.strict:
                    target_dataset += '_strict'
                try:
                    probabilities['in-domain'].update(json.load(open(os.path.join(args.results_dir, '{}_calibrator_{}_{}_source_probs.json'.format(args.model_name, expose_dataset, target_dataset)))))
                    probabilities['OOD'].update(json.load(open(os.path.join(args.results_dir, '{}_calibrator_{}_{}_target_probs.json'.format(args.model_name, expose_dataset, target_dataset)))))
                    correctness['in-domain'].update(json.load(open(os.path.join(args.results_dir, '{}_calibrator_{}_{}_source_em.json'.format(args.model_name, expose_dataset, target_dataset)))))
                    correctness['OOD'].update(json.load(open(os.path.join(args.results_dir, '{}_calibrator_{}_{}_target_em.json'.format(args.model_name, expose_dataset, target_dataset)))))
                except:
                    print("Couldn't find files needed. Check that all results were generated.")

    else:
        name = 'MaxProb'
        for dataset in datasets:
            if args.strict:
                dataset += '_strict'
            try:
                probabilities['in-domain'].update(json.load(open(os.path.join(args.results_dir, '{}_maxprob_{}_source_probs.json'.format(args.model_name, dataset)))))
                probabilities['OOD'].update(json.load(open(os.path.join(args.results_dir, '{}_maxprob_{}_target_probs.json'.format(args.model_name, dataset)))))
                correctness['in-domain'].update(json.load(open(os.path.join(args.results_dir, '{}_maxprob_{}_source_em.json'.format(args.model_name, dataset)))))
                correctness['OOD'].update(json.load(open(os.path.join(args.results_dir, '{}_maxprob_{}_target_em.json'.format(args.model_name, dataset)))))
            except:
                print("Couldn't find files needed. Check that all results were generated.")


    get_histogram_graph(args, datasets, probabilities, name)
    plt.clf()
    get_calibration_graph(args, datasets, probabilities, correctness, name)


if __name__ == '__main__':
    main()


