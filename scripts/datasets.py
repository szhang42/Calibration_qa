"""
Dataset classes.

"""

import os
import pdb
import utils
import numpy as np
import random
import math


class Dataset(object):
    def _calculate_maxprobs(self):
        """
        Calculate the maxprobs from nbest data.
        Args:
            None
        Returns:
            a dict of {QID: maxprob}
        """
        raise NotImplementedError

    def _generate_predictions(self):
        """
        Generate predictions from nbest data.
        Predictions are non-null.
        Args:
            None
        Returns:
            a dict of {QID: "pred"}
        """
        raise NotImplementedError

    def _calculate_em(self):
        """
        Calculate the EM dictionary from nbest data.
        Args:
            None
        Returns:
            a dict of {QID: 1[EM]}
        """
        raise NotImplementedError

    def get_features(self):
        """
        Generate dataset features.
        Args:
            None
        Returns:
            A dict of {QID: [features]}
        """
        raise NotImplementedError

    def precompute_all(self, nbest_flag=True, \
                        substring_flag=False, hidden_flag=False):
        """
        Precompute nbest, substring and hidden
        features of the model for the dataset,
        per flags.
        Stores JSONs in self.model_dir.
        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

class QaDataset(Dataset):
    def __init__(self, args, dataset_prefix, split_no):
        gold_data_train_path = \
            'data_splits/{}_train_split_{}.jsonl'.format(\
            dataset_prefix, split_no)
        self.gold_data = {'train': {}, 'dev': {}}

        if args.mode not in ['minimum', 'maxprob', 'maxprob_squad_only', 'ttdo']:
            self.gold_data['train'] = utils.read_gold_data(args.task, gold_data_train_path,\
                dataset_prefix)

        if args.mode == 'extrapolate' and not args.expose_prefix \
            and dataset_prefix=='squad1.1':
            # Need to train calibrator on double
            # the usual amount of SQuAD examples
            # one split already read, above.
            self._get_double_squad(args, dataset_prefix, split_no)

        gold_data_dev_path = \
            'data_splits/{}_test_split_{}.jsonl'.format(\
            dataset_prefix, 0)
        self.gold_data['dev'] = utils.read_gold_data(args.task, gold_data_dev_path,\
            dataset_prefix)

        percentage = 1.0
        if args.fraction_id:
            if 'squad' in dataset_prefix: 
                percentage = args.fraction_id
            else:
                percentage = 1 - args.fraction_id

        num_train = int(percentage * len(self.gold_data['train'].keys()))
        
        self.train_guid_list = list(self.gold_data['train'].keys())[:num_train]

        if dataset_prefix == 'squad2.0':
            np.random.seed(42)
            self.dev_guid_list = np.random.choice(list(self.gold_data['dev'].keys()), 4000, replace=False)
            unanswerable_guids = [guid for guid in self.gold_data['dev'].keys() if self.gold_data['dev'][guid]['answers'][0]==""]
            #print("Fraction of unanswerable questions in the entire dataset = {}".format(float(len(unanswerable_guids) / len(self.gold_data['dev'].keys()))))
            unanswerable_guids = [guid for guid in unanswerable_guids if guid in self.dev_guid_list]
            #print("Fraction of unanswerable questions in the selected dev dataset = {}".format(float(len(unanswerable_guids) / len(self.dev_guid_list))))
            # If you only want to evaluate on unanswerable questions:
            # unanswerable_guids = [guid for guid in self.gold_data['dev'].keys() if self.gold_data['dev'][guid]['answers'][0]==""]
            # assert len(unanswerable_guids) >= 4000
            # self.dev_guid_list = np.random.choice(unanswerable_guids, 4000, replace=False)
        else:
            num_dev = int(percentage * len(self.gold_data['dev'].keys()))
            self.dev_guid_list = list(self.gold_data['dev'].keys())[:num_dev]

        nbest_path = os.path.join(args.model_dir, \
                                    '{}-nbest_predictions.json'.format\
                                    (dataset_prefix))
        try:
            self.nbest_data = utils.read_nbest_data(nbest_path, self.dev_guid_list)
        except:
            self.nbest_data = utils.read_nbest_data_from_long(nbest_path+'l', self.dev_guid_list)
        
        if args.mode in ['minimum', 'maxprob', 'maxprob_squad_only', 'ttdo']:
            pass
        elif dataset_prefix != 'squad1.1' and dataset_prefix != 'squad2.0':
            nbest_path = os.path.join(args.model_dir, \
                                    '{}_train-nbest_predictions.jsonl'.format\
                                    (dataset_prefix))
            self.nbest_data.update(utils.read_nbest_data_from_long(nbest_path, \
                                        self.train_guid_list))
        elif dataset_prefix != 'squad2.0':
            try:
                self.nbest_data.update(utils.read_nbest_data(nbest_path, self.train_guid_list))
            except:
                self.nbest_data.update(utils.read_nbest_data_from_long(nbest_path+'l', self.train_guid_list))

        # call calc maxprob
        self.maxprobs = self._calc_maxprobs()
        self.second_maxprobs = self._calc_maxprobs(2)
        self.third_maxprobs = self._calc_maxprobs(3)
        self.fourth_maxprobs = self._calc_maxprobs(4)
        self.fifth_maxprobs = self._calc_maxprobs(5)

        self.temp_maxprobs = self._calc_softmax_temp_maxprob()
        self.answer_entropy= self._calc_answer_entropy()
        self.dataset_prefix = dataset_prefix

        # call gen preds
        self.preds = self._generate_predictions()
        if args.ttdo_calibrator:
            #top 5 probability
            self.more_preds = [self._generate_more_predictions(i) for i in range(0,5)]
        # call calc EM
        if args.strict_eval:
            self.em_dict = self._calc_em_strict('train')
            self.em_dict.update(self._calc_em_strict('dev'))
        else:
            self.em_dict = self._calc_em('train')
            self.em_dict.update(self._calc_em('dev'))

    def _calc_maxprobs(self, i=1):
        maxprobs = {}
        for k, v in self.nbest_data.items():
            try:
                if v[0]['text'] != "":
                    maxprobs[k] = float(v[0+i-1]['probability'])
                else:
                    maxprobs[k] = float(v[1+i-1]['probability'])
            except:
                maxprobs[k] = 0
        return maxprobs


    def _calc_answer_entropy(self, i=1):
        answer_entropy = {}
        for k, v in self.nbest_data.items():
            list_ans_entr = []
            # try:
            prob_list = []
            for a in v[:5]:
                if a['text'] != "":
                    prob = float(a['probability'])
                    prob_list.append(prob)
                    ans_entr = prob * np.log(prob)
                    list_ans_entr.append(ans_entr)
                else:
                    pass

            prob_remain= 1 - sum(prob_list)
            ans_entr = prob_remain * np.log(prob_remain + 1e-10)
            list_ans_entr.append(ans_entr)
            answer_entropy[k] = float((sum(list_ans_entr))*(-1.0))
            # except:
            #     answer_entropy[k] = 0
        # print('lolololo', answer_entropy)
        return answer_entropy

    def _calc_softmax_temp_maxprob(self, i=1):

        # total_scores = []
        # best_non_null_entry = None
        # for entry in nbest:
        #     total_scores.append(entry.start_logit + entry.end_logit)
        #     if not best_non_null_entry:
        #         if entry.text:
        #             best_non_null_entry = entry
        #
        # max_score = None
        # for score in scores:
        #     if max_score is None or score > max_score:
        #         max_score = score
        #
        # exp_scores = []
        # total_sum = 0.0
        # for score in scores:
        #     x = math.exp(score - max_score)
        #     exp_scores.append(x)
        #     total_sum += x
        #
        # probs = []
        # for score in exp_scores:
        #     probs.append(score / total_sum)

        temp_maxprobs = {}
        topk = 5
        for k, v in self.nbest_data.items():
            probs = []
            max_logit = None
            exp_scores = []
            for a in v:
                if a['text'] != "":
                    start_logit = float(a['start_logit'])
                    end_logit = float(a['end_logit'])
                    final_logit = start_logit + end_logit
                    if max_logit is None or final_logit > max_logit:
                        max_logit = final_logit
                    # total_sum += final_logit
            for a in v:
                if a['text'] != "":
                    start_logit = float(a['start_logit'])
                    end_logit = float(a['end_logit'])
                    final_logit = start_logit + end_logit
                    x = math.exp(final_logit/1.15 - max_logit)
                    exp_scores.append(x)

            norm_term = sum(exp_scores)
            for score in exp_scores[:topk]:
                probs.append(score / norm_term)
            temp_maxprobs[k] = probs

        # def softmax(x):
        #     """Compute softmax values for each sets of scores in x."""
        #     e_x = np.exp(x - np.max(x))
        #     return e_x / e_x.sum(axis=0)  # only difference


        # temp_maxprobs = {}
        # for k, v in self.nbest_data.items():
        #     probs_list = []
        #     # try:
        #     for a in v[:5]:
        #         if a['text'] != "":
        #             start_logit = float(a['start_logit'])
        #             end_logit = float(a['end_logit'])
        #             final_logit = start_logit + end_logit
        #             probs = softmax(final_logit/ 1.15)
        #             probs_list.append(probs)
        #         else:
        #             pass
        #     temp_maxprobs[k] = float(sum(probs_list))
            # except:
            #     temp_maxprobs[k] = 0
        return temp_maxprobs

    def _generate_predictions(self):
        preds = {k: v[0]['text'] if v[0]['text']!=""
                    else v[1]['text']
                    for k, v in self.nbest_data.items()}
        return preds


    def _generate_more_predictions(self, i):
        # print('nbest', i)

        i_preds = {}
        for k, v in self.nbest_data.items():
            try:
                if v[0]['text']!="":
                    i_preds[k] = v[i]['text']
                else:
                    i_preds[k] = v[i + 1]['text']
            except:
                i_preds[k] = 0.
        # i_preds = {k: v[i]['text'] if v[0]['text']!=""
        #             else v[i+1]['text']
        #             for k, v in self.nbest_data.items()}
        return i_preds

    def _calc_em(self, train_or_dev):
        em_dict = {}
        if train_or_dev == 'train':
            guid_list = self.train_guid_list
        else:
            guid_list = self.dev_guid_list
        for guid in guid_list:
            try:
                m = max(utils.exact_match_score(x, self.preds[guid]) \
                            for x in self.gold_data[train_or_dev][guid]['answers'])
                em_dict[guid] = 1 if m else 0
            except:
                print('missing guid guid guid new new')
                em_dict[guid] =0
        return em_dict

    def _calc_em_strict(self, train_or_dev):
        # Only consider the first answer correct
        em_dict = {}
        if train_or_dev == 'train':
            guid_list = self.train_guid_list
        else:
            guid_list = self.dev_guid_list
        for guid in guid_list:
            m = utils.exact_match_score(self.gold_data[train_or_dev][guid]['answers'][0], self.preds[guid])
            em_dict[guid] = 1 if m else 0
        return em_dict

    def _get_double_squad(self, args, dataset_prefix, split_no):
        gold_data_train_path = \
                'data_splits/{}_train_split_{}.jsonl'.format(\
                dataset_prefix, (split_no+1)%10)
        self.gold_data['train'].update(utils.read_gold_data(args.task, gold_data_train_path,\
            dataset_prefix))
        gold_data_train_path = \
            'data_splits/{}_train_split_{}.jsonl'.format(\
            dataset_prefix, (split_no+2)%10)
        self.gold_data['train'].update(utils.read_gold_data(args.task, gold_data_train_path,\
            dataset_prefix))
        gold_data_train_path = \
            'data_splits/{}_train_split_{}.jsonl'.format(\
            dataset_prefix, (split_no+3)%10)
        self.gold_data['train'].update(utils.read_gold_data(args.task, gold_data_train_path,\
            dataset_prefix))
        np.random.seed(42)
        chosen_train = np.random.choice(list(self.gold_data['train'].keys()), 3200, replace=False)
        self.gold_data['train'] = {k: v for k, v in self.gold_data['train'].items() if k in chosen_train}
        
    def generate_features(self, args, train_or_dev):
        if train_or_dev == 'train':
            guid_list = self.train_guid_list
        else:
            guid_list = self.dev_guid_list

        if args.ttdo_calibrator:
            mean_dict_list, var_dict_list = \
                    utils.get_more_prob_stats_test(args, guid_list, \
                                self.more_preds, self.dataset_prefix, \
                                train_or_dev)

        features = {}
        for guid in guid_list:
            features[guid] = []
            if args.ablate != 'context_len':
                features[guid].append(len( \
                                self.gold_data[train_or_dev][guid]['context'].split()))
            if args.ablate != 'answer_entropy':
                features[guid].append(self.answer_entropy[guid])

            if args.ablate != 'softmax_temp':
                # print(self.temp_maxprobs[guid])
                # self.temp_maxprobs is a list
                features[guid] += self.temp_maxprobs[guid]

            if not args.ttdo_calibrator:
                if args.ablate != 'all_prob':
                    if args.ablate != 'maxprob':
                        features[guid].append(self.maxprobs[guid])
                    if args.ablate != 'other_prob':
                        features[guid].append(self.second_maxprobs[guid])
                        features[guid].append(self.third_maxprobs[guid])
                        features[guid].append(self.fourth_maxprobs[guid])
                        features[guid].append(self.fifth_maxprobs[guid])
            if args.ablate != 'pred_len':
                features[guid].append(len(self.preds[guid].split()))
            # print('ttdo', args.ttdo_calibrator)
            if args.ttdo_calibrator:
                for i in range(5):
                    features[guid].append(mean_dict_list[i][guid])
                    features[guid].append(var_dict_list[i][guid])
        return features


    def generate_features_jsonfiles(self, args, train_or_dev):
        if train_or_dev == 'train':
            guid_list = self.train_guid_list
        else:
            guid_list = self.dev_guid_list

        # if args.ttdo_calibrator:
        # if train_or_dev == 'dev':

        # top 5 probability
        self.more_preds = [self._generate_more_predictions(i) for i in range(0, 5)]
        # else:
        #     self.more_preds = [self._generate_more_predictions()]
        mean_dict_list, var_dict_list = \
                utils.get_more_prob_stats_test(args, guid_list, \
                            self.more_preds, self.dataset_prefix, \
                            train_or_dev)


        context_len_features={}
        answer_entropy_features = {}
        softmax_temp_features={}
        max_features = {}
        other_features={}
        pred_len_features={}
        ttdo_mean_features={}
        ttdo_var_features ={}



        for guid in guid_list:

            context_len_features[guid]=[]
            answer_entropy_features[guid]=[]
            softmax_temp_features[guid] =[]
            max_features[guid] = []
            other_features[guid] = []
            pred_len_features[guid]=[]
            ttdo_mean_features[guid]=[]
            ttdo_var_features[guid] =[]


            context_len_features[guid].append(len(self.gold_data[train_or_dev][guid]['context'].split()))
            answer_entropy_features[guid].append(self.answer_entropy[guid])
            softmax_temp_features[guid].append(self.temp_maxprobs[guid])
            max_features[guid].append(self.maxprobs[guid])
            other_features[guid].append(self.second_maxprobs[guid])
            other_features[guid].append(self.third_maxprobs[guid])
            other_features[guid].append(self.fourth_maxprobs[guid])
            other_features[guid].append(self.fifth_maxprobs[guid])
            pred_len_features[guid].append(len(self.preds[guid].split()))
            for i in range(5):
                ttdo_mean_features[guid].append(mean_dict_list[i][guid])
                ttdo_var_features[guid].append(var_dict_list[i][guid])


            # utils.write_features_output(args, output_features)


            # if args.write_features == 'context_len':
            #     max_features[guid].append(len( \
            #                     self.gold_data[train_or_dev][guid]['context'].split()))
            # if args.ablate != 'answer_entropy':
            #     features[guid].append(self.answer_entropy[guid])
            #
            # if args.ablate != 'softmax_temp':
            #     features[guid].append(self.temp_maxprobs[guid])
            #
            # if not args.ttdo_calibrator:
            #     if args.ablate != 'all_prob':
            #         if args.ablate != 'maxprob':
            #             features[guid].append(self.maxprobs[guid])
            #         if args.ablate != 'other_prob':
            #             features[guid].append(self.second_maxprobs[guid])
            #             features[guid].append(self.third_maxprobs[guid])
            #             features[guid].append(self.fourth_maxprobs[guid])
            #             features[guid].append(self.fifth_maxprobs[guid])
            # if args.ablate != 'pred_len':
            #     features[guid].append(len(self.preds[guid].split()))
            # if args.ttdo_calibrator:
            #     for i in range(5):
            #         features[guid].append(mean_dict_list[i][guid])
            #         features[guid].append(var_dict_list[i][guid])
        return context_len_features, answer_entropy_features,softmax_temp_features, max_features, \
               other_features,pred_len_features,ttdo_mean_features,ttdo_var_features