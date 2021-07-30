# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import pdb
from io import open
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer

from bert_squad_dataset_utils import read_squad_examples, read_mrqa_examples, convert_examples_to_features, RawResult, write_predictions, write_predictions_long

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    # If training on SQuAD and MRQA, put (orig) SQuAD JSON in train_file 
    # and MRQA in mrqa_train_file. Add --train_both,
    # specifying --amount_train_both.
    # If training on MRQA alone, put it in train_file and
    # add --train_mrqa and --train_mrqa_only.
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--mrqa_train_file", default=None, type=str, help="MRQA json for training for both, else put in train_file.")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json, or MRQA jsonl for predictions.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_mrqa", action='store_true', help="Whether to train on MRQA train data")
    parser.add_argument("--train_mrqa_only", action='store_true', help="Whether to train on MRQA train data ONLY")
    parser.add_argument("--train_both", action='store_true', help="Whether to train on both train data")

    parser.add_argument("--adverse_squad2", action='store_true', help="Whether to only include SQUAD 2.0 unanswerable questions")

    parser.add_argument("--amount_train_both", default=0, type=int, help="How much OOD data to include in train")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--predict_mrqa", action='store_true', help="Whether to run eval on MRQA dev data.")
    parser.add_argument("--use_pretrained", action='store_true', help="Whether to use a pretrained model from out_dir.")
    # Specify this for prediction json name
    parser.add_argument("--output_name", default="out", type=str, help="Prefix for eval output files.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    # This is important
    parser.add_argument("--test_time_dropout", action='store_true', help='Activate test time dropout')
    parser.add_argument('--dropout_seed', type=int, default=None, help='dropout test time seed')
    # If you're predicting on MRQA train
    parser.add_argument("--predict_on_train", type=str, default=None, help='Dataset QID PATH if predicting on MRQA train')
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    #print(args)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.WARN)
                        #level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    if args.use_pretrained:
        model = BertForQuestionAnswering.from_pretrained(args.output_dir)
    else:
        model = BertForQuestionAnswering.from_pretrained(args.bert_model)

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        # Prepare data loader
        if args.train_mrqa:
            print("Reading MRQA examples...")
            train_examples = read_mrqa_examples(input_file=args.train_file, is_training=True)
            print("Number of train examples: {}".format(len(train_examples)))
        elif args.train_both or args.train_mrqa_only:
            if args.train_both:
                print("Reading SQuAD examples from {}...".format(args.train_file))
                train_examples = read_squad_examples(
                    input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative, adverse_squad2=args.adverse_squad2)
            #train_examples = read_mrqa_examples(
            #    input_file=args.train_file, is_training=True)
            if args.train_mrqa_only:
                train_examples = []
            print("Number of SQuAD examples = {}".format(len(train_examples)))
            print("Reading MRQA examples from {}...".format(args.mrqa_train_file))
            mrqa_examples = read_mrqa_examples(input_file=args.mrqa_train_file, is_training=True)
            np.random.seed(42)
            mrqa_examples = np.random.choice(mrqa_examples, args.amount_train_both, replace=False)
            print("Number of MRQA examples = {}".format(len(mrqa_examples)))
            train_examples = train_examples + list(mrqa_examples)
            random.shuffle(train_examples)
            print("All {} examples were shuffled.".format(len(train_examples)))
        else:
            train_examples = read_squad_examples(
                input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative, adverse_squad2=args.adverse_squad2)

        cached_train_features_file = args.train_file+'_{0}_{1}_{2}_{3}'.format(
            list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        # Don't load cached features or Augmented QA models will
        # all look the same, per the above naming scheme
        print("Converting examples to features...")
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving train features into cached file %s", cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        # if args.local_rank != -1:
        #     num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        if args.fp16:
            try:
                from apex.contrib.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if args.local_rank in [-1, 0]:
                        if not args.fp16:  # https://github.com/huggingface/pytorch-pretrained-BERT/commit/59cefd4f985b7221846189690ead3300ff864b3d
                            tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', loss.item(), global_step)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForQuestionAnswering.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

        # Good practice: save your training arguments together with the trained model
        output_args_file = os.path.join(args.output_dir, 'training_args.bin')
        torch.save(args, output_args_file)
    elif args.use_pretrained:
        model = BertForQuestionAnswering.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForQuestionAnswering.from_pretrained(args.bert_model)

    model.to(device)


    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if args.predict_mrqa:
            if args.predict_on_train:
                f = open(args.predict_on_train, 'r')
                qid_list = [line.strip() for line in f]
                eval_examples = read_mrqa_examples(
                    input_file=args.predict_file, is_training=False,
                    train_qid_list=qid_list)
            else:
                eval_examples = read_mrqa_examples(
                    input_file=args.predict_file, is_training=False)
        else:
            eval_examples = read_squad_examples(
                input_file=args.predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative, adverse_squad2=args.adverse_squad2)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
        
        if args.test_time_dropout:        
            def apply_dropout(m):
                if type(m) == torch.nn.Dropout:
                    m.train()
        
        model.eval()
        
        if args.test_time_dropout:
            torch.manual_seed(args.dropout_seed)
            model.apply(apply_dropout)
        
        all_results = []
        logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits, sequence_output_medium = model(input_ids, segment_ids, input_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                sequence_output = sequence_output_medium[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits, sequence_output=sequence_output))
        if args.test_time_dropout:
            dropout_path = os.path.join(args.output_dir, 'new_dropout')
            if not os.path.exists(dropout_path):
                os.makedirs(dropout_path)
            output_prediction_file = os.path.join(dropout_path, "{}{}-predictions.json".format(args.output_name, args.dropout_seed))
            output_nbest_file = os.path.join(dropout_path, "{}{}-nbest_predictions.json".format(args.output_name, args.dropout_seed))
        
        else:
            output_prediction_file = os.path.join(args.output_dir, "{}-predictions.json".format(args.output_name))
            output_nbest_file = os.path.join(args.output_dir, "{}-nbest_predictions.json".format(args.output_name))

        #output_prediction_file = os.path.join(args.output_dir, "{}-predictions.json".format(args.output_name))
        #output_nbest_file = os.path.join(args.output_dir, "{}-nbest_predictions.json".format(args.output_name))
        output_null_log_odds_file = os.path.join(args.output_dir, "{}-null_odds.json".format(args.output_name))
        

        if len(eval_examples) < 10: #20000:
            print("Writing to {}".format(output_nbest_file))
            write_predictions(eval_examples, eval_features, all_results,
                              args.n_best_size, args.max_answer_length,
                              args.do_lower_case, output_prediction_file,
                              output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                              args.version_2_with_negative, args.null_score_diff_threshold)
        else:
            print("Writing to {}".format(output_nbest_file+'l'))
            write_predictions_long(eval_examples, eval_features, all_results,
                              args.n_best_size, args.max_answer_length,
                              args.do_lower_case, output_prediction_file,
                              output_nbest_file+'l', output_null_log_odds_file, args.verbose_logging,
                              args.version_2_with_negative, args.null_score_diff_threshold)


if __name__ == "__main__":
    main()
