# Knowing More About Questions Can Help: Improving Calibration in Question Answering

This is the official GitHub repository for the following paper:

> **[Knowing More About Questions Can Help: Improving Calibration in Question Answering.]()**  
> Shujian Zhang, Chengyue Gong, and Eunsol Choi.  
> _Findings of ACL_, 2021.  

## Setup
Install packages:

```
pip install -r requirements.txt
```

Download data:

We use the data provided in the Selective Question Answering under Domain Shift (ACL 2020). For more details on how to download data, please see the associated [CodaLab Worksheet](https://worksheets.codalab.org/worksheets/0xea5a522788f743acb4fbf9e60065be8f).


## Running experiments
Training the base QA model:

```
python src/bert_squad.py --bert_model bert-base-uncased --do_train --train_file=data/squad/train-v1.1.json --do_predict --predict_file=data/squad/dev-v1.1.json --fp16 --loss_scale 128 --do_lower_case --output_dir=out --output_name=model_squad --learning_rate=3e-5 --num_train_epochs=2 --max_seq_length=384 --doc_stride=128 --train_batch_size=16 --predict_batch_size=16
```
Above is an example that the base QA model is trained on SQuAD1.1 dataset. 

Data augmentation via paraphrasing:

```
python backtranslation/back_translation_squad_question.py
```
Above is an example that the paraphrasing through the backtranslation on the question of the examples.

Training calibrator:

```
python backtranslation/train_calibrator.py
```


## Paraphrasing
The `backtranslation` directory contains scripts that backtranslate either question or context of examples and train the calibrator with features. 

## Data Set Up
Standard RC: We test two in domain settings and two out of domain settings. We randomly sample 4K examples from each of the datasets included in the training portion of the MRQA
shared task.
Adversarial RC (SQuAD 1.1 Adversarial): we sample 2K examples from the development portion of the SQuAD 1.1 (Jia and
Liang, 2017) AddSent dataset.
Unanswerable RC (SQuAD 2.0): we sampled 2K examples from the development portion of the SQuAD 2.0 dataset (Rajpurkar et al., 2018).
Please refer to our paper for more details.


## Citation
```bibtex
@article{zhang2021knowing,
  title={Knowing More About Questions Can Help: Improving Calibration in Question Answering},
  author={Zhang, Shujian and Gong, Chengyue and Choi, Eunsol},
  journal={arXiv preprint arXiv:2106.01494},
  year={2021}
}