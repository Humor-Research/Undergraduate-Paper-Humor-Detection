import pandas as pd
import sys
import json

import pandas as pd
import datasets
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os
import sys
from transformers import EarlyStoppingCallback
from transformers import TextClassificationPipeline

from hri_tools import HumorDataset, HRI_PAPER_2023_DATASETS

import textattack
from textattack.attack_recipes.input_reduction_feng_2018 import InputReductionFeng2018

TRAIN_DATASET = str(sys.argv[1])

with open("best_model_path.json") as f:
    best_model_path = json.load(f)

model_path = best_model_path[TRAIN_DATASET]
tokenizer_path = '/home/ambaranov/roberta-base/'
true_path = '/home/ambaranov/You_Told_Me_That_Joke_Twice/task_3/reports'

model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_length = 512, truncation=True)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=0, max_length = 512, truncation=True)

model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

attack = InputReductionFeng2018.build(model_wrapper)

d_names = {'one_liners-693': 'one_liners', 'pun_of_the_day-23': 'pun_of_the_day', 'semeval_2017_task_7-693': 'semeval_2017_task_7', 'short_jokes-453': 'short_jokes', 'reddit_jokes_last_laught-693': 'reddit_jokes_last_laught', 'semeval_2021_task_7-47': 'semeval_2021_task_7', 'funlines_and_human_microedit_paper_2023-23': 'funlines_and_human_microedit_paper_2023', 'unfun_me-23': 'unfun_me', 'the_naughtyformer-693': 'the_naughtyformer', 'meta_dataset-977': 'comb'}

hd = HumorDataset(d_names[TRAIN_DATASET])
hd.load()
df_ = hd.get_test()

attacked_texts = []
df = df_[df_['label'] == 1]
if TRAIN_DATASET == 'short_jokes-453' or TRAIN_DATASET == 'the_naughtyformer-693':
    df = df.sample(n=min(len(df), 10000), random_state=42)
df = df.reset_index(drop=True)

print(len(df))

for example in df.iterrows():
	idx = example[0]
	text, label = example[1].text, example[1].label
	try:
		attack_result = attack.attack(text, label)
		attacked_texts.append(attack_result.perturbed_result.attacked_text)
	except:
		attacked_texts.append('UNCHANGED')		
		print(text, label)
		continue

df['attacked_text'] = attacked_texts
df.to_csv(f"./results/{TRAIN_DATASET}_input_reduction_positive.csv")

df = df_[df_['label'] == 0]
if TRAIN_DATASET == 'the_naughtyformer-693':
	sys.exit()

if TRAIN_DATASET == 'short_jokes-453':
	df = df.sample(n=min(len(df), 10000), random_state=42)
df = df.reset_index(drop=True)
attacked_texts = []

print(len(df))

for example in df.iterrows():
	idx = example[0]
	text, label = example[1].text, example[1].label
	try:
		attack_result = attack.attack(text, label)
		attacked_texts.append(attack_result.perturbed_result.attacked_text)
	except:
		attacked_texts.append('UNCHANGED')
		print(text, label)
		continue	

df['attacked_text'] = attacked_texts
df.to_csv(f"./results/{TRAIN_DATASET}_input_reduction_negative.csv")

