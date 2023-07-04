import copy
from hashlib import new
import json
import nltk
import random
import tqdm
import Levenshtein
import torch
import argparse
import os
from transformers import BertTokenizer, BertModel

tag = 20
input_raw = 'datasets/cast-%d/queries.raw.tsv'%tag
input_manual = 'datasets/cast-%d/queries.manual.tsv'%tag
with open(input_raw, 'r') as f:
    raw_queries = f.readlines()
with open(input_manual, 'r') as f:
    manual_queries = f.readlines()
raw_queries = [item.strip().split('\t')[1] for item in raw_queries]
manual_queries = [item.strip().split('\t')[1] for item in manual_queries]
raw2manual = {}
for raw, manual in zip(raw_queries, manual_queries):
    raw2manual[raw] = manual

for i in range(5):
    # records = []
    inpur_eval_topic_path = 'datasets/cast-%d/eval_topics.jsonl.%d' % (tag, i)
    # with open(inpur_eval_topic_path, encoding="utf-8") as f:
    #     for line in f:
    #         record = json.loads(line)
    #         records.append(record)
    # with open(inpur_eval_topic_path, 'w') as f:
    #     json.dump(records, f, ensure_ascii=False, indent=2)

    with open(inpur_eval_topic_path, 'r') as f:
        inpur_eval_topic = json.load(f)
    for item in inpur_eval_topic:
        print(item)
        input_sents = item['input']
        # print(input_sents)
        input_sents.reverse()
        new_input_sents = []
        for sent in input_sents: 
            new_input_sents = [sent] + new_input_sents
            if raw2manual[sent.strip()] == sent.strip():
                break
        item['input'] = new_input_sents 
    with open('datasets/cast-%d/eval_topics_cut.jsonl.%d' % (tag, i), 'w') as f:
        json.dump(inpur_eval_topic, f, ensure_ascii=False, indent=2)

