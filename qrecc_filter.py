# from builtins import breakpoint, print
# from builtins import input
from tracemalloc import start
from urllib import response
import tqdm
import json
import nltk
import copy
import sys



input_path = 'QRECC/query_refomulation.json'
output_path = 'QRECC/query_refomulation_filter.json'


with open(input_path, 'r') as f:
    data_source = json.load(f)
# new_records = []

k = 0
# print(len(data_source))
cand_topics = list(set([item['topic_number'] for item in data_source if item['input'][0] == item['target']])) 
new_records = [item for item in data_source if item['topic_number'] in cand_topics]
bad_records = [item for item in data_source if item['topic_number'] not in cand_topics]
print(bad_records[:5])

# print(len(new_records))
# with open(output_path, 'w') as f:
#         json.dump(new_records, f, ensure_ascii=False, indent=2)

# for item in data_source:
#     input_sents = item['input']
#     if input_sents[0] != item['target']:
#         print(input_sents)
#         breakpoint()