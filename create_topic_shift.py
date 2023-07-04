# from builtins import breakpoint, print
# from builtins import input
from tracemalloc import start
import tqdm
import json
import nltk
import copy
import random
import sys


def process(tag):
    if tag == 19:
        input_path = 'QRECC/query_refomulation.json'
        output_path = 'QRECC/query_refomulation_topicshift_19.json'
    elif tag == 20:
        input_path = 'QRECC/query_refomulation.json'
        output_path = 'QRECC/query_refomulation_topicshift_20.json'
    else:
        print("wrong tag")
        sys.exit()
    with open(input_path, 'r') as f:
        data_source = json.load(f)
    new_records = []
    random.seed(42)
    k = 0
    print(len(data_source))
    # cand_topics = []
    cand_topics = list(set([item['topic_number'] for item in data_source if item['input'][0] == item['target']])) 
    neg_topics = list(set([item['topic_number'] for item in data_source]) - set(cand_topics))  

    # print(cand_topics[:10])
    # print(neg_topics[:10])
    # breakpoint()
    cand_sesstions = {}
    for item in data_source:
        if item['topic_number'] in cand_topics:
            if item['topic_number'] not in cand_sesstions:
                cand_sesstions[item['topic_number']] = item
            else:
                if len(item['input']) > len(cand_sesstions[item['topic_number']]['input']):
                    cand_sesstions[item['topic_number']] = item
    cand_sesstions = [cand_sesstions[topic_number] for topic_number in cand_sesstions]


    for item in tqdm.tqdm(data_source):
    
        input_sents = item['input']
        target_sent = item['target']
        if item['topic_number'] in neg_topics:
            topic = [0] *len(input_sents)
        else:
            extern_session = random.sample(cand_sesstions, k=1)[0]['input'] 
            cut_populations = list(range(1, len(extern_session)+1))
            cut_weights = list(range(1, len(extern_session)+1))
            cut_weights.reverse()
            cut_num = random.choices(cut_populations, weights=cut_weights)[0]
            topic = [1] * cut_num + [0] * len(item['input'])
            item['input'] = extern_session[:cut_num] + item['input']
        if tag == 20:
            #add response topic
            if len(item['automatic_response']) >= 2:
                topic = topic + [0]
                assert len(topic) == len(item['input']) + 1
            else:
                assert len(topic) == len(item['input'])
        else:
            assert len(topic) == len(item['input'])
        item['topic'] = copy.deepcopy(topic)

        new_records.append(copy.deepcopy(item))

    # with open(input_path, 'r') as f:
    #     data_source = json.load(f)  

    # for item in tqdm.tqdm(data_source):
    
    #     input_sents = item['input']
    #     target_sent = item['target']
    #     topic = [0] *len(input_sents)
    #     if tag == 20:
    #         #add response topic
    #         if len(item['automatic_response']) >= 2:
    #             topic = topic + [0]
    #             assert len(topic) == len(item['input']) + 1
    #         else:
    #             assert len(topic) == len(item['input'])
    #     else:
    #         assert len(topic) == len(item['input'])
    #     item['topic'] = copy.deepcopy(topic)

    #     new_records.append(copy.deepcopy(item))


    print(len(new_records))
    with open(output_path, 'w') as f:
        json.dump(new_records, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    tags = [19, 20]
    for tag in tags:
        process(tag)