# from builtins import breakpoint, print
# from builtins import input
from tracemalloc import start
from urllib import response
import tqdm
import json
import nltk
import copy
import sys

def process(tag):
    if tag == 19:
        input_path = 'QRECC/query_refomulation_topicshift_19.json'
        output_path = 'QRECC/query_refomulation_aug_19.json'
    elif tag == 20:
        input_path = 'QRECC/query_refomulation_topicshift_20.json'
        output_path = 'QRECC/query_refomulation_aug_20.json'
    else:
        print("wrong tag")
        sys.exit()
    

    # if tag == 19:
    #     input_path = 'QRECC/query_refomulation.json'
    #     output_path = 'QRECC/query_refomulation_mark_19.json'
    # elif tag == 20:
    #     input_path = 'QRECC/query_refomulation.json'
    #     output_path = 'QRECC/query_refomulation_mark_20.json'
    # else:
    #     print("wrong tag")
    #     sys.exit()

    with open(input_path, 'r') as f:
        data_source = json.load(f)
    new_records = []

    k = 0
    print(len(data_source))
    for item in tqdm.tqdm(data_source):

        input_sents = item['input']
        target_sent = item['target']
        responses = item['automatic_response']
        # if len(input_sents) == 1:
        #     # item['mark'] = [1]
        #     continue
        # print(input_sents, target_sent)
        refo_words = list(set(nltk.word_tokenize(target_sent)) - set(nltk.word_tokenize(input_sents[-1])))
        if tag == 20:
            if len(responses) >= 2:
                input_sents = input_sents[:-1] + [responses[-2]] + [input_sents[-1]]
                # print(input_sents)
                # breakpoint()
        
        if len(refo_words) == 0:
            mark = [0] * len(input_sents)
        else:
            mark = []
            for sent in input_sents[::-1]:
                if len(list(set(refo_words) - set(nltk.word_tokenize(sent)))) == 0:
                    mark = [1] + mark
                else:
                    mark = [0] + mark
            # mark[-1] = 1

        # if sum(mark) != 0:
        item['mark'] = copy.deepcopy(mark)
        new_records.append(copy.deepcopy(item))
        # item['mark'] = copy.deepcopy(mark)
        # new_records.append(copy.deepcopy(item))
        # else:
        #     print(input_sents, target_sent)
        #     breakpoint()

    print(len(new_records))
    with open(output_path, 'w') as f:
            json.dump(new_records, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    tags = [19, 20]
    for tag in tags:
        process(tag)
    