import copy
import json
from xxlimited import new
import nltk
import random
import tqdm
import Levenshtein
import torch
import argparse
import os
from transformers import BertTokenizer, BertModel

# input_path = 'data/cast19_split/raw/train_data4rank_extra.json'
# out_path = 'data/cast19_split/raw/train_data4rank_extra_aug.json'
from nltk.corpus import wordnet as wn

# wn.synsets('throat').lemma_names()   
# tokenizer = BertTokenizer.from_pretrained('bert/')
# bert = BertModel.from_pretrained('bert/', state_dict=torch.load(os.path.join('bert/', 'pytorch_model.bin'), map_location="cpu"))
# bert_embedding = bert.embeddings.word_embeddings
# vocab_weights = bert_embedding.weight
# with open('bert/vocab.txt', 'r') as f:
#     vocabs = f.readlines()
# vocabs = [word.strip() for word in vocabs]
# valid_index = []
# bad_chars = ['#', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# for i, vocab in enumerate(vocabs):
#     if i < 2000:
#         continue
#     tag = True
#     for char in bad_chars:
#         if char in vocab:
#             tag = False
#             break
#     if tag:
#         valid_index.append(i)

# vocabs = [vocabs[i] for i in valid_index]
# vocab_weights = vocab_weights[valid_index]

def replace_seq(input_seq):
    return ['apple', 'peer', 'eye',  'melon', 'cat', 'orange', 'dog', 'mouth']
    # return ['apple', 'peer', 'orange'] #best
    # return ['apple', 'peer', 'orange','melon', 'cat']
    # return ['[PAD%d]' % i for i in range(50)]
    # return ['[PAD]']
    # input_seq = "throat cancer"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_seq))
    input_ids = torch.tensor(input_ids).long() 
    input_vector = bert_embedding(input_ids)
    return_seqs = []
    inner_product = torch.mm(input_vector, vocab_weights.T)
    inner_product = torch.mean(inner_product, dim=0)
    # print(inner_product)
    # breakpoint()
    values, indices = torch.topk(inner_product, k=5,)
    # print(values)
    # breakpoint()
    for i in range(indices.shape[0]):
        # tokens = []
        # tokens.append(vocabs[int(indices[j][i].item())])  
        # return_seqs.append(' '.join(tokens))
        return_seqs.append(vocabs[int(indices[i].item())])
    # print(return_seqs)
    # print(return_seqs)
    # breakpoint()
    # token_id = 
    # print(vocabs)
    return return_seqs
# replace_seq("Neverending Story ")

# breakpoint()


def get_refomulation_string(raw, target):
    if raw[-1] in ['.', '?']:
        raw = raw[:-1]
    if target[-1] in ['.', '?']:
        target = target[:-1]
    
    target = target.split(' ')
    raw = raw.split(' ')
    
    head = 0
    tail = -1
    while head<len(raw):
        if target[head] == raw[head]:
            head += 1
        else:
            break
    
    if head == len(raw):
        if  head != len(target):
            
            target = [word for word in target[head:] if word not in raw]
                    
            return ' '.join(target)
        else:
            return ''

    while True:
        if target[tail] == raw[tail]:
            tail -= 1
        else:
            break
    # print(target)
    # print(head, tail)
    if tail == -1:
        target = [word for word in target[head:] if word not in raw]
        results = ' '.join(target)
    else:
        target = [word for word in target[head:tail+1] if word not in raw]
        results = ' '.join(target)
    # print(results)
    return results
        
# get_refomulation_string('Where and when was the first invented?', 'Where and when was the first toilet invented?')
# breakpoint()
from nltk import pos_tag, word_tokenize
# tags = pos_tag('US Electoral'.split(' '))

def get_key_nn(input_sents):
    key_nn = []
    for input_sent in input_sents:
        word_tags = pos_tag(word_tokenize(input_sent))
        for word, tag in word_tags:
            if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                key_nn.append(word)
    return key_nn

def get_nn(sent):
    tags = pos_tag(sent.split(' '))
    start = 0
    word_list = []
    # for i, item in enumerate(tags):
    #     if item[0] in ['it', 'its', 'they', 'them', 'their']:
    #         word_list.append(item[0])
    # if len(word_list) == 0:
    for i, item in enumerate(tags):
        if i < start:
            continue
        if item[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            word = item[0]
            start = i+1
            for j in range(i-1, -1, -1):
                if tags[j][1] in ['NN', 'NNS', 'JJ']:
                    word = tags[j][0] + ' ' + word
                else:
                    break
            for j in range(i+1, len(tags)):
                if tags[j][1] in ['NN', 'NNS']:
                    word = word + ' ' + tags[j][0]
                    start = j+1
                else:
                    break
            word_list.append(word)
    if len(word_list) == 0:
        return ''
    else:
        return ' '.join(word_list)


def same_word(word, word_list):
    for ref_word in word_list:
        dis = Levenshtein.distance(word, ref_word)
        if dis <= 1:
            return True
    return False

def run(input_path, out_path):
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    print(len(input_data))
    # print(len(input_data))
    # breakpoint()
    # input_data = input_data[:10000]
    out_data = []
    
    # for item in tqdm.tqdm(input_data):
    #     # continue
    #     input_sents = item['input']
    #     target_sent = item['target']
    #     cut_num = 4
    #     # choose_item =random.sample(input_data, k=1)[0]
    #     # add_sent = choose_item['input']
    #     while True:
    #         choose_item =random.sample(input_data, k=1)[0]
    #         if len(choose_item['input']) >= cut_num:
    #             break
    #     if len(input_sents) == 1:
    #         continue
    #     add_sent = choose_item['input'][:cut_num]
    #     new_input_sents = copy.deepcopy(input_sents)
    #     new_target_sent = copy.deepcopy(target_sent)
    #     new_input_sents = add_sent + new_input_sents
    #     new_item = copy.deepcopy(item)
    #     new_item['input'] = new_input_sents
    #     new_item['target'] = new_target_sent
    #     out_data.append(item)

    

        # for sent in input_sents[:-1]:
        #     new_input_sents = copy.deepcopy(input_sents)
        #     new_target_sent = copy.deepcopy(target_sent)
        #     new_input_sents = [sent] + new_input_sents
        #     new_item = copy.deepcopy(item)
        #     new_item['input'] = new_input_sents
        #     new_item['target'] = new_target_sent
        #     out_data.append(item)

    
    k = 0
    for item in tqdm.tqdm(input_data):
        # continue
        input_sents = item['input']
        target_sent = item['target']
        automatic_responses = item['automatic_response'] 
        refomulation_string = get_refomulation_string(input_sents[-1], target_sent)
        refomulation_string = get_nn(refomulation_string)
        if refomulation_string == '':
            continue 
        # if item['topic_number'] == "31" and item['query_number'] == "4" :
        #     print(refomulation_string)
        #     breakpoint()
        k += 1
        cand_seqs = replace_seq(refomulation_string)
        for keyword in cand_seqs:
            new_input_sents = copy.deepcopy(input_sents)
            new_target_sent = copy.deepcopy(target_sent)
            new_responses = copy.deepcopy(automatic_responses)
            new_input_sents = [nltk.word_tokenize(sent) for sent in new_input_sents]
            new_responses = [nltk.word_tokenize(sent) for sent in new_responses]
            new_target_sent = nltk.word_tokenize(new_target_sent)
            len_refomulation_string = len(refomulation_string.split(' '))
            tag = False   
            for i in range(len(new_input_sents)-1):
                new_input_sent = new_input_sents[i]
                for j in range(len(new_input_sent)-len_refomulation_string+1):
                    if ' '.join(new_input_sent[j:j+len_refomulation_string]) == refomulation_string:
                        new_input_sent = new_input_sent[:j] + [keyword] + new_input_sent[j+len_refomulation_string:]
                        new_input_sents[i] = new_input_sent
                        tag = True
                        break

            for i in range(len(new_responses)-1):
                new_response = new_responses[i]
                for j in range(len(new_response)-len_refomulation_string+1):
                    if ' '.join(new_response[j:j+len_refomulation_string]) == refomulation_string:
                        new_response = new_response[:j] + [keyword] + new_response[j+len_refomulation_string:]
                        new_responses[i] = new_response
                        tag = True
                        break
            if not tag:
                continue
            for j in range(len(new_target_sent)-len_refomulation_string+1):
                if ' '.join(new_target_sent[j:j+len_refomulation_string]) == refomulation_string:
                    new_target_sent = new_target_sent[:j] + [keyword] + new_target_sent[j+len_refomulation_string:]

            new_input_sents = [' '.join(sent) for sent in new_input_sents]
            new_responses = [' '.join(sent) for sent in new_responses]
            new_target_sent = ' '.join(new_target_sent)
            new_item = copy.deepcopy(item)
            new_item['input'] = new_input_sents
            new_item['target'] = new_target_sent
            new_item['automatic_response'] = new_responses
            out_data.append(new_item)
    k = 0
    for item in tqdm.tqdm(input_data):
        k += 1
        continue
        input_sents = item['input']
        target_sent = item['target']
        cut_num = 3
        # choose_item =random.sample(input_data, k=1)[0]
        # add_sent = choose_item['input']
        # while True:
        choose_item =random.sample(input_data, k=1)[0]
        if len(choose_item['input']) >= cut_num:
            break
        add_sent = random.sample(choose_item['input'], k=1)
        
        # add_sent = choose_item['input']
        new_input_sents = copy.deepcopy(input_sents)
        new_target_sent = copy.deepcopy(target_sent)
        new_input_sents = add_sent + new_input_sents
        new_item = copy.deepcopy(item)
        new_item['input'] = new_input_sents
        new_item['target'] = new_target_sent
        out_data.append(new_item)

    # for item in tqdm.tqdm(input_data):
    #     # continue
    #     input_sents = item['input']
    #     target_sent = item['target']
    #     cut_num = 2
    #     # choose_item =random.sample(input_data, k=1)[0]
    #     # add_sent = choose_item['input']
    #     while True:
    #         choose_item =random.sample(input_data, k=1)[0]
    #         if len(choose_item['input']) >= cut_num:
    #             break
    #     add_sent = choose_item['input'][:cut_num]

    #     # choose_item = random.sample(input_data, k=1)[0]
    #     # add_sent = choose_item['input']
    #     new_input_sents = copy.deepcopy(input_sents)
    #     new_target_sent = copy.deepcopy(target_sent)
    #     new_input_sents = add_sent + new_input_sents
    #     new_item = copy.deepcopy(item)
    #     new_item['input'] = new_input_sents
    #     new_item['target'] = new_target_sent
    #     out_data.append(new_item)

   
    
    for item in tqdm.tqdm(input_data):
        continue
        input_sents = item['input']
        target_sent = item['target']
        # key_nn = get_key_nn(input_sents)
        cut_num = 6
        # add_sent = []
        # if len(key_nn) == 0:
        #     continue
        # for i in range(max_num):
        #     add_sent += random.sample(key_nn, k=1)
        # add_sent = [' '.join(add_sent)]
        # print(add_sent)
        # continue
        if len(input_sents) <= cut_num + 1:
            continue
        add_sent = random.sample(input_sents[:-1], k=cut_num)
        
        new_input_sents = copy.deepcopy(input_sents)
        new_target_sent = copy.deepcopy(target_sent)
        new_input_sents = add_sent + new_input_sents
        new_item = copy.deepcopy(item)
        new_item['input'] = new_input_sents
        new_item['target'] = new_target_sent
        out_data.append(new_item)

    
    # breakpoint()
    out_data.extend(input_data)
    out_data.extend(input_data)
    out_data.extend(input_data)

    

    print(len(out_data))
    with open(out_path, 'w') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=int)
    args = parser.parse_args()
    random.seed(42)
    for i in range(5):
        if args.tag == 20:
            input_path = 'datasets/cast-%d/eval_topics.jsonl.%d'% (args.tag, i)
         
        else:
            input_path = 'datasets/cast-%d/eval_topics_cut.jsonl.%d'% (args.tag, i)
        # records = []
        # with open(input_path, encoding="utf-8") as f:
        #     for line in f:
        #         record = json.loads(line)
        #         records.append(record)
        # with open(input_path, 'w') as f:
        #     json.dump(records, f, ensure_ascii=False, indent=2)
        out_path = 'datasets/cast-%d/eval_topics_aug.jsonl.%d'% (args.tag, i)
        run(input_path, out_path)
