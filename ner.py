# import nltk
# from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer

# train_text = state_union.raw("2005-GWBush.txt")
# sample_text = state_union.raw("2006-GWBush.txt")

# custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# tokenized = custom_sent_tokenizer.tokenize(sample_text)

# def process_content():
#     try:
#         for i in tokenized[5:]:
#             words = nltk.word_tokenize(i)
#             tagged = nltk.pos_tag(words)
#             namedEnt = nltk.ne_chunk(tagged, binary=True)
#             namedEnt.draw()
#     except Exception as e:
#         print(str(e))


# process_content()
from nltk import pos_tag, word_tokenize
def get_nn(sent):
    tags = pos_tag(word_tokenize(sent))
    start = 0
    word_list = []
    for i, item in enumerate(tags):
        if item[0] in ['it', 'its', 'they', 'them', 'their']:
            word_list.append(item[0])
    # if len(word_list) == 0:
    for i, item in enumerate(tags):
        if i < start:
            continue
        if item[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            word = item[0]
            start = i+1
            for j in range(i-1, -1, -1):
                if tags[j][1] in ['NN', 'JJ']:
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
    return word_list
    
# for process_his(sents):
# print(pos_tag)
tags = pos_tag(word_tokenize("What is [MASK] GDP "))
print(tags)
# pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
#  'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', ' ']
# pos_tag_dict = {}
# for i in range(len(pos_tags)):
#     pos_tag_dict[pos_tags[i]] = i

# print(pos_tag_dict)
# print(len(pos_tag_dict))
breakpoint()


input_sent = "Tell me about lung cancer."
input_file = 'datasets/cast-19/queries.raw.tsv'
with open(input_file, 'r') as f:
    all_sents = f.readlines()

all_sents = [sent.strip().split('\t')[1] for sent in all_sents]


for sent in all_sents:
    word_list = get_nn(sent)
    print(sent, word_list)


