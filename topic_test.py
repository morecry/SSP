from hashlib import new
import json 
import random


def update_item(item, extern_sessions, k=9):
    while True:
        choose_session = extern_sessions[random.choice(list(extern_sessions.keys()))]['input']
        if len(choose_session) >= k:
            break
    item['input'] = choose_session[:k] + item['input']
    return item
   

def get_topic_sessions(records):
    sessions = {}
    for item in records:
        if item['topic_number'] not in sessions:
            sessions[item['topic_number']] = item
        elif sessions[item['topic_number']]['query_number'] < item['query_number']:
            sessions[item['topic_number']] = item
    return sessions



def process(i, file_paths, new_file_paths):
    with open(file_paths[i], 'r') as f:
        raw_records = json.load(f)
    external_records = []
    for j in range(5):
        if j != i:
            with open(file_paths[j], 'r') as f:
                external_records.extend(json.load(f))
    extern_sessions = get_topic_sessions(external_records)
    new_records = []
    for item in raw_records:
        new_item = update_item(item, extern_sessions)
        new_records.append(new_item)
    with open(new_file_paths[i], 'w') as f:
        json.dump(new_records, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    random.seed(42)
    tag = 19
    file_paths = ["datasets/cast-%d/eval_topics.jsonl.%d" % (tag, i) for i in range(5)]
    new_file_paths = ["datasets/cast-%d/eval_topics_add.jsonl.%d" % (tag, i) for i in range(5)]

    for i in range(5):
        process(i, file_paths, new_file_paths)