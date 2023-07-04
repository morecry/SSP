import json
input_path = 'datasets/cast-20/eval_topics.jsonl'
output_path = 'datasets/cast-20/eval_topics.jsonl'
record = []
with open(input_path, 'r') as f:
    for line in f:
        
        record.append(json.loads(line))

with open(output_path, 'w') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
