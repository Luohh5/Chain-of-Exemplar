import json

prediction = json.load(open('/data/luohh/acl23/Qwen-VL/infer/pred_gpt_qg_3shot/1.json'))
count = 0
for example in prediction:
    print(example['id'])
    count += 1
print(count)