import json

problems = json.load(open('/data/luohh/acl23/Qwen-VL/infer/pred_csc_detect.json'))
for problem in problems:
    if problem['id'] == 'identity_8033':
        print(problem['response'])