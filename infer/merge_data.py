import json
import os
import glob
import pdb

target = 'pred_gpt_qg_0shot'
files = glob.glob(f'/data/luohh/acl23/Qwen-VL/infer/{target}/*.json')
files = sorted(files)
pred_list = []
for file in files:
    print(file)
    pred = json.load(open(file))
    pred_list.extend(pred)
with open(target+'.json', 'w') as fp:
    json.dump(pred_list, fp)