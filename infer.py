import pdb
from tqdm import tqdm
# from flask import Flask, request, jsonify
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import json
# app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained('output_qwen_planning_10train',local_files_only=True, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(
    'output_qwen_planning_10train', # path to the output directory
    device_map="cuda",
    trust_remote_code=True
).eval()
split = '3'
task = 'planning'
set = 'test'
examples = json.load(open(f'data/Mind2web_{set}_{task}.json'))
# examples = json.load(open(f'/data/luohh/acl23/Qwen-VL/csc_detect_test.json'))
# examples = json.load(open(f'data/ScienceQA_{set}_{task}_blip2xl_new.json'))
questions = []
save_root = f'infer/pred_test_planning_10train.json'
# save_root = f'infer/pred_{set}_{task}_blip2xl_new.json'
count = 0
for i, example in tqdm(enumerate(examples)):
    query = example['conversations'][0]['value']
    # query = 'Screenshot: <img>/data/luohh/agent/SeeAct/data/data/raw_dump/task/9c9e89c1-fdb9-424c-b544-b9fd2f1ef46e/processed/snapshots/27502e8e-1ee0-49f3-a0ed-60b044dd585c_before.jpg</img>\nTask: Browse spider-man toys for kids and sort by lowest price.\nImagine that you are imitating humans doing web navigation for a task step by step. At each stage, you can see the webpage like humans by a screenshot. Combined with the screenshot, make a plan to accomplish the task. The plan should contain a complete flow of actions.\n\nConclude your plan using the format below.Ensure your answer is strictly adhering to the format provided below.\n1: step1 of your plan.\n2: step2 of your plan.\n...'
    id = example['id']
    count += 1
    response, history = model.chat(tokenizer, query=query, history=None)
    # questions.append({'id':id,'response':response})
    # print(query)
    # print('*******************TARGET*********************')
    # print(example['id'])
    # print(f"{example['conversations'][1]['value']}")
    # print('*******************PREDICT*********************')
    print(response)
    # print('\n\n')
    # with open(save_root, 'w') as fp:
    #     json.dump(questions, fp)
