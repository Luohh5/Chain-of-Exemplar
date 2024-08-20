# [
#   {
#     "id": "identity_1",
#     "conversations": [
#       {
#         "from": "user",
#         "value": "Picture 1: <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\n图中的狗是什么品种？"
#       },
#       {
#         "from": "assistant",
#         "value": "图中是一只拉布拉多犬。"
#       },
#       {
#         "from": "user",
#         "value": "框出图中的格子衬衫"
#       },
#       {
#         "from": "assistant",
#         "value": "<ref>格子衬衫</ref><box>(588,499),(725,789)</box>"
#       }
#     ]
#   }
# ]

import json
import pdb
import jsonlines
import os
from tqdm import tqdm

def build_example(problem,id):
    answer = 'Answer: ' + problem['choices'][problem['answer']] + '\n'
    question = problem['question']
    rationale = problem['lecture'] + problem['solution']
    choices = problem['choices'].copy()
    choices.remove(problem['choices'][problem['answer']])
    distractors = ''
    for i, choice in enumerate(choices):
        distractors += f'({i + 1}) {choice}\n'
    if problem['image'] != None:
        image = 'Picture: <img>' + \
                os.path.join(data_root, problem['split'], str(id), problem[
                    'image']) + f'</img>\n'
        context = f'Context: ' + problem['hint'] + '\n' if problem['hint'] != '' else ''
        # qg_prompt = 'Please generate a question from this picture, context and the corresponding answer.' if \
        # problem['hint'] != '' else 'Please generate a question from this picture and the corresponding answer.'
        # rg_prompt = 'According to this picture, context and question with its answer, how to make a reasoning to answer the question?' if \
        # problem[
        #     'hint'] != '' else 'According to this picture and question with its answer, how to make a reasoning to answer the question?'
        # dg_prompt = 'According to this picture, context and question with its answer obtained through reasoning, please generate at least 1 plausible yet incorrect answers which should be similar and grammatically consistent with the correct answer and seperate them with numbers like (1) (2) (3). \n' if \
        # problem[
        #     'hint'] != '' else 'According to this picture and question with its answer obtained through reasoning, please generate at least 1 plausible yet incorrect answers which should be similar and grammatically consistent with the correct answer and seperate them with numbers like (1) (2) (3).\n'
        qg = '\nExample:\n' + image + context + answer + f'Question: {question}'
        rg = '\nExample:\n' + image + context + f'Question: {question}\n' + answer + f'Reasoning: {rationale}'
        dg = '\nExample:\n' + image + context + f'Question: {question}\n' + answer + f'Distractors: {distractors}'
    else:
        context = 'Context: ' + problem['hint'] + '\n' if problem['hint'] != '' else ''
        # qg_prompt = 'Please generate a question from this context and the corresponding answer.' if \
        #     problem[
        #         'hint'] != '' else 'Please generate a question from the corresponding answer.'
        # rg_prompt = 'According to this context and question with its answer, how to make a reasoning to answer the question?' if \
        #     problem[
        #         'hint'] != '' else 'According to this question with its answer, how to make a reasoning to answer the question?'
        # dg_prompt = 'According to this context and question with its answer obtained through reasoning, please generate at least 1 plausible yet incorrect answers which should be similar and grammatically consistent with the correct answer and seperate them with numbers like (1) (2) (3). \n' if \
        #     problem[
        #         'hint'] != '' else 'According to this question with its answer obtained through reasoning, please generate at least 1 plausible yet incorrect answers which should be similar and grammatically consistent with the correct answer and seperate them with numbers like (1) (2) (3).\n'
        qg = '\nExample:\n' + context + answer + f'Question: {question}'
        rg = '\nExample:\n' + context + f'Question: {question}\n' + answer + f'Reasoning: {rationale}'
        dg = '\nExample:\n' + context + f'Question: {question}\n' + answer + f'Distractors: {distractors}'
    return qg, rg, dg
save_list = []
split = 'train'
save_root = f'data/Mind2web_{split}_pl.json'
data_root = '/data/luohh/agent/aaai25/data/train_pl'
for root, dirs, files in os.walk(data_root):
    for id, file in enumerate(files):
        # if file == 'train_5.json' or file == 'train_10.json':
        #     continue
        examples = json.load(open(os.path.join(data_root, file)))
        # save_list = json.load(open('/data/luohh/acl23/Qwen-VL/data/Mind2web_train_planning.json'))

        for example in tqdm(examples):
            task = f"Task: {example['confirmed_task']}\n"
            keypoints = example['keypoints']
            for a_id, action in enumerate(example['actions']):
                dict = {"id": f"{example['annotation_id']}-{action['action_uid']}"}
                image = 'Screenshot: <img>' + \
                        os.path.join('/data/luohh/agent/SeeAct/data/data/raw_dump/task',example['annotation_id'],'processed/snapshots',action['action_uid']+'_before.jpg') + f'</img>\n'
                prompt = f"""\nYou are a proficient planner for web navigation task. Given several keypoints extracted from task description and based on the web screenshot, please give me a plan for the web navigation task, including specific element and action (e.g. Zip code 123456 -> Type). Note that all the elements in your plan should be strictly derived from the keypoints. You must stritly adhere to the following format given in the example using 1. 2. 3. as separator. Don't include any specific detailed steps like [span]!!!
                
                Refer to the following examples and imitate their strategy to guide your planning.
                **Example 1:**
                Task: Find the cheapest women's plus size brown color loungewear in 3xl size.
                Keypoints: 
                (1) cheapest
                (2) women's
                (3) brown
                (4) loungewear
                (5) 3xl
                Plan: 
                1. women's -> HOVER
                2. loungewear -> CLICK
                3. 3xl size -> CLICK
                4. cheapest -> CLICK
                5. brown color -> CLICK
    
                **Example 2:**
                Task: see Nissan and Honda cars for sale near Kentwood, MI 49512
                Keypoints: 
                (1) Nissan
                (2) Honda
                (3) cars for sale
                (4) near Kentwood, MI
                (5) 49512
                Plan: 
                1. cars for sale -> CLICK
                2. near Kentwood, MI -> CLICK
                3. 49512 -> TYPE
                4. Nissan -> CLICK
                5. Honda -> CLICK
                """
                user_value = image + task + keypoints + prompt
                assistant_value = example['plan']
                # if assistant_value.find('Fishing') != -1:
                #     continue
                conversations = [
                    {
                        'from': 'user',
                        'value': user_value
                    },
                    {
                        'from': 'assistant',
                        'value': assistant_value
                    }
                ]
                dict.update({'conversations':conversations})
                # print(dict)
                # pdb.set_trace()
                save_list.append(dict)
with open(save_root, 'w') as fp:
    json.dump(save_list, fp)
