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
import random
random.seed(2024)
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
        dg = '\nExample:\n' + image + context + f'Question: {question}\n' + f'Reasoning: {rationale}\n' + answer + f'Distractors: {distractors}'
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
        dg = '\nExample:\n' + context + f'Question: {question}\n' + f'Reasoning: {rationale}\n' + answer + f'Distractors: {distractors}'
    return qg, rg, dg

# problems = json.load(open('/data/luohh/Qwen-VL/csc_detect.json'))
# for problem in problems:
#     print(problem)
#     pdb.set_trace()
data_root = '/data/luohh/dataset'
tgt_list = []
save_list = []
save_root = f'csc_detect_test.json'
with open(os.path.join(data_root, 'tgt_new.txt'), 'r') as file:
    lines = file.readlines()
    for line in lines:
        tgt_list.append(line.split()[1])
with open(os.path.join(data_root, 'src_new.txt'), 'r') as file:
    lines = file.readlines()
    for num, line in enumerate(lines):
        if num >= 8000:
            dict = {"id": f"identity_{num}"}
            example = line.split()
            image, target = example[0], example[1]
            image_input = 'Picture: <img>' + \
                    os.path.join(data_root, 'pic', image) + f'</img>\n'
            error = ''
            with open(os.path.join(data_root, 'singleOCR', image.replace('.jpg','.txt')), 'r') as positions:
                position = positions.readlines()
                i = 1
                for n, pos in enumerate(position):
                    pos_list = pos.split(', ')
                    if pos_list[-1] == 'X\n':
                        error += f'<ref>错字{i}</ref><box>({pos_list[0]},{pos_list[1]}),({pos_list[2]},{pos_list[3]})</box>\n'
                        # true_word = tgt_list[num][n]
                        # print(true_word)
                        # pdb.set_trace()
                        i += 1
            if error == '':
                error = '这段句子没有错字'
            prompt = '这是一张字迹潦草的手写体作文图片，请识别出其中的完整句子，将写错的字用X代替，并将每个X的位置框出'
            user_value = image_input + prompt
            assistant_value = target + '\n' + error
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
            # true_sentence = tgt_list[num]
            dict.update({'conversations': conversations})
            # print(dict)
            # pdb.set_trace()
            save_list.append(dict)
with open(save_root, 'w') as fp:
    json.dump(save_list, fp)
pdb.set_trace()
save_list = []
split = 'val'
save_root = f'data/ScienceQA_{split}_qg_angle_2ep.json'
id = 0
sample_ids = []
for qid in problems:
    if problems[qid]['split'] == split:
        sample_ids.append(qid)
for qid in problems:
    if problems[qid]['split'] == split:
        dict = {"id": f"identity_{qid}"}
        id += 1
        if not problems[qid]["relevant_question"]:
            prefix = ''
            qg_example_1, rg_example_1, dg_example_1, qg_example_2, rg_example_2, dg_example_2 = '', '', '', '', '', ''
        else:
            prefix = 'Refer to the following example, '
            most_relevant_question = problems[qid]["relevant_question"][0]
            # most_relevant_question = random.choice(sample_ids)
            # print(most_relevant_question)
            # if most_relevant_question == qid:
            #     most_relevant_question = random.choice(sample_ids)
            # qg_example, rg_example, dg_example = build_example(problems[most_relevant_question],most_relevant_question)
            qg_example_1, rg_example_1, dg_example_1 = build_example(problems[most_relevant_question],
                                                                     most_relevant_question)
            if len(problems[qid]["relevant_question"]) > 1:
                most_relevant_question = problems[qid]["relevant_question"][1]
                qg_example_2, rg_example_2, dg_example_2 = build_example(problems[most_relevant_question],
                                                                         most_relevant_question)
            else:
                qg_example_2, rg_example_2, dg_example_2 = '', '', ''
        if problems[qid]['image'] != None:
            image = 'Picture: <img>' + \
                    os.path.join(data_root, problems[qid]['split'], str(qid), problems[qid]['image']) + f'</img>\n'
            context = f'Context: ' + problems[qid]['hint'] + '\n' if problems[qid]['hint']!='' else ''
            answer = 'Answer: ' + problems[qid]['choices'][problems[qid]['answer']] + '\n'
            question = problems[qid]['question']
            prompt = 'generate a question based on the above picture, context and the corresponding answer.' if problems[qid]['hint']!='' else 'generate a question based on the above picture and the corresponding answer.'
            user_value = image + context + answer + prefix + prompt + qg_example_1 + qg_example_2
            assistant_value = question
        else:
            context = 'Context: ' + problems[qid]['hint'] + '\n' if problems[qid]['hint']!='' else ''
            answer = 'Answer: ' + problems[qid]['choices'][problems[qid]['answer']] + '\n'
            question = problems[qid]['question']
            prompt = 'generate a question based on the above context and the corresponding answer.' if problems[qid]['hint']!='' else 'generate a question based on the corresponding answer.'
            user_value = context + answer + prefix + prompt + qg_example_1 + qg_example_2
            assistant_value = question
        if len(user_value.split()) > 2048:
            user_value = " ".join(user_value.split()[:2048])
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
# with jsonlines.open(save_root, mode='w') as writer:
#     for item in save_list:
#         writer.write(item)