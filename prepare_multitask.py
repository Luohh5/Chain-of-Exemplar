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
def image_caption(problem, use_caption):
    if use_caption:
        caption = f'Picture describes {problem["image_caption"]}\n'
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
        # dg = '\nExample:\n' + image + context + f'Question: {question}\n' + f'Reasoning: {rationale}\n' + answer + f'Distractors: {distractors}'
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
        # dg = '\nExample:\n' + context + f'Question: {question}\n' + f'Reasoning: {rationale}\n' + answer + f'Distractors: {distractors}'
        dg = '\nExample:\n' + context + f'Question: {question}\n' + answer + f'Distractors: {distractors}'
    return qg, rg, dg
data_root = '/data/luohh/acl23/data/scienceqa/'
problems = json.load(open(os.path.join(data_root, 'problems_blip2xl_angle.json')))
save_list = []
split = 'train'
save_root = f'data/ScienceQA_{split}_multitask_norationale.json'
id = 0
for qid in problems:
    if problems[qid]['split'] == split:
        answer = 'Answer: ' + problems[qid]['choices'][problems[qid]['answer']] + '\n'
        question = problems[qid]['question']
        rationale = problems[qid]['lecture'] + problems[qid]['solution']
        choices = problems[qid]['choices'].copy()
        choices.remove(problems[qid]['choices'][problems[qid]['answer']])
        distractors = ''
        for i, choice in enumerate(choices):
            distractors += f'({i + 1}) {choice}\n'
        id += 1
        if not problems[qid]["relevant_question"]:
            prefix = ''
            qg_example, rg_example, dg_example = '','',''
        else:
            prefix = 'Refer to the following example, '
            most_relevant_question = problems[qid]["relevant_question"][0]
            qg_example, rg_example, dg_example = build_example(problems[most_relevant_question],most_relevant_question)
        if problems[qid]['image'] != None:
            image = 'Picture: <img>' + \
                    os.path.join(data_root, problems[qid]['split'], str(qid), problems[qid]['image']) + f'</img>\n'
            context = f'Context: ' + problems[qid]['hint'] + '\n' if problems[qid]['hint']!='' else ''
            qg_prompt = 'generate a question based on the above picture, context and the corresponding answer.' \
                if problems[qid]['hint']!='' else 'generate a question based on the above picture and the corresponding answer.'
            rg_prompt = 'how to make a reasoning to answer the question based on the above picture (with description), context and question with its answer?' \
                if problems[qid]['hint'] != '' else 'how to make a reasoning to answer the question based on the above picture (with description) and question with its answer?'
            # dg_prompt = 'based on the above picture (with description), context and question with its answer obtained through reasoning, generate at least 1 plausible yet incorrect answers which should be similar and grammatically consistent with the correct answer and seperate them with numbers like (1) (2) (3).' \
            #     if problems[qid]['hint'] != '' else 'based on the above picture (with description) and question with its answer obtained through reasoning, generate at least 1 plausible yet incorrect answers which should be similar and grammatically consistent with the correct answer and seperate them with numbers like (1) (2) (3).'
            dg_prompt = 'based on the above picture, context and question with its answer, generate at least 1 plausible yet incorrect answers which should be similar and grammatically consistent with the correct answer and seperate them with numbers like (1) (2) (3).' \
                if problems[qid][
                       'hint'] != '' else 'based on the above picture and question with its answer, generate at least 1 plausible yet incorrect answers which should be similar and grammatically consistent with the correct answer and seperate them with numbers like (1) (2) (3).'
            qg_user_value = image + context + answer + prefix + qg_prompt + qg_example
            qg_assistant_value = question
            rg_user_value = image + context + f'Question: {question}\n' + answer + prefix + rg_prompt + rg_example
            rg_assistant_value = rationale
            dg_user_value = image + context + f'Question: {question}\n' + answer + prefix[:-2] + ' and ' + dg_prompt + dg_example
            dg_assistant_value = distractors
        else:
            context = 'Context: ' + problems[qid]['hint'] + '\n' if problems[qid]['hint']!='' else ''
            qg_prompt = 'generate a question based on the above context and the corresponding answer.' if \
            problems[qid][
                'hint'] != '' else 'generate a question based on the corresponding answer.'
            rg_prompt = 'how to make a reasoning to answer the question based on the above context and question with its answer?' if \
            problems[qid][
                'hint'] != '' else 'how to make a reasoning to answer the question based on the above question with its answer?'
            dg_prompt = 'based on the above context and question with its answer, generate at least 1 plausible yet incorrect answers which should be similar and grammatically consistent with the correct answer and seperate them with numbers like (1) (2) (3).' if \
            problems[qid][
                'hint'] != '' else 'based on the above question with its answer, generate at least 1 plausible yet incorrect answers which should be similar and grammatically consistent with the correct answer and seperate them with numbers like (1) (2) (3).'

            qg_user_value = context + answer + prefix + qg_prompt + qg_example
            qg_assistant_value = question
            rg_user_value = context + f'Question: {question}\n' + answer + prefix + rg_prompt + rg_example
            rg_assistant_value = rationale
            dg_user_value = context + f'Question: {question}\n' + answer + prefix[:-2] + ' and ' + dg_prompt + dg_example
            dg_assistant_value = distractors
        qg_conversations = [
                            {
                                'from': 'user',
                                'value': qg_user_value
                            },
                            {
                                'from': 'assistant',
                                'value': qg_assistant_value
                            }
                        ]
        rg_conversations = [
            {
                'from': 'user',
                'value': rg_user_value
            },
            {
                'from': 'assistant',
                'value': rg_assistant_value
            }
        ]
        dg_conversations = [
            {
                'from': 'user',
                'value': dg_user_value
            },
            {
                'from': 'assistant',
                'value': dg_assistant_value
            }
        ]
        qg_dict = {"id": f"qg_{id}","conversations":qg_conversations}
        rg_dict = {"id": f"rg_{id}","conversations":rg_conversations}
        dg_dict = {"id": f"dg_{id}", "conversations": dg_conversations}
        # if qid == '7909':
        #     print(qg_dict)
        #     print(rg_dict)
        #     print(dg_dict)
        #     pdb.set_trace()
        save_list.extend([qg_dict,dg_dict])
with open(save_root, 'w') as fp:
    json.dump(save_list, fp)
# with jsonlines.open(save_root, mode='w') as writer:
#     for item in save_list:
#         writer.write(item)