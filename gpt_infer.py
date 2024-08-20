
import json
import pdb
import jsonlines
import os
from tqdm import tqdm
import openai
API_KEY = 'sk-rTTXBQseFSTLa9xwbhnVcOPfXKZXYIOFDMVKKxA8KhiIR474'
openai.api_key = API_KEY
openai.api_base = "https://api.chatanywhere.com.cn/v1"

def build_example(problem,id,num):
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
        qg = f'\nExample {num}:\n' + image + context + answer + f'Question: {question}'
        rg = f'\nExample {num}:\n' + image + context + f'Question: {question}\n' + answer + f'Reasoning: {rationale}'
        dg = f'\nExample {num}:\n' + image + context + f'Question: {question}\n' + f'Reasoning: {rationale}' + answer + f'Distractors: {distractors}'
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
        qg = f'\nExample {num}:\n' + context + answer + f'Question: {question}'
        rg = f'\nExample {num}: \n' + context + f'Question: {question}\n' + answer + f'Reasoning: {rationale}'
        dg = f'\nExample {num}:\n' + context + f'Question: {question}\n' + f'Reasoning: {rationale}' + answer + f'Distractors: {distractors}'
    return qg, rg, dg
shot = 0
task = 'qg'
split = 'test'
data_root = '/data/luohh/acl23/data/scienceqa/'
problems = json.load(open(os.path.join(data_root, 'problems_blip2xl_angle.json')))
save_root = f'/data/luohh/acl23/Qwen-VL/infer/pred_{split}_gpt_{task}_{shot}shot/2.json'
save_list = []
id = 0
for qid in tqdm(problems):
    if problems[qid]['split'] == split:
        save_dict = {"id": f"identity_{qid}"}
        if int(qid) <= 8670:
            continue
        id += 1
        if not problems[qid]["relevant_question"]:
            prefix = ''
            qg_example, rg_example, dg_example = '','',''
            qg_example_1, qg_example_2, qg_example_3 = '', '', ''
            dg_example_1, dg_example_2, dg_example_3 = '','',''
        # elif len(problems[qid]["relevant_question"]) < 3:
        #     prefix = f'Refer to the following {len(problems[qid]["relevant_question"])} examples, '
        #     # relevant_question = []
        #     qg_example, rg_example, dg_example = [], [], []
        #     for i in range(len(problems[qid]["relevant_question"])):
        #         relevant_question = problems[qid]["relevant_question"][i]
        #         qg_example_1, rg_example_1, dg_example_1 = build_example(problems[relevant_question],
        #                                                                  relevant_question, i+1)
        #         qg_example.append(qg_example_1)
        #         rg_example.append(rg_example_1)
        #         dg_example.append(dg_example_1)
        #     qg_example_1, rg_example_1, dg_example_1 = qg_example[0], rg_example[0], dg_example[0]
        #     qg_example_3, rg_example_3, dg_example_3 = '', '', ''
        #     if len(problems[qid]["relevant_question"]) == 2:
        #         qg_example_2, rg_example_2, dg_example_2 = qg_example[1], rg_example[1], dg_example[1]
        #     else:
        #         qg_example_2, rg_example_2, dg_example_2 = '','',''
        else:
            prefix = f'Refer to the following example, '
            most_relevant_question = problems[qid]["relevant_question"][0]
            # second_relevant_question = problems[qid]["relevant_question"][1]
            # third_relevant_question = problems[qid]["relevant_question"][2]
            qg_example_1, rg_example_1, dg_example_1 = build_example(problems[most_relevant_question],most_relevant_question,1)
            # qg_example_2, rg_example_2, dg_example_2 = build_example(problems[second_relevant_question],
            #                                                          second_relevant_question,2)
            # qg_example_3, rg_example_3, dg_example_3 = build_example(problems[third_relevant_question],
            #                                                          third_relevant_question,3)
        # prefix = ''
        # qg_example, rg_example, dg_example = '','',''
        rationale = 'Reasoning: ' + problems[qid]['lecture'] + problems[qid]['solution'] + '\n'
        question = 'Question: ' + problems[qid]['question'] + '\n'
        answer = 'Answer: ' + problems[qid]['choices'][problems[qid]['answer']] + '\n'
        if problems[qid]['image'] != None:
            image = f'Picture: {problems[qid]["image_caption"]}\n'
            context = f'Context: ' + problems[qid]['hint'] + '\n' if problems[qid]['hint']!='' else ''
            qg_prompt = 'generate a question based on the above picture, context and the corresponding answer.' if problems[qid]['hint']!='' else 'generate a question based on the above picture and the corresponding answer.'
            qg_value = image + context + answer + qg_prompt
            # dg_prompt = 'generate a question based on the above picture, context and the corresponding answer.' \
            #     if problems[qid]['hint']!='' else 'generate a question based on the above picture and the corresponding answer.'
            # dg_value = image + context + answer + prefix + qg_prompt + qg_example_1 + qg_example_2 + qg_example_3
        else:
            context = 'Context: ' + problems[qid]['hint'] + '\n' if problems[qid]['hint']!='' else ''
            qg_prompt = 'generate a question based on the above context and the corresponding answer.' if problems[qid]['hint']!='' else 'generate a question based on the corresponding answer.'
            qg_value = context + answer + qg_prompt
            # dg_prompt = 'generate a question based on the above context and the corresponding answer.' if \
            # problems[qid][
            #     'hint'] != '' else 'generate a question based on the corresponding answer.'
            # dg_value = context + answer + prefix + qg_prompt + qg_example_1 + qg_example_2 + qg_example_3
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "a professional question generator"},
                {"role": "user", "content": qg_value}
            ]
        )
        result = response.choices[0].message['content']
        # print(type(result))
        # pdb.set_trace()
        # conversations = [
        #                     {
        #                         'from': 'user',
        #                         'value': user_value
        #                     },
        #                     {
        #                         'from': 'assistant',
        #                         'value': assistant_value
        #                     }
        #                 ]
        # print(result)
        save_dict.update({'response':result})
        # print(dict)
        # pdb.set_trace()
        save_list.append(save_dict)
        with open(save_root, 'w') as fp:
            json.dump(save_list, fp)

# prompt = 'Answer: The snoring is loud.\nPlease generate a question from the corresponding answer.'
# response = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#         {"role": "system", "content": "a professional question generator"},
#         {"role": "user", "content": prompt}
#     ]
# )
# result = response.choices[0].message['content']
# print(result)