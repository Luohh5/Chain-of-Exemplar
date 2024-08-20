import pdb
import time
from sentence_transformers import SentenceTransformer, util
import json, os
from tqdm import tqdm
import pandas as pd

from angle_emb import AnglE, Prompts

angle = AnglE.from_pretrained('Llama-2-7b-chat-hf', pretrained_lora_path='angle-llama-7b-nli-v2')

print('All predefined prompts:', Prompts.list_prompts())
angle.set_prompt(prompt=Prompts.A)
print('prompt:', angle.prompt)
vec = angle.encode({'text': 'hello world'}, to_numpy=True)
print(vec.shape)
vecs = angle.encode([{'text': 'hello world1'}, {'text': 'hello world2'}], to_numpy=True)
print(vecs.shape)
pdb.set_trace()
def top_k(my_dict, k):
    # lst_with_index = [(idx, val) for idx, val in enumerate(similar_list)]
    # # 按照值从大到小排序
    # sorted_lst = sorted(lst_with_index, key=lambda x: x[1], reverse=True)
    # # 取前k个元素的索引和值
    # top_k_index = [x[0] for x in sorted_lst[:k]]
    series_obj = pd.Series(my_dict)
    sorted_series = series_obj.sort_values(ascending=False)
    sorted_dict = sorted_series.to_dict()
    top_k_index = []
    for key in sorted_dict.keys():
        top_k_index.append(key)
        if len(top_k_index) >= k:
            break
    return top_k_index

model = SentenceTransformer('all-mpnet-base-v2')
def calculate_similar(sentence1, sentence2):
    # tokenizer = AutoTokenizer.from_pretrained('LinkBERT-large',local_files_only=True, trust_remote_code=True)
    # model = AutoModel.from_pretrained('LinkBERT-large', trust_remote_code=True)
    # sentence_1 = tokenizer("A", return_tensors="pt")
    # output_1 = model(**sentence_1).last_hidden_state
    # sentence_2 = tokenizer("A woman watches TV", return_tensors="pt")
    # output_2 = model(**sentence_2).last_hidden_state
    # print(output_1.shape)
    # print(output_2.shape)
    # pdb.set_trace()
    # cosine_scores = util.cos_sim(output_1, output_2)

    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)[0][0].item()

    return cosine_scores

context_save_root = 'retrieve/context_embedding.json'
answer_save_root = 'retrieve/answer_embedding.json'
question_save_root = 'retrieve/question_embedding.json'

def save_embedding(problems):
    train_dict = {}
    context_dict = {}
    answer_dict = {}
    question_dict = {}
    for k,v in problems.items():
        if v['split'] == 'train':
            train_dict.update({k:v})
    for qid in tqdm(train_dict):
        context = train_dict[qid]['hint'] if train_dict[qid]['hint'] != '' else ''
        answer = train_dict[qid]['choices'][train_dict[qid]['answer']]
        question = train_dict[qid]['question']
        context_emb = model.encode(context, convert_to_tensor=False).tolist()
        answer_emb = model.encode(answer, convert_to_tensor=False).tolist()
        question_emb = model.encode(question, convert_to_tensor=False).tolist()
        context_dict.update({qid:context_emb})
        answer_dict.update({qid: answer_emb})
        question_dict.update({qid: question_emb})
    with open(context_save_root, 'w') as fp:
        json.dump(context_dict, fp)
    with open(answer_save_root, 'w') as fp:
        json.dump(answer_dict, fp)
    with open(question_save_root, 'w') as fp:
        json.dump(question_dict, fp)

def retrieve_example():
    data_root = '/data/luohh/acl23/data/scienceqa/'
    problems = json.load(open(os.path.join(data_root, 'problems.json')))
    # context_embedding = json.load(open(context_save_root))
    answer_embedding = json.load(open(answer_save_root))
    question_embedding = json.load(open(question_save_root))
    thres = 0.6
    k = 5
    split = 'train'
    new_problems = {}
    for qid in tqdm(problems):
        if problems[qid]['split'] == split:
            if problems[qid]['hint'] != '' or problems[qid]['image'] != None:
                continue
            # context_1 = context_embedding[qid]
            answer_1 = answer_embedding[qid]
            question_1 = question_embedding[qid]
            similar_dict = {}
            for qid_new in answer_embedding:
                if qid_new != qid:
                    answer_2 = answer_embedding[qid_new]
                    question_2 = question_embedding[qid_new]
                    # if problems[qid_new]['hint'] == '':
                    #     context_similar = 0
                    # else:
                        # context_2 = context_embedding[qid_new]
                        # context_similar = util.cos_sim(context_1, context_2)[0][0].item()
                    answer_similar = util.cos_sim(answer_1, answer_2)[0][0].item()
                    question_similar = util.cos_sim(question_1, question_2)[0][0].item()
                    # Signals Maximum
                    # similar = max(context_similar,answer_similar,question_similar)
                    # similar = max(answer_similar, question_similar)
                    # Signals Combined
                    # similar = answer_similar + question_similar + context_similar
                    similar = answer_similar + question_similar
                else:
                    similar = 0
                if similar >= thres * 2:
                    similar_dict.update({qid_new:similar})
            topk = top_k(similar_dict, k)
            # for example in topk:
            #     if problems[example]['hint'] != '' or problems[example]['image'] != None:
            #         print(problems[qid])
            #         print(problems[example])
            #         pdb.set_trace()
            problems[qid].update({'relevant_question':topk})
        else:
            problems[qid].update({'relevant_question': []})
        new_problems.update({qid:problems[qid]})
    with open(f'/data/luohh/acl23/data/scienceqa/problems.json', 'w') as fp:
        json.dump(new_problems, fp)

retrieve_example()
# sentences1 = 'chemical change'
# sentences2 = 'physical change'
# print(calculate_similar(model,sentences1,sentences2))