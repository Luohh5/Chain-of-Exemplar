import json
import pdb
from distinct_n import distinct_n_corpus_level, distinct_n_sentence_level
from paddlenlp.metrics import Distinct
distinct = Distinct(n_size=1)
cand = ["The","cat","The","cat","on","the","mat"]
#update the states
# distinct.add_inst(cand)
# print(distinct.score())
# 0.8333333333333334
# pdb.set_trace()
sentences = ''
pred = json.load(open('infer/pred_gpt_qg_3shot.json'))
for i, example in enumerate(pred):
    sentences += example['response']
# print(sentences)
distinct.add_inst(sentences.split())
print(distinct.score())
a = 'the cat sat on the mat the cat sat on the mat the cat sat on the mat the cat sat on the mat' * 10000
# sentences = a.split()
# print(distinct_n_sentence_level(sentences, 4))
pdb.set_trace()
pred = json.load(open('infer/pred_val_qg.json'))
gt = json.load(open('data/ScienceQA_val_qg.json'))
problems = json.load(open('/data/luohh/acl23/data/scienceqa/problems.json'))
for i, example in enumerate(gt):
    print('gt: ', example['conversations'][1]['value'])
    print('pr: ', pred[i])
    print('\n')
    pdb.set_trace()
# tasks = {'closed choice':0, 'yes or no':0, 'true-or false':0}
# answer_only = 0
# for example in problems.values():
#     if example['hint'] == '' and example['image'] == None:
#         answer_only += 1
# print(answer_only)