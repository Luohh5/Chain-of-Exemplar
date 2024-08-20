# using mplug environment

import pdb
import json
import jsonlines
import nltk
import os, string, re, json, argparse, evaluate
# from sentence_transformers import SentenceTransformer, util
# from angle_emb import AnglE, Prompts
# from bleu.bleu import Bleu
import pandas as pd
from tqdm import tqdm
from evaluate.utils.file_utils import DownloadConfig
from evaluate import load
from metrics.bleu import bleu as Bleu
from metrics.rouge import rouge as Rouge
from metrics.meteor import meteor as Meteor
from metrics.bleurt import bleurt as Bleurt
import datasets
# nltk.download('wordnet')
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7,8,9'
def semantic_similarity(vec, vecs):
    vec = angle.encode({'text': vec}, to_numpy=True)
    vec = torch.tensor(vec[0])
    vecs = angle.encode({'text': vecs}, to_numpy=True)
    vecs = torch.tensor(vecs[0])
    return util.cos_sim(vec, vecs)[0][0].item()

def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def grade_score(df, bleurt):
    nls = []
    for curit, (q, gq) in enumerate(zip(df['question'], df['generated_question'])):
        result = bleurt.compute(predictions=[normalize(gq)], references=[normalize(q)])
        nls.append(result)
    return nls


def get_batch(iterable, n=1):
    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def ceildiv(a, b):
    # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python/17511341#17511341
    return -(a // -b)


def grade_score_with_batching(df, bleurt, batch_size=64, normalize_flag=True):
    # Add batching to speed up BLEURT model computation
    # Note: BLEURT metric is non commutative, therefore predictions must match questions generated
    # df['target'] = df['target'].apply(normalize)
    # if normalize_flag:
    #     df['generated_question'] = df['generated_question'].apply(normalize)

    # ref_q = df['target'].tolist()
    # gen_q = df['generated_question'].tolist()
    ref_q = df['target']
    gen_q = df['generated_question']

    scores = []
    num_batches = ceildiv(len(ref_q), batch_size)
    for ref_q_batch, gen_q_batch in tqdm( zip(get_batch(ref_q, batch_size), get_batch(gen_q, batch_size)), total=num_batches ):
        batch_scores = bleurt.compute(predictions=gen_q_batch, references=ref_q_batch)
        scores.extend(batch_scores["scores"])

    return scores
def ml_metrics(results):
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    bleurt = evaluate.load('bleurt', 'bleurt-20')

    bleu4s, meteors, rouges = [], [], []

    bleurt_scores = grade_score_with_batching(results, bleurt, 64)

    for _, ans in tqdm(results.iterrows(), total=results.shape[0]):
        ref = ans['target']
        hyp = ans['generated_question']
        bleu4s.append(bleu.compute(predictions=[hyp], references=[ref])['bleu'])
        meteors.append(meteor.compute(predictions=[hyp], references=[ref])['meteor'])
        rouges.append(rouge.compute(predictions=[hyp], references=[ref])['rougeL'])

    res_len = len(bleu4s)
    # b1, b2, b3, b4 = sum(bleu1s) / res_len, sum(bleu2s) / res_len, sum(bleu3s) / res_len, sum(bleu4s) / res_len
    b4 = sum(bleu4s) / res_len
    meteor_score = sum(meteors) / res_len
    rouge_l = sum(rouges) / res_len
    bleurt_score = sum(bleurt_scores) / res_len

    # print("BLEU-N-grams: 1-{:.4f}, 2-{:.4f}, 3-{:.4f}, 4-{:.4f}".format(b1, b2, b3, b4))
    print("BLEU-4: {:.4f}".format(b4))
    print("METEOR: {:.4f}".format(meteor_score))
    print("ROUGE-L: {:.4f}".format(rouge_l))
    print("BLEURT: {:.4f}".format(bleurt_score))

    return {'bleu_4': b4, 'meteor': meteor_score, 'rouge': rouge_l, 'bleurt': bleurt_score}
def show_metric(results, bleurt, bleu4s, meteors, rouges, task):
    bleurt_scores = grade_score_with_batching(results, bleurt, 64)
    res_len = len(bleurt_scores)
    # b1, b2, b3, b4 = sum(bleu1s) / res_len, sum(bleu2s) / res_len, sum(bleu3s) / res_len, sum(bleu4s) / res_len
    b4 = sum(bleu4s) / res_len
    meteor_score = sum(meteors) / res_len
    rouge_l = sum(rouges) / res_len
    bleurt_score = sum(bleurt_scores) / res_len
    print(task,": ")
    # print("BLEU-N-grams: 1-{:.4f}, 2-{:.4f}, 3-{:.4f}, 4-{:.4f}".format(b1, b2, b3, b4))
    print("BLEU-4: {:.4f}".format(b4))
    print("METEOR: {:.4f}".format(meteor_score))
    print("ROUGE-L: {:.4f}".format(rouge_l))
    print("BLEURT: {:.4f}".format(bleurt_score))
    print('bleu_4', b4, 'meteor', meteor_score, 'rouge', rouge_l, 'bleurt', bleurt_score)

task = 'qg'
split = 'val'
shot = '0'
gt = json.load(open(f'/data/luohh/acl23/Qwen-VL/data/ScienceQA_{split}_{task}_blip2xl_angle.json'))
# generated_question = json.load(open(f'/data/luohh/acl23/Qwen-VL/infer/pred_{split}_{task}_textonly.json'))
generated_question = json.load(open(f'/data/luohh/acl23/Qwen-VL/infer/pred_gpt_{task}_{shot}shot.json'))
problems = json.load(open(f'/data/luohh/acl23/data/scienceqa/problems.json'))
target = []
for example in tqdm(gt):
    target.append(example['conversations'][1]['value'])

assert len(target) == len(generated_question), f'target len: {len(target)} != predict len: {len(generated_question)}!!'

bleu = evaluate.load('/data/luohh/evaluate/metrics/bleu')
rouge = evaluate.load('/data/luohh/evaluate/metrics/rouge')
meteor = evaluate.load('/data/luohh/evaluate/metrics/meteor')
bleurt = evaluate.load('/data/luohh/evaluate/metrics/bleurt', 'bleurt-20')
bleu4s, meteors, rouges, refs, hyps = [], [], [], [], []
bleu4s_language, meteors_language, rouges_language, refs_language, hyps_language = [], [], [], [], []
bleu4s_social, meteors_social, rouges_social, refs_social, hyps_social = [], [], [], [], []
bleu4s_natural, meteors_natural, rouges_natural, refs_natural, hyps_natural = [], [], [], [], []
bleu4s_g16, meteors_g16, rouges_g16, refs_g16, hyps_g16 = [], [], [], [], []
bleu4s_g712, meteors_g712, rouges_g712, refs_g712, hyps_g712 = [], [], [], [], []
bleu4s_image, meteors_image, rouges_image, refs_image, hyps_image = [], [], [], [], []
bleu4s_text, meteors_text, rouges_text, refs_text, hyps_text = [], [], [], [], []
bleu4s_no, meteors_no, rouges_no, refs_no, hyps_no = [], [], [], [], []
for i, example in tqdm(enumerate(gt)):
    # if example['conversations'][0]['value'][:6] == 'Answer':
    #     continue
    qid = generated_question[i]['id'][9:]
    subject = problems[qid]['subject']
    grade = int(problems[qid]['grade'][5:])
    image = problems[qid]['image']
    text = problems[qid]['hint']
    hyp = generated_question[i]['response']
    ref = target[i]
    if hyp == '':
        hyp = 'None'
    if ref == '':
        ref = 'None'
    bl = bleu.compute(predictions=[hyp], references=[ref])['bleu']
    me = meteor.compute(predictions=[hyp], references=[ref])['meteor']
    ro = rouge.compute(predictions=[hyp], references=[ref])['rougeL']
    rf = normalize(ref)
    hy = normalize(hyp)
    bleu4s.append(bl)
    meteors.append(me)
    rouges.append(ro)
    refs.append(rf)
    hyps.append(hy)
    if subject == "social science":
        bleu4s_social.append(bl)
        meteors_social.append(me)
        rouges_social.append(ro)
        refs_social.append(rf)
        hyps_social.append(hy)
    elif subject == "natural science":
        bleu4s_natural.append(bl)
        meteors_natural.append(me)
        rouges_natural.append(ro)
        refs_natural.append(rf)
        hyps_natural.append(hy)
    elif subject == "language science":
        bleu4s_language.append(bl)
        meteors_language.append(me)
        rouges_language.append(ro)
        refs_language.append(rf)
        hyps_language.append(hy)
    if grade < 7:
        bleu4s_g16.append(bl)
        meteors_g16.append(me)
        rouges_g16.append(ro)
        refs_g16.append(rf)
        hyps_g16.append(hy)
    elif grade >= 7:
        bleu4s_g712.append(bl)
        meteors_g712.append(me)
        rouges_g712.append(ro)
        refs_g712.append(rf)
        hyps_g712.append(hy)
    if image == None and text == '':
        bleu4s_no.append(bl)
        meteors_no.append(me)
        rouges_no.append(ro)
        refs_no.append(rf)
        hyps_no.append(hy)
    elif image != None and text == '':
        bleu4s_image.append(bl)
        meteors_image.append(me)
        rouges_image.append(ro)
        refs_image.append(rf)
        hyps_image.append(hy)
    elif image == None and text != '':
        bleu4s_text.append(bl)
        meteors_text.append(me)
        rouges_text.append(ro)
        refs_text.append(rf)
        hyps_text.append(hy)
results = {'target':refs, 'generated_question':hyps}
results_social = {'target':refs_social, 'generated_question':hyps_social}
results_language = {'target':refs_language, 'generated_question':hyps_language}
results_natural = {'target':refs_natural, 'generated_question':hyps_natural}
results_g16 = {'target':refs_g16, 'generated_question':hyps_g16}
results_g712 = {'target':refs_g712, 'generated_question':hyps_g712}
results_image = {'target':refs_image, 'generated_question':hyps_image}
results_text = {'target':refs_text, 'generated_question':hyps_text}
results_no = {'target':refs_no, 'generated_question':hyps_no}
bleurt_scores = grade_score_with_batching(results, bleurt, 64)
res_len = len(bleurt_scores)
# b1, b2, b3, b4 = sum(bleu1s) / res_len, sum(bleu2s) / res_len, sum(bleu3s) / res_len, sum(bleu4s) / res_len
b4 = sum(bleu4s) / res_len
meteor_score = sum(meteors) / res_len
rouge_l = sum(rouges) / res_len
bleurt_score = sum(bleurt_scores) / res_len
print('Total: ')
# print("BLEU-N-grams: 1-{:.4f}, 2-{:.4f}, 3-{:.4f}, 4-{:.4f}".format(b1, b2, b3, b4))
print("BLEU-4: {:.4f}".format(b4))
print("METEOR: {:.4f}".format(meteor_score))
print("ROUGE-L: {:.4f}".format(rouge_l))
print("BLEURT: {:.4f}".format(bleurt_score))
print('bleu_4', b4, 'meteor', meteor_score, 'rouge', rouge_l, 'bleurt', bleurt_score)
show_metric(results_social, bleurt, bleu4s_social, meteors_social, rouges_social, '\nSocial')
show_metric(results_language, bleurt, bleu4s_language, meteors_language, rouges_language, '\nLanguage')
show_metric(results_natural, bleurt, bleu4s_natural, meteors_natural, rouges_natural, '\nNatural')
show_metric(results_g16, bleurt, bleu4s_g16, meteors_g16, rouges_g16, '\nG1-6')
show_metric(results_g712, bleurt, bleu4s_g712, meteors_g712, rouges_g712, '\nG7-12')
show_metric(results_image, bleurt, bleu4s_image, meteors_image, rouges_image, '\nImage')
show_metric(results_text, bleurt, bleu4s_text, meteors_text, rouges_text, '\nText')
show_metric(results_no, bleurt, bleu4s_no, meteors_no, rouges_no, '\nNo')
# print('bleu_4', b4, 'meteor', meteor_score, 'rouge', rouge_l)
# pdb.set_trace()
# with jsonlines.open('score_val' + ".jsonl", mode='w') as writer:
#     for item in log:
#         writer.write(item)

# reference = ["There is a cat on the mat"]
# reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog'],
#              ['this', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
# candidate = ["There is a cat on the mat"]
#
# scores = scorer.score(references=reference, candidates=candidate)
# assert isinstance(scores, list) and len(scores) == 1
# print(scores)
# score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
# print(score)
# rouge = Rouge()
# scores = rouge.get_scores(candidate,reference)
# print(scores[0]["rouge-1"]["f"])