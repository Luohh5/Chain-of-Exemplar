from sentence_transformers import SentenceTransformer, util
from angle_emb import AnglE, Prompts
import json
from tqdm import tqdm
import torch
angle = AnglE.from_pretrained('Llama-2-7b-chat-hf', pretrained_lora_path='angle-llama-7b-nli-v2')

print('All predefined prompts:', Prompts.list_prompts())
angle.set_prompt(prompt=Prompts.A)
print('prompt:', angle.prompt)
def semantic_similarity(vec, vecs):
    vec = angle.encode({'text': vec}, to_numpy=True)
    vec = torch.tensor(vec[0])
    vecs = angle.encode({'text': vecs}, to_numpy=True)
    vecs = torch.tensor(vecs[0])
    return util.cos_sim(vec, vecs)[0][0].item()

split = 'val'
problems = json.load(open('/data/luohh/acl23/data/scienceqa/problems.json'))
gt = json.load(open(f'data/ScienceQA_{split}_dg_blip2xl_angle.json'))
generated_question = json.load(open(f'infer/pred_{split}_dg_blip2xl_angle.json'))
target = []
for example in tqdm(gt):
    target.append(example['conversations'][1]['value'])
semantics = []
semantics_social, semantics_language, semantics_natural, semantics_g16, semantics_g712, semantics_image, semantics_text, semantics_no = [], [], [], [], [], [], [], []
for i, example in tqdm(enumerate(gt)):
    # if example['conversations'][0]['value'][:6] == 'Answer':
    #     continue
    qid = example['id'][9:]
    subject = problems[qid]["subject"]
    grade = int(problems[qid]['grade'][5:])
    image = problems[qid]['image']
    text = problems[qid]['hint']
    hyp = generated_question[i]['response']
    ref = target[i]
    if hyp == '':
        hyp = 'None'
    if ref == '':
        ref = 'None'
    semantic = semantic_similarity(hyp,ref)
    semantics.append(semantic)
    if subject == "social science":
        semantics_social.append(semantic)
    elif subject == "natural science":
        semantics_natural.append(semantic)
    elif subject == "language science":
        semantics_language.append(semantic)
    if grade < 7:
        semantics_g16.append(semantic)
    elif grade >= 7:
        semantics_g712.append(semantic)
    if image == None and text == '':
        semantics_no.append(semantic)
    elif image != None and text == '':
        semantics_image.append(semantic)
    elif image == None and text != '':
        semantics_text.append(semantic)
semantic_score = sum(semantics) / len(semantics)
social_score = sum(semantics_social) / len(semantics_social)
language_score = sum(semantics_language) / len(semantics_language)
natural_score = sum(semantics_natural) / len(semantics_natural)
g16_score = sum(semantics_g16) / len(semantics_g16)
g712_score = sum(semantics_g712) / len(semantics_g712)
image_score = sum(semantics_image) / len(semantics_image)
text_score = sum(semantics_text) / len(semantics_text)
no_score = sum(semantics_no) / len(semantics_no)
print('semantic_score: ',semantic_score)
print('social_score: ',social_score)
print('language_score: ',language_score)
print('natural_score: ',natural_score)
print('g16_score: ',g16_score)
print('g712_score: ',g712_score)
print('image_score: ',image_score)
print('text_score: ',text_score)
print('no_score: ',no_score)
