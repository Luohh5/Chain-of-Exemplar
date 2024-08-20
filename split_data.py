import pdb
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import json
import math
task = 'dg'
split = 'val'
examples = json.load(open(f'data/ScienceQA_{split}_{task}_imageonly.json'))
question_files = []
questions = []
save_root = f'data/ScienceQA_{split}_{task}_imageonly'
sub_list_length = math.ceil(len(examples)/4)
sub_lists = [examples[i:i+sub_list_length] for i in range(0, len(examples), sub_list_length)]
for i, sub_list in enumerate(sub_lists):
    print(len(sub_list))
    with open(save_root+f'/{i}.json', 'w') as fp:
        json.dump(sub_list, fp)