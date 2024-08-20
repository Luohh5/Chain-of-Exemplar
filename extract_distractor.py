import json
import pdb
distractors_list = []
raw_distractors = json.load(open('infer/pred_gpt_dg_3shot.json'))
for raw_distractor in raw_distractors:
    distractors = []
    for num in range(1,11):
        number = f'({num}) '
        start = raw_distractor['response'].find(number)
        if start == -1:
            continue
        start += len(number)
        number = f'({num+1}) '
        end = raw_distractor['response'].find(number)
        if end == -1:
            distractors.append(raw_distractor['response'][start:])
        else:
            distractors.append(raw_distractor['response'][start:end-1])
    # starts = raw_distractor['response'].split(') ')
    # ends = raw_distractor['response'].split('\n')
    # for end in ends:
    #     distractor = end.split(') ')
    #     if len(distractor) == 1:
    #         if distractor[0] not in distractors:
    #             distractors.append(distractor[0])
    #     else:
    #         distractors.append(distractor[1])
    distractors_list.append(distractors)
problems = json.load(open('/data/luohh/acl23/data/scienceqa/problems.json'))
problems_dict = {}
iter = 0
for qid in problems:
    if problems[qid]['split'] == 'val':
        correct_answer = problems[qid]['choices'][problems[qid]['answer']]
        choice = distractors_list[iter]
        choice.append(correct_answer)
        problems[qid]['choices'] = choice
        problems[qid]['answer'] = len(choice) - 1
        iter += 1
    problems_dict.update({qid:problems[qid]})
with open('/data/luohh/acl23/data/scienceqa/problems_pred_gpt3shot.json', 'w') as fp:
    json.dump(problems_dict, fp)