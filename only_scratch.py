import os

cache_root = 'yours'
hf_token = 'yours'  # Recommended: use environment variables or CLI login instead of hardcoding tokens in the repo
device = '0'
# 1) Set environment variables before any imports
os.environ["HF_TOKEN"] = hf_token
os.environ["HF_HOME"] = cache_root                 # Recommended: use a unified root directory
os.environ["CUDA_VISIBLE_DEVICES"] = device
import random

import pickle as pkl
import re
import json
from load_utils import load_model, model_name_func
name =  'Qwen3-0.6B'# 'Qwen3-0.6B'# 'Qwen3-0.6B'# 'ds-qwen-1.5B'  # 'ds-qwen-1.5B'# 'Qwen3-0.6B'
goal_name =  'Qwen-1.5B' # 'ds-qwen-1.5B'
goal_name2 =  'ds-qwen-1.5B' # 'ds-qwen-1.5B'
train_dataset_name = 'gsm8k' # "gsm8k" # 'prontoQA' # "gsm8k" # 'prontoQA' # "gsm8k"
edge_ratio = 0.5
node_ratio = 0.4
l1_co = 0.00005
save_path = f'yours/{name}_{train_dataset_name}_{edge_ratio}_{node_ratio}_{l1_co}/'
save_goal_path = f'yours/{goal_name}_{train_dataset_name}_{edge_ratio}_{node_ratio}_{l1_co}/'
save_goal_path2 = f'yours/{goal_name2}_{train_dataset_name}_{edge_ratio}_{node_ratio}_{l1_co}/'
save_graph_path = f'yours/graph_{name}_{train_dataset_name}_{edge_ratio}_{node_ratio}_{l1_co}/'
save_graph_goal_path = f'yours/graph_{goal_name}_{train_dataset_name}_{edge_ratio}_{node_ratio}_{l1_co}/'
import os
idx_list = []
files = [int(f.split('.')[0]) for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
print(len(files))
# model_name = model_name_func(name)
# model = load_model(name,device=f'cuda:{device}')
from load_utils import load_model, model_name_func

from transformers import AutoTokenizer, AutoModelForCausalLM

def get_text_before_second_question_correct(text):
    sub = 'Answer:'
    text = text.split(sub)[0] + sub +  text.split(sub)[1]
    return text
save_json = f'./data_{train_dataset_name}/{name}_answer.json'
with open(save_json,'r') as f:
    test_data = json.load(f)

data_idx = []
data_list = []
for files in os.listdir(save_path):
    with open(save_path + f'{files}', 'rb') as f:
        data = pkl.load(f)
    data_list.append(data)
    data_idx.append(int(files.split('.')[0]))
# len(data_list)
# print(data_list)
# exit()
# data_list = data_list[:100]
model_name =  model_name_func(name)# 'meta-llama/Llama-3.2-1B'# "Qwen/Qwen2.5-1.5B"
model = load_model(name)
tokenizer = model.tokenizer

if name =='Qwen3-0.6B' or name == 'Qwen-1.5B':
    unk_id = tokenizer(' ', return_tensors="pt")["input_ids"][0]
else:
    unk_id = tokenizer(' ')["input_ids"][1]
print('unk id',unk_id)
saved_file = f'./pertur_data_{train_dataset_name}/'
if os.path.exists(saved_file) == False:
    os.makedirs(saved_file)


text_list = []
count = 0

def answer_judge_string(data):
    text = data['gold_ans']
    match = re.search(r"####\s*(\d+)", text)
    if match is not None:
        match = match.group(1)
        if match not in data['ans']:
            return 0, match
    if data['judge'] is not None:
        judge_prompt = data['judge'].lower()
    else:
        return 0, match
    if 'yes' in judge_prompt and 'no' not in judge_prompt:
        return 1, match
    else:
        return 0, match

files = [int(f.split('.')[0]) for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
goal_files = [int(f.split('.')[0]) for f in os.listdir(save_goal_path) if os.path.isfile(os.path.join(save_goal_path, f))]
goal_files2 = [int(f.split('.')[0]) for f in os.listdir(save_goal_path2) if os.path.isfile(os.path.join(save_goal_path2, f))]

print(len(files))
print('ds',len(goal_files))
print('0.6',len(goal_files2))
result = list(set(files) & set(goal_files))
print('1.5 vs ds 1.5',len(result))
result = list(set(files) & set(goal_files2))
print('1.5 vs 0.6',len(result))
result = list(set(goal_files) & set(goal_files2))
print('ds 1.5 vs 0.6',len(result))
result = list(set(files) & set(goal_files) & set(goal_files2))
print(len(result))
# print(len(result))
# for i in files:
#     if i not in goal_files:
#         print(i)
exit()

def question_generates_prontoQA(data):
    question, ans = data['question'], data['answer']

    prompt = question + "\nLet's think step by step\nAnswer:\n<think>\n"
    max_token = 900
    return prompt, question, max_token, ans


def question_generate_arc(data):
    choices, question, ans = data['choices'], data['question'], data['answerKey']
    context = ''
    for idx, option in enumerate(choices['text']):
        context += choices['label'][idx] + ' ) ' + option + ' '
    prompt = "Context: " + str(context) + " Question: " + str(question) + "\nAnswer:\n<think>"
    if name == 'Qwen-1.5B':
        prompt = "Context: " + str(context) + " Question: " + str(question) + "\nLet's think step by step\nAnswer:\n"
    question = "Context: " + str(context) + " Question: " + str(question)
    max_token = 500
    if 'ds' in name:
        max_token = 600
    return prompt, question, max_token, ans


def generate_gsm8k(data):
    question, ans = data['question'], data['answer']
    if 'ds' in name:
        prompt = "Question: " + str(question) + "\nLet's think step by step\nAnswer:\n<think>\n"
        max_token = 500
    else:
        prompt = "Question: " + str(question) + "\nLet's think step by step\nAnswer:\n"
        max_token = 300
    return question, prompt, ans, max_token

def generate_MAWPS(data):
    question, ans, numbers = data['Question'], data['Answer'], data['Numbers']
    numbers = data['Numbers'].split()
    for i, number in enumerate(numbers):
        # print(number)
        placeholder = f'N_{i:02d}'
        question = re.sub(rf'\b{placeholder}\b', number, question)
    if 'ds' in name:
        prompt = "Question: " + str(question) + "\nLet's think step by step\nAnswer:\n<think>\n"
        max_token = 400
    else:
        prompt = "Question: " + str(question) + "\nLet's think step by step\nAnswer:\n"
        max_token = 200
    return question, prompt, ans, max_token

# for idx, data in enumerate(data_list):
#     question = test_data[data_idx[idx]]['question']
#     ans = test_data[data_idx[idx]]['ans']
#     # print(ans)
#     # if test_data[data_idx[idx]]['judge']: continue
#     # ans_judge, _ = answer_judge_string(test_data[data_idx[idx]])
#     # if ans_judge != True: continue
#     # if data_idx[idx] not in files or data_idx[idx] not in goal_files: continue
#     count += 1

#     # with open(save_graph_path+f'{idx}.pkl', 'rb') as f:
#     #     graph_data = pkl.load(f)
#     # if len(graph_data) < 10: continue
#     # with open(save_graph_goal_path+f'{idx}.pkl', 'rb') as f:
#     #     graph_data = pkl.load(f)
#     # if len(graph_data) < 10: continue

#     # if 'ds' in name:
#     #     prompt = "Question: " + str(question) + "\nLet's think step by step\nAnswer:\n<think>\n"
#     #     max_token = 500
#     # else:
#     #     prompt = "Question: " + str(question) + "\nLet's think step by step\nAnswer:\n"
#     #     max_token = 300
#     if train_dataset_name == 'gsm8k':
#         question, prompt, ans, max_token = generate_gsm8k(data)
#     if train_dataset_name == 'MAWPS':
#         question, prompt, ans, max_token = generate_MAWPS(data)
#     if train_dataset_name == 'prontoQA':
#         question, prompt, ans, max_token = question_generates_prontoQA(data)
#     if train_dataset_name == 'arc':
#         question, prompt, ans, max_token = question_generate_arc(data)
#     ans = test_data[data_idx[idx]]['ans']
#     print(question)
#     question_tokens = tokenizer(prompt, return_tensors="pt")
#     input_ids = question_tokens ["input_ids"].clone()
#     inv_input_ids = question_tokens ["input_ids"].clone()
#     random_input_ids = question_tokens ["input_ids"].clone()

#     known_pos = []
#     masked_pos = []
#     for node in data.keys():
#         if node == '**ans**':
#             continue
#         double_check = set()

#         for key in data[node].keys():
#             for words in data[node][key]:
#                 words, layer, pos, flag = words.split('_')
#                 pos = int(pos)
#                 known_pos.append(pos)

#     for i in range(input_ids.shape[1]):
#         if i not in known_pos:
#             masked_pos.append(i)

#     marked_pos_count = 0
#     for pos in masked_pos:
#         if pos < input_ids.shape[1]:
#             input_ids[0, pos] = unk_id
#             marked_pos_count += 1
#     for i in range(marked_pos_count):
#         random_inc = random.randint(0, input_ids.shape[1]-1)
#         random_input_ids[0, random_inc] = unk_id

#     # for pos in known_pos:
#     #     if pos < input_ids.shape[1]:
#     #         inv_input_ids[0, pos] = unk_id
#             # marked_pos_count += 1

#     sparsity = marked_pos_count/input_ids.shape[1]
#     print('idx',idx,'sparsity',sparsity)
#     # if sparsity == 0:continue

#     # if count == 100:break
#     # break
# print(count)
# # exit()



from datasets import load_from_disk, load_dataset, concatenate_datasets

if train_dataset_name == "gsm8k":
    dataset = load_dataset("gsm8k", "main", cache_dir=cache_root)
    test = dataset["test"]
elif train_dataset_name == 'MAWPS':
    dataset = load_dataset("mwpt5/MAWPS")
    test = dataset["train"]
elif train_dataset_name == 'prontoQA':
    ds = load_dataset("fengyang0317/prontoqa")
    subset = ds['train'].select(range(500))
    print(ds.keys())
    print(len(ds['validation']))
    test = concatenate_datasets([ds['validation'], ds['test'], subset])
elif train_dataset_name == 'arc':
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
    print(ds.keys())
    test = ds['test']

count = 0
for idx,data in enumerate(test):
    # question = test_data[data_idx[idx]]['question']
    # if test_data[data_idx[idx]]['judge']: continue
    # ans_judge, _ = answer_judge_string(test_data[data_idx[idx]])
    # if ans_judge != True: continue
    if idx not in files or idx not in goal_files or idx not in goal_files2: continue
    count += 1


    # if 'ds' in name:
    #     prompt = "Question: " + str(question) + "\nLet's think step by step\nAnswer:\n<think>\n"
    #     max_token = 500
    # else:
    #     prompt = "Question: " + str(question) + "\nLet's think step by step\nAnswer:\n"
    #     max_token = 300
    # ans = test_data[data_idx[idx]]['ans']
    # print(question)
    if train_dataset_name == 'gsm8k':
        question, prompt, ans, max_token = generate_gsm8k(data)
    if train_dataset_name == 'MAWPS':
        question, prompt, ans, max_token = generate_MAWPS(data)
    if train_dataset_name == 'prontoQA':
        prompt, question, max_token, ans = question_generates_prontoQA(data)
    if train_dataset_name == 'arc':
        prompt, question, max_token, ans = question_generate_arc(data)
    print(question)
    question_tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = question_tokens ["input_ids"].clone()
    inv_input_ids = question_tokens ["input_ids"].clone()
    random_input_ids = question_tokens ["input_ids"].clone()

    known_pos = []
    masked_pos = []
    data_dicts = data_list[data_idx.index(idx)]
    for node in data_dicts.keys():
        if node == '**ans**':
            continue
        double_check = set()

        for key in data_dicts[node].keys():
            for words in data_dicts[node][key]:
                words, layer, pos, flag = words.split('_')
                pos = int(pos)
                known_pos.append(pos)

    for i in range(input_ids.shape[1]):
        if i not in known_pos:
            masked_pos.append(i)

    marked_pos_count = 0
    for pos in masked_pos:
        if pos < input_ids.shape[1]:
            input_ids[0, pos] = unk_id
            marked_pos_count += 1
    for i in range(marked_pos_count):
        random_inc = random.randint(0, input_ids.shape[1]-1)
        random_input_ids[0, random_inc] = unk_id

    # for pos in known_pos:
    #     if pos < input_ids.shape[1]:
    #         inv_input_ids[0, pos] = unk_id
            # marked_pos_count += 1

    sparsity = 1 - marked_pos_count/input_ids.shape[1]
    print(count)
    print('idx',idx,'sparsity',sparsity)

    new_text = tokenizer.decode(input_ids[0])
    # print(new_text)
    # print(tokenizer.encode(new_text))
    inv_new_text = tokenizer.decode(inv_input_ids[0])
    random_new_text = tokenizer.decode(random_input_ids[0])

    generated_text = model.generate(
        new_text,
        max_new_tokens=max_token,    # generate max_token tokens
        temperature=0,      # temperature
    )


    random_generated_text = model.generate(
        random_new_text,
        max_new_tokens=max_token,    # generate max_token tokens
        temperature=0,      # temperature
    )
    print(generated_text)
    # ans = test_data[idx]['ans']
    text_list.append({'question': question, 'gold_ans': test_data[idx]['gold_ans'],'ori_ans':test_data[idx]['ans'], 'ans':generated_text,
                      'random':random_generated_text, 'sparsity':sparsity, 'question_masked':new_text, 'question_random':random_new_text})
    if count == 100:break
    # break
# exit()
with open(saved_file +  f'{name}_{node_ratio}_{edge_ratio}_answer.json','w') as f:
    json.dump(text_list, f)
