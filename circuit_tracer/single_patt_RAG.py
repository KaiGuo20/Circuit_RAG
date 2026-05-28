#single_pattern_RAG.py
import os
import gc
hf_token = 'yours'
os.environ["HF_TOKEN"] = hf_token
os.environ["TOKENIZERS_PARALLELISM"] = "false"
cache_root = 'yours'
hf_token = 'yours'  # Recommended: use environment variables or CLI login instead of hardcoding tokens in the repo

os.environ["HF_HOME"] = cache_root
from transformer_lens import HookedTransformer, HookedTransformerConfig
from dataclasses import dataclass
import torch
import os
import sys
from abc import ABC
import json

import pickle
from training import train_sae_on_language_model
from transformers import PreTrainedTokenizerFast

# sys.path.append('../')
from transcoder.activation_functions import JumpReLU, Relu
from transcoder.single_layer_transcoder import SingleLayerTranscoder
from transcoder.activations_store import ActivationsStore
from my_replacement_model import ReplacementModel
# from transcoder.short_activations_stores import Short_ActivationsStore

from configs import Configs

from redefined_datasets import MyDataset, MyIterableDataset

sys.path.append('../')
from load_utils import load_model, model_name_func
# base_path = f'yours'
# path =  os.path.join(base_path, "model_weights.pt")


cache_dir = 'yours'
activate_cache_dir = 'yours'

if os.path.exists(activate_cache_dir) == False:
    os.makedirs(activate_cache_dir)

import random
import numpy as np
import torch

seed = 0
# Python built-in random
random.seed(seed)

# NumPy random
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


device = 'cuda:1'
# torch.cuda.set_device(device)  # Set default CUDA device

# # Ensure all new tensors default to cuda:3
# torch.set_default_tensor_type(torch.cuda.FloatTensor)  # optional

total_training_tokens = 1_000_000 * 20
acts_func = 'relu'
transcoder_config = {}
from_pretrained = False
name = 'llama3_8b' # 
train_dataset_name = "musique" #manu_musique, musique, 2wiki, hotpotqa
# Flag to control whether to use wrong=True for hotpotqa, 2wiki, musique datasets
wrong_answer_flag = False
if name == 'ds-llama':
    edge_ratio = 0.99
    node_ratio = 0.99
elif name == 'llama3_8b':
    edge_ratio = 0.99
    node_ratio = 0.99
elif name == 'llama3_8b_finetune':
    edge_ratio = 0.99
    node_ratio = 0.99
else:
    edge_ratio = 0.99
    node_ratio = 0.99

# if train_dataset_name == 'prontoQA':
#     edge_ratio = 0.4
#     node_ratio = 0.3


# name = 'Qwen3-0.6B'
l1_co = 0.0005
if 'ds' in name:
    max_length = 500
else:
    max_length = 500

model_name = model_name_func(name)
print('model_name', model_name)
if train_dataset_name == 'hotpotqa':
    model, hf_model = load_model(name, device=device)
elif train_dataset_name == '2wiki':
    model, hf_model = load_model(name, device=device)
elif train_dataset_name == 'musique':
    model, hf_model = load_model(name, device=device)
elif train_dataset_name == 'manu_musique':
    model, hf_model = load_model(name, device=device)

else:
    model = load_model(name, device=device)

print('model---', model)
tokenizer = model.tokenizer

# for name, param in model.named_parameters():
#     print(f"{name:60} | {str(param.shape):20} | requires_grad={param.requires_grad}")

transcoder_config['model_name'] = model_name
transcoder_config['train_dataset_name'] = train_dataset_name
transcoder_config['max_length'] = max_length
transcoder_config['method'] = 'real'
transcoder_config['pattern_name'] = train_dataset_name
transcoder_config['is_tokenized'] = True
transcoder_config['dataset_path'] = None

record_scores_list = []
# for target_layer in range(n_layer):
target_layer = 4
n_layer = model.cfg.n_layers
d_model = model.cfg.d_model
configs = Configs(
    target_layer = target_layer,
    out_hook_point_layer = target_layer,
    epoch = 1,
    d_in = d_model, # 2048, #  768, # 2048, #  2304, # llama 2048
    d_out = d_model, # 2048, # 768, # 2048, # 2304,
    d_transcoder = d_model, # 2048, # 768, # 2048, # 2304,
    train_batch_size = 1024, # 2048,
    context_size = 300, # 64, # 128, # 64, # 128, # 256, # 512,
    dataset_path = transcoder_config['dataset_path'],
    is_dataset_tokenized = transcoder_config['is_tokenized'],
    pattern_name = transcoder_config['pattern_name'],
    method = transcoder_config['method'],
    max_length = transcoder_config['max_length'],
    
    hook_point = f"blocks.{target_layer}.ln2.hook_normalized",
    out_hook_point = f"blocks.{target_layer}.hook_mlp_out",# f"blocks.{target_layer + 1}.hook_mlp_out" if target_layer + 1 <= n_layer - 1 else f"blocks.{n_layer - 1}.hook_mlp_out", #f"blocks.{target_layer}.hook_mlp_out",# f"blocks.{n_layer - 1}.hook_mlp_out", # f"blocks.{target_layer}.hook_mlp_out",# f"blocks.{n_layer}.hook_mlp_out",# f"blocks.{target_layer}.hook_mlp_out",
    is_transcoder = True,
    is_sae = False,
    use_cached_activations = False,
    cached_activations_path = activate_cache_dir,
    
    n_batches_in_buffer = 32,
    total_training_tokens = total_training_tokens,
    store_batch_size = 16, # 256,
    
    act_store_device = device,
    device = device,
    seed = 42,
    # dtype = torch.float32,
    hook_point_head_index = None,
    
    b_dec_init_method = "mean",
    l1_coefficient = l1_co,
    activate_func= 'relu',
    
    model_name = 'llama3', # 'llama3',# 'llama3',# 'gpt2',# 'llama3',
    dataset_name = 'graph', # 'opentext', # 'opentext',# 'graph',# 'opentext',
    
    batch_size = 1024,
    lr = 1e-3, #  0.0004,
    dead_feature_window = d_model, # 1000,# 100,  # unless this window is larger feature sampling,
    dead_feature_estimation_method = 'no_fire',
    dead_feature_threshold = 1e-8,
    resample_batches = 1024,
    is_sparse_connection = False,
    checkpoint_path =f'yours/{acts_func}_{name}_{train_dataset_name}_{n_layer}',
    lr_scheduler_name = 'constantwithwarmup',
    lr_warm_up_steps = 5000,
    from_pretrained = from_pretrained
    
)



# tokenizer_path = os.path.join(dataset_path, f"{pattern_name}_{method}_baby_tokenizer.json")

transcoder_cache_dir = os.path.join(configs.checkpoint_path, f"{name}_{train_dataset_name}_{configs.l1_coefficient}",'configs.json')

transcoder = os.path.join(configs.checkpoint_path, f"{configs.dead_feature_window}_{configs.l1_coefficient}")

configs.load(os.path.join(transcoder_cache_dir,'configs.json'))
print('configs.d_transcoder', configs.d_transcoder)

configs.d_transcoder = d_model * 2 # 192 # 2048 # 768 # 2048 # 2304

d_emb = d_model # model.embed.W_E.shape[1]
# configs.d_in = d_emb
# configs.d_out = d_emb 
# configs.d_model = d_emb 
# configs.d_transcoder = d_emb
configs.n_layers = n_layer


print('name', name)
model = ReplacementModel.from_self_pretrained_and_transcoders(cfg=configs, model_name=name, model_path=cache_root, transcoders_path=transcoder, device=device)
print(model)
# model = model.to(configs.act_store_device) # modified 2024-12-16
model = model.to(configs.device)





from circuit_tracer import attribute
from circuit_tracer.utils import create_graph_files
model.eval()  # Set the model to evaluation mode
from transformers import PreTrainedTokenizerFast
from redefined_datasets import MyDataset, MyTestDataset


# exit()


# Add special tokens + set semantic labels
# tokenizer.add_special_tokens({
#     "pad_token": "<PAD>",
#     "bos_token": "<START_Q>",
#     "eos_token": "<END_Q>",
#     "unk_token": "<UNK>",
# })

model.tokenizer = tokenizer

# save_json = f'../data_{train_dataset_name}/{name}_answer.json'
# with open(save_json,'r') as f:
#     test_data = json.load(f)


# Read file
save_json = f'../data_{train_dataset_name}/{name}_answer.json'
with open(save_json, 'r') as f:
    test_data = json.load(f)

# # Modify the ans field for each entry
# for item in test_data:
#     question = item['question']
#     original_ans = item['ans']

#     # Build new ans format
#     item['ans'] = f"Question: {question}\n Answer:\n {original_ans}"

# # Save back to file
# with open(save_json, 'w') as f:
#     json.dump(test_data, f, indent=2, ensure_ascii=False)

# print(f"Updated {len(test_data)} items in {save_json}")
import re



def strip_based_on_context(text: str) -> str:
    """
    Remove everything between 'Based on the context:' and '\n Answer:' (including the labels).
    Fallback: if '\n Answer:' is not found, remove from 'Based on the context:' to end.
    """
    start = text.find("Based on the context:")
    if start == -1:
        return text

    end = text.find("\n Answer:", start)
    if end == -1:
        end = len(text)

    return text[:start] + text[end:]

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

def answer_judge_string_bool(data):
    text = str(data['gold_ans'])
    if data['judge']:
        return 1, text
    else:
        return 0, text

def answer_judge_string_list(data):
    text = str(data['gold_ans'])
    if data['judge']:
        return 1, text
    else:
        return 0, text

def answer_judge_webqsp(data, wrong=False):
    """
    Judge function for WebQSP dataset.
    gold_ans: list format ["answer1", "answer2"]
    judge: string "Yes" or "No"
    """
    gold_ans = data['gold_ans']  # list
    judge_text = data['judge']    # string

    # Convert list to string (for display)
    gold_ans_str = ', '.join(gold_ans) if isinstance(gold_ans, list) else str(gold_ans)

    # Judge logic: judge contains 'Yes' and does not contain 'No'
    if judge_text:
        judge_lower = judge_text.strip().lower()
        if wrong:
            is_correct = ('no' in judge_lower) and ('yes' not in judge_lower)
        else:
            is_correct = ('yes' in judge_lower) and ('no' not in judge_lower)
        return (1 if is_correct else 0), gold_ans_str
    else:
        return 0, gold_ans_str

def answer_judge_hotpotqa(data, wrong=False):
    """
    Judge function for HotpotQA dataset.
    gold_ans: list format ["answer1", "answer2"]
    judge: string "Yes" or "No"
    """
    gold_ans = data['gold_ans']  # list
    judge_text = data['judge']    # string

    # Convert list to string (for display)
    gold_ans_str = ', '.join(gold_ans) if isinstance(gold_ans, list) else str(gold_ans)

    # Judge logic: judge contains 'Yes' and does not contain 'No'
    if judge_text:
        judge_lower = judge_text.strip().lower()
        if wrong:
            is_correct = ('no' in judge_lower) and ('yes' not in judge_lower)
        else:
            is_correct = ('yes' in judge_lower) and ('no' not in judge_lower)
        return (1 if is_correct else 0), gold_ans_str
    else:
        return 0, gold_ans_str

def answer_judge_2wiki(data, wrong=False):
    """
    Judge function for 2WikiMultiHopQA dataset.
    gold_ans: list format ["answer1", "answer2"]
    judge: string "Yes" or "No"
    """
    gold_ans = data['gold_ans']  # list
    judge_text = data['judge']    # string

    # Convert list to string (for display)
    gold_ans_str = ', '.join(gold_ans) if isinstance(gold_ans, list) else str(gold_ans)

    # Judge logic: judge contains 'Yes' and does not contain 'No'
    if judge_text:
        judge_lower = judge_text.strip().lower()
        if wrong:
            is_correct = ('no' in judge_lower) and ('yes' not in judge_lower)
        else:
            is_correct = ('yes' in judge_lower) and ('no' not in judge_lower)
        return (1 if is_correct else 0), gold_ans_str
    else:
        return 0, gold_ans_str

def answer_judge_musique(data, wrong=False):
    """
    Judge function for MuSiQue dataset.
    gold_ans: list format ["answer1", "answer2"]
    judge: string "Yes" or "No"
    """
    gold_ans = data['gold_ans']  # list
    judge_text = data['judge']    # string

    # Convert list to string (for display)
    gold_ans_str = ', '.join(gold_ans) if isinstance(gold_ans, list) else str(gold_ans)

    # Judge logic: judge contains 'Yes' and does not contain 'No'
    if judge_text:
        judge_lower = judge_text.strip().lower()
        if wrong:
            is_correct = ('no' in judge_lower) and ('yes' not in judge_lower)
        else:
            is_correct = ('yes' in judge_lower) and ('no' not in judge_lower)
        return (1 if is_correct else 0), gold_ans_str
    else:
        return 0, gold_ans_str
def answer_judge_bool(data, dicts):
    judge = data['judge']
    ans = data['gold_ans']
    return judge, dicts[ans]


def find_last_uppercase_simple(s):
    for i in range(len(s) - 1, -1, -1):
        if s[i].isupper():
            return s[i]
    return None

correct_idx = []
match_collect = {}
for idx, data in enumerate(test_data):
    # if '**Final Answer:**' not in data['ans']: continue
    if data['judge'] is None:
        print(f'None idx {idx}')
        continue
    if train_dataset_name == 'gsm8k':
        judge, match = answer_judge_string(data)
    elif train_dataset_name == 'WebQSP' or train_dataset_name == 'WebQSP_ROG':
        judge, match = answer_judge_webqsp(data)
    elif train_dataset_name == 'hotpotqa':
        judge, match = answer_judge_hotpotqa(data, wrong=wrong_answer_flag)
    elif train_dataset_name == '2wiki':
        judge, match = answer_judge_2wiki(data, wrong=wrong_answer_flag)
    elif train_dataset_name == 'musique':
        judge, match = answer_judge_musique(data, wrong=wrong_answer_flag)
    elif train_dataset_name == 'manu_musique':
        judge, match = answer_judge_musique(data, wrong=wrong_answer_flag)
    elif train_dataset_name == 'boolQA':
        judge_dicts = {bool(0): 'False', bool(1): 'True'}
        judge, match = answer_judge_bool(data, judge_dicts)

    elif train_dataset_name == 'MAWPS':
        judge, match = answer_judge_string_bool(data)
    elif train_dataset_name == 'qasc' or train_dataset_name == 'arc':
        judge, match = answer_judge_string_list(data)

    elif train_dataset_name == 'prontoQA':
        judge_dicts = {'True': 'True', 'False': 'False'}
        judge, match = answer_judge_bool(data, judge_dicts)

    if judge:
        # print('idx',idx)
        # print('match', match)
        correct_idx.append(idx)
        # print(match_collect)
        match_collect[idx] = match
    else:
        if train_dataset_name == 'gsm8k' or train_dataset_name == 'MAWPS':
            if '**Final Answer:**' in data['ans']:
                # print(data['ans'].split('**Final Answer:**')[1])
                if 'llama' in name:
                    txt = data['ans']
                    num_txt = re.findall(r"\d+", txt)
                    if len(num_txt) > 0:
                        match_collect[idx] = num_txt[-1]
                    else: 
                        match_collect[idx] = txt
                else:
                    if name == 'Qwen3-0.6B':
                        txt = data['ans'].split('**Final Answer:**')[1].split(' ')[1]
                    elif 'ds' in name:
                        txt = data['ans'].split('**Final Answer:**')[1]
                    num_txt = re.findall(r"\d+", txt)
                    if len(num_txt) > 0:
                        match_collect[idx] = num_txt[0]
                    else: 
                        match_collect[idx] = txt
                    
        elif train_dataset_name == 'boolQA':
            if data['gold_ans'] == True:
                match_collect[idx] = 'False'
            else:
                match_collect[idx] = 'True'
        elif train_dataset_name == 'qasc' or train_dataset_name == 'arc':
            match_collect[idx] = find_last_uppercase_simple(data['ans'])
        elif train_dataset_name == 'prontoQA':
            if 'ink>' not in data['ans']: continue
            else:
                tmp_ans = data['ans'].split('ink>')[1]
                if 'true' in tmp_ans:
                    match_collect[idx] = 'true'
                elif 'false' in tmp_ans:
                    match_collect[idx] = 'false'
        
    # exit()
# print(match_collect)            
# exit()
# print(correct_idx)
# print(len(correct_idx)/len(test_data))
max_n_logits = 10   # How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_desired_log_prob); see below for the latter
desired_logit_prob = 0.95  # Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)
max_feature_nodes = 1024#2048  # Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.
batch_size=4#256
offload = 'cpu'# 'disk' if IN_COLAB else 'cpu' # Offload various parts of the model during attribution to save memory. Can be 'disk', 'cpu', or None (keep on GPU)
verbose = True 

def nodes_at_n_hop(G, source, n, filter = ''):
    current_layer = set([source])
    all_traversed = set([source])
    for _ in range(n):
        next_layer = set()
        for node in current_layer:
            # print(f"Expanding node: {node}")
            if node not in G:
                continue
            next_layer.update(G.successors(node))
            all_traversed.update(G.successors(node))
        # print(f"Current layer: {current_layer}, Next layer: {next_layer}")
        current_layer = next_layer
    if filter != '':
        all_traversed = {node for node in all_traversed if node.split('_')[1] == filter}
    return all_traversed
# exit()
import networkx as nx
def selects_prompt(prompt):
    graph_prompts, ans = prompt.split(' T ')
    ans = ans.split(' END_P')[0].split(' , ')
    graph_prompts = graph_prompts + ' T '
    return graph_prompts, ans



acc_list = []
layer_list = []  

import pickle

from tqdm import tqdm
from circuit_tracer.frontend.graph_models import Metadata, Model, Node, QParams
from circuit_tracer.frontend.utils import add_graph_metadata, process_token
from circuit_tracer.graph import Graph, prune_graph
from circuit_tracer.utils.create_graph_files import create_nodes, create_used_nodes_and_edges



bos_token = "<|begin_of_text|> "
embeds_take_layers = {}
logit_take_layers = {}
embeds_key_words = {}
logit_key_words = {}
logit_key_words_layers = {}
emb_key_words_layers = {}
contribute_counts = {}
max_count = 50
count = 0

def translate_node_ids(graph, node_id, given_type):
    node_id = node_id.split('_')
    if given_type == 'embedding':
        vocab_id = int(node_id[1])
        pos = int(node_id[2])
        id = tokenizer.decode(vocab_id)
        layer = 'Emb'
    if given_type == "mlp reconstruction error":
        _, layer, pos = int(node_id[0]), int(node_id[1]), int(node_id[2])
        id = 'block_inner'
    if given_type == "logit":
        layer, vocab_id, pos = int(node_id[0]), int(node_id[1]), int(node_id[2])
        id = tokenizer.decode(vocab_id)
    if given_type == "cross layer transcoder":
        layer, feat_idx, pos = int(node_id[0]), int(node_id[1]), int(node_id[2])
        id = tokenizer.decode(graph.input_tokens[pos])
    return id, pos, layer
import random

# test_data = test_data[:200]

print(len(test_data))

if train_dataset_name == 'gsm8k'or train_dataset_name == 'MAWPS':
    if name == 'ds-qwen-1.5B':
        tokenizer_Answer = tokenizer('Answer')['input_ids'][1]
    elif name == 'Qwen3-0.6B':
        tokenizer_Answer = tokenizer('Answer')['input_ids'][0]
    elif 'Qwen-1.5' in name:
        tokenizer_Answer = tokenizer('Answer')['input_ids'][0]
    elif 'llama' in name.lower():
        tokenizer_Answer  = tokenizer('Answer')['input_ids'][1]
elif train_dataset_name == 'WebQSP' or train_dataset_name == 'WebQSP_ROG':
    if name == 'ds-qwen-1.5B':
        tokenizer_Answer = tokenizer('Answer')['input_ids'][1]
    elif name == 'Qwen3-0.6B':
        tokenizer_Answer = tokenizer('Answer')['input_ids'][0]
    elif 'Qwen-1.5' in name:
        tokenizer_Answer = tokenizer('Answer')['input_ids'][0]
    elif 'llama' in name.lower():
        tokenizer_Answer  = tokenizer(' Answer')['input_ids'][1]  
        print('tokenizer_Answer', tokenizer_Answer)

elif train_dataset_name == 'hotpotqa':
    if name == 'ds-qwen-1.5B':
        tokenizer_Answer = tokenizer('Answer')['input_ids'][1]
    elif name == 'Qwen3-0.6B':
        tokenizer_Answer = tokenizer('Answer')['input_ids'][0]
    elif 'Qwen-1.5' in name:
        tokenizer_Answer = tokenizer('Answer')['input_ids'][0]
    elif 'llama' in name.lower():
        tokenizer_Answer  = tokenizer(' Answer')['input_ids'][1]  
        print('tokenizer_Answer', tokenizer_Answer)
elif train_dataset_name == '2wiki':
    if name == 'ds-qwen-1.5B':
        tokenizer_Answer = tokenizer('Answer')['input_ids'][1]
    elif name == 'Qwen3-0.6B':
        tokenizer_Answer = tokenizer('Answer')['input_ids'][0]
    elif 'Qwen-1.5' in name:
        tokenizer_Answer = tokenizer('Answer')['input_ids'][0]
    elif 'llama' in name.lower():
        tokenizer_Answer  = tokenizer(' Answer')['input_ids'][1]  
        print('tokenizer_Answer', tokenizer_Answer)
elif train_dataset_name == 'musique' or train_dataset_name == 'manu_musique':
    if name == 'ds-qwen-1.5B':
        tokenizer_Answer = tokenizer('Answer')['input_ids'][1]
    elif name == 'Qwen3-0.6B':
        tokenizer_Answer = tokenizer('Answer')['input_ids'][0]
    elif 'Qwen-1.5' in name:
        tokenizer_Answer = tokenizer('Answer')['input_ids'][0]
    elif 'llama' in name.lower():
        tokenizer_Answer  = tokenizer(' Answer')['input_ids'][1]  
        print('tokenizer_Answer', tokenizer_Answer)
elif train_dataset_name == 'prontoQA' or train_dataset_name == 'boolQA' or train_dataset_name == 'arc' or train_dataset_name == 'qasc':
    if name == 'Qwen3-0.6B':
        tokenizer_Answer = tokenizer('<think>')['input_ids'][0]
    elif name == 'ds-qwen-1.5B':
        tokenizer_Answer = tokenizer('<think>')['input_ids'][1]
    elif 'Qwen-1.5' in name:
        tokenizer_Answer = tokenizer('Answer')['input_ids'][0]
# print(tokenizer.decode(128000))
# print(tokenizer.decode(16533))
# print('token answer',tokenizer_Answer)
# print('token answer',tokenizer_Answer)

# exit()
def get_text_before_second_question_correct(text):
    sub = 'Answer:'
    text = text.split(sub)[0] + sub +  text.split(sub)[1]  
    return text
    # parts = text.split("Question")
    # if len(parts) < 3:
    #     return text  # If fewer than two Questions, return original text
    # return "Question".join(parts[:2])

# def get_text_before_second_question_correct(text):
#     sub = 'Answer:'
#     text = text.split(sub)[0] + sub +  text.split(sub)[1]  
#     return text

def get_text_before_second_question_wrong(text):
    sub = '**Final Answer:**'
    if text.count(sub) >= 2:
        text = text.split(sub)[0] + sub +  text.split(sub)[1]    
    return text

def anwswer_pos_detect(token_list, matched_token):
    for idx, token in enumerate(reversed(token_list)):
        if matched_token in token:
            return len(token_list) - idx - 1

def re_detect_string( ori_text,offsets, position):
    chunk = offsets[position][-1]
    return ori_text[:chunk]


def implicit_route(prompt_str, ans_pos):  # adds negative edges
    print('implicit_route')

    n_ctx = getattr(model.cfg, "n_ctx", None)
    if n_ctx is None:
        n_ctx = getattr(tokenizer, "model_max_length", 2048)

    enc = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=n_ctx)
    input_ids = enc["input_ids"].to(configs.device)

    graph = attribute(
        prompt=None,
        model=model,
        input_ids=input_ids,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose,
        print_log=False,
    )

    node_threshold = node_ratio
    edge_threshold = edge_ratio

    node_mask, edge_mask, cumulative_scores = (
        el.cpu() for el in prune_graph(graph, node_threshold, edge_threshold)
    )

    scan = graph.scan
    nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores, scan)
    used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)

    nodes_dicts = {}
    for n in used_nodes:
        nodes_dicts[n.node_id] = {
            "features": n.feature_type,
            "clerp": n.clerp
        }

    # =========================
    # Key change: store pos/neg/abs/sign in one graph
    # =========================
    result_graph_G = nx.DiGraph()
    logit_nodes = []

    for e in used_edges:
        if nodes_dicts[e['source']]['features'] == "mlp reconstruction error":
            continue
        if nodes_dicts[e['target']]['features'] == "mlp reconstruction error":
            continue
        if nodes_dicts[e['target']]['features'] == "logit" and nodes_dicts[e['source']]['features'] == 'embedding':
            continue

        w = float(e["weight"])
        if w == 0.0:
            continue

        source_node, source_pos, source_layer = translate_node_ids(
            graph, e['source'], nodes_dicts[e['source']]['features']
        )
        target_node, target_pos, target_layer = translate_node_ids(
            graph, e['target'], nodes_dicts[e['target']]['features']
        )

        start_node = f'{source_node}_{source_layer}_{source_pos}'
        end_node   = f'{target_node}_{target_layer}_{target_pos}'

        pos_w = w if w > 0 else 0.0
        neg_w = (-w) if w < 0 else 0.0
        abs_w = abs(w)

        # Accumulate if edge already exists
        if result_graph_G.has_edge(end_node, start_node):
            result_graph_G[end_node][start_node]["pos_weight"]  += pos_w
            result_graph_G[end_node][start_node]["neg_weight"]  += neg_w
            result_graph_G[end_node][start_node]["abs_weight"]  += abs_w
            result_graph_G[end_node][start_node]["sign_weight"] += w
        else:
            result_graph_G.add_edge(
                end_node, start_node,
                pos_weight=pos_w,
                neg_weight=neg_w,
                abs_weight=abs_w,
                sign_weight=w,
            )

        if nodes_dicts[e['target']]['features'] == 'logit':
            logit_nodes.append(end_node)

    # =========================
    # marked_pos: traverse only positive edges (don't let negative edges pollute evidence paths)
    # =========================
    marked_pos = []
    repeat_check = set()

    for target_node in set(logit_nodes):
        target_layers = nodes_at_n_hop_pos(result_graph_G, target_node, n_layer + 2)
        for t in target_layers:
            if len(t.split('_')) != 3:
                continue
            word, layer, pos = t.split('_')
            pos_int = int(pos)

            tmp_t = f'{word}_{layer}_{pos}_' + ('1' if pos_int > ans_pos else '0')
            if tmp_t in repeat_check:
                continue
            repeat_check.add(tmp_t)

            if pos_int > ans_pos and pos_int not in marked_pos:
                marked_pos.append(pos_int)

    del graph
    gc.collect()
    torch.cuda.empty_cache()

    return marked_pos, repeat_check, result_graph_G
def nodes_at_n_hop_pos(G, source, n, filter=''):
    current_layer = {source}
    all_traversed = {source}
    for _ in range(n):
        next_layer = set()
        for node in current_layer:
            if node not in G:
                continue
            for succ in G.successors(node):
                # Only traverse positive-contribution edges
                if G[node][succ].get("pos_weight", 0.0) > 0:
                    next_layer.add(succ)
        all_traversed |= next_layer
        current_layer = next_layer

    if filter != '':
        all_traversed = {node for node in all_traversed if node.split('_')[1] == filter}
    return all_traversed




if wrong_answer_flag:
    save_path = f'yours/minus50_wrong22_{name}_{train_dataset_name}_{edge_ratio}_{node_ratio}_{l1_co}/'
    save_graph_path = f'yours/minus50_wrong22_graph_{name}_{train_dataset_name}_{edge_ratio}_{node_ratio}_{l1_co}/'
else:
    save_path = f'yours/minus50_correct22_{name}_{train_dataset_name}_{edge_ratio}_{node_ratio}_{l1_co}/'
    save_graph_path = f'yours/minus50_correct22_graph_{name}_{train_dataset_name}_{edge_ratio}_{node_ratio}_{l1_co}/'
import os

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_graph_path):
    os.makedirs(save_graph_path)

start = 000 #int(3*len(test_data)/4)
end = len(test_data) # int(3*len(test_data)/4)
test_data = test_data[start:end]
# print(tokenizer('<think>')['input_ids'])



for idx, data in enumerate(test_data):  # only save first graph
    print('idx', idx)
    files = [int(f.split('.')[0]) for f in os.listdir(save_graph_path)
             if os.path.isfile(os.path.join(save_graph_path, f))]
    real_id = start + idx
    print('real_id', real_id)
    print(count)

    if real_id in files:
        continue

    for node_thre in [node_ratio]:
        overall_acc = []
        layer_counts = {}
        counts = 0

        # --------- Select ans (keep original logic) ----------
        if train_dataset_name == 'gsm8k':
            if 'ds' in name:
                prompt = "Question: " + data['question'] + "\nLet's think step by step\nAnswer:\n<think>\n"
            else:
                prompt = "Question: " + data['question'] + "\nLet's think step by step\nAnswer:\n"
            if real_id in correct_idx:
                ans = get_text_before_second_question_correct(data['ans'])
            else:
                ans = get_text_before_second_question_wrong(data['ans'])

        elif train_dataset_name == 'MAWPS':
            if 'ds' in name:
                prompt = "Question: " + data['question'] + "\nLet's think step by step\nAnswer:\n<think>\n"
            else:
                prompt = "Question: " + data['question'] + "\nLet's think step by step\nAnswer:\n"
            ans = data['ans']

        elif train_dataset_name == 'prontoQA':
            prompt = "Question: " + data['question'] + "\nLet's think step by step\nAnswer:\n<think>\n"
            ans = data['ans']

        elif train_dataset_name == 'boolQA':
            prompt = "Question: " + data['question'] + "\nLet's think step by step\nAnswer:\n<think>\n"
            ans = data['ans']

        elif train_dataset_name == 'WebQSP' or train_dataset_name == 'WebQSP_ROG':
            ans = data['ans']

        elif train_dataset_name == 'hotpotqa':
            ans = data['ans']

        elif train_dataset_name == '2wiki':
            ans = data['ans']

        elif train_dataset_name == 'musique':
            ans = data['ans']
        
        elif train_dataset_name == 'manu_musique':
            ans = data['ans']

        elif train_dataset_name == 'qasc' or train_dataset_name == 'arc':
            prompt = "Question: " + data['question'] + "\nAnswer:\n<think>\n"
            if 'Qwen-1.5' in name:
                prompt = "Question: " + data['question'] + "\nLet's think step by step\nAnswer:\n"
            ans = data['ans']

        # --------- Only generate the first graph ----------
        try:
            tokens = tokenizer(ans, return_offsets_mapping=True)
            offsets = tokens['offset_mapping']
            ids = tokens['input_ids']

            # Use tokenizer_Answer to find ans_position (preserved from original logic)
            if tokenizer_Answer not in ids:
                # Avoid ValueError: xxx is not in list
                print(f"[WARN] tokenizer_Answer not found in ids, skip real_id={real_id}")
                continue
            ans_position = ids.index(tokenizer_Answer)

            if real_id not in match_collect:
                continue

            # # Find the last occurrence of the answer string position (original logic)
            # last_pos_ans = ans.rfind(match_collect[real_id])
            # if last_pos_ans < 0:
            #     print(f"[WARN] match string not found in ans, skip real_id={real_id}")
            #     print(f"  [DEBUG] match_collect[{real_id}] = {repr(match_collect[real_id])}")
            #     print(f"  [DEBUG] ans = {repr(ans[:1000])}...")  # print first 1000 chars
            #     print(f"  [DEBUG] is_correct = {real_id in correct_idx}")
            #     print("-" * 50)
            #     continue
            ans = strip_based_on_context(ans)

            last_pos_ans = len(ans)

            prefix = ans[:last_pos_ans]  # Only do attribution on the prefix before the answer (original logic)
            print('start implicit_route (single graph only)')
            tok_len = len(tokenizer(prefix)["input_ids"])
            print("tok_len =", tok_len, "model_n_ctx =", getattr(model.cfg, "n_ctx", None))

            # Run implicit_route only once
            marked_pos, repeat_check, graph = implicit_route(prefix, ans_position)

            # Original used len(marked_pos)>50 to break; now no recursion, just skip directly
            # if len(marked_pos) > 50:
            #     print(f"[WARN] marked_pos too large ({len(marked_pos)}), skip real_id={real_id}")
            #     continue

            # --------- Save dicts (optional: keep most critical info) ----------
            dicts = {}
            dicts['**ans**'] = match_collect[real_id]

            # Original used last char as key; preserved
            last_char = ans[last_pos_ans:last_pos_ans+1]
            dicts[last_char] = {}
            dicts[last_char]['last'] = repeat_check

            # Also record marked_pos, but no longer recursively generate graphs
            dicts['marked_pos'] = marked_pos

            # graph_list always has exactly one graph
            graph_list = [graph]

            # If you want "don't save when no marked_pos", keep original logic; otherwise remove this check
            # if len(marked_pos) <= 0:
            #     print(f'len(marked_pos) <= 0, real_id={real_id}')
            #     continue

            with open(save_path + f'{real_id}.pkl', 'wb') as f:
                # print('save dicts')
                print(f'save dicts, real_id={real_id}')
                pickle.dump(dicts, f)

            with open(save_graph_path + f'{real_id}.pkl', 'wb') as f:
                print('save graph_list (single)')
                pickle.dump(graph_list, f)

        except Exception as e:
            print(f"Error at idx={idx}, real_id={real_id}: {type(e).__name__}: {e}")
            continue

    count += 1

print(emb_key_words_layers)
