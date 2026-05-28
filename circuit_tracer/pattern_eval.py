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

seed = 4
# Python built-in random
random.seed(seed)

# NumPy random
np.random.seed(seed)

device = 'cuda:4'


total_training_tokens = 1_000_000 * 20
acts_func = 'relu'
transcoder_config = {}
from_pretrained = False
name = 'ds-qwen-1.5B'

if 'ds' in name:
    l1_co = 0.00005
    max_length = 400
else:
    l1_co = 0.0005
    max_length = 300

model_name = model_name_func(name)
model = load_model(name)

tokenizer = model.tokenizer
train_dataset_name = "gsm8k"
# for name, param in model.named_parameters():
#     print(f"{name:60} | {str(param.shape):20} | requires_grad={param.requires_grad}")

transcoder_config['model_name'] = model_name
transcoder_config['train_dataset_name'] = train_dataset_name
transcoder_config['max_length'] = max_length
transcoder_config['method'] = 'real'
transcoder_config['pattern_name'] = 'gsm8k'
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
    
    model_name = 'gpt2', # 'llama3',# 'llama3',# 'gpt2',# 'llama3',
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
print(configs.d_transcoder)

configs.d_transcoder = d_model * 2 # 192 # 2048 # 768 # 2048 # 2304

d_emb = d_model # model.embed.W_E.shape[1]
# configs.d_in = d_emb
# configs.d_out = d_emb 
# configs.d_model = d_emb 
# configs.d_transcoder = d_emb
configs.n_layers = n_layer



model = ReplacementModel.from_self_pretrained_and_transcoders(cfg=configs,model_name=name, model_path = cache_root, transcoders_path = transcoder)
print(model)

model = model.to(configs.act_store_device)

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

save_json = f'../data/{name}_answer.json'
with open(save_json,'r') as f:
    test_data = json.load(f)

import re
def answer_judge(data):
    text = data['gold_ans']
    match = re.search(r"####\s*(\d+)", text)
    if match is not None:
        match = match.group(1)
        if match not in data['ans']:
            return 0
    if data['judge'] is not None:
        judge_prompt = data['judge'].lower()
    else:
        return 0
    if 'yes' in judge_prompt and 'no' not in judge_prompt:
        return 1
    else:
        return 0

correct_idx = []

for idx, data in enumerate(test_data):
    if data['judge'] is None:
        print(f'None idx {idx}')
        continue
    judge = answer_judge(data)
    if judge:
        # print(idx)
        correct_idx.append(idx)
        # print('gold ans')
        # print(data['gold_ans'])
        # print('pred ans')
        # print(data['ans'])

print(correct_idx)
print(len(correct_idx)/len(test_data))
max_n_logits = 10   # How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_desired_log_prob); see below for the latter
desired_logit_prob = 0.95  # Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)
max_feature_nodes = 2048  # Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.
batch_size=256
offload = 'cpu'# 'disk' if IN_COLAB else 'cpu' # Offload various parts of the model during attribution to save memory. Can be 'disk', 'cpu', or None (keep on GPU)
verbose = True 

def nodes_at_n_hop(G, source, n, filter = ''):
    current_layer = set([source])
    all_traversed = set([source])
    for _ in range(n):
        next_layer = set()
        for node in current_layer:
            # print(f"Expanding node: {node}")
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

def translate_node_ids(node_id, given_type):
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

def get_text_before_second_question(text):
    text = "Answer:" + text.split("Answer:")[1]
    parts = text.split("Question")
    if len(parts) < 3:
        return text  # If fewer than two Questions, return original text
    return "Question".join(parts[:2])

for idx, data in enumerate(test_data):
    # if idx not in correct_idx:continue
    # if count >= max_count: break
    print(count)
    for node_thre in [0.5]:
        overall_acc = []
        layer_counts = {}
        counts = 0
        if 'ds' in name:
            prompt = str(bos_token) + "\nQuestion: " + data['question'] + "\n<think>\n"
        # prompt = bos_token + "\nQuestion: " + data['question'] + "\nLet's think step by step\n"
        else:
            prompt = "\nQuestion: " + data['question'] + "\nLet's think step by step\nAnswer:\n"
        ans = get_text_before_second_question(data['ans'])
        
        # print(ans)
        # ans_prompt = tokenizer.tokenize(ans)
        # print(ans_prompt)
        ans = ans.replace('\n',' \n ')
        ans = ans.replace(':',' :')
        ans = ans.replace(',',' ,')
        ans = ans.replace('.',' .')
        ans = ans.replace('?',' ?')
        ans = ans.replace('!',' !')
        ans = ans.replace('$','$ ')
        
        ans_prompt = ans.split(' ')
        print(ans_prompt)
        random_indices = random.sample(range(len(ans_prompt)), 2)
        # random_indices = list(range(len(ans_prompt)))
        # prompt = tokenizer.tokenize(prompt)
        prompt = prompt.split(' ')
        for ans_idx, a in tqdm(enumerate(ans_prompt), total = len(ans_prompt)):
            prompt.append(a)
            
            # print(prompt)
            if ans_idx not in random_indices:continue
            try:
                prompt_str = ' '.join(prompt)
                graph = attribute(
                    prompt=prompt_str,
                    model=model,
                    # input_ids=input_ids,
                    input_ids=None,
                    max_n_logits=max_n_logits,
                    desired_logit_prob=desired_logit_prob,
                    batch_size=batch_size,
                    max_feature_nodes=max_feature_nodes,
                    offload=offload,
                    verbose=verbose,
                    print_log=False,
                )

                

                node_threshold = node_thre
                edge_threshold = 1# 0.99

                # print(prune_graph(graph, node_threshold=0.4, edge_threshold=0.5))

                node_mask, edge_mask, cumulative_scores = (
                    el.cpu() for el in prune_graph(graph, node_threshold, edge_threshold)
                )
                nonzero_indices = torch.nonzero(node_mask)
                # from transformers import AutoTokenizer

                # tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name, cache_dir=cache_dir)
                scan = graph.scan
                nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores, scan)
                used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
                nodes_dicts = {}



                for n in used_nodes:
                    nodes_dicts[n.node_id] = {}
                    nodes_dicts[n.node_id]['features'] = n.feature_type
                    nodes_dicts[n.node_id]['clerp'] = n.clerp


                source_nodes = {}
                target_nodes = {}
                
                result_graph = {}
                result_graph_G = nx.DiGraph()
                marked_source = set()
                marked_target = set()
                logit_nodes = []
                # print(used_edges)
                for e in used_edges:
                    if nodes_dicts[e['source']]['features'] == "mlp reconstruction error": continue
                    if nodes_dicts[e['target']]['features'] == "mlp reconstruction error": continue
                    if nodes_dicts[e['target']]['features'] == "logit" and nodes_dicts[e['source']]['features'] == 'embedding': continue
                    # if nodes_dicts[e['source']]['features'] == "cross layer transcoder":
                    weights = e['weight']
                    if weights<0. :continue
                    source_node, source_pos, source_layer = translate_node_ids(e['source'], nodes_dicts[e['source']]['features'])
                    target_node, target_pos, target_layer = translate_node_ids(e['target'], nodes_dicts[e['target']]['features'])
                    # print(f"Edge from {e['source']} ({nodes_dicts[e['source']]['features']}, id: {source_node}, pos: {source_pos}, layer: {source_layer}) to {e['target']} ({nodes_dicts[e['target']]['features']}, id: {target_node}, pos: {target_pos}, layer: {target_layer}) with weight {weights}")
                    
                    start_node = f'{source_node}_{source_layer}_{source_pos}'
                    end_node = f'{target_node}_{target_layer}_{target_pos}'
                    if end_node not in result_graph:
                        result_graph[end_node] = {}
                    if start_node not in result_graph[end_node]:
                        result_graph[end_node][start_node] = 0
                    result_graph[end_node][start_node] += weights
                    
                    if nodes_dicts[e['source']]['features'] == 'embedding':
                        if nodes_dicts[e['target']]['features'] == 'embedding': continue
                        if e['target'] not in marked_source:
                            marked_source.add(e['target'])
                        else:
                            continue
                        if target_layer not in embeds_take_layers:
                            embeds_take_layers[target_layer] = 0
                        if target_node not in embeds_key_words:    
                            embeds_key_words[target_node] = 0
                            emb_key_words_layers[target_node] = []
                        embeds_take_layers[target_layer] += 1
                        embeds_key_words[target_node] += 1
                        emb_key_words_layers[target_node].append(int(target_layer))
                        
                    if nodes_dicts[e['target']]['features'] == 'logit':
                        if e['source'] not in marked_target:
                            marked_target.add(e['source'])
                        else:
                            continue
                        if source_layer not in logit_take_layers:
                            logit_take_layers[source_layer] = 0
                        if source_node not in logit_key_words:
                            logit_key_words[source_node] = 0
                            logit_key_words_layers[source_node] = []
                        logit_key_words[source_node] += 1
                        logit_take_layers[source_layer] += 1
                        logit_key_words_layers[source_node].append(int(source_layer))
                        logit_nodes.append(end_node)
                    
                
                result_graph = {outer_k: {inner_k: inner_v 
                        for inner_k, inner_v in outer_v.items() if inner_v > 0} 
                for outer_k, outer_v in result_graph.items()}
                result_graph = {k: v for k, v in result_graph.items() if v}
                for key in result_graph.keys():
                    for inner_key in result_graph[key].keys():
                        result_graph_G.add_edge(key, inner_key, weight=result_graph[key][inner_key])
                # print(result_graph)
                
                repeat_check = set()
                for target_node in list(set(logit_nodes)):
                    # print(result_graph_G.in_edges(target_node))
                    
                    target_layers = nodes_at_n_hop(result_graph_G, target_node, n_layer + 2)
                    for t in target_layers:
                        word, layer, pos = t.split('_')
                        tmp_t = f'{word}_{layer}'
                        if tmp_t in repeat_check:continue
                        repeat_check.add(tmp_t)
                        word, layer, pos = t.split('_')
                        if layer not in contribute_counts:
                            contribute_counts[layer] = {}
                        # print(word, layer, pos)
                        if word not in contribute_counts[layer]:
                            contribute_counts[layer][word] = 0
                        contribute_counts[layer][word] += 1
                #     print('target', target_node)
                #     print(target_layers)
                # print(contribute_counts)
                # exit()
                # model = model.to('cpu')
                del graph
                gc.collect() 
                torch.cuda.empty_cache()
                # model = model.to(device)
            
            except:continue
            # exit()
    count += 1

record_save = f'./records/{name}_{train_dataset_name}_{node_thre}_{seed}.pkl'
with open(record_save,'wb') as f:
    pickle.dump((embeds_take_layers, embeds_key_words, emb_key_words_layers, logit_take_layers, logit_key_words, logit_key_words_layers, contribute_counts),f)


print(emb_key_words_layers)