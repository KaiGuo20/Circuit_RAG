import os
import gc
import faiss
hf_token = 'yours'
os.environ["HF_TOKEN"] = hf_token
os.environ["TOKENIZERS_PARALLELISM"] = "false"
cache_root = 'yours'
hf_token = 'yours'  # Recommended: use environment variables or CLI login instead of hardcoding tokens in the repo

os.environ["HF_HOME"] = cache_root

import os
import pickle

dataset_name = 'gsm8k'
name = 'Qwen-0.5B'
# k = 30
# i = 23
# cluster_file = f'yours'
# with open(cluster_file,'rb') as f:
#     cluster_data = pickle.load(f)

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



device = 'cuda:1'


total_training_tokens = 1_000_000 * 20
acts_func = 'relu'
transcoder_config = {}
from_pretrained = False
name = 'Qwen-0.5B'

if 'ds' in name:
    l1_co = 0.00005
    max_length = 400
else:
    l1_co = 0.0005
    max_length = 300

model_name = model_name_func(name)
ori_model = load_model(name)

tokenizer = ori_model.tokenizer
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
n_layer = ori_model.cfg.n_layers
d_model = ori_model.cfg.d_model
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
        correct_idx.append(idx)

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

max_count = 50
count = 0

@torch.no_grad()
def get_mlp_and_rpost_all_layers(model: HookedTransformer, text: str, layers="all"):
    tokens = model.to_tokens(text).to(model.cfg.device)
    n_layers = model.cfg.n_layers
    if layers == "all":
        layer_ids = list(range(n_layers))
    elif isinstance(layers, int):
        layer_ids = [layers]
    else:
        layer_ids = list(layers)

    # Only cache the two types we need
    names = (
        [f"blocks.{l}.mlp.hook_post"   for l in layer_ids] +
        [f"blocks.{l}.hook_resid_post" for l in layer_ids]
    )

    logits, cache = model.run_with_cache(tokens, names_filter=names)

    mlp_list   = [cache[f"blocks.{l}.mlp.hook_post"][0]   for l in layer_ids]
    rpost_list = [cache[f"blocks.{l}.hook_resid_post"][0] for l in layer_ids]

    mlp_stack   = torch.stack(mlp_list,   dim=0)  # [n_layers, seq, d_model]
    rpost_stack = torch.stack(rpost_list, dim=0)
    return mlp_stack, rpost_stack

def get_mlp_and_rpost_all_layers_tokens(model, tokens, layers=0):
    # tokens = model.to_tokens(text).to(model.cfg.device)
    n_layers = model.cfg.n_layers
    if layers == "all":
        layer_ids = list(range(n_layers))
    elif isinstance(layers, int):
        layer_ids = [layers]
    else:
        layer_ids = list(layers)

    # Only cache the two types we need
    names = (
        [f"blocks.{l}.mlp.hook_post"   for l in layer_ids] +
        [f"blocks.{l}.hook_resid_post" for l in layer_ids]
    )
    print(tokens.shape)
    logits, cache = model.run_with_cache(tokens, names_filter=names)

    mlp_list   = [cache[f"blocks.{l}.mlp.hook_post"][0]   for l in layer_ids]
    rpost_list = [cache[f"blocks.{l}.hook_resid_post"][0] for l in layer_ids]

    mlp_stack   = torch.stack(mlp_list,   dim=0)  # [n_layers, seq, d_model]
    rpost_stack = torch.stack(rpost_list, dim=0)
    

    return mlp_stack, rpost_stack




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

node_thres = 0.3
k = 50
emb_pkl_file = f'yours/{dataset_name}_{name}_{k}_{node_thres}_last_mlp_emb_kmeans.pkl'
with open(emb_pkl_file,'rb') as f:
    layer_wise_list, center_list, sentence_label_dicts = pickle.load(f)

print(len(center_list),center_list[0].shape)
    
record_save = f'./records/{name}_{dataset_name}_{node_thres}.pkl'
with open(record_save,'rb') as f:
    embeds_take_layers, embeds_key_words, emb_key_words_layers, logit_take_layers, logit_key_words, logit_key_words_layers = pickle.load(f)
embeds_key_words = sorted(embeds_key_words.items(), key=lambda x: x[1], reverse=True)
top_keys = []
for idx,key in enumerate(embeds_key_words):
    if idx >= 50:break
    top_keys.append(key[0])

counts_same = 0
import numpy as np
def split_by_mask(x: torch.Tensor, mask: torch.Tensor):
    # Find indices of True values
    cut_points = mask.nonzero(as_tuple=True)[0].tolist()
    slices = []
    start = 0
    for p in cut_points:
        print(start,p)
        slices.append(x[start:p+1])  # include this True element
        start = p+1
    if start < len(x):  # trailing segment
        slices.append(x[start:])
    return slices

def deduplicate_adjacent(lst):
    result = []
    for x in lst:
        if not result or result[-1] != x:
            result.append(x)
    return result

for idx, data in enumerate(test_data):
    # if idx not in correct_idx:continue
    # if count >= max_count: break
    # print(count)
    for node_thre in [node_thres]:
        overall_acc = []
        layer_counts = {}
        counts = 0
        if 'ds' in name:
            prompt = str(bos_token) + "\nQuestion: " + data['question'] + "\n<think>\n"
        # prompt = bos_token + "\nQuestion: " + data['question'] + "\nLet's think step by step\n"
        else:
            prompt = "\nQuestion: " + data['question'] + "\nLet's think step by step\nAnswer:\n"
        
        ans_prompt = data['ans'][len(prompt):]# .split(' ')
        # print(ans_prompt)
        prompt_list =  "Answer:\n" + data['ans']
        prompt_list_token = model.to_tokens(prompt_list).to(model.cfg.device)
        mlp_stack, rpost_stack = get_mlp_and_rpost_all_layers(ori_model, prompt_list)
        
        ans_prompt = tokenizer.tokenize(ans_prompt)
        prompt_list = tokenizer.tokenize(prompt_list)
        
        bool_mask = [x in top_keys for x in prompt_list]
        # print(bool_mask)
        
        
        
        ans_check = False
        last_slides_idx = -1
        current_slides_idx = -1
        chunk_idx = 0
        print(len(prompt_list), len(bool_mask), mlp_stack.shape)
        slices = split_by_mask(prompt_list_token[0], torch.tensor(bool_mask))
        print(slices)
        counts_ori = 0
        counts_slices = 0
        given_labels = deduplicate_adjacent(sentence_label_dicts[0][idx])# [counts_slices - 2] 
        for slice_idx, slice in enumerate(slices):
            # if slice_idx == 2:break
            print("slice:", slice)
            mlp_stack, rpost_stack = get_mlp_and_rpost_all_layers_tokens(ori_model, slice)
            embs = mlp_stack[:,-1,:].detach().cpu().numpy()# .unsqueeze(0).detach().cpu().numpy()
            # print(embs)
            # print(embs.shape)
            centroids = center_list[0] # torch.from_numpy(center_list[0]).cuda()
            # print(centroids.shape)
            d = embs.shape[-1]
            # Build an index and add the centroids
            index_centroids = faiss.IndexFlatL2(d)
            # print(centroids)
            index_centroids.add(centroids)  # centroids: (k, d)

            # Search for nearest centroid
            D, I = index_centroids.search(embs, 3)  # 1 means find the nearest one
            # print(I.shape)
            given_label = given_labels[slice_idx] # sentence_label_dicts[0][idx][counts_slices - 2]
            if I[0][0] == given_label:
                counts_ori += 1
            print(sentence_label_dicts[0][idx])
            print("Nearest centroid index:", I, sentence_label_dicts[0][idx][counts_slices - 2] )
            print("Distance:", D)
            counts_slices += len(slice)
            
        
        # exit()
        print('====================')
        bool_mask = torch.tensor(bool_mask).nonzero(as_tuple=True)[0].tolist()
        print(bool_mask)
        overall_tokens = []
        counts_slices = 0
        for slice_idx, token_slice in enumerate(slices):
            # token_slice = token_slice.unsqueeze(0)
            # print(token_slice)
            
            given_label = given_labels[slice_idx]# sentence_label_dicts[0][idx][counts_slices - 2] 
            for token_id in range(token_slice.shape[0]):
                mlp_stack, rpost_stack = get_mlp_and_rpost_all_layers_tokens(ori_model, token_slice[:token_id + 1])
                embs = mlp_stack[0, -1].unsqueeze(0).detach().cpu().numpy()
                centroids = center_list[0] # torch.from_numpy(center_list[0]).cuda()
                # print(centroids.shape)
                d = embs.shape[-1]
                # Build an index and add the centroids
                index_centroids = faiss.IndexFlatL2(d)
                index_centroids.add(centroids)  # centroids: (k, d)

                # Search for nearest centroid
                D, I = index_centroids.search(embs, 3)  # 1 means find the nearest one
                # print(I.shape)

                if I[0][0] == given_label:
                    counts_same += 1
                print("Nearest centroid index:", I[0][0], given_label)
                print("Distance:", D[0][0])
            counts_slices += len(token_slice)
            # overall_tokens.append(token_slice)
            # results = torch.cat(overall_tokens, dim=0)
        continue
        for ans_idx, a in tqdm(enumerate(prompt_list_token[0]), total = len(prompt_list_token[0])):
                # if ans_idx < 30: continue
            # if ans_idx == 30: break
            
            given_label = sentence_label_dicts[0][idx][ans_idx] 
            
            
            if ans_idx in bool_mask:
                print('split')
                if chunk_idx == 0:
                    last_slides_idx = current_slides_idx
                else:
                    last_slides_idx = current_slides_idx + 1
                current_slides_idx = ans_idx
                chunk_idx += 1
            print(len(sentence_label_dicts[0][idx]), len(prompt_list), len(np.unique(sentence_label_dicts[0][idx])))
            print(last_slides_idx, ans_idx)
            if last_slides_idx != ans_idx:
                print('compares',prompt_list_token[0][last_slides_idx + 1:ans_idx + 1])
            else:
                print('compares',prompt_list_token[0][last_slides_idx:ans_idx + 1])
            # print(a)
            continue
            if 'step' in a: ans_check = True
            if ans_check == False: continue
            
            mlp_stack, rpost_stack = get_mlp_and_rpost_all_layers_tokens(ori_model, prompt_list_token[0][last_slides_idx - 1:ans_idx + 2].unsqueeze(0))
            # x_new = torch.randn(d).cpu().numpy().astype('float32').reshape(1, -1)
            # print(mlp_stack.shape)
            embs = mlp_stack[0, -1].unsqueeze(0).detach().cpu().numpy()
            centroids = center_list[0] # torch.from_numpy(center_list[0]).cuda()
            # print(centroids.shape)
            d = embs.shape[-1]
            # Build an index and add the centroids
            index_centroids = faiss.IndexFlatL2(d)
            index_centroids.add(centroids)  # centroids: (k, d)

            # Search for nearest centroid
            D, I = index_centroids.search(embs, 1)  # 1 means find the nearest one
            # print(I.shape)
            if I[0][0] == given_label:
                counts_same += 1
            print("Nearest centroid index:", I[0][0], given_label)
            print("Distance:", D[0][0])

            
            
            
            # embs = mlp_stack[0, ans_idx].unsqueeze(0)
            
            # new_x = torch.nn.functional.normalize(embs, dim=1)
            # centroids = torch.nn.functional.normalize(torch.from_numpy(center_list[0]), dim=1).cuda()
            # # new_x = embs
            # # centroids = torch.from_numpy(center_list[0]).cuda()
            # sim = new_x @ centroids.T        # (m, k)
            # labels = sim.argmax(dim=1)
            # token_slice = tokens[last_slides_idx: ans_idx]
            # print(labels, given_label)
            
            # break
            
            # if bos_token in a or 'Question:' in a:break
            
            
            # prompt += a
            # try:
            #     graph = attribute(
            #         prompt=prompt,
            #         model=model,
            #         # input_ids=input_ids,
            #         input_ids=None,
            #         max_n_logits=max_n_logits,
            #         desired_logit_prob=desired_logit_prob,
            #         batch_size=batch_size,
            #         max_feature_nodes=max_feature_nodes,
            #         offload=offload,
            #         verbose=verbose,
            #         print_log=False,
                    
            #     )

                

            #     node_threshold = node_thre
            #     edge_threshold = 1# 0.99

            #     # print(prune_graph(graph, node_threshold=0.4, edge_threshold=0.5))

            #     node_mask, edge_mask, cumulative_scores = (
            #         el.cpu() for el in prune_graph(graph, node_threshold, edge_threshold)
            #     )
            #     nonzero_indices = torch.nonzero(node_mask)
            #     # from transformers import AutoTokenizer

            #     # tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name, cache_dir=cache_dir)
            #     scan = graph.scan
            #     nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores, scan)
            #     used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
            #     nodes_dicts = {}



            #     for n in used_nodes:
            #         nodes_dicts[n.node_id] = {}
            #         nodes_dicts[n.node_id]['features'] = n.feature_type
            #         nodes_dicts[n.node_id]['clerp'] = n.clerp


            #     source_nodes = {}
            #     target_nodes = {}
                
            #     result_graph = {}
            #     marked_source = set()
            #     marked_target = set()

            #     for e in used_edges:
            #         if nodes_dicts[e['source']]['features'] == "mlp reconstruction error": continue
            #         if nodes_dicts[e['target']]['features'] == "mlp reconstruction error": continue
            #         if nodes_dicts[e['target']]['features'] == "logit" and nodes_dicts[e['source']]['features'] == 'embedding': continue
            #         # if nodes_dicts[e['source']]['features'] == "cross layer transcoder":
            #         weights = e['weight']
            #         if weights<0. :continue
            #         source_node, source_pos, source_layer = translate_node_ids(e['source'], nodes_dicts[e['source']]['features'])
            #         target_node, target_pos, target_layer = translate_node_ids(e['target'], nodes_dicts[e['target']]['features'])
                    
            #         if nodes_dicts[e['source']]['features'] == 'embedding':
            #             if nodes_dicts[e['target']]['features'] == 'embedding': continue
            #             if e['target'] not in marked_source:
            #                 marked_source.add(e['target'])
            #             else:
            #                 continue
            #             if target_layer not in embeds_take_layers:
            #                 embeds_take_layers[target_layer] = 0
            #             if target_node not in embeds_key_words:    
            #                 embeds_key_words[target_node] = 0
            #                 emb_key_words_layers[target_node] = []
            #             embeds_take_layers[target_layer] += 1
            #             embeds_key_words[target_node] += 1
            #             emb_key_words_layers[target_node].append(int(target_layer))
                        
            #         if nodes_dicts[e['target']]['features'] == 'logit':
            #             if e['source'] not in marked_target:
            #                 marked_target.add(e['source'])
            #             else:
            #                 continue
            #             if source_layer not in logit_take_layers:
            #                 logit_take_layers[source_layer] = 0
            #             if source_node not in logit_key_words:
            #                 logit_key_words[source_node] = 0
            #                 logit_key_words_layers[source_node] = []
            #             logit_key_words[source_node] += 1
            #             logit_take_layers[source_layer] += 1
            #             logit_key_words_layers[source_node].append(int(source_layer))
                
            #     # model = model.to('cpu')
            #     del graph
            #     gc.collect() 
            #     torch.cuda.empty_cache()
            #     # model = model.to(device)
                
            # except:continue
        break
    break
    count += 1
print(counts_ori)
print(counts_same)
print(len(slices))

print(center_list[0])