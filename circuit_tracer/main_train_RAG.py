import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch

device = 'cuda:0'

# Directly reserve 135GB (leaving some headroom from 140GB)
gb_to_reserve = 120
_placeholder = torch.empty(gb_to_reserve * 1024**3 // 4, dtype=torch.float32, device=device)
print(f"Reserved {gb_to_reserve} GB")
del _placeholder

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
# from transcoder.short_activations_stores import Short_ActivationsStore

from configs import Configs

from redefined_datasets import MyDataset, MyIterableDataset

sys.path.append('../')
from load_utils import load_model, model_name_func

# base_path = f'yours/baby_models'
# path =  os.path.join(base_path, "model_weights.pt")


# cache_dir = 'yours'
cache_dir = 'yours'
activate_cache_dir = 'yours'
# activate_cache_dir = 'yours'

if os.path.exists(activate_cache_dir) == False:
    os.makedirs(activate_cache_dir)



# device = 'cuda:1'



total_training_tokens = 2_000_000 * 10
acts_func = 'relu'
transcoder_config = {}
from_pretrained = False
name = 'llama3_8b' # 'ds-llama'

model_name = model_name_func(name)
if name == 'llama3_8b_finetune' or name == 'llama3_8b':
    model, hf_model = load_model(name, device=device)
else:
    model = load_model(name, device=device)

tokenizer = model.tokenizer
train_dataset_name = 'manu_musique'# "gsm8k" "2wiki" 'hotpotqa' 'RAMDOCS' 'WebQSP' 'WebQSP_ROG' 'musique'

if 'ds' in name:
    max_length = 500
else:
    max_length = 300

transcoder_config['model_name'] = model_name
transcoder_config['train_dataset_name'] = train_dataset_name
transcoder_config['max_length'] = max_length
transcoder_config['method'] = 'real'
transcoder_config['pattern_name'] = train_dataset_name
transcoder_config['is_tokenized'] = True
transcoder_config['dataset_path'] = None


save_json = f'../data_{train_dataset_name}/{name}_answer.json'
with open(save_json,'r') as f:
    test_data = json.load(f)
def _is_new_format(entry):
    question = entry.get('question', '')
    path = entry.get('path', '')
    ans = entry.get('ans', '')
    expected_prefix = f"Question: {question}\n Based on the context: {path}\n Answer:\n"
    return ans.startswith(expected_prefix)


already_formatted = any(_is_new_format(item) for item in test_data)

if not already_formatted:
    for item in test_data:
        question = item['question']
        original_ans = item['ans']
        path = item['path']

        item['ans'] = f"Question: {question}\n Based on the context: {path}\n Answer:\n {original_ans}"

    with open(save_json, 'w') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"Updated {len(test_data)} items in {save_json}")
else:
    print(f"Skipped formatting; detected existing formatted entries in {save_json}")

# for item in test_data:
#     question = item['question']
#     original_ans = item['ans']
#     path = item['path']
#
#     # Build new ans format
#     item['ans'] = f"Question: {question}\n Based on the context: {path}\n Answer:\n {original_ans}"

# # Save back to file
# with open(save_json, 'w') as f:
#     json.dump(test_data, f, indent=2, ensure_ascii=False)

# print(f"Updated {len(test_data)} items in {save_json}")
training_corpus = []
for data in test_data:
    training_corpus.append(data['ans'])
valid_corpus = training_corpus[:10]


train_dataset = MyIterableDataset(training_corpus, tokenizer,stack=True)
valid_dataset = MyIterableDataset(valid_corpus, tokenizer,stack=True)

n_layer = model.cfg.n_layers
d_model = model.cfg.d_model
    

record_scores_list = []
print(n_layer)
if 'llama' in name.lower():
    l1_co = 0.0005
else:
    l1_co = 0.00005
# l1_co = 0.0005



if 'ds' in name:
    max_length = 500
    
elif 'gemma-2B' in name:
    max_length = 300

if 'llama' in name.lower():
    total_training_tokens = total_training_tokens / 10
#     n_batches = 8
#     store_batch = 4
#     batch_size = 256
# else:


if train_dataset_name == 'hotpotqa':
    max_length = 500
    total_training_tokens = 10_000_000 
if train_dataset_name == '2wiki':
    max_length = 500
    total_training_tokens = 10_000_000 
if train_dataset_name == 'musique':
    max_length = 500
    total_training_tokens = 10_000_000 

if train_dataset_name == 'manu_musique':
    max_length = 500
    total_training_tokens = 10_000_000 


n_batches = 16#32
store_batch = 16#16
batch_size = 1024

print(n_batches, store_batch, batch_size)

model = model.to(device)
for target_layer in range(8,n_layer):
    configs = Configs(
        target_layer = target_layer,
        out_hook_point_layer = target_layer,# target_layer + 1 if target_layer + 1 <= n_layer - 1 else n_layer - 1,  # n_layer - 1, # target_layer,# n_layer - 1, # target_layer,# n_layer, # target_layer,
        # max_layer = 4,
        epoch = 1,
        d_in = d_model, # 2048, #  768, # 2048, #  2304, # llama 2048
        d_out = d_model, # 2048, # 768, # 2048, # 2304,
        d_transcoder = d_model, # 2048, # 768, # 2048, # 2304,
        train_batch_size = batch_size, # 2048,
        context_size = max_length, # 64, # 128, # 64, # 128, # 256, # 512,
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
        
        n_batches_in_buffer = n_batches,
        total_training_tokens = total_training_tokens,
        store_batch_size = store_batch, # 256,
        
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
        
        batch_size = batch_size,
        lr = 1e-4,# 1e-4,# 1e-3, # 1e-3, #  0.0004,
        dead_feature_window = d_model,# 100,# 50, # 100,  # unless this window is larger feature sampling,
        dead_feature_estimation_method = 'no_fire',
        dead_feature_threshold = 1e-8,
        resample_batches = batch_size,
        is_sparse_connection = False,
        checkpoint_path = f'yours/{acts_func}_{name}_{train_dataset_name}_{n_layer}',
        lr_scheduler_name = 'constantwithwarmup',
        lr_warm_up_steps = 500,
        from_pretrained = from_pretrained,
        
        
    )
    

    ###########
    # model = model.to(configs.act_store_device)
    configs.d_in = d_model
    configs.d_out = d_model
    configs.d_transcoder = d_model * 2
    configs.tokenizer_name = None
    
    activation_stores = ActivationsStore(configs, model, dataset=train_dataset, tokenizer=tokenizer)

    if acts_func == 'relu':
        transcoder = SingleLayerTranscoder(configs, Relu())# JumpReLU(0.0, 0.1))

    record_scores = train_sae_on_language_model(configs, model, transcoder, activation_stores, use_eval=False)
    print(f"Target Layer: {target_layer}, Record Scores: {record_scores}")
    record_scores_list.append(record_scores)
    del activation_stores
    del transcoder
    gc.collect()
    torch.cuda.empty_cache()
transcoder_cache_dir = os.path.join(configs.checkpoint_path, f"{name}_{train_dataset_name}_{configs.l1_coefficient}",'configs.json')# f'yours/{acts_func}_{pattern_name}_{method}_{n_layer}'
if os.path.exists(transcoder_cache_dir) is False:
    os.makedirs(transcoder_cache_dir, exist_ok=True)
configs.save(os.path.join(transcoder_cache_dir,'configs.json'))
configs.load(os.path.join(transcoder_cache_dir,'configs.json'))
print(configs)
path = f'records/{acts_func}_{configs.pattern_name}_{configs.method}_{name}_{train_dataset_name}_{configs.l1_coefficient}_record_scores.json'
with open(path, "w") as f:
    json.dump(record_scores_list, f)
print(record_scores_list)