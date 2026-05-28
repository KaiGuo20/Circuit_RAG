
import os

cache_root = 'yours'
hf_token = 'yours'  # Recommended: use environment variables or CLI login instead of hardcoding tokens in the repo

# 1) Set environment variables before any imports
os.environ["HF_TOKEN"] = hf_token
os.environ["HF_HOME"] = cache_root                 # Recommended: use a unified root directory

# 2) Then import libraries
from transformers import AutoModelForCausalLM, AutoTokenizer,  BitsAndBytesConfig
import torch
from transformer_lens import HookedTransformer, ActivationCache


def model_name_func(name):
    if name == 'llama3_8b':
        return 'meta-llama/Llama-3.1-8B-Instruct'


def load_tokenizer(name):
    model_name = model_name_func(name)
    if 'llama3_8b_finetune' in name:
        path = 'yours' #webqsp
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    return tokenizer

def load_model(name, device='cuda'):
    model_name = model_name_func(name)
    if 'ds' in name:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )

        if 'qwen-1.5B' in name:
            backbone_model = "Qwen/Qwen2.5-1.5B"
        if 'llama' in name:
            backbone_model = "meta-llama/Llama-3.1-8B"
        model = HookedTransformer.from_pretrained(
            backbone_model,
            hf_model=hf_model,
            tokenizer=tok,
            device=device,
            dtype=torch.float16,
            trust_remote_code=True,
        )
    elif 'llama3_8b' in name:
        backbone_model = "meta-llama/Llama-3.1-8B-instruct"
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        model = HookedTransformer.from_pretrained(
            backbone_model,
            hf_model=hf_model,
            tokenizer=tok,
            device=device,
            dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_name=model_name,
            device=device,
        )
    if 'llama3_8b' in name:
        return model, hf_model
    else:
        return model
