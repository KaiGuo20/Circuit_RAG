
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
    if name == 'gemma-2B':
        return 'google/gemma-2-2b'
    if name == 'gemma-1B':
        return 'google/gemma-3-1b'
    if name == 'Qwen-0.5B':
        return "Qwen/Qwen2.5-0.5B"
    if name == 'Qwen3-0.6B':
        return "Qwen/Qwen3-0.6B"
    if name == 'Qwen3-4B':
        return "Qwen/Qwen3-4B"
    if name == 'Qwen-1.5B':
        return "Qwen/Qwen2.5-1.5B"
    if name == 'Llama-1B':
        return 'meta-llama/Llama-3.2-1B'
    if name == 'Llama-3B':
        return 'meta-llama/Llama-3.2-3B'
    if name == 'Llama-8B':
        return 'meta-llama/Llama-3.1-8B'
    if name == 'Llama-8BI':
        return 'meta-llama/Llama-3.1-8B-Instruct'
    if name == 'ds-qwen-1.5B':
        return 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    if name == 'ds-llama':
        return 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    if name == 'llama3_8b_finetune':
        return 'meta-llama/Llama-3.1-8B-Instruct'
    if name == 'llama3_8b':
        return 'meta-llama/Llama-3.1-8B-Instruct'
    if name == 'qwen3_14b':
        return 'Qwen/Qwen3-14B'
    if name == 'qwen2.5_14b':
        return 'Qwen/Qwen2.5-14B'
    if name == 'llama3_70b':
        return 'meta-llama/Llama-3.1-70B-Instruct'

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
    elif 'llama3_8b_finetune' in name:
        path = 'yours'
        tok = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        backbone_model = "meta-llama/Llama-3.1-8B-Instruct"
        model = HookedTransformer.from_pretrained(
            backbone_model,
            hf_model=hf_model,
            tokenizer=tok,
            device=device,
            dtype=torch.float16,
            trust_remote_code=True,
        )
    elif 'qwen3_14b' in name:
        backbone_model = "Qwen/Qwen3-14B"
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
    elif 'qwen2.5_14b' in name:
        backbone_model = "Qwen/Qwen2.5-14B"
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
    elif 'llama3_70b' in name:
        backbone_model = "meta-llama/Llama-3.1-70B-instruct"
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        # )
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        model = HookedTransformer.from_pretrained_no_processing(
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
    if 'llama3_8b_finetune' in name or 'llama3_8b' in name or 'qwen3_14b' in name or 'qwen2.5_14b' in name or 'llama3_70b' in name:
        return model, hf_model
    else:
        return model
