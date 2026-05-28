
import os
from tqdm import tqdm
from datasets import load_dataset, Dataset
import json
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cache_root = 'yours'
hf_token = 'yours'  # Recommended: use environment variables or CLI login instead of hardcoding tokens in the repo

os.environ["HF_TOKEN"] = hf_token
os.environ["HF_HOME"] = cache_root
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from load_utils import model_name_func, load_model, load_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformer_lens import HookedTransformer, ActivationCache
import re
from load_utils import load_model, model_name_func
INVALID_ANS = "[invalid]"

def _to_path(paragraphs):
    """Keep only idx/title/paragraph_text as path output format."""
    return [
        {
            "idx": p.get("idx"),
            "title": p.get("title", ""),
            "paragraph_text": p.get("paragraph_text", ""),
        }
        for p in paragraphs
    ]

def build_manu_musique_dataset(raw_ds, n_total=2000, seed=42):
    """
    Build a new dataset from raw MuSiQue samples:
    - Total count: n_total
    - Three types: all_support / mix / all_unsupport
    - Each sample gets new fields:
        - path: List[dict]
        - mau_type: str
    """
    random.seed(seed)

    base = n_total // 3
    rem  = n_total - base * 3
    target = {
        "all_support": base + (1 if rem > 0 else 0),
        "mix":         base + (1 if rem > 1 else 0),
        "all_unsupport": base
    }

    indices = list(range(len(raw_ds)))
    random.shuffle(indices)

    buckets = {"all_support": [], "mix": [], "all_unsupport": []}

    for i in indices:
        if all(len(buckets[t]) >= target[t] for t in target):
            break

        ex = dict(raw_ds[i])
        paras = ex.get("paragraphs", [])
        if not paras:
            continue

        paras_sorted = sorted(paras, key=lambda x: x.get("idx", 10**9))
        supporting   = [p for p in paras_sorted if p.get("is_supporting", False)]
        nonsupport   = [p for p in paras_sorted if not p.get("is_supporting", False)]

        if len(buckets["all_support"]) < target["all_support"]:
            if len(supporting) > 0:
                new_ex = {k: v for k, v in ex.items()}
                new_ex["path"] = _to_path(supporting)
                new_ex["mau_type"] = "all_support"
                buckets["all_support"].append(new_ex)

        if len(buckets["mix"]) < target["mix"]:
            if len(supporting) + len(nonsupport) >= 5 and len(supporting) > 0:
                sel = supporting[:5]
                if len(sel) < 5:
                    sel += nonsupport[: (5 - len(sel))]
                sel = sel[:5]
                new_ex = {k: v for k, v in ex.items()}
                new_ex["path"] = _to_path(sel)
                new_ex["mau_type"] = "mix"
                buckets["mix"].append(new_ex)

        if len(buckets["all_unsupport"]) < target["all_unsupport"]:
            if len(nonsupport) >= 5:
                sel = nonsupport[:5]
                new_ex = {k: v for k, v in ex.items()}
                new_ex["path"] = _to_path(sel)
                new_ex["mau_type"] = "all_unsupport"
                buckets["all_unsupport"].append(new_ex)

    selected = buckets["all_support"] + buckets["mix"] + buckets["all_unsupport"]
    print("built mau_musique counts:",
          {k: len(v) for k, v in buckets.items()},
          "total:", len(selected))
    return Dataset.from_list(selected)


name = 'qwen2.5_14b'  # llama3_8b, qwen3_14b
dataset_name = 'manu_musique'  # hotpotqa, 2wiki, musique, manu_musique
model_name = model_name_func(name)
model, hf_model = load_model(name)
tokenizer = load_tokenizer(name)
LLAMA3_CHAT_TEMPLATE = (
    "<|begin_of_text|>"
    "{% for message in messages %}"
    "{% set role = message['role'] %}"
    "{% if role == 'system' %}"
    "<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
    "{% elif role == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
    "{% elif role == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
    "{% else %}"
    "<|start_header_id|>{{ role }}<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "{% endif %}"
)
if not getattr(tokenizer, "chat_template", None):
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
print(model.cfg.n_layers)

if dataset_name == 'hotpotqa':
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        name="fullwiki",
        split="train"
    ).select(range(1000))
    test = dataset
elif dataset_name == "2wiki":
    dataset = load_dataset(
        "framolfese/2WikiMultihopQA",
        split="train"
    ).select(range(1000))
    test = dataset
elif dataset_name == "musique":
    dataset = load_dataset(
        "dgslibisey/MuSiQue",
        split="train"
    ).select(range(2000))
    test = dataset
elif dataset_name == "manu_musique":
    raw = load_dataset(
        "dgslibisey/MuSiQue",
        split="train"
    ).select(range(8000))
    dataset = build_manu_musique_dataset(raw, n_total=2000, seed=42)
    test = dataset

save_json = f'./data_{dataset_name}/'
text = []


def build_prompt_hotpotqa(question: str, path: str, use_ds: bool) -> tuple[str, str, int]:
    user_content = (
        f"Use the information available under context to answer. "
        f"Give brief reasoning, then end with 'The answer is:' followed by the answer in brackets.\n\n"
        f"The answer should be short phrase, not a full sentence.\n\n"
        f"Example:\n"
        f"Question: What are the three primary colors?\n"
        f"Let's think step by step:\n"
        f"1. Primary colors are the base colors that cannot be created by mixing other colors.\n"
        f"2. In traditional color theory, these are the fundamental colors.\n"
        f"3. They are used to create all other colors.\n"
        f"The answer is: [Red, Blue, Yellow]\n\n"
        f"Now answer this question:\n"
        f"Question: {question}\n"
        f"Context: {path}\n\n"
        f"Based on the Context, please answer the given question step by step.\n"
        f"Remember: End with 'The answer is:' followed by the phrase-level answer in brackets.\n"
    )
    messages = [{"role": "user", "content": user_content}]
    chat_formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    max_new_tokens = 500
    return user_content, chat_formatted, max_new_tokens

def build_prompt_2wiki(question: str, path: str, use_ds: bool) -> tuple[str, str, int]:
    user_content = (
        f"Use the information available under context to answer. "
        f"Give brief reasoning, then end with 'The answer is:' followed by the answer in brackets.\n\n"
        f"The answer should be short phrase, not a full sentence.\n\n"
        f"Example:\n"
        f"Question: What are the three primary colors?\n"
        f"Let's think step by step:\n"
        f"1. Primary colors are the base colors that cannot be created by mixing other colors.\n"
        f"2. In traditional color theory, these are the fundamental colors.\n"
        f"3. They are used to create all other colors.\n"
        f"The answer is: [Red, Blue, Yellow]\n\n"
        f"Now answer this question:\n"
        f"Question: {question}\n"
        f"Context: {path}\n\n"
        f"Based on the Context, please answer the given question step by step.\n"
        f"Remember: End with 'The answer is:' followed by the phrase-level answer in brackets.\n"
    )
    messages = [{"role": "user", "content": user_content}]
    chat_formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    max_new_tokens = 500
    return user_content, chat_formatted, max_new_tokens

def build_prompt_musique(question: str, path: str, use_ds: bool) -> tuple[str, str, int]:
    user_content = (
        f"Use the information available under context to answer. "
        f"Give brief reasoning, then end with 'The answer is:' followed by the answer in brackets.\n\n"
        f"The answer should be short phrase, not a full sentence.\n\n"
        f"Example:\n"
        f"Question: What are the three primary colors?\n"
        f"Let's think step by step:\n"
        f"1. Primary colors are the base colors that cannot be created by mixing other colors.\n"
        f"2. In traditional color theory, these are the fundamental colors.\n"
        f"3. They are used to create all other colors.\n"
        f"The answer is: [Red, Blue, Yellow]\n\n"
        f"Now answer this question:\n"
        f"Question: {question}\n"
        f"Context: {path}\n\n"
        f"Based on the Context, please answer the given question step by step.\n"
        f"Remember: End with 'The answer is:' followed by the phrase-level answer in brackets.\n"
    )
    messages = [{"role": "user", "content": user_content}]
    chat_formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    max_new_tokens = 10
    return user_content, chat_formatted, max_new_tokens


def generate_hotpotqa(data, use_ds: bool = False):
    question = data["question"]
    path = data["context"]
    gold_ans = data["answer"]
    user_content, prompt, max_tok = build_prompt_hotpotqa(question, path, use_ds)
    inputs = model.to_tokens(prompt)
    max_input_len = model.cfg.n_ctx - max_tok
    if inputs.shape[1] > max_input_len:
        inputs = inputs[:, :max_input_len]
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_tok, temperature=0.0, do_sample=False)
    generated_text = model.to_string(outputs[0, inputs.shape[1]:]).replace("<|eot_id|>", "").strip()
    print("generated_text:", generated_text)
    print("gold_ans:", gold_ans)
    return generated_text, question, prompt, gold_ans, max_tok, path

def generate_2wiki(data, use_ds: bool = False):
    question = data["question"]
    path = data["context"]
    gold_ans = data["answer"]
    user_content, prompt, max_tok = build_prompt_2wiki(question, path, use_ds)
    inputs = model.to_tokens(prompt)
    max_input_len = model.cfg.n_ctx - max_tok
    if inputs.shape[1] > max_input_len:
        inputs = inputs[:, :max_input_len]
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_tok, temperature=0.0, do_sample=False)
    generated_text = model.to_string(outputs[0, inputs.shape[1]:]).replace("<|eot_id|>", "").strip()
    print("generated_text:", generated_text)
    print("gold_ans:", gold_ans)
    return generated_text, question, prompt, gold_ans, max_tok, path

def _select_5_paragraphs(paragraphs, k: int = 5, no_true_prob: float = 0.3, seed: int | None = None):
    if seed is not None:
        random.seed(seed)
    paragraphs_sorted = sorted(paragraphs, key=lambda x: x.get("idx", 10**9))
    supporting = [p for p in paragraphs_sorted if p.get("is_supporting", False)]
    nonsupport = [p for p in paragraphs_sorted if not p.get("is_supporting", False)]
    if random.random() < no_true_prob:
        if len(nonsupport) >= k:
            selected = nonsupport[:k]
        else:
            selected = nonsupport + supporting[: (k - len(nonsupport))]
    else:
        selected = supporting[:k]
        if len(selected) < k:
            selected += nonsupport[: (k - len(selected))]
    random.shuffle(selected)
    return [
        {"idx": p.get("idx"), "title": p.get("title", ""), "paragraph_text": p.get("paragraph_text", "")}
        for p in selected
    ]

def generate_musique(data, idx, use_ds: bool = False):
    question = data["question"]
    path = _select_5_paragraphs(data["paragraphs"], k=5, no_true_prob=0.3, seed=42 + idx)
    gold_ans = data["answer"]
    user_content, prompt, max_tok = build_prompt_musique(question, path, use_ds)
    inputs = model.to_tokens(prompt)
    max_input_len = model.cfg.n_ctx - max_tok
    if max_input_len < 1:
        raise ValueError(f"max_tok={max_tok} is too large for n_ctx={model.cfg.n_ctx}")
    if inputs.shape[1] > max_input_len:
        inputs = inputs[:, -max_input_len:]
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_tok, temperature=0.0, do_sample=False)
    generated_text = model.to_string(outputs[0, inputs.shape[1]:]).replace("<|eot_id|>", "").strip()
    print("generated_text:", generated_text)
    print("gold_ans:", gold_ans)
    return generated_text, question, prompt, gold_ans, max_tok, path

def generate_manu_musique(data, use_ds: bool = False):
    question = data["question"]
    path = data["path"]
    gold_ans = data["answer"]
    mau_type = data["mau_type"]
    user_content, prompt, max_tok = build_prompt_musique(question, path, use_ds)
    inputs = model.to_tokens(prompt)
    max_input_len = model.cfg.n_ctx - max_tok
    if inputs.shape[1] > max_input_len:
        inputs = inputs[:, -max_input_len:]
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_tok, temperature=0.0, do_sample=False)
    generated_text = model.to_string(outputs[0, inputs.shape[1]:]).replace("<|eot_id|>", "").strip()
    return generated_text, question, prompt, gold_ans, max_tok, path, mau_type


for idx, data in tqdm(enumerate(test), total=len(test), desc="Testing", unit="batch"):
    print(idx)
    if dataset_name == 'hotpotqa':
        generated_text, question, prompt, ans, max_token, path = generate_hotpotqa(data)
    elif dataset_name == '2wiki':
        generated_text, question, prompt, ans, max_token, path = generate_2wiki(data)
    elif dataset_name == 'musique':
        generated_text, question, prompt, ans, max_token, path = generate_musique(data, idx)
    elif dataset_name == 'manu_musique':
        generated_text, question, prompt, ans, max_token, path, mau_type = generate_manu_musique(data)

    if dataset_name == 'manu_musique':
        text.append({
            'question': question,
            'path': path,
            'gold_ans': ans,
            'ans': generated_text,
            'type': mau_type,
        })
    else:
        text.append({'question': question, 'path': path, 'gold_ans': ans, 'ans': generated_text})

if not os.path.exists(save_json):
    os.makedirs(save_json)
with open(save_json + f'{name}_answer.json', 'w') as f:
    json.dump(text, f)
