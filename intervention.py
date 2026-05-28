import os, re, json, pickle as pkl
import networkx as nx

# ========================================
# MIX type: attention intervention experiment
# Perturb attention to steer toward the question / away from distracting context,
# then compare baseline vs. controlled generation.
# ========================================

print("="*80)
print("MIX type: attention intervention experiment")
print("="*80)

# ==================== Configuration ====================
CONFIG = {
    'name': 'llama3_8b',
    'dataset': 'manu_musique',
    'edge_ratio': 0.99,
    'node_ratio': 0.99,
    'l1_co': 0.0005,
    'n_layers': 32,
}

# ==================== Load Data ====================
def load_graphs_from_path(path):
    graphs = {}
    for fname in os.listdir(path):
        if fname.endswith(".pkl"):
            idx = int(fname.split('.')[0])
            with open(os.path.join(path, fname), "rb") as f:
                obj = pkl.load(f)
            G = obj[0] if isinstance(obj, list) else obj
            if isinstance(G, (nx.Graph, nx.DiGraph)):
                graphs[idx] = G
    return graphs

# Load JSON
json_path = f'./data_{CONFIG["dataset"]}/{CONFIG["name"]}_answer.json'
with open(json_path, 'r') as f:
    test_data = json.load(f)
print(f"Loaded {len(test_data)} samples")

# Build idx -> type mapping
idx_to_type = {}
for i, item in enumerate(test_data):
    t = item.get('type', 'unknown')
    idx_to_type[i] = t

# Load Graphs
base_path = 'yours'
wrong_graphs = load_graphs_from_path(f'{base_path}/minus50_wrong22_graph_{CONFIG["name"]}_{CONFIG["dataset"]}_{CONFIG["edge_ratio"]}_{CONFIG["node_ratio"]}_{CONFIG["l1_co"]}/')
correct_graphs = load_graphs_from_path(f'{base_path}/minus50_correct22_graph_{CONFIG["name"]}_{CONFIG["dataset"]}_{CONFIG["edge_ratio"]}_{CONFIG["node_ratio"]}_{CONFIG["l1_co"]}/')

# Filter MIX type
mix_wrong_graphs = {idx: G for idx, G in wrong_graphs.items() if idx_to_type.get(idx) == 'mix'}
mix_correct_graphs = {idx: G for idx, G in correct_graphs.items() if idx_to_type.get(idx) == 'mix'}

print(f"MIX Wrong graphs: {len(mix_wrong_graphs)}")
print(f"MIX Correct graphs: {len(mix_correct_graphs)}")

import torch
import re
import json
import time
from typing import List, Dict, Callable, Tuple, Optional
from tqdm import tqdm
from datetime import datetime

# ============================================================
# Attention Intervention Hook class
# ============================================================

class AttentionInterventionHook:
    """
    Attention control Hook - for TransformerLens model

    Control strategy:
    - Low layers (0-9): boost Q->Q, suppress Context->Answer
    - High layers (10-31): boost Q->Answer
    """

    def __init__(
        self,
        tokenizer,
        q_q_boost: float = 1.25,      # Q->Q boost factor (low layers)
        ctx_suppress: float = 0.85,    # Context->Answer suppression factor (low layers)
        q_ans_boost: float = 1.15,    # Q->Answer boost factor (high layers)
        low_layer_end: int = 10,      # end of low layers (0-9)
        ctx_layer_end: int = 8,       # end of context suppression layers (0-7)
    ):
        self.tokenizer = tokenizer
        self.q_q_boost = q_q_boost
        self.ctx_suppress = ctx_suppress
        self.q_ans_boost = q_ans_boost
        self.low_layer_end = low_layer_end
        self.ctx_layer_end = ctx_layer_end
        
        # Token region positions
        self.question_pos: List[int] = []
        self.context_pos: List[int] = []
        self.answer_start: int = 0

    def set_regions(self, prompt_text: str, input_ids: torch.Tensor):
        """Set token region positions."""
        encoding = self.tokenizer(prompt_text, return_offsets_mapping=True)
        offsets = encoding.get('offset_mapping', [])

        # Identify region boundaries
        q_start = prompt_text.find("Question:")
        q_end = self._find_end(prompt_text, q_start, ["Context:", "Based on", "\n\n"], 300)

        ctx_start = self._find_start(prompt_text, ["Context:", "Context paths:"])
        ctx_end = self._find_end(prompt_text, ctx_start, ["Based on the", "Please answer", "\n\nNow"], len(prompt_text))

        # Assign tokens to regions
        self.question_pos = []
        self.context_pos = []
        
        for i, offset in enumerate(offsets):
            if offset is None or offset == (0, 0):
                continue
            char_start, _ = offset
            if q_start != -1 and q_start <= char_start < q_end:
                self.question_pos.append(i)
            elif ctx_start != -1 and ctx_start <= char_start < ctx_end:
                self.context_pos.append(i)
        
        self.answer_start = input_ids.shape[1]
        return self
    
    def _find_start(self, text: str, markers: List[str]) -> int:
        for m in markers:
            pos = text.find(m)
            if pos != -1:
                return pos
        return -1
    
    def _find_end(self, text: str, start: int, markers: List[str], default_offset: int) -> int:
        if start == -1:
            return 0
        for m in markers:
            pos = text.find(m, start + 10)
            if pos != -1:
                return pos
        return min(start + default_offset, len(text))
    
    def create_layer_hook(self, layer_idx: int) -> Callable:
        """Create attention hook for a single layer."""
        def hook_fn(attn_pattern: torch.Tensor, hook) -> torch.Tensor:
            modified = attn_pattern.clone()
            query_len = modified.shape[-2]
            key_len = modified.shape[-1]

            if query_len == 1:
                # Generating new token
                if layer_idx < self.low_layer_end:
                    for j in self.question_pos:
                        if j < key_len:
                            modified[:, :, 0, j] *= self.q_q_boost
                
                if layer_idx < self.ctx_layer_end:
                    for j in self.context_pos:
                        if j < key_len:
                            modified[:, :, 0, j] *= self.ctx_suppress
                
                if layer_idx >= self.low_layer_end:
                    for j in self.question_pos:
                        if j < key_len:
                            modified[:, :, 0, j] *= self.q_ans_boost
            else:
                # First forward pass
                if layer_idx < self.low_layer_end and len(self.question_pos) >= 2:
                    for i in self.question_pos:
                        for j in self.question_pos:
                            if i < query_len and j < key_len:
                                modified[:, :, i, j] *= self.q_q_boost
            
            modified = modified / (modified.sum(dim=-1, keepdim=True) + 1e-9)
            return modified
        
        return hook_fn
    
    def get_hooks(self, n_layers: int = 32) -> List[Tuple[str, Callable]]:
        """Get hooks for all layers."""
        return [(f"blocks.{i}.attn.hook_pattern", self.create_layer_hook(i)) for i in range(n_layers)]

# ============================================================
# Generation functions
# ============================================================

def build_prompt_musique(question: str, path: str, tokenizer) -> str:
    """Build a musique-format prompt."""
    user_content = (
        f"Use the information available under context to answer. "
        f"Give brief reasoning, then end with 'The answer is:' followed by the answer in brackets.\n\n"
        f"The answer should be short phrase, not a full sentence.\n\n"
        f"Question: {question}\n"
        f"Context: {path}\n\n"
        f"Based on the Context, please answer the given question step by step.\n"
        f"Remember: End with 'The answer is:' followed by the phrase-level answer in brackets.\n"
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_with_intervention(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 500,
    hook: Optional[AttentionInterventionHook] = None
) -> str:
    """Generate with attention control."""
    input_ids = model.to_tokens(prompt)
    
    with torch.no_grad():
        if hook is not None:
            hook.set_regions(prompt, input_ids)
            hooks = hook.get_hooks(n_layers=model.cfg.n_layers)
            with model.hooks(fwd_hooks=hooks):
                outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.0)
        else:
            outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.0)
    
    generated = model.to_string(outputs[0, input_ids.shape[1]:])
    return generated.replace("<|eot_id|>", "").strip()

def simple_match(generated: str, gold: str) -> bool:
    """Simple answer matching (fallback)."""
    gen_lower = generated.lower()
    gold_lower = gold.lower()
    match = re.search(r'\[([^\]]+)\]', generated)
    if match:
        gen_answer = match.group(1).lower()
        return gold_lower in gen_answer or gen_answer in gold_lower
    return gold_lower in gen_lower

# ============================================================
# Gemini judge functions
# ============================================================

from google import genai

GEMINI_API = 'yours'
gemini_client = genai.Client(api_key=GEMINI_API)

def gemini_judge(question: str, gold_ans: str, pred_ans: str, max_retries: int = 3) -> bool:
    """Use Gemini API to judge whether the predicted answer is correct."""
    prompt = (
        f"Given the question: {question}\n"
        f"Ground truth answer: {gold_ans}\n"
        f"Predicted answer: {pred_ans}\n\n"
        f"Is the prediction COVERS the gold answer? Only select Yes or No."
    )
    
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )
            judge = response.text.strip().lower()
            return judge in ['yes', 'yes.', 'yes!']
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"Gemini API error: {e}")
                return simple_match(pred_ans, gold_ans)

# ============================================================
# Load model
# ============================================================

print("=" * 80)
print("Loading model")
print("=" * 80)

try:
    model
    print("Model already loaded")
except NameError:
    print("Loading model (using local cache)...")
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    # Local path and TransformerLens model name
    local_model_path = 'yours'
    backbone_model = "meta-llama/Llama-3.1-8B-Instruct"

    # Set HF cache directory
    cache_root = 'yours'
    os.environ["HF_HOME"] = cache_root
    os.environ["TRANSFORMERS_CACHE"] = cache_root

    print(f"  Loading from local: {local_model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True, trust_remote_code=True, local_files_only=True)

    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
    )

    # Convert to HookedTransformer
    model = HookedTransformer.from_pretrained(
        backbone_model,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device='cuda:0',
        dtype=torch.float16,
        trust_remote_code=True,
    )

    # Set chat template
    LLAMA3_CHAT_TEMPLATE = (
        "<|begin_of_text|>"
        "{% for message in messages %}"
        "{% set role = message['role'] %}"
        "{% if role == 'system' %}<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
        "{% elif role == 'user' %}<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
        "{% elif role == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
        "{% else %}<|start_header_id|>{{ role }}<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n{% endif %}"
    )
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    print("Model loading complete")

# ============================================================
# Run experiment
# ============================================================

print("=" * 80)
print("Attention control experiment: all MIX samples")
print("=" * 80)

# Filter all MIX type samples
mix_all_samples = [item for item in test_data if item.get('type') == 'mix']
print(f"\nMIX total samples: {len(mix_all_samples)}")

# Get the index set of original mix_wrong (to distinguish original wrong/correct)
mix_wrong_indices_set = set(mix_wrong_graphs.keys())

# Select number of test samples
N_TEST = min(1000, len(mix_all_samples))  # at most 200
test_samples = mix_all_samples[:N_TEST]
print(f"Test samples: {N_TEST}")

# Record whether each sample was originally wrong or correct
original_labels = []
for item in test_samples:
    idx = test_data.index(item)
    original_labels.append('wrong' if idx in mix_wrong_indices_set else 'correct')

n_orig_wrong = sum(1 for l in original_labels if l == 'wrong')
n_orig_correct = sum(1 for l in original_labels if l == 'correct')
print(f"  - Original MIX Wrong: {n_orig_wrong}")
print(f"  - Original MIX Correct: {n_orig_correct}")

# Create attention hook (tunable parameters)
hook = AttentionInterventionHook(
    tokenizer,
    q_q_boost=1.5,        # boost Q->Q (low layers)
    ctx_suppress=0.5,     # suppress Context->Answer (low layers)
    q_ans_boost=1.5,      # boost Q->Answer (high layers)
    low_layer_end=10,     # low layers: 0-9
    ctx_layer_end=8       # context suppression layers: 0-7
)

print(f"\nControl parameters:")
print(f"  Q->Q boost (layers 0-9): {hook.q_q_boost}x")
print(f"  Context suppression (layers 0-7): {hook.ctx_suppress}x")
print(f"  Q->Answer boost (layers 10-31): {hook.q_ans_boost}x")

# Run experiment
results_baseline = []
results_controlled = []

print(f"\nStarting test...")
for idx, item in enumerate(tqdm(test_samples, desc="Testing")):
    question = item["question"]
    path = item["path"][:8192]  # truncate overly long context
    gold = item["gold_ans"]

    prompt = build_prompt_musique(question, path, tokenizer)

    # Baseline (no control)
    gen_base = generate_with_intervention(model, tokenizer, prompt, hook=None)
    results_baseline.append({"gen": gen_base, "gold": gold, "q": question})

    # Controlled (with control)
    gen_ctrl = generate_with_intervention(model, tokenizer, prompt, hook=hook)
    results_controlled.append({"gen": gen_ctrl, "gold": gold, "q": question})

# Judge all results with Gemini
print(f"\nJudging answers with Gemini...")
for i, (base, ctrl) in enumerate(tqdm(zip(results_baseline, results_controlled), total=len(results_baseline), desc="Judging")):
    base['correct'] = gemini_judge(base['q'], base['gold'], base['gen'])
    ctrl['correct'] = gemini_judge(ctrl['q'], ctrl['gold'], ctrl['gen'])
    time.sleep(0.1)  # avoid API rate limiting

# Aggregate results
n_tested = len(results_baseline)
base_acc = sum(1 for r in results_baseline if r['correct'])
ctrl_acc = sum(1 for r in results_controlled if r['correct'])

print("\n" + "=" * 80)
print("Experiment results (Gemini judge) - all MIX samples")
print("=" * 80)
print(f"Baseline (no control): {base_acc}/{n_tested} ({base_acc/n_tested*100:.1f}%)")
print(f"Controlled (with control): {ctrl_acc}/{n_tested} ({ctrl_acc/n_tested*100:.1f}%)")
print(f"Improvement: {ctrl_acc - base_acc:+d} samples ({(ctrl_acc-base_acc)/n_tested*100:+.1f}%)")

# Breakdown statistics
wrong_to_correct = sum(1 for b, c in zip(results_baseline, results_controlled) if not b['correct'] and c['correct'])
correct_to_wrong = sum(1 for b, c in zip(results_baseline, results_controlled) if b['correct'] and not c['correct'])
both_correct = sum(1 for b, c in zip(results_baseline, results_controlled) if b['correct'] and c['correct'])
both_wrong = sum(1 for b, c in zip(results_baseline, results_controlled) if not b['correct'] and not c['correct'])

print(f"\nDetailed breakdown:")
print(f"  correct->correct (both right): {both_correct}")
print(f"  wrong->wrong (both wrong): {both_wrong}")
print(f"  wrong->correct (improved): {wrong_to_correct}")
print(f"  correct->wrong (degraded): {correct_to_wrong}")

# ============================================================
# Breakdown by original class (MIX Wrong vs MIX Correct)
# ============================================================
print("\n" + "=" * 80)
print("Detailed statistics by original class")
print("=" * 80)

# Statistics for MIX Wrong and MIX Correct separately
for orig_label in ['wrong', 'correct']:
    indices = [i for i, l in enumerate(original_labels) if l == orig_label]
    n_sub = len(indices)
    if n_sub == 0:
        continue

    sub_base_acc = sum(1 for i in indices if results_baseline[i]['correct'])
    sub_ctrl_acc = sum(1 for i in indices if results_controlled[i]['correct'])

    sub_w2c = sum(1 for i in indices if not results_baseline[i]['correct'] and results_controlled[i]['correct'])
    sub_c2w = sum(1 for i in indices if results_baseline[i]['correct'] and not results_controlled[i]['correct'])
    sub_both_c = sum(1 for i in indices if results_baseline[i]['correct'] and results_controlled[i]['correct'])
    sub_both_w = sum(1 for i in indices if not results_baseline[i]['correct'] and not results_controlled[i]['correct'])

    label_str = "Original MIX WRONG" if orig_label == 'wrong' else "Original MIX CORRECT"
    print(f"\n{label_str} ({n_sub} samples):")
    print(f"   Baseline: {sub_base_acc}/{n_sub} ({sub_base_acc/n_sub*100:.1f}%)")
    print(f"   Controlled: {sub_ctrl_acc}/{n_sub} ({sub_ctrl_acc/n_sub*100:.1f}%)")
    print(f"   Improvement: {sub_ctrl_acc - sub_base_acc:+d} ({(sub_ctrl_acc-sub_base_acc)/n_sub*100:+.1f}%)")
    print(f"   correct->correct: {sub_both_c}  |  wrong->wrong: {sub_both_w}  |  wrong->correct: {sub_w2c}  |  correct->wrong: {sub_c2w}")

# Show improved examples
improved = [(i, results_baseline[i], results_controlled[i])
            for i in range(n_tested)
            if not results_baseline[i]['correct'] and results_controlled[i]['correct']]

if improved:
    print(f"\nImproved examples (Baseline wrong -> Controlled right): {len(improved)}")
    for idx, (i, base, ctrl) in enumerate(improved[:3]):
        print(f"\n--- Example {idx+1} ---")
        print(f"Question: {base['q'][:80]}...")
        print(f"Gold: {base['gold']}")
        print(f"Baseline: {base['gen'][:150]}...")
        print(f"Controlled: {ctrl['gen'][:150]}...")

# Show degraded examples
degraded = [(i, results_baseline[i], results_controlled[i])
            for i in range(n_tested)
            if results_baseline[i]['correct'] and not results_controlled[i]['correct']]
if degraded:
    print(f"\nDegraded samples: {len(degraded)}")
    for idx, (i, base, ctrl) in enumerate(degraded[:2]):
        print(f"\n--- Degraded example {idx+1} ---")
        print(f"Question: {base['q'][:80]}...")
        print(f"Gold: {base['gold']}")
        print(f"Baseline: {base['gen'][:100]}...")
        print(f"Controlled: {ctrl['gen'][:100]}...")

# ============================================================
# Save all results to file
# ============================================================
import json
from datetime import datetime

# Build complete results list
all_results = []
for i in range(n_tested):
    idx_in_test_data = test_data.index(test_samples[i])
    result = {
        "idx": i,
        "idx_in_test_data": idx_in_test_data,
        "original_label": original_labels[i],  # original label: wrong or correct
        "question": results_baseline[i]['q'],
        "gold_answer": results_baseline[i]['gold'],
        "baseline_answer": results_baseline[i]['gen'],
        "controlled_answer": results_controlled[i]['gen'],
        "baseline_correct": results_baseline[i]['correct'],
        "controlled_correct": results_controlled[i]['correct'],
        "category": "both_correct" if results_baseline[i]['correct'] and results_controlled[i]['correct']
                    else "both_wrong" if not results_baseline[i]['correct'] and not results_controlled[i]['correct']
                    else "improved" if not results_baseline[i]['correct'] and results_controlled[i]['correct']
                    else "degraded"
    }
    all_results.append(result)

# Save path
save_path = f'{base_path}/attention_intervention_results_{CONFIG["dataset"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
# Save results
save_data = {
    "config": {
        "dataset": CONFIG["dataset"],
        "model": CONFIG["name"],
        "n_tested": n_tested,
        "n_orig_wrong": n_orig_wrong,
        "n_orig_correct": n_orig_correct,
        "hook_params": {
            "q_q_boost": hook.q_q_boost,
            "ctx_suppress": hook.ctx_suppress,
            "q_ans_boost": hook.q_ans_boost,
            "low_layer_end": hook.low_layer_end,
            "ctx_layer_end": hook.ctx_layer_end,
        }
    },
    "summary": {
        "overall": {
            "baseline_accuracy": base_acc / n_tested,
            "controlled_accuracy": ctrl_acc / n_tested,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "improved": wrong_to_correct,
            "degraded": correct_to_wrong,
        },
        "by_original_label": {}
    },
    "results": all_results
}

# Add breakdown by original label
for orig_label in ['wrong', 'correct']:
    indices = [i for i, l in enumerate(original_labels) if l == orig_label]
    n_sub = len(indices)
    if n_sub == 0:
        continue
    sub_base_acc = sum(1 for i in indices if results_baseline[i]['correct'])
    sub_ctrl_acc = sum(1 for i in indices if results_controlled[i]['correct'])
    sub_w2c = sum(1 for i in indices if not results_baseline[i]['correct'] and results_controlled[i]['correct'])
    sub_c2w = sum(1 for i in indices if results_baseline[i]['correct'] and not results_controlled[i]['correct'])
    sub_both_c = sum(1 for i in indices if results_baseline[i]['correct'] and results_controlled[i]['correct'])
    sub_both_w = sum(1 for i in indices if not results_baseline[i]['correct'] and not results_controlled[i]['correct'])
    
    save_data["summary"]["by_original_label"][orig_label] = {
        "n_samples": n_sub,
        "baseline_accuracy": sub_base_acc / n_sub,
        "controlled_accuracy": sub_ctrl_acc / n_sub,
        "both_correct": sub_both_c,
        "both_wrong": sub_both_w,
        "improved": sub_w2c,
        "degraded": sub_c2w,
    }

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {save_path}")
print(f"   Contains {n_tested} records, each record includes:")
print(f"   - question, gold_answer")
print(f"   - baseline_answer, controlled_answer")
print(f"   - baseline_correct, controlled_correct, category")