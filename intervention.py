# %matplotlib inline
import os, re, json, pickle as pkl
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from transformers import AutoTokenizer
from scipy.stats import mannwhitneyu

# ========================================
# MIX type: External vs Internal pos_weight analysis
# Using pos_mean (average) for fair comparison
# ========================================

print("="*80)
print("MIX type: External vs Internal edge pos_weight analysis (using pos_mean)")
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

# Load Tokenizer
cache_root = 'yours'
hf_token = 'yours'
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=cache_root, token=hf_token)
except:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=cache_root, local_files_only=True)
print("Tokenizer loaded")

# Load Graphs
base_path = 'yours'
wrong_graphs = load_graphs_from_path(f'{base_path}/minus50_wrong22_graph_{CONFIG["name"]}_{CONFIG["dataset"]}_{CONFIG["edge_ratio"]}_{CONFIG["node_ratio"]}_{CONFIG["l1_co"]}/')
correct_graphs = load_graphs_from_path(f'{base_path}/minus50_correct22_graph_{CONFIG["name"]}_{CONFIG["dataset"]}_{CONFIG["edge_ratio"]}_{CONFIG["node_ratio"]}_{CONFIG["l1_co"]}/')

# Filter MIX type
mix_wrong_graphs = {idx: G for idx, G in wrong_graphs.items() if idx_to_type.get(idx) == 'mix'}
mix_correct_graphs = {idx: G for idx, G in correct_graphs.items() if idx_to_type.get(idx) == 'mix'}

print(f"MIX Wrong graphs: {len(mix_wrong_graphs)}")
print(f"MIX Correct graphs: {len(mix_correct_graphs)}")


# ==================== Strip Function ====================
def strip_based_on_context(text: str) -> str:
    start = text.find("Based on the context:")
    if start == -1:
        return text
    end = text.find("\n Answer:", start)
    if end == -1:
        end = len(text)
    return text[:start] + text[end:]


# ==================== Analyzer for pos_weight ====================
class MixAnalyzerPosWeight:
    def __init__(self, tokenizer, n_layers=32):
        self.tokenizer = tokenizer
        self.n_layers = n_layers
        self.stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'just', 'don', 'now', 'and', 'but', 'or', 'because', 'if',
            'that', 'this', 'these', 'those', 'it', 'its', 'he', 'she',
            'they', 'them', 'his', 'her', 'their', 'what', 'which', 'who',
        }
        
    def parse_node(self, node_name):
        parts = node_name.rsplit('_', 2)
        if len(parts) < 3:
            return {'token': node_name, 'layer': -1, 'position': -1, 'valid': False}
        try:
            return {'token': parts[0], 'layer': int(parts[1]), 'position': int(parts[2]), 'valid': True}
        except:
            return {'token': node_name, 'layer': -1, 'position': -1, 'valid': False}
    
    def extract_external_text(self, text):
        if 'Based on the context:' in text:
            ext_start = text.find('Based on the context:') + len('Based on the context:')
            ext_end = text.find('\n Answer:', ext_start)
            if ext_end == -1:
                ext_end = len(text)
            return text[ext_start:ext_end].strip()
        return ''
    
    def get_external_words(self, external_text):
        clean = re.sub(r'[^\w\s]', ' ', external_text.lower())
        words = clean.split()
        return {w for w in words if w not in self.stopwords and len(w) > 2}
    
    def extract_regions_from_stripped(self, stripped_text):
        regions = {'question_range': (-1, -1), 'answer_range': (-1, -1)}
        if 'Question:' in stripped_text:
            q_start = stripped_text.find('Question:') + len('Question:')
            q_end = stripped_text.find('\n Answer:')
            if q_end == -1:
                q_end = len(stripped_text)
            regions['question_range'] = (q_start, q_end)
        if '\n Answer:' in stripped_text:
            ans_start = stripped_text.find('\n Answer:') + len('\n Answer:')
            regions['answer_range'] = (ans_start, len(stripped_text))
        return regions
    
    def get_token_positions(self, stripped_text):
        tokens = self.tokenizer(stripped_text, return_offsets_mapping=True)
        offsets = tokens['offset_mapping']
        regions = self.extract_regions_from_stripped(stripped_text)
        
        token_regions = {'question': [], 'answer': []}
        q_start, q_end = regions['question_range']
        ans_start, ans_end = regions['answer_range']
        
        for i, (char_start, char_end) in enumerate(offsets):
            if char_start is None:
                continue
            if q_start != -1 and q_start <= char_start < q_end:
                token_regions['question'].append(i)
            elif ans_start != -1 and ans_start <= char_start:
                token_regions['answer'].append(i)
        
        return token_regions
    
    def is_external_sourced(self, token_str, external_words):
        clean_token = re.sub(r'[^\w]', '', token_str.lower())
        if len(clean_token) <= 2 or clean_token in self.stopwords:
            return False
        return clean_token in external_words
    
    def analyze_graph_pos_weight(self, G, original_text):
        """Analyze pos_weight of graph, distinguishing external/internal."""
        external_text = self.extract_external_text(original_text)
        external_words = self.get_external_words(external_text)
        stripped_text = strip_based_on_context(original_text)
        token_regions = self.get_token_positions(stripped_text)
        
        # Classify nodes
        nodes_info = {}
        for node in G.nodes():
            info = self.parse_node(node)
            if not info['valid']:
                continue
            pos = info['position']
            token_str = info['token']

            if pos in token_regions['question']:
                info['region'] = 'question'
            elif pos in token_regions['answer']:
                if self.is_external_sourced(token_str, external_words):
                    info['region'] = 'answer_ext'
                else:
                    info['region'] = 'answer_int'
            else:
                info['region'] = 'unknown'
            nodes_info[node] = info

        # Count pos_weight by edge type
        result = {
            'n_edges': G.number_of_edges(),
            'external_words_count': len(external_words),
            # pos_weight stats
            'q_to_ans_ext_pos': [], 'q_to_ans_int_pos': [],
            'ans_ext_to_ans_ext_pos': [], 'ans_ext_to_ans_int_pos': [],
            'ans_int_to_ans_ext_pos': [], 'ans_int_to_ans_int_pos': [],
            # neg_weight stats
            'q_to_ans_ext_neg': [], 'q_to_ans_int_neg': [],
            'ans_ext_to_ans_ext_neg': [], 'ans_ext_to_ans_int_neg': [],
            'ans_int_to_ans_ext_neg': [], 'ans_int_to_ans_int_neg': [],
            # node stats
            'nodes_by_region': Counter(),
        }
        
        for node, info in nodes_info.items():
            result['nodes_by_region'][info['region']] += 1
        
        for u, v, data in G.edges(data=True):
            pos_w = data.get('pos_weight', 0.0)
            neg_w = data.get('neg_weight', 0.0)
            
            u_info = nodes_info.get(u)
            v_info = nodes_info.get(v)
            if not u_info or not v_info:
                continue
            
            src_region = v_info['region']
            tgt_region = u_info['region']
            
            # Question → Answer
            if src_region == 'question':
                if tgt_region == 'answer_ext':
                    result['q_to_ans_ext_pos'].append(pos_w)
                    result['q_to_ans_ext_neg'].append(neg_w)
                elif tgt_region == 'answer_int':
                    result['q_to_ans_int_pos'].append(pos_w)
                    result['q_to_ans_int_neg'].append(neg_w)
            # Answer_ext → Answer
            elif src_region == 'answer_ext':
                if tgt_region == 'answer_ext':
                    result['ans_ext_to_ans_ext_pos'].append(pos_w)
                    result['ans_ext_to_ans_ext_neg'].append(neg_w)
                elif tgt_region == 'answer_int':
                    result['ans_ext_to_ans_int_pos'].append(pos_w)
                    result['ans_ext_to_ans_int_neg'].append(neg_w)
            # Answer_int → Answer
            elif src_region == 'answer_int':
                if tgt_region == 'answer_ext':
                    result['ans_int_to_ans_ext_pos'].append(pos_w)
                    result['ans_int_to_ans_ext_neg'].append(neg_w)
                elif tgt_region == 'answer_int':
                    result['ans_int_to_ans_int_pos'].append(pos_w)
                    result['ans_int_to_ans_int_neg'].append(neg_w)
        
        return result


analyzer = MixAnalyzerPosWeight(tokenizer, CONFIG['n_layers'])
print("Analyzer ready")


# ==================== Analyze MIX samples ====================
def analyze_mix_samples(graphs, test_data, analyzer, label):
    results = []
    for idx, G in graphs.items():
        if idx >= len(test_data):
            continue
        try:
            r = analyzer.analyze_graph_pos_weight(G, test_data[idx]['ans'])
            
            row = {
                'idx': idx,
                'label': label,
                'n_edges': r['n_edges'],
                'external_words_count': r['external_words_count'],
                'n_question': r['nodes_by_region'].get('question', 0),
                'n_ans_ext': r['nodes_by_region'].get('answer_ext', 0),
                'n_ans_int': r['nodes_by_region'].get('answer_int', 0),
            }
            
            # pos_weight statistics
            for key in ['q_to_ans_ext', 'q_to_ans_int',
                        'ans_ext_to_ans_ext', 'ans_ext_to_ans_int',
                        'ans_int_to_ans_ext', 'ans_int_to_ans_int']:
                pos_list = r[f'{key}_pos']
                neg_list = r[f'{key}_neg']
                row[f'{key}_pos_sum'] = sum(pos_list)
                row[f'{key}_pos_mean'] = np.mean(pos_list) if pos_list else 0
                row[f'{key}_pos_count'] = len([x for x in pos_list if x > 0])
                row[f'{key}_neg_sum'] = sum(neg_list)
                row[f'{key}_neg_mean'] = np.mean(neg_list) if neg_list else 0
                row[f'{key}_neg_count'] = len([x for x in neg_list if x > 0])
            
            results.append(row)
        except Exception as e:
            continue
    
    return pd.DataFrame(results)


print("\nAnalyzing MIX samples...")
wrong_df = analyze_mix_samples(mix_wrong_graphs, test_data, analyzer, 'WRONG')
correct_df = analyze_mix_samples(mix_correct_graphs, test_data, analyzer, 'CORRECT')
all_df = pd.concat([wrong_df, correct_df])

print(f"Analyzed: WRONG {len(wrong_df)}, CORRECT {len(correct_df)}")


# ==================== Cliff's Delta ====================
def cliffs_delta(x, y):
    from bisect import bisect_left, bisect_right
    x, y = np.asarray(x), np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    more = less = 0
    y_sorted = np.sort(y)
    n1, n2 = len(x), len(y)
    for val in np.sort(x):
        more += bisect_left(y_sorted, val)
        less += (n2 - bisect_right(y_sorted, val))
    return (more - less) / (n1 * n2)

def effect_label(d):
    d = abs(d)
    if d >= 0.474: return 'large'
    elif d >= 0.33: return 'medium'
    elif d >= 0.147: return 'small'
    else: return 'negligible'


# ==================== Statistical Comparison (using pos_mean) ====================
print("\n" + "="*80)
print("MIX: External vs Internal pos_mean statistical comparison (fair per-edge average weight comparison)")
print("="*80)

edge_types = ['q_to_ans_ext', 'q_to_ans_int', 
              'ans_ext_to_ans_ext', 'ans_ext_to_ans_int',
              'ans_int_to_ans_ext', 'ans_int_to_ans_int']

# Focus on pos_mean as the primary comparison metric
comparison_results = []
for edge_type in edge_types:
    for metric in ['pos_mean', 'neg_mean', 'pos_count']:  # mainly care about mean
        col = f'{edge_type}_{metric}'
        if col not in wrong_df.columns:
            continue
        
        w_vals = wrong_df[col].dropna().values
        c_vals = correct_df[col].dropna().values
        
        if len(w_vals) < 3 or len(c_vals) < 3:
            continue
        
        try:
            stat, pval = mannwhitneyu(c_vals, w_vals, alternative='two-sided')
            delta = cliffs_delta(c_vals, w_vals)
        except:
            continue
        
        comparison_results.append({
            'edge_type': edge_type,
            'metric': metric,
            'wrong_mean': np.mean(w_vals),
            'correct_mean': np.mean(c_vals),
            'diff': np.mean(c_vals) - np.mean(w_vals),
            'rel_diff_pct': (np.mean(c_vals) - np.mean(w_vals)) / (abs(np.mean(w_vals)) + 1e-9) * 100,
            'cliffs_delta': delta,
            'effect': effect_label(delta),
            'p_value': pval,
        })

comparison_df = pd.DataFrame(comparison_results).sort_values('p_value')

print("\n" + "-"*120)
print(f"{'Edge Type':<25} {'Metric':<12} {'WRONG':>12} {'CORRECT':>12} {'Diff':>12} {'Δ':>8} {'Effect':<10} {'p-value':<12} {'Direction':<10}")
print("-"*120)

for _, row in comparison_df.iterrows():
    direction = "Correct↑" if row['diff'] > 0 else "Wrong↑"
    sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
    print(f"{row['edge_type']:<25} {row['metric']:<12} {row['wrong_mean']:>12.4f} {row['correct_mean']:>12.4f} "
          f"{row['diff']:>+12.4f} {row['cliffs_delta']:>+8.3f} {row['effect']:<10} {row['p_value']:.2e} {sig:<3} {direction:<10}")


# ==================== Visualization (using pos_mean) ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('MIX Type: External vs Internal pos_mean Analysis (Wrong vs Correct)\nPer-edge average attention weight (fair comparison)',
             fontsize=14, fontweight='bold')

# pos_mean comparison for main edge types
main_edge_types = ['q_to_ans_ext', 'q_to_ans_int', 'ans_ext_to_ans_int',
                   'ans_int_to_ans_int', 'ans_ext_to_ans_ext', 'ans_int_to_ans_ext']
labels_map = {
    'q_to_ans_ext': 'Q→Ans_EXT',
    'q_to_ans_int': 'Q→Ans_INT',
    'ans_ext_to_ans_int': 'EXT→INT',
    'ans_int_to_ans_int': 'INT→INT',
    'ans_ext_to_ans_ext': 'EXT→EXT',
    'ans_int_to_ans_ext': 'INT→EXT',
}

for i, edge_type in enumerate(main_edge_types):
    ax = axes[i//3, i%3]
    col = f'{edge_type}_pos_mean'  # switch to pos_mean
    
    if col in wrong_df.columns and col in correct_df.columns:
        w_vals = wrong_df[col].dropna().values
        c_vals = correct_df[col].dropna().values
        
        bp = ax.boxplot([w_vals, c_vals], tick_labels=['Wrong', 'Correct'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][1].set_facecolor('#2ecc71')
        
        ax.set_title(f'{labels_map[edge_type]} (pos_mean)', fontsize=11, fontweight='bold')
        ax.set_ylabel('pos_weight Mean (per-edge average)')
        
        # p-value and direction
        try:
            _, pval = mannwhitneyu(c_vals, w_vals, alternative='two-sided')
            w_mean = np.mean(w_vals)
            c_mean = np.mean(c_vals)
            direction = "C↑" if c_mean > w_mean else "W↑"
            color = 'green' if (pval < 0.05 and c_mean > w_mean) else 'red' if pval < 0.05 else 'black'
            ax.text(0.5, 0.95, f'p={pval:.2e} {direction}', transform=ax.transAxes, ha='center', fontsize=9, color=color)
        except:
            pass

plt.tight_layout()
plt.savefig(f'{base_path}/graph_compare_6types_{CONFIG["name"]}_{CONFIG["dataset"]}_{CONFIG["edge_ratio"]}_{CONFIG["node_ratio"]}_{CONFIG["l1_co"]}/mix_external_internal_pos_mean.png', 
            dpi=150, bbox_inches='tight')
plt.show()


# ==================== Summary Table (pos_mean) ====================
print("\n" + "="*80)
print("Summary: External vs Internal (pos_mean - per-edge average weight)")
print("="*80)

print(f"\n{'Edge Flow':<20} {'WRONG':>12} {'CORRECT':>12} {'Diff':>12} {'Direction':<12} {'Significant':<10}")
print("-"*90)

summary_data = []
for edge_type in main_edge_types:
    col = f'{edge_type}_pos_mean'
    if col in wrong_df.columns and col in correct_df.columns:
        w_vals = wrong_df[col].dropna().values
        c_vals = correct_df[col].dropna().values
        w = np.mean(w_vals)
        c = np.mean(c_vals)
        diff = c - w
        direction = "Correct↑" if diff > 0 else "Wrong↑"
        
        try:
            _, pval = mannwhitneyu(c_vals, w_vals, alternative='two-sided')
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        except:
            pval = 1.0
            sig = ""
        
        summary_data.append({
            'edge_type': edge_type,
            'label': labels_map[edge_type],
            'wrong': w,
            'correct': c,
            'diff': diff,
            'direction': direction,
            'pval': pval,
            'sig': sig
        })
        print(f"{labels_map[edge_type]:<20} {w:>12.4f} {c:>12.4f} {diff:>+12.4f} {direction:<12} {sig:<10}")


# ==================== Key Findings ====================
print("\n" + "="*80)
print("Key Findings: MIX External vs Internal (pos_mean analysis)")
print("="*80)

# Analyze results
correct_higher = [s for s in summary_data if s['diff'] > 0 and s['pval'] < 0.05]
wrong_higher = [s for s in summary_data if s['diff'] < 0 and s['pval'] < 0.05]

print("\n[Edge types where Correct has higher per-edge average weight] (indicates more focused attention in Correct):")
if correct_higher:
    for s in correct_higher:
        print(f"  {s['label']}: Correct={s['correct']:.4f}, Wrong={s['wrong']:.4f} ({s['diff']:+.4f})")
else:
    print("  (no significant difference)")

print("\n[Edge types where Wrong has higher per-edge average weight] (indicates more focused attention on these edges in Wrong):")
if wrong_higher:
    for s in wrong_higher:
        print(f"  {s['label']}: Wrong={s['wrong']:.4f}, Correct={s['correct']:.4f} ({abs(s['diff']):+.4f})")
else:
    print("  (no significant difference)")

# Comparison with overall
print("\n" + "-"*80)
print("Comparison with overall pos_mean:")
print("-"*80)
print(f"  Overall pos_mean: Correct=8.67, Wrong=7.51 -> Correct higher (+15.5%)")

if correct_higher:
    print(f"\n  -> These edge types are consistent with the overall trend (Correct attention is stronger):")
    for s in correct_higher:
        print(f"     - {s['label']}")

if wrong_higher:
    print(f"\n  -> These edge types are contrary to the overall trend (Wrong attention is stronger):")
    for s in wrong_higher:
        print(f"     - {s['label']}")

print("\n" + "="*80)
print("Conclusion")
print("="*80)
if correct_higher and not wrong_higher:
    print("  After pos_mean analysis, all edge types show Correct higher or no significant difference")
    print("  This is consistent with the overall pos_mean trend: Correct samples have stronger per-edge attention")
elif wrong_higher:
    print(f"  {len(wrong_higher)} edge type(s) have higher pos_mean in Wrong:")
    for s in wrong_higher:
        print(f"     - {s['label']}: may indicate over-attention in Wrong samples on these connections")

# Save results
comparison_df.to_csv(f'{base_path}/graph_compare_6types_{CONFIG["name"]}_{CONFIG["dataset"]}_{CONFIG["edge_ratio"]}_{CONFIG["node_ratio"]}_{CONFIG["l1_co"]}/mix_ext_int_pos_mean_comparison_rebuttal.csv', index=False)
print(f"\nResults saved")


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