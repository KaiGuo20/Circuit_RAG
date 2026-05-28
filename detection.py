import copy
import os
import pickle as pkl
import random
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, TransformerConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax


@dataclass(frozen=True)
class Config:
    name: str = "llama3_8b"
    dataset: str = "musique"
    edge_ratio: float = 0.99
    node_ratio: float = 0.99
    l1_co: float = 0.0005
    graph_root: str = 'yours'

    sample_size: int = 500
    train_groups_per_class: int = 250
    valid_groups_per_class: int = 50
    seed: int = 42

    batch_train: int = 32
    batch_eval: int = 64

    max_epochs: int = 250
    patience: int = 30
    min_delta: float = 1e-4
    label_smoothing: float = 0.01
    grad_clip: float = 1.0

    hidden: int = 128
    num_layers: int = 2
    heads: int = 8
    dropout: float = 0.10
    lr: float = 8e-4
    weight_decay: float = 1e-4
    metrics: Tuple[str, ...] = (
        "dag_longest_path_len",
        "avg_degree",
        "triad_003_ratio",
        "density",
        "triad_021U_ratio",
        "max_pagerank",
    )


CFG = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NODE_TYPES = [
    "Question",
    "Answer",
    "Context",
    "ContentEmb",
    "Content",
    "QuestionWord",
    "Punct",
    "Special",
    "Other",
]
TYPE2ID = {node_type: idx for idx, node_type in enumerate(NODE_TYPES)}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(CFG.seed)


def natural_key(text: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def list_graph_files(kind: str) -> List[str]:
    folder = (
        f"minus50_{kind}_graph_{CFG.name}_{CFG.dataset}_"
        f"{CFG.edge_ratio}_{CFG.node_ratio}_{CFG.l1_co}"
    )
    path = os.path.join(CFG.graph_root, folder)
    files = [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith(".pkl")]
    files.sort(key=lambda fp: natural_key(os.path.basename(fp)))
    return files


def ensure_graph(obj):
    if isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        return obj
    if isinstance(obj, list) and obj:
        return ensure_graph(obj[0])
    if isinstance(obj, dict) and obj:
        return ensure_graph(next(iter(obj.values())))
    raise TypeError(f"Unsupported pickle payload type: {type(obj)}")


def load_graphs(kind: str) -> List[nx.Graph]:
    graphs = []
    for file_path in list_graph_files(kind):
        with open(file_path, "rb") as handle:
            payload = pkl.load(handle)
        graphs.append(ensure_graph(payload))
    return graphs


def parse_node_type(node_name: str) -> str:
    name = str(node_name)
    lower_name = name.lower()
    if "question" in lower_name:
        return "Question"
    if "answer" in lower_name:
        return "Answer"
    if "context" in lower_name:
        return "Context"
    if "<|" in name:
        return "Special"

    question_words = {"what", "who", "where", "when", "which", "how", "why", "do", "does", "is", "are"}
    first_part = name.split("_")[0].strip().lower()
    if first_part in question_words:
        return "QuestionWord"
    if first_part in {":", "?", ".", ",", "(", ")"}:
        return "Punct"
    return "ContentEmb" if "_Emb_" in name else "Content"


def onehot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


def fit_g_stats(g_feats_raw: Sequence[np.ndarray], train_row_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g_train = np.array([g_feats_raw[idx] for idx in train_row_indices], dtype=np.float32)
    mean_fill = np.nanmean(g_train, axis=0)
    mean_fill = np.where(np.isnan(mean_fill), 0.0, mean_fill)

    g_train_filled = np.where(np.isnan(g_train), mean_fill[None, :], g_train)
    mean = g_train_filled.mean(axis=0)
    std = g_train_filled.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean_fill, mean, std


def transform_g_feats(g_feats_raw: Sequence[np.ndarray], mean_fill: np.ndarray, mean: np.ndarray, std: np.ndarray) -> List[np.ndarray]:
    transformed = []
    for g_feat in g_feats_raw:
        cur = np.array(g_feat, dtype=np.float32)
        cur = np.where(np.isnan(cur), mean_fill, cur)
        cur = (cur - mean) / std
        transformed.append(cur.astype(np.float32))
    return transformed


def get_metric_columns(dfm: pd.DataFrame) -> List[str]:
    return list(CFG.metrics)


def to_pyg_data(graph: nx.Graph, label: int, g_feat: np.ndarray, sample_id: int) -> Data:
    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    if graph.is_directed():
        indeg = dict(graph.in_degree())
        outdeg = dict(graph.out_degree())
        deg = {node: indeg.get(node, 0) + outdeg.get(node, 0) for node in nodes}
    else:
        deg = dict(graph.degree())
        indeg = deg
        outdeg = deg

    try:
        pagerank = nx.pagerank(graph, alpha=0.85)
    except Exception:
        pagerank = {node: 0.0 for node in nodes}

    max_deg = max(deg.values()) if deg else 1.0
    max_pr = max(pagerank.values()) if pagerank else 1.0

    node_features = []
    for node in nodes:
        node_type = parse_node_type(node)
        type_id = TYPE2ID.get(node_type, TYPE2ID["Other"])
        feat = np.concatenate(
            [
                onehot(type_id, len(NODE_TYPES)),
                np.array(
                    [
                        float(indeg.get(node, 0)) / (max_deg + 1e-6),
                        float(outdeg.get(node, 0)) / (max_deg + 1e-6),
                        float(deg.get(node, 0)) / (max_deg + 1e-6),
                        float(pagerank.get(node, 0.0)) / (max_pr + 1e-6),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        node_features.append(feat)

    x = torch.tensor(np.vstack(node_features), dtype=torch.float)

    edge_index = []
    edge_attr = []

    def squash(value: float) -> float:
        return float(np.tanh(value))

    def get_weight(data: dict, key: str, default: float = 0.0) -> float:
        try:
            return float(data.get(key, default))
        except Exception:
            return float(default)

    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        iterator = graph.edges(keys=True, data=True)
        for src, dst, _, data in iterator:
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]
            attr = np.array([squash(get_weight(data, "pos_weight", 0.0))], dtype=np.float32)
            edge_index.append([src_idx, dst_idx])
            edge_attr.append(attr)
            if not graph.is_directed():
                edge_index.append([dst_idx, src_idx])
                edge_attr.append(attr)
    else:
        iterator = graph.edges(data=True)
        for src, dst, data in iterator:
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]
            attr = np.array([squash(get_weight(data, "pos_weight", 0.0))], dtype=np.float32)
            edge_index.append([src_idx, dst_idx])
            edge_attr.append(attr)
            if not graph.is_directed():
                edge_index.append([dst_idx, src_idx])
                edge_attr.append(attr)

    if edge_index:
        edge_index_tensor = torch.tensor(np.array(edge_index).T, dtype=torch.long)
        edge_attr_tensor = torch.tensor(np.vstack(edge_attr), dtype=torch.float)
    else:
        edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.zeros((0, 1), dtype=torch.float)

    g_tensor = torch.tensor(np.asarray(g_feat, dtype=np.float32).reshape(1, -1), dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor,
        y=torch.tensor([label], dtype=torch.long),
        g=g_tensor,
        sample_id=torch.tensor([sample_id], dtype=torch.long),
    )


class GraphTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        edge_dim: int = 4,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.10,
        g_dim: int = 6,
    ):
        super().__init__()
        self.dropout = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.local_convs = nn.ModuleList()
        self.global_convs = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            self.local_convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms1.append(nn.LayerNorm(hidden))
            self.global_convs.append(
                TransformerConv(
                    hidden,
                    hidden // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms2.append(nn.LayerNorm(hidden))

        self.g_proj = nn.Sequential(
            nn.Linear(g_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.node_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 5, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.input_proj(x)

        for local_conv, global_conv, norm1, norm2 in zip(
            self.local_convs, self.global_convs, self.norms1, self.norms2
        ):
            x_local = local_conv(x, edge_index, edge_attr)
            x = norm1(x + F.dropout(x_local, p=self.dropout, training=self.training))

            x_global = global_conv(x, edge_index, edge_attr)
            x = norm2(x + F.dropout(x_global, p=self.dropout, training=self.training))

        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_max = global_max_pool(x, batch)
        attn_score = self.node_attn(x).squeeze(-1)
        attn_weight = pyg_softmax(attn_score, batch)
        x_att = global_add_pool(x * attn_weight.unsqueeze(-1), batch)

        x_graph = torch.cat([x_mean, x_sum, x_max, x_att], dim=-1)
        g_emb = self.g_proj(data.g)
        x_all = torch.cat([x_graph, g_emb], dim=-1)
        return self.head(x_all)


def build_group_pairs(dfm: pd.DataFrame) -> pd.DataFrame:
    metric_cols = get_metric_columns(dfm)
    needed = {"sample_id", "label", "graph_id", "file_path"}.union(metric_cols)
    missing = sorted(needed.difference(dfm.columns))
    if missing:
        raise ValueError(f"Missing columns in metrics csv: {missing}")

    sampled_frames = []
    rng = np.random.default_rng(CFG.seed)
    for label in (0, 1):
        cur = dfm[dfm["label"] == label].reset_index(drop=True)
        take = min(CFG.sample_size, len(cur))
        selected = rng.choice(len(cur), size=take, replace=False)
        sampled_frames.append(cur.iloc[selected])

    sampled = pd.concat(sampled_frames, ignore_index=True).reset_index(drop=True)

    pair_counts = sampled.groupby("sample_id")["label"].nunique()
    valid_sample_ids = pair_counts[pair_counts == 2].index
    paired = sampled[sampled["sample_id"].isin(valid_sample_ids)].copy()
    if paired.empty:
        raise ValueError("No paired sample_id with both labels found after sampling.")

    # Keep exactly one row per (sample_id, label), using deterministic order.
    paired = (
        paired.sort_values(["sample_id", "label", "graph_id", "file_path"])
        .drop_duplicates(subset=["sample_id", "label"], keep="first")
        .reset_index(drop=True)
    )

    counts = paired.groupby("sample_id")["label"].nunique()
    bad_ids = counts[counts != 2]
    if not bad_ids.empty:
        raise ValueError(f"Some sample_id are still unpaired: {bad_ids.index[:10].tolist()}")

    return paired


def make_group_split(sample_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_ids = np.array(sorted(set(int(sample_id) for sample_id in sample_ids)), dtype=int)
    rng = np.random.default_rng(CFG.seed)

    n_train = min(CFG.train_groups_per_class, len(unique_ids))
    train_full_ids = rng.choice(unique_ids, size=n_train, replace=False)

    n_valid = min(CFG.valid_groups_per_class, len(train_full_ids))
    valid_ids = rng.choice(train_full_ids, size=n_valid, replace=False)

    train_group_set = set(int(val) for val in train_full_ids.tolist())
    valid_group_set = set(int(val) for val in valid_ids.tolist())
    train_ids = train_group_set - valid_group_set
    test_ids = set(int(val) for val in unique_ids.tolist()) - train_group_set

    train_idx = np.array([idx for idx, sample_id in enumerate(sample_ids) if int(sample_id) in train_ids], dtype=int)
    valid_idx = np.array([idx for idx, sample_id in enumerate(sample_ids) if int(sample_id) in valid_group_set], dtype=int)
    test_idx = np.array([idx for idx, sample_id in enumerate(sample_ids) if int(sample_id) in test_ids], dtype=int)

    return train_idx, valid_idx, test_idx


def build_dataset(dfm: pd.DataFrame, wrong_all: Sequence[nx.Graph], correct_all: Sequence[nx.Graph]):
    paired = build_group_pairs(dfm)
    metric_cols = get_metric_columns(paired)
    sample_ids = paired["sample_id"].astype(int).tolist()
    labels = paired["label"].astype(int).tolist()
    graph_ids = paired["graph_id"].astype(int).tolist()

    train_idx, valid_idx, test_idx = make_group_split(sample_ids)

    raw_g = paired[metric_cols].to_numpy(dtype=np.float32)
    mean_fill, g_mean, g_std = fit_g_stats(raw_g, train_idx)
    g_feats = transform_g_feats(raw_g, mean_fill, g_mean, g_std)

    graphs = []
    for label, graph_id in zip(labels, graph_ids):
        graph = wrong_all[graph_id] if label == 0 else correct_all[graph_id]
        graphs.append(graph)

    dataset = [
        to_pyg_data(graph, label, g_feat, sample_id)
        for graph, label, g_feat, sample_id in zip(graphs, labels, g_feats, sample_ids)
    ]

    split_keys = {
        "train": {(sample_ids[idx], labels[idx]) for idx in train_idx},
        "valid": {(sample_ids[idx], labels[idx]) for idx in valid_idx},
        "test": {(sample_ids[idx], labels[idx]) for idx in test_idx},
    }
    assert not (split_keys["train"] & split_keys["valid"])
    assert not (split_keys["train"] & split_keys["test"])
    assert not (split_keys["valid"] & split_keys["test"])

    group_splits = {
        "train": {sample_ids[idx] for idx in train_idx},
        "valid": {sample_ids[idx] for idx in valid_idx},
        "test": {sample_ids[idx] for idx in test_idx},
    }
    assert not (group_splits["train"] & group_splits["valid"])
    assert not (group_splits["train"] & group_splits["test"])
    assert not (group_splits["valid"] & group_splits["test"])

    return dataset, labels, sample_ids, metric_cols, train_idx, valid_idx, test_idx


def compute_class_weights(labels: Sequence[int], train_idx: Sequence[int]) -> torch.Tensor:
    selected = [labels[idx] for idx in train_idx]
    n0 = sum(1 for label in selected if label == 0)
    n1 = sum(1 for label in selected if label == 1)
    total = len(selected)
    w0 = total / (2 * max(1, n0))
    w1 = total / (2 * max(1, n1))
    return torch.tensor([w0, w1], dtype=torch.float)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, class_weights: torch.Tensor) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(
            logits,
            batch.y.view(-1),
            weight=class_weights,
            label_smoothing=CFG.label_smoothing,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, class_weights: torch.Tensor):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_prob1 = []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y.view(-1), weight=class_weights)
        total_loss += float(loss.item()) * batch.num_graphs
        prob1 = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        all_prob1.append(prob1)
        all_labels.append(batch.y.view(-1).detach().cpu().numpy())

    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=int)
    prob1 = np.concatenate(all_prob1) if all_prob1 else np.array([], dtype=np.float32)
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, labels, prob1


def metrics_at_threshold(labels: np.ndarray, prob1: np.ndarray, threshold: float) -> Tuple[float, float, np.ndarray]:
    pred = (prob1 >= threshold).astype(int)
    acc = accuracy_score(labels, pred) if len(labels) else 0.0
    bal_acc = balanced_accuracy_score(labels, pred) if len(labels) else 0.0
    return acc, bal_acc, pred


def select_best_threshold(labels: np.ndarray, prob1: np.ndarray) -> Tuple[float, float, float]:
    thresholds = np.unique(np.clip(prob1, 0.0, 1.0))
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0
    candidates = np.unique(np.concatenate(([0.3, 0.4, 0.5, 0.6, 0.7], thresholds)))

    best_threshold = 0.5
    best_bal = -1.0
    best_acc = -1.0
    for threshold in candidates:
        acc, bal, _ = metrics_at_threshold(labels, prob1, float(threshold))
        if bal > best_bal + 1e-12 or (abs(bal - best_bal) <= 1e-12 and acc > best_acc):
            best_threshold = float(threshold)
            best_bal = bal
            best_acc = acc
    return best_threshold, best_acc, best_bal


def main() -> None:
    print("=" * 100)
    print("Grouped Detection — sample_id grouped split + independent classification")
    print("=" * 100)
    print(f"Device: {DEVICE}")

    metrics_path = (
        f"{CFG.graph_root}/graph_compare_{CFG.name}_{CFG.dataset}_"
        f"{CFG.edge_ratio}_{CFG.node_ratio}_{CFG.l1_co}/align_all_metrics.csv"
    )
    dfm = pd.read_csv(metrics_path)

    print("Loading graphs...")
    wrong_all = load_graphs("wrong33")#hotpotqa_wrong44
    correct_all = load_graphs("correct33")
    print(f"Loaded wrong graphs:   {len(wrong_all)}")
    print(f"Loaded correct graphs: {len(correct_all)}")

    dataset, labels, sample_ids, metric_cols, train_idx, valid_idx, test_idx = build_dataset(dfm, wrong_all, correct_all)

    print(f"Rows after sample_id pairing: {len(dataset)}")
    print(f"Unique sample_id: {len(set(sample_ids))}")
    print(f"Graph-level configured features: {len(metric_cols)} -> {metric_cols}")
    print(
        f"Split rows: train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)} | "
        f"split groups: train={len(set(sample_ids[i] for i in train_idx))} "
        f"valid={len(set(sample_ids[i] for i in valid_idx))} "
        f"test={len(set(sample_ids[i] for i in test_idx))}"
    )

    train_loader = DataLoader([dataset[idx] for idx in train_idx], batch_size=CFG.batch_train, shuffle=True)
    valid_loader = DataLoader([dataset[idx] for idx in valid_idx], batch_size=CFG.batch_eval, shuffle=False)
    test_loader = DataLoader([dataset[idx] for idx in test_idx], batch_size=CFG.batch_eval, shuffle=False)

    class_weights = compute_class_weights(labels, train_idx).to(DEVICE)
    print(f"Class weights (train only): [0]={class_weights[0].item():.3f}, [1]={class_weights[1].item():.3f}")

    in_dim = dataset[0].x.size(1)
    model = GraphTransformer(
        in_dim=in_dim,
        hidden=CFG.hidden,
        edge_dim=1,
        num_layers=CFG.num_layers,
        heads=CFG.heads,
        dropout=CFG.dropout,
        g_dim=len(metric_cols),
    ).to(DEVICE)

    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        threshold=1e-4,
        min_lr=1e-6,
    )

    best_state = None
    best_epoch = -1
    best_val_bal = -1.0
    best_val_loss = float("inf")
    best_threshold = 0.5
    bad_epochs = 0

    for epoch in range(1, CFG.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, class_weights)
        val_loss, val_labels, val_prob1 = collect_predictions(model, valid_loader, class_weights)
        tuned_threshold, val_acc, val_bal = select_best_threshold(val_labels, val_prob1)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_bal)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < prev_lr:
            print(f"[LR Decay] {prev_lr:.2e} -> {current_lr:.2e}")

        improved = False
        if val_bal > best_val_bal + CFG.min_delta:
            improved = True
        elif abs(val_bal - best_val_bal) <= CFG.min_delta and val_loss < best_val_loss - 1e-5:
            improved = True

        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_bal = val_bal
            best_val_loss = val_loss
            best_threshold = tuned_threshold
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} lr={current_lr:.2e} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_bal_acc={val_bal:.4f} "
                f"best_threshold={tuned_threshold:.4f} "
                f"(best_bal={best_val_bal:.4f}, bad={bad_epochs}/{CFG.patience})"
            )

        if bad_epochs >= CFG.patience:
            print(f"[EarlyStop] No validation improvement for {CFG.patience} epochs. Stop at epoch={epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"[Best] Loaded checkpoint from epoch={best_epoch} with val_bal_acc={best_val_bal:.4f}, "
            f"val_loss={best_val_loss:.4f}, threshold={best_threshold:.4f}"
        )
    else:
        print("[Warn] No best checkpoint found; using current model state.")

    test_loss, test_labels, test_prob1 = collect_predictions(model, test_loader, class_weights)
    test_acc, test_bal, test_pred = metrics_at_threshold(test_labels, test_prob1, best_threshold)

    print("\n" + "=" * 100)
    print("FINAL RESULT — Grouped split by sample_id, 6 graph features, pos_weight edge, independent classification")
    print("=" * 100)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Decision Threshold: {best_threshold:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Balanced Accuracy: {test_bal:.4f}")
    print(classification_report(test_labels, test_pred, digits=4))


if __name__ == "__main__":
    main()
