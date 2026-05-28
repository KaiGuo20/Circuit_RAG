from dataclasses import dataclass
import torch
import os
import sys
from abc import ABC
from typing import Optional
import json
import yaml

@dataclass
class Configs(ABC):
    target_layer: int
    out_hook_point_layer: int
    # max_layer: int
    d_in: int
    epoch: int
    d_out: int
    d_transcoder: int  # Dimension of the transcoder output
    context_size: int
    hook_point_head_index: Optional[int]
    
    
    train_batch_size: int
    dataset_path: str
    is_dataset_tokenized: bool
    
    hook_point: str
    out_hook_point: str
    is_transcoder: bool
    is_sae: bool
    use_cached_activations: bool
    cached_activations_path: str
    
    n_batches_in_buffer: int
    total_training_tokens: int
    store_batch_size: int
    
    act_store_device: str
    device: str
    seed: int
    # dtype: torch.dtype # = torch.float32
    b_dec_init_method: str
    l1_coefficient: float
    
    model_name: str
    dataset_name: str
    pattern_name: str
    method: str
    max_length: int  # Default value, can be overridden
    activate_func:str
    
    batch_size: int
    lr: float
    dead_feature_window: int
    dead_feature_estimation_method: str
    dead_feature_threshold: float
    resample_batches: int
    is_sparse_connection: bool
    checkpoint_path: str
    lr_scheduler_name: str
    lr_warm_up_steps: int
    # max_layer: int
    from_pretrained: bool
    tokenizer_name: Optional[str] = None
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)
    def load_json_file(self, path: str):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return Configs(**config_dict)