# my_replacement_model.py

from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union, Literal, NamedTuple

import torch
from torch import nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
import logging

from circuit_tracer.transcoder.single_layer_transcoder import (
    SingleLayerTranscoder,
    load_trained_transcoder,
    load_transcoder_set,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformer_lens.utils as utils


def model_name_func(name: str) -> str:
    if name == "gemma-2B":
        return "google/gemma-2-2b"
    if name == "gemma-1B":
        return "google/gemma-3-1b"
    if name == "Qwen-0.5B":
        return "Qwen/Qwen2.5-0.5B"
    if name == "Qwen3-0.6B":
        return "Qwen/Qwen3-0.6B"
    if name == "Qwen-1.5B":
        return "Qwen/Qwen2.5-1.5B"
    if name == "Llama-1B":
        return "meta-llama/Llama-3.2-1B"
    if name == "Llama-3B":
        return "meta-llama/Llama-3.2-3B"
    if name == "Llama-8B":
        return "meta-llama/Llama-3.1-8B"
    if name == "Llama-8BI":
        return "meta-llama/Llama-3.1-8B-Instruct"
    if name == "ds-qwen-1.5B":
        return "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    if name == "ds-llama":
        return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    if name == "llama3_8b_finetune":
        return "meta-llama/Llama-3.1-8B-Instruct"
    if name == "llama3_8b":
        return "meta-llama/Llama-3.1-8B-Instruct"
    return name


class Output(NamedTuple):
    logits: torch.Tensor
    loss: torch.Tensor


class ReplacementMLP(nn.Module):
    """Wrapper for a TransformerLens MLP layer that adds in extra hooks"""

    def __init__(self, old_mlp: nn.Module):
        super().__init__()
        self.old_mlp = old_mlp
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_in(x)
        mlp_out = self.old_mlp(x)
        return self.hook_out(mlp_out)


class ReplacementUnembed(nn.Module):
    """Wrapper for a TransformerLens Unembed layer that adds in extra hooks"""

    def __init__(self, old_unembed: nn.Module):
        super().__init__()
        self.old_unembed = old_unembed
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    @property
    def W_U(self):
        return self.old_unembed.W_U

    @property
    def b_U(self):
        return self.old_unembed.b_U

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_pre(x)
        x = self.old_unembed(x)
        return self.hook_post(x)


class ReplacementModel(HookedTransformer):
    d_transcoder: int
    transcoders: nn.ModuleList
    feature_input_hook: str
    feature_output_hook: str
    skip_transcoder: bool
    scan: Optional[Union[str, List[str]]]
    device_map: Optional[Dict[int, torch.device]] = None

    # ========= Construction =========

    @classmethod
    def from_config(
        cls,
        config: HookedTransformerConfig,
        transcoders: Dict[int, SingleLayerTranscoder],
        feature_input_hook: str = "mlp.hook_in",
        feature_output_hook: str = "mlp.hook_out",
        scan: Optional[str] = None,
        **kwargs,
    ) -> "ReplacementModel":
        model = cls(config, **kwargs)
        model._configure_replacement_model(
            transcoders, feature_input_hook, feature_output_hook, scan
        )
        return model

    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: Dict[int, SingleLayerTranscoder],
        feature_input_hook: str = "mlp.hook_in",
        feature_output_hook: str = "mlp.hook_out",
        scan: str = None,
        **kwargs,
    ) -> "ReplacementModel":
        model = super().from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            **kwargs,
        )
        model._configure_replacement_model(
            transcoders, feature_input_hook, feature_output_hook, scan
        )
        return model

    @classmethod
    def from_self_pretrained_and_transcoders(
        cls,
        cfg: HookedTransformerConfig,
        model_name: str,
        model_path: str,
        transcoders_path: str,
        feature_input_hook: str = "mlp.hook_in",
        feature_output_hook: str = "mlp.hook_out",
        scan: str = None,
        device: str = "cuda",
        **kwargs,
    ) -> "ReplacementModel":
        """
        Adapter for external calling pattern:
        ReplacementModel.from_self_pretrained_and_transcoders(
            cfg=configs,
            model_name=name,
            model_path=cache_root,
            transcoders_path=transcoder,
            device=base_device,
        )
        """
        name_alias = model_name
        hf_name = model_name_func(name_alias)

        # DeepSeek-R1 special case (requires loading hf_model / tokenizer via transformers first)
        if "ds" in name_alias:
            tok = AutoTokenizer.from_pretrained(
                hf_name, use_fast=True, trust_remote_code=True
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
            )

            if "qwen-1.5B" in name_alias:
                backbone_model = "Qwen/Qwen2.5-1.5B"
            elif "llama" in name_alias:
                backbone_model = "meta-llama/Llama-3.1-8B"
            else:
                backbone_model = hf_name

            model = super().from_pretrained(
                backbone_model,
                hf_model=hf_model,
                tokenizer=tok,
                dtype=torch.bfloat16,
                cache_dir=model_path,
                device="cpu",
                move_to_device=False,
                **kwargs,
            )
        else:
            # Load directly from HF to CPU; use set_device_map to assign cards later
            model = super().from_pretrained(
                hf_name,
                dtype=torch.bfloat16,
                cache_dir=model_path,
                device="cpu",
                move_to_device=False,
                **kwargs,
            )

        # cfg contains transcoder configuration (n_layers / d_model etc.)
        cfg.n_layers = model.cfg.n_layers
        transcoder_sets = load_trained_transcoder(
            configs=cfg, transcoder_config_file=transcoders_path
        )

        model._configure_replacement_model(
            transcoder_sets, feature_input_hook, feature_output_hook, scan
        )

        return model

    @classmethod
    def from_self_defined_models(
        cls,
        model_name: str,
        transcoder_path: str,
        device: str,
        config: Optional[HookedTransformerConfig] = None,
        dtype: Optional[torch.dtype] = torch.float32,
        **kwargs,
    ) -> "ReplacementModel":
        transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
            transcoder_path, device=device, dtype=dtype
        )
        return cls.from_pretrained_and_transcoders(
            model_name,
            transcoders,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        transcoder_set: str,
        device: Optional[torch.device] = torch.device("cuda"),
        dtype: Optional[torch.dtype] = torch.float32,
        **kwargs,
    ) -> "ReplacementModel":
        transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
            transcoder_set, device=device, dtype=dtype
        )
        return cls.from_pretrained_and_transcoders(
            model_name,
            transcoders,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    # ========= Multi-GPU sharding =========

    def set_device_map(self, device_map: Dict[int, torch.device]):
        """
        device_map: { layer_idx -> device }, length must equal cfg.n_layers
        """
        assert (
            len(device_map) == self.cfg.n_layers
        ), "device_map need equal to layer num"
        self.device_map = dict(device_map)
        n_layers = self.cfg.n_layers

        first_dev = self.device_map[0]
        last_dev = self.device_map[n_layers - 1]

        # Place embed / pos_embed on the first GPU
        for attr in ["embed", "pos_embed"]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                getattr(self, attr).to(first_dev)
                self.cfg.device = first_dev

        # Place ln_final / unembed on the last GPU
        if hasattr(self, "ln_final") and self.ln_final is not None:
            self.ln_final.to(last_dev)
        if hasattr(self, "unembed") and self.unembed is not None:
            self.unembed.to(last_dev)

        # Place each block on its corresponding GPU
        for i, block in enumerate(self.blocks):
            block.to(self.device_map[i])

        # Distribute transcoders by layer as well, with verification
        if hasattr(self, "transcoders"):
            for i, tr in enumerate(self.transcoders):
                target_device = self.device_map[i]
                tr.to(target_device)

                # Verify all parameters are on the correct device
                for name, param in tr.named_parameters():
                    if param.device != target_device:
                        print(f"[WARN] transcoder[{i}].{name} on {param.device}, expected {target_device}")
                        param.data = param.data.to(target_device)

                # Also check buffers
                for name, buf in tr.named_buffers():
                    if buf.device != target_device:
                        print(f"[WARN] transcoder[{i}] buffer {name} on {buf.device}, expected {target_device}")
                        buf.data = buf.data.to(target_device)

    def forward_sharded(
        self,
        input,
        return_type: Optional[str] = "logits",
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = utils.USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = utils.USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: torch.Tensor | None = None,
        shortformer_pos_embed: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        stop_at_layer: int | None = None,
        past_kv_cache=None,
    ):
        def _device_for_layer(layer_idx: int, fallback: torch.device):
            return (
                self.device_map[layer_idx]
                if self.device_map is not None
                else fallback
            )

        def _module_device(m: nn.Module, default: torch.device):
            try:
                p = next(m.parameters())
                return p.device
            except StopIteration:
                return default
            except Exception:
                return default

        with utils.LocallyOverridenDefaults(
            self,
            prepend_bos=prepend_bos,
            padding_side=padding_side,
        ):
            # 1. embed
            if start_at_layer is None:
                # tokens can be on CPU; input_to_embed will move them to the right GPU
                if (
                    isinstance(input, torch.Tensor)
                    and input.dtype in (torch.int64, torch.int32, torch.int16)
                    and input.device.type != "cpu"
                ):
                    input = input.cpu()
                residual, tokens, shortformer_pos_embed, attention_mask = (
                    self.input_to_embed(
                        input,
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        attention_mask=attention_mask,
                        past_kv_cache=past_kv_cache,
                    )
                )
                start_at_layer = 0
            else:
                assert isinstance(
                    input, torch.Tensor
                ), "When start_at_layer is set, input must be residual Tensor"
                residual = input

            # 2. blocks
            blocks_and_idxs = list(zip(range(self.cfg.n_layers), self.blocks))
            for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:
                dev_i = _device_for_layer(i, residual.device)

                # Move everything for the current layer to dev_i
                if residual.device != dev_i:
                    residual = residual.to(dev_i, non_blocking=True)
                if (
                    shortformer_pos_embed is not None
                    and shortformer_pos_embed.device != dev_i
                ):
                    shortformer_pos_embed = shortformer_pos_embed.to(
                        dev_i, non_blocking=True
                    )
                if (
                    attention_mask is not None
                    and attention_mask.device != dev_i
                ):
                    attention_mask = attention_mask.to(dev_i, non_blocking=True)

                pkv_entry = (
                    past_kv_cache[i] if past_kv_cache is not None else None
                )

                residual = block(
                    residual,
                    past_kv_cache_entry=pkv_entry,
                    shortformer_pos_embed=shortformer_pos_embed,
                    attention_mask=attention_mask,
                )

            if stop_at_layer is not None:
                return residual

            # 3. ln_final
            if self.cfg.normalization_type is not None:
                if hasattr(self, "ln_final") and self.ln_final is not None:
                    dev_ln = _module_device(self.ln_final, residual.device)
                    if residual.device != dev_ln:
                        residual = residual.to(dev_ln, non_blocking=True)
                    residual = self.ln_final(residual)

            if return_type is None:
                return None

            # 4. unembed
            dev_unembed = _module_device(self.unembed, residual.device)
            if residual.device != dev_unembed:
                residual = residual.to(dev_unembed, non_blocking=True)
            logits = self.unembed(residual)

            if (
                getattr(self.cfg, "output_logits_soft_cap", 0.0)
                and self.cfg.output_logits_soft_cap > 0.0
            ):
                softcap = self.cfg.output_logits_soft_cap
                logits = softcap * torch.tanh(logits / softcap)

            if return_type == "logits":
                return logits

            assert tokens is not None, "tokens must be passed if return_type is 'loss' or 'both'"
            loss = self.loss_fn(
                logits, tokens, attention_mask, per_token=loss_per_token
            )

            if return_type == "loss":
                return loss
            elif return_type == "both":
                try:
                    return Output(logits, loss)
                except NameError:
                    return (logits, loss)
            else:
                logging.warning(f"Invalid return_type passed in: {return_type}")
                return None

    # ========= Replacement / Transcoder configuration =========

    def _configure_replacement_model(
        self,
        transcoders: Dict[int, SingleLayerTranscoder],
        feature_input_hook: str,
        feature_output_hook: str,
        scan: Optional[Union[str, List[str]]],
    ):
        # Only change dtype, not force device; device is managed by set_device_map
        for transcoder in transcoders.values():
            transcoder.to(dtype=self.cfg.dtype)

        # n_layers follows the current model
        assert (
            self.cfg.n_layers == len(transcoders)
        ), f"cfg.n_layers={self.cfg.n_layers}, but got {len(transcoders)} transcoders"

        self.add_module(
            "transcoders",
            nn.ModuleList([transcoders[i] for i in range(self.cfg.n_layers)]),
        )

        any_tr = self.transcoders[0]
        self.d_transcoder = any_tr.d_transcoder
        self.feature_input_hook = feature_input_hook
        self.original_feature_output_hook = feature_output_hook
        self.feature_output_hook = feature_output_hook + ".hook_out_grad"
        self.skip_transcoder = any_tr.W_skip is not None
        self.scan = scan

        # Replace mlp / unembed for each layer
        for block in self.blocks:
            block.mlp = ReplacementMLP(block.mlp)

        self.unembed = ReplacementUnembed(self.unembed)

        self._configure_gradient_flow()
        self.setup()

    def _configure_gradient_flow(self):
        for layer, transcoder in enumerate(self.transcoders):
            self._configure_skip_connection(self.blocks[layer], transcoder)

        def stop_gradient(acts, hook):
            return acts.detach()

        for block in self.blocks:
            block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)
            block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)
            block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)
            if hasattr(block, "ln1_post"):
                block.ln1_post.hook_scale.add_hook(
                    stop_gradient, is_permanent=True
                )
            if hasattr(block, "ln2_post"):
                block.ln2_post.hook_scale.add_hook(
                    stop_gradient, is_permanent=True
                )
        self.ln_final.hook_scale.add_hook(stop_gradient, is_permanent=True)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        def enable_gradient(acts, hook):
            acts.requires_grad = True
            return acts

        self.hook_embed.add_hook(enable_gradient, is_permanent=True)

    def _configure_skip_connection(self, block, transcoder):
        cached = {}

        def cache_activations(acts, hook):
            cached["acts"] = acts

        def add_skip_connection(
            acts: torch.Tensor, hook: HookPoint, grad_hook: HookPoint
        ):
            skip_input_activation = cached.pop("acts")
            
            # Ensure skip_input_activation is on the correct device
            if skip_input_activation.device != acts.device:
                skip_input_activation = skip_input_activation.to(acts.device, non_blocking=True)

            # Ensure transcoder parameters are also on the correct device
            transcoder_device = next(transcoder.parameters()).device
            if skip_input_activation.device != transcoder_device:
                skip_input_activation = skip_input_activation.to(transcoder_device, non_blocking=True)

            if transcoder.W_skip is not None:
                skip = transcoder.compute_skip(skip_input_activation)
                # Move skip back to acts' device
                if skip.device != acts.device:
                    skip = skip.to(acts.device, non_blocking=True)
            else:
                skip = torch.zeros_like(acts)
            
            return grad_hook(skip + (acts - skip).detach())

        # feature input hook
        output_hook_parts = self.feature_input_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.add_hook(cache_activations, is_permanent=True)

        # feature output hook
        output_hook_parts = self.original_feature_output_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.hook_out_grad = HookPoint()
        subblock.add_hook(
            partial(add_skip_connection, grad_hook=subblock.hook_out_grad),
            is_permanent=True,
        )

    # ========= Activation cache (note: cross-layer cache stored on CPU) =========

    def _get_activation_caching_hooks(
        self,
        zero_bos: bool = False,
        sparse: bool = False,
        apply_activation_function: bool = True,
    ):
        activation_matrix: List[torch.Tensor] = [None] * self.cfg.n_layers  # type: ignore

        def cache_activations(acts, hook, layer, zero_bos_flag):
            transcoder_acts = (
                self.transcoders[layer]
                .encode(acts, apply_activation_function=apply_activation_function)
                .detach()
                .squeeze(0)
            )
            if zero_bos_flag:
                transcoder_acts[0] = 0

            # Key: move all to CPU to avoid cross-GPU stack operations
            transcoder_acts = transcoder_acts.to("cpu")
            activation_matrix[layer] = (
                transcoder_acts.to_sparse() if sparse else transcoder_acts
            )

        activation_hooks = [
            (
                f"blocks.{layer}.{self.feature_input_hook}",
                partial(
                    cache_activations, layer=layer, zero_bos_flag=zero_bos
                ),
            )
            for layer in range(self.cfg.n_layers)
        ]
        return activation_matrix, activation_hooks

    def get_activations(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = False,
        apply_activation_function: bool = True,
    ):
        activation_cache, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse,
            zero_bos=zero_bos,
            apply_activation_function=apply_activation_function,
        )
        with torch.inference_mode(), self.hooks(activation_hooks):
            logits = self(inputs)
        activation_cache = torch.stack(activation_cache)
        if sparse:
            activation_cache = activation_cache.coalesce()
        return logits, activation_cache

    @contextmanager
    def zero_softcap(self):
        current_softcap = self.cfg.output_logits_soft_cap
        try:
            self.cfg.output_logits_soft_cap = 0.0
            yield
        finally:
            self.cfg.output_logits_soft_cap = current_softcap

    # ========= Attribution (pay close attention to device here) =========

    @torch.no_grad()
    def setup_attribution(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = True,
    ):
        if isinstance(inputs, torch.Tensor):
            tokens = inputs.squeeze(0)
            assert tokens.ndim == 1, "Tokens must be a 1D tensor"
        else:
            assert isinstance(inputs, str), "Inputs must be a string"
            tokenized = self.tokenizer(
                inputs, return_tensors="pt"
            ).input_ids.to("cpu")
            tokens = tokenized.squeeze(0)

        # special tokens for zero_bos
        special_tokens = []
        for special_token in self.tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                special_tokens.extend(special_token)
            else:
                special_tokens.append(special_token)
        special_token_ids = self.tokenizer.convert_tokens_to_ids(
            special_tokens
        )
        zero_bos = zero_bos and tokens[0].item() in special_token_ids

        # cache transcoder activations
        activation_matrix, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse, zero_bos=zero_bos
        )
        mlp_in_cache, mlp_in_caching_hooks, _ = self.get_caching_hooks(
            lambda name: self.feature_input_hook in name
        )

        error_vectors = torch.zeros(
            [self.cfg.n_layers, len(tokens), self.cfg.d_model],
            device="cpu",
            dtype=self.cfg.dtype,
        )
        fvu_values = torch.zeros(
            [self.cfg.n_layers, len(tokens)],
            device="cpu",
            dtype=torch.float32,
        )

        def compute_error_hook(acts, hook, layer):
            try:
                in_hook = f"blocks.{layer}.{self.feature_input_hook}"
                cached_acts = mlp_in_cache[in_hook]
                
                # Get the device of the transcoder
                transcoder_device = next(self.transcoders[layer].parameters()).device

                # Move cached activations to the transcoder's device
                if cached_acts.device != transcoder_device:
                    cached_acts = cached_acts.to(transcoder_device, non_blocking=True)

                # Only call encode and decode, no loss computation
                with torch.no_grad():
                    latents = self.transcoders[layer].encode(cached_acts, apply_activation_function=True)
                    reconstruction = self.transcoders[layer].decode(latents)

                # Move reconstruction to the device of acts
                if reconstruction.device != acts.device:
                    reconstruction = reconstruction.to(acts.device, non_blocking=True)
                
                error = acts - reconstruction

                err_cpu = error.to("cpu", non_blocking=True)
                error_vectors[layer] = err_cpu

                total_variance = (
                    (acts - acts.mean(dim=-2, keepdim=True))
                    .pow(2)
                    .sum(dim=-1)
                )
                fvu_values[layer] = (
                    (error.pow(2).sum(dim=-1) / total_variance).to("cpu", non_blocking=True)
                )
            except Exception as e:
                print(f"[ERROR in compute_error_hook layer {layer}] {type(e).__name__}: {e}")
                print(f"  acts device: {acts.device}")
                print(f"  cached_acts device: {mlp_in_cache[in_hook].device}")
                print(f"  transcoder device: {next(self.transcoders[layer].parameters()).device}")
                raise

        error_hooks = [
            (
                f"blocks.{layer}.{self.feature_output_hook}",
                partial(compute_error_hook, layer=layer),
            )
            for layer in range(self.cfg.n_layers)
        ]

        try:
            logits = self.run_with_hooks(
                tokens,
                fwd_hooks=activation_hooks + mlp_in_caching_hooks + error_hooks,
            )
        except Exception as e:
            print(f"[ERROR in setup_attribution] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

        if zero_bos:
            error_vectors[:, 0] = 0

        activation_matrix = torch.stack(activation_matrix)
        if sparse:
            activation_matrix = activation_matrix.coalesce()

        token_vectors = self.W_E[tokens].detach().to("cpu")
        return logits, activation_matrix, error_vectors, token_vectors

    # ========= Intervention (can be ignored if not currently needed) =========

    def setup_intervention_with_freeze(
        self, inputs: Union[str, torch.Tensor], direct_effects: bool = False
    ) -> List[Tuple[str, Callable]]:
        if direct_effects:
            hookpoints_to_freeze = [
                "hook_pattern",
                "hook_scale",
                self.feature_output_hook,
            ]
            if self.skip_transcoder:
                hookpoints_to_freeze.append(self.feature_input_hook)
        else:
            hookpoints_to_freeze = ["hook_pattern"]

        freeze_cache, cache_hooks, _ = self.get_caching_hooks(
            names_filter=lambda name: any(
                hookpoint in name for hookpoint in hookpoints_to_freeze
            )
        )
        self.run_with_hooks(inputs, fwd_hooks=cache_hooks)

        def freeze_hook(activations, hook):
            cached_values = freeze_cache[hook.name]

            if "hook_pattern" in hook.name and activations.shape[2:] != cached_values.shape[2:]:
                new_activations = activations.clone()
                new_activations[:, :, : cached_values.shape[2], : cached_values.shape[3]] = cached_values
                return new_activations

            elif (
                "hook_scale" in hook.name or self.feature_output_hook in hook.name
            ) and activations.shape[1] != cached_values.shape[1]:
                new_activations = activations.clone()
                new_activations[:, : cached_values.shape[1]] = cached_values
                return new_activations

            assert (
                activations.shape == cached_values.shape
            ), f"Shape mismatch at {hook.name}"
            return cached_values

        fwd_hooks = [
            (hookpoint, freeze_hook)
            for hookpoint in freeze_cache.keys()
            if self.feature_input_hook not in hookpoint
        ]

        if not direct_effects:
            return fwd_hooks

        if self.skip_transcoder:
            skip_diffs: Dict[int, torch.Tensor] = {}

            def diff_hook(activations, hook, layer: int):
                frozen_skip = self.transcoders[layer].compute_skip(
                    freeze_cache[hook.name]
                )
                normal_skip = self.transcoders[layer].compute_skip(
                    activations
                )
                skip_diffs[layer] = normal_skip - frozen_skip

            def add_diff_hook(activations, hook, layer: int):
                return activations + skip_diffs[layer]

            fwd_hooks += [
                (
                    f"blocks.{layer}.{self.feature_input_hook}",
                    partial(diff_hook, layer=layer),
                )
                for layer in range(self.cfg.n_layers)
            ]
            fwd_hooks += [
                (
                    f"blocks.{layer}.{self.feature_output_hook}",
                    partial(add_diff_hook, layer=layer),
                )
                for layer in range(self.cfg.n_layers)
            ]

        return fwd_hooks

    def _get_feature_intervention_hooks(
        self,
        inputs: Union[str, torch.Tensor],
        interventions: List[
            Tuple[int, Union[int, slice, torch.Tensor], int, Union[int, torch.Tensor]]
        ],
        direct_effects: bool = False,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
    ):
        interventions_by_layer: Dict[int, List] = defaultdict(list)
        for layer, pos, feature_idx, value in interventions:
            interventions_by_layer[layer].append((pos, feature_idx, value))

        # To avoid cross-GPU stack(), activation_cache is stored per layer on its corresponding device;
        # then moved to CPU and stacked inside feature_intervention.
        activation_cache: List[torch.Tensor] = [None] * self.cfg.n_layers  # type: ignore

        def cache_for_intervention(acts, hook, layer: int):
            transcoder_acts = (
                self.transcoders[layer]
                .encode(acts, apply_activation_function=apply_activation_function)
                .detach()
                .squeeze(0)
            )
            activation_cache[layer] = transcoder_acts  # keep on the GPU of that layer

        activation_hooks = [
            (
                f"blocks.{layer}.{self.feature_input_hook}",
                partial(cache_for_intervention, layer=layer),
            )
            for layer in range(self.cfg.n_layers)
        ]

        def intervention_hook(
            activations, hook, layer: int, layer_interventions
        ):
            transcoder_activations = activation_cache[layer]
            if not apply_activation_function:
                transcoder_activations = self.transcoders[layer].activation_function(
                    transcoder_activations.unsqueeze(0)
                ).squeeze(0)
            transcoder_output = self.transcoders[layer].decode(
                transcoder_activations
            )
            for pos, feature_idx, value in layer_interventions:
                transcoder_activations[pos, feature_idx] = value
            new_transcoder_output = self.transcoders[layer].decode(
                transcoder_activations
            )
            steering_vector = new_transcoder_output - transcoder_output
            return activations + steering_vector

        intervention_hooks = [
            (
                f"blocks.{layer}.{self.feature_output_hook}",
                partial(
                    intervention_hook,
                    layer=layer,
                    layer_interventions=layer_interventions,
                ),
            )
            for layer, layer_interventions in interventions_by_layer.items()
        ]

        all_hooks: List[Tuple[str, Callable]] = []
        if freeze_attention or direct_effects:
            all_hooks += self.setup_intervention_with_freeze(
                inputs, direct_effects=direct_effects
            )
        all_hooks += activation_hooks + intervention_hooks

        return all_hooks, activation_cache

    @torch.no_grad()
    def feature_intervention(
        self,
        inputs: Union[str, torch.Tensor],
        interventions: List[
            Tuple[int, Union[int, slice, torch.Tensor], int, Union[int, torch.Tensor]]
        ],
        direct_effects: bool = False,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
    ):
        hooks, activation_cache = self._get_feature_intervention_hooks(
            inputs,
            interventions,
            direct_effects=direct_effects,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
        )

        with self.hooks(hooks):
            logits = self(inputs)

        # Move all to CPU before stacking to avoid cross-GPU issues
        activation_cache_cpu = [
            ac.to("cpu") for ac in activation_cache
        ]
        activation_cache_cpu = torch.stack(activation_cache_cpu)

        return logits, activation_cache_cpu
