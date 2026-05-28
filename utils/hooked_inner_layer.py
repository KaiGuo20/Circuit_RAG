from torch import Tensor, nn
import torch.nn.functional as F
import torch
from typing import Callable, Optional

import torch.nn as nn

class LinkPrediction(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.d_in = configs['d_in']
        self.d_trans_hid = configs['d_trans_hid']
        self.n_layer = configs['n_layers']
        
        self.lin = torch.nn.Linear(self.d_trans_hid, 1)
    def forward(self, z, edge_index):
        src, dst = edge_index
        h = torch.cat([z[src], z[dst]], dim=1)
        return torch.sigmoid(self.lin(h)).squeeze()
    
    

class SingleLayerTranscoder(nn.Module):
    # d_model: int
    # d_transcoder: int
    # layer_idx: int
    # W_enc: nn.Parameter
    # W_dec: nn.Parameter
    # b_enc: nn.Parameter
    # b_dec: nn.Parameter
    # W_skip: Optional[nn.Parameter]
    # activation_function: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        configs,
        skip_connection = False,
        device = 'cuda'
    ):
        """Single layer transcoder implementation, adapted from the JumpReLUSAE implementation here:
        https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp

        Args:
            d_model (int): The dimension of the model.
            d_transcoder (int): The dimension of the transcoder.
            activation_function (nn.Module): The activation function.
            layer_idx (int): The layer index.
            skip_connection (bool): Whether there is a skip connection,
                as in https://arxiv.org/abs/2501.18823
        """
        super().__init__()

        self.d_model = configs['d_model']# d_model
        self.d_transcoder = configs['d_trans_hid']# d_transcoder
        self.dtype = torch.float32 # configs.dtype
        self.device = device

        # self.W_enc = nn.Parameter(torch.zeros(self.d_model, self.d_transcoder))
        # self.W_dec = nn.Parameter(torch.zeros(self.d_transcoder, self.d_model))
        # self.b_enc = nn.Parameter(torch.zeros(self.d_transcoder))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))
        
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_model, self.d_transcoder, dtype=self.dtype, device=self.device)
            )   
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_transcoder, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_transcoder, self.d_model, dtype=self.dtype, device=self.device)
            )
        )
        
        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_model, dtype=self.dtype, device=self.device)
        )
        

        # self.b_dec_out = None
        # self.b_dec_out = nn.Parameter(
        #     torch.zeros(self.d_model, dtype=self.dtype, device=self.device)
        # )


        if skip_connection:
            self.W_skip = nn.Parameter(torch.zeros(self.d_model, self.d_model))
        else:
            self.W_skip = None

        # self.activation_function = activation_function

    def encode(self, input_acts, return_pres = False, apply_activation_function: bool = True):
        # print('input_acts', input_acts.shape, 'W_enc', self.W_enc.shape, 'b_enc', self.b_enc.shape)
        pre_acts = input_acts.to(self.W_enc.dtype) @ self.W_enc + self.b_enc
        if not apply_activation_function:
            return pre_acts
        acts = F.relu(pre_acts)
        
        if return_pres:
            return acts, pre_acts
        else:
            return acts

    def decode(self, acts):
        if acts.is_sparse:
            return (
                torch.bmm(acts, self.W_dec.unsqueeze(0).expand(acts.size(0), *self.W_dec.size()))
                + self.b_dec
            )
        else:
            # print('acts', acts.shape, 'W_dec', self.W_dec.shape, 'b_dec', self.b_dec.shape)
            return acts @ self.W_dec + self.b_dec
    
    def forward(self, activation):
        self.transcoder_acts, hidden_pre = self.encode(activation, return_pres = True)
        decoded = self.decode(self.transcoder_acts)
        self.sparsity_loss = torch.abs(self.transcoder_acts).sum(dim=-1).mean(dim=(0,))
        return decoded
        
    def __call__(self, activation, hook):
        return self.forward(activation)
        
    # def forward(self, input_acts, dead_neuron_mask = None, mse_target = None):
    #     # print('input_acts', input_acts)
    #     transcoder_acts, hidden_pre = self.encode(input_acts, return_pres = True)
    #     # print('transcoder_acts', transcoder_acts)
    #     # print('hidden_pre', hidden_pre)
    #     decoded = self.decode(transcoder_acts)
    #     # print('decoded',decoded)
        
    #     if self.training == False:
    #         decoded = decoded.detach()
    #         decoded.requires_grad = True

    #     if self.W_skip is not None:
    #         skip = self.compute_skip(input_acts)
    #         decoded = decoded + skip
            
    #     if mse_target is None:
    #         mse_loss = (torch.pow((decoded-input_acts.float()), 2) / (input_acts**2).sum(dim=-1, keepdim=True).sqrt())
    #     else:
    #         mse_loss = (torch.pow((decoded-mse_target.float()), 2) / (mse_target**2).sum(dim=-1, keepdim=True).sqrt())
    #     mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)
            
    #     if self.training and dead_neuron_mask.sum() > 0:
    #         assert dead_neuron_mask is not None 
            
    #         # ghost protocol
            
    #         # 1.
    #         residual = input_acts - decoded
            
    #         l2_norm_residual = torch.norm(residual, dim=-1)
    #         feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
    #         ghost_out =  feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask,:]
    #         l2_norm_ghost_out = torch.norm(ghost_out, dim = -1)
    #         norm_scaling_factor = l2_norm_residual / (1e-6+ l2_norm_ghost_out* 2)
    #         ghost_out = ghost_out*norm_scaling_factor[:, None].detach()
            
    #         # 3. 
    #         mse_loss_ghost_resid = (
    #             torch.pow((ghost_out - residual.detach().float()), 2) / (residual.detach()**2).sum(dim=-1, keepdim=True).sqrt()
    #         )
    #         mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
    #         mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid
        
    #     mse_loss_ghost_resid = mse_loss_ghost_resid.mean()
    #     mse_loss = mse_loss.mean()
    #     # print(feature_acts.shape)
    #     sparsity = torch.abs(transcoder_acts).sum(dim=-1).mean(dim=(0,))
    #     l1_loss = self.l1_coefficient * sparsity
    #     loss = mse_loss + l1_loss + mse_loss_ghost_resid
        

    #     return decoded, transcoder_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid



class JointTransformerWithHooks(nn.Module):
    def __init__(self, transformer: nn.Module, transcoder: nn.Module, device):
        super().__init__()
        self.transformer = transformer.to(device)  # HookedTransformer
        self.transcoder = transcoder.to(device)    # MultiLayerTranscoder

        # Automatically register hooks
        self.transcoder.register_hooks(self.transformer)

    def forward(self, tokens, edges = None, training = True):
        """
        tokens: [batch, seq_len] input token IDs
        """
        output = self.transformer(tokens)  # normal forward
        if training == True:
            loss = self.transcoder.total_loss(edges)  # per-layer hook loss
            return output, loss
        else: return output

class MultiLayerTranscoder(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.n_layers = configs['n_layers']
        self.lin = torch.nn.Linear(configs['d_trans_hid'], 1)
        self.transcoders = nn.ModuleList([
            SingleLayerTranscoder(configs, device=device) for _ in range(configs['n_layers'])
        ])

    def register_hooks(self, model):
        for i, transcoder in enumerate(self.transcoders):
            model.add_hook(f'blocks.{i}.ln2.hook_normalized', transcoder)
        return model
    
    def total_loss(self, edge_index_list):
        out1_list = []
        out2_list = []
        scores = []
        for i in range(self.n_layers - 1):
            out1 = self.transcoders[i].transcoder_acts
            out2 = self.transcoders[i + 1].transcoder_acts
            score = torch.sigmoid(torch.matmul(out1, out2.transpose(1,2)))
            scores.append(score)
        scores = torch.stack(scores)
        scores, _ = torch.max(scores, dim=0)
        # print(scores.shape, edge_index_list.shape)
        connection_loss = F.binary_cross_entropy_with_logits(scores, edge_index_list)
        # for edge_index in edge_index_list:
        sparsity_losses = [mod.sparsity_loss for mod in self.transcoders if mod.sparsity_loss is not None]
        # return torch.stack(losses).mean() if losses else None
        sparsity_losses = torch.cat(sparsity_losses, dim=0)
        # print(connection_loss, torch.sum(sparsity_losses))
        sparsity_losses = torch.sum(sparsity_losses)
        return 0.05*connection_loss + 0.0005*sparsity_losses