import torch
from torch import nn
from typing import Optional, Union
import torch.nn.functional as F


class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.xavier_normal_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor, lora_scale) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))

        if isinstance(lora_scale, torch.Tensor):
            mid = torch.diag_embed(lora_scale)
            down_hidden_states = down_hidden_states @ mid

        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        if isinstance(lora_scale, float):
            return lora_scale * up_hidden_states.to(orig_dtype)
        return up_hidden_states.to(orig_dtype)
    
    
class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):

        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer
        self._lora_scale = 1.0

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer
        
    def set_scale_tensor(self, scale: torch.Tensor):
        self._lora_scale = scale

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.dtype, fused_weight.device

        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device).float()

        unfused_weight = fused_weight.float() - (self._lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lora_layer is None:
            out = super(LoRACompatibleLinear, self).forward(hidden_states)
            return out
        else:
            out = super(LoRACompatibleLinear, self).forward(hidden_states) + self.lora_layer(hidden_states, self._lora_scale)
            return out
        
        
class LoRAMultiheadAttention(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rank: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rank = rank

        self.in_proj_down = nn.Linear(3*embed_dim, rank, bias=False, device=device, dtype=dtype)
        self.in_proj_up = nn.Linear(rank, embed_dim, bias=False, device=device, dtype=dtype)
        self.out_proj_down = nn.Linear(embed_dim, rank, bias=False, device=device, dtype=dtype)
        self.out_proj_up = nn.Linear(rank, embed_dim, bias=False, device=device, dtype=dtype)

        # Initialize weights
        nn.init.xavier_normal_(self.in_proj_down.weight)
        nn.init.zeros_(self.in_proj_up.weight)

        nn.init.xavier_normal_(self.out_proj_down.weight)
        nn.init.zeros_(self.out_proj_up.weight)

        
class LoRACompatibleMultiheadAttention(nn.MultiheadAttention):

    def __init__(self, embed_dim: int, num_heads: int, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):
        
        super().__init__(embed_dim, num_heads, *args, **kwargs)
        self.lora_layer = lora_layer
        self._lora_scale = 1.0
        self.dropout = 0.0
        self.add_zero_attn = False
        self.training = True

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer
        
    def set_scale_tensor(self, scale: torch.Tensor):
        self._lora_scale = scale

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask= None,
        need_weights=True,
        attn_mask= None,
        average_attn_weights: bool = True,
        is_causal: bool = False,) -> torch.Tensor:
        # Standard MultiheadAttention forward pass
        orig_dtype = query.dtype
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.T

        if self.lora_layer is not None:
            in_proj_orig = self.in_proj_weight
            in_proj_up = self.lora_layer.in_proj_up.weight
            in_proj_down = self.lora_layer.in_proj_down.weight
            
            out_proj_orig = self.out_proj.weight
            out_proj_up = self.lora_layer.out_proj_up.weight
            out_proj_down = self.lora_layer.out_proj_down.weight
                
            mid = torch.diag_embed(self._lora_scale).squeeze()
            #print(in_proj_up.shape, mid.shape, in_proj_down.shape, in_proj_orig.shape)
            #print(in_proj_orig.shape, out_proj_orig.shape)

            fused_weight_in = in_proj_orig + (in_proj_up @ mid @ in_proj_down).T
            fused_weight_out = out_proj_orig + (out_proj_up @ mid @ out_proj_down).T
            
        else:
            fused_weight_in = self.in_proj_weight
            fused_weight_out = self.out_proj.weight
        
        # Compute the attention scores
        attn_output, attn_output_weights = F.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    fused_weight_in,
                    self.in_proj_bias,
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout,
                    fused_weight_out,
                    self.out_proj.bias,
                    self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                    attn_mask = attn_mask,
                    use_separate_proj_weight=False,
                    q_proj_weight=self.q_proj_weight,
                    k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal,
                )

        return attn_output.to(orig_dtype), attn_output_weights
        
 
 
 
 
        
def LoRALinearLayerforward(self, hidden_states: torch.Tensor):
    orig_dtype = hidden_states.dtype
    dtype = self.down.weight.dtype

    down_hidden_states = self.down(hidden_states.to(dtype))

    if isinstance(self._lora_scale, torch.Tensor):
        mid = torch.diag_embed(self._lora_scale)
        down_hidden_states = down_hidden_states @ mid

    up_hidden_states = self.up(down_hidden_states)

    if self.network_alpha is not None:
        up_hidden_states *= self.network_alpha / self.rank

    if isinstance(self._lora_scale, float):
        return self._lora_scale * up_hidden_states.to(orig_dtype)
    return up_hidden_states.to(orig_dtype)


def LoRACompatibleLinearforward(self, hidden_states: torch.Tensor):
    if self.lora_layer is None:
        print('no lora linear')
        out = super(LoRACompatibleLinear, self).forward(hidden_states)
        return out
    else:
        out = super(LoRACompatibleLinear, self).forward(hidden_states) + self.lora_layer(hidden_states, self._lora_scale)
        print('lora forward')
        return out


def LoRAMultiHeadAttentionforward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:

    orig_dtype = query.dtype
    dtype = self.in_proj_weight.dtype

    if self.lora_layer is not None:
        in_proj_orig = self.in_proj_weight.data.float()
        in_proj_up = self.lora_layer.in_proj_up.weight.data.float()
        in_proj_down = self.lora_layer.in_proj_down.weight.data.float()
        
        out_proj_orig = self.out_proj.weight.data.float()
        out_proj_up = self.lora_layer.out_proj_up.weight.data.float()
        out_proj_down = self.lora_layer.out_proj_down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            in_proj_up = in_proj_up * self.lora_layer.network_alpha / self.lora_layer.rank
            out_proj_up = out_proj_up * self.lora_layer.network_alpha / self.lora_layer.rank
            
        if isinstance(self._lora_scale, torch.Tensor):
            mid = torch.diag_embed(self._lora_scale)
            in_proj_up = in_proj_up @ mid
            out_proj_up = out_proj_up @ mid
            print('lora attention')

        fused_weight_in = in_proj_orig + (torch.bmm(in_proj_up[None, :], in_proj_down[None, :])[0])
        fused_weight_out = out_proj_orig + (torch.bmm(out_proj_up[None, :], out_proj_down[None, :])[0])
        
    else:
        fused_weight_in = self.in_proj_weight
        fused_weight_out = self.out_proj.weight
    
    # Compute the attention scores
    attn_output, _ = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                fused_weight_in,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                fused_weight_out,
                self.out_proj.bias,
                self.training,
                need_weights=False,
                use_separate_proj_weight=False,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )



    return attn_output.to(orig_dtype)





