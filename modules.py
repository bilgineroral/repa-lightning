from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from config import PretrainConfig
from rope import RoPE, apply_rotary_pos_emb, prepend_prefix_tokens

class PatchEmbed(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.hidden_size, 
            kernel_size=config.patch_size, stride=config.patch_size
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """ img: [B,C_in,H,W] """
        return self.patch_embed(img).flatten(2).transpose(1, 2) # [B,N,D]

class DropPath(nn.Module):
    def __init__(self, drop_p: Optional[float] = None) -> None:
        super().__init__()
        self.drop_p = drop_p if drop_p is not None else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_p == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        if init_value is not None and init_value > 0:
            self.lambda1 = nn.Parameter(init_value * torch.ones(dim))
        else:
            self.lambda1 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.lambda1 is None else (x * self.lambda1)

class SelfAttention(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dim = config.hidden_size

        self.attn_drop_p = config.attn_drop

        self.q_proj = nn.Linear(config.hidden_size, self.dim, bias=config.qkv_bias)
        self.kv_proj = nn.Linear(config.hidden_size, 2*self.dim, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.drop = nn.Dropout(config.drop)

        if config.qk_norm:
            self.q_norm = nn.LayerNorm(
                self.head_dim, eps=config.layer_norm_eps,
                elementwise_affine=config.qk_norm_affine, bias=config.qk_norm_bias
            )
            self.k_norm = nn.LayerNorm(
                self.head_dim, eps=config.layer_norm_eps,
                elementwise_affine=config.qk_norm_affine, bias=config.qk_norm_bias
            )
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x_q: torch.Tensor, 
        x_kv: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, # (cos, sin)
        attn_mask: Optional[torch.Tensor] = None,
        kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """ x_q: [B, Nq, D], x_kv: [B, Nk, D] or None, attn_mask: [Nq, Nk] bool, False = masked"""
        if x_kv is None:
            x_kv = x_q

        B, Nq, D = x_q.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x_q)
        q = q.view(B, Nq, H, d).transpose(1, 2)  # [B,H,Nq,d]
        q = self.q_norm(q)

        if kv is None:
            Nk = x_kv.shape[1]
            k, v = self.kv_proj(x_kv).split(D, dim=-1)
            k = k.view(B, Nk, H, d).transpose(1, 2)  # [B,H,Nk,d]
            v = v.view(B, Nk, H, d).transpose(1, 2)  # [B,H,Nk,d]
            k = self.k_norm(k)

            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            k, v = kv
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, _ = apply_rotary_pos_emb(q, None, cos, sin)

        mask = None if attn_mask is None else attn_mask[None, None, :, :] # [1,1,Nq,Nk]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_drop_p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)
        out = self.proj(out)
        out = self.drop(out)
        
        if return_kv:
            return out, k, v
        return out

class MLP(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.drop)
        
        self.use_gated_mlp = config.use_gated_mlp
        if self.use_gated_mlp:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)

        if config.hidden_act == "gelu":
            self.act_fn = F.gelu
        elif config.hidden_act == "swish":
            self.act_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.up_proj(x)
        x = self.act_fn(self.gate_proj(x)) * x_up if self.use_gated_mlp else self.act_fn(x_up)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x

class REPALayer(nn.Module):
    def __init__(self, config: PretrainConfig, drop_path_rate: float = 0.0):
        super().__init__()
        self.attention = SelfAttention(config)
        self.mlp = MLP(config)

        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.ls1_h = LayerScale(config.hidden_size, config.layerscale_value)
        self.ls2_h = LayerScale(config.hidden_size, config.layerscale_value)
        if config.use_query_stream:
            self.ls1_g = LayerScale(config.hidden_size, config.layerscale_value)
            self.ls2_g = LayerScale(config.hidden_size, config.layerscale_value)

        self.ln1_h = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2_h = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.use_query_stream:
            self.ln1_g = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ln2_g = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        h: torch.Tensor,  # content stream [B,1+N,D] (CLS + patches)
        g: Optional[torch.Tensor],  # query stream [B,N,D], None without the query stream
        position_embeddings_h: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_g: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask_h: Optional[torch.Tensor] = None,
        attn_mask_g: Optional[torch.Tensor] = None,
    ):
        """ h: [B,1+N,D], g: [B,N,D] or None """
        h_in, g_in = h, g
        h_norm = self.ln1_h(h_in)
        g_norm = self.ln1_g(g_in) if g is not None else None # before the content attention: keeps the op order

        # self-attention
        h_attn, k, v = self.attention(
            x_q=h_norm, x_kv=h_norm,
            position_embeddings=position_embeddings_h,
            attn_mask=attn_mask_h,
            return_kv=True,
        )
        h = h_in + self.drop_path1(self.ls1_h(h_attn))

        if g is None: # content stream only
            h = h + self.drop_path2(self.ls2_h(self.mlp(self.ln2_h(h))))
            return h, None

        g_attn = self.attention(
            x_q=g_norm, x_kv=h_norm,
            position_embeddings=position_embeddings_g,
            attn_mask=attn_mask_g,
            kv=(k, v),
        )
        g = g_in + self.drop_path1(self.ls1_g(g_attn))

        # mlp
        h_ff = self.mlp(self.ln2_h(h))
        h = h + self.drop_path2(self.ls2_h(h_ff))

        g_ff = self.mlp(self.ln2_g(g))
        g = g + self.drop_path2(self.ls2_g(g_ff))

        return h, g
    
class REPAEncoder(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        dpr = np.linspace(0, config.drop_path, config.num_hidden_layers).tolist()
        self.layers = nn.ModuleList([REPALayer(config, drop_path_rate=dpr[i]) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        h: torch.Tensor, g: Optional[torch.Tensor],
        position_embeddings_h: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_g: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask_h: Optional[torch.Tensor] = None,
        attn_mask_g: Optional[torch.Tensor] = None,
    ):
        for layer in self.layers:
            h, g = layer(h, g, position_embeddings_h, position_embeddings_g, attn_mask_h, attn_mask_g)
        return h, g

class REPA(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbed(config)
        self.dropout = nn.Dropout(config.drop)

        num_patches = (config.image_size // config.patch_size) ** 2
        if config.use_rope:
            self.rope_embed = RoPE(config) # spatial position
            self.pos_embed = None
        else:
            self.rope_embed = None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
            nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=config.initializer_range)

        self.encoder = REPAEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # final norm, as NEPA

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=config.initializer_range)
        if config.use_query_stream:
            self.w = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) # query stream input
            nn.init.trunc_normal_(self.w, mean=0.0, std=config.initializer_range)
        else:
            self.w = None

        self.apply(self._init_weights)

    def _make_attn_masks(self, N: int, device: torch.device):
        # key masks over [CLS, x_1..x_N]: the content stream is causal, the query for target t sees CLS and x_<t
        attn_mask_h = torch.tril(torch.ones(1 + N, 1 + N, dtype=torch.bool, device=device), diagonal=0)
        attn_mask_g = torch.tril(torch.ones(N, 1 + N, dtype=torch.bool, device=device), diagonal=0)
        return attn_mask_h, attn_mask_g

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm, LayerScale]):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, LayerScale):
            if module.lambda1 is not None:
                module.lambda1.data.fill_(self.config.layerscale_value)

    def forward(self, img: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """ img: [B,C_in,H,W], position_ids: [B,N] or None """
        patches = self.patch_embed(img)
        B, N, D = patches.shape

        if position_ids is None:
            tgt = patches
        else:
            tgt = patches.gather(1, position_ids[:, :, None].expand(B, N, D))
        cls = self.cls_token.expand(B, -1, -1)

        # Position embeddings: RoPE applied in attention (CLS unrotated), APE added to patch tokens
        if self.rope_embed is not None:
            position_embeds_g = self.rope_embed(img, position_ids=position_ids)
            position_embeds_h = prepend_prefix_tokens(position_embeds_g)
            h0 = torch.cat([cls, tgt], dim=1)
            g0 = self.w.expand(B, N, D) if self.w is not None else None
        else:
            # APE: gather in permuted order and add to both streams (not target)
            if position_ids is None:
                pos = self.pos_embed.expand(B, -1, -1)
            else:
                pos = self.pos_embed.expand(B, -1, -1).gather(1, position_ids[:, :, None].expand(B, N, D))
            h0 = torch.cat([cls, tgt + pos], dim=1)
            g0 = self.w.expand(B, N, D) + pos if self.w is not None else None
            position_embeds_h = position_embeds_g = None

        h0 = self.dropout(h0)  # content stream sees the CLS token and patch embeddings

        attn_mask_h, attn_mask_g = self._make_attn_masks(N, tgt.device)

        h, g = self.encoder(
            h0, g0,
            position_embeddings_h=position_embeds_h,
            position_embeddings_g=position_embeds_g,
            attn_mask_h=attn_mask_h,
            attn_mask_g=attn_mask_g,
        )

        if g is None:
            # NEPA: the content stream at CLS, x_1..x_{N-1} predicts x_1..x_N
            return tgt, self.norm(h[:, :-1])
        return tgt, self.norm(g) + h.sum() * 0.0

class SingleStreamREPALayer(nn.Module):
    """ Content stream of a REPALayer, run with bidirectional attention """
    def __init__(self, layer: REPALayer, drop_path: float = 0.0):
        super().__init__()
        self.attention = layer.attention
        self.mlp = layer.mlp
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls1 = layer.ls1_h
        self.ls2 = layer.ls2_h

        self.ln1 = layer.ln1_h
        self.ln2 = layer.ln2_h

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        x = x + self.drop_path(self.ls1(self.attention(self.ln1(x), position_embeddings=position_embeddings)))
        x = x + self.drop_path(self.ls2(self.mlp(self.ln2(x))))
        return x

class SingleStreamREPAEncoder(nn.Module):
    def __init__(self, encoder: REPAEncoder, drop_path: float = 0.0):
        super().__init__()
        dpr = np.linspace(0, drop_path, len(encoder.layers)).tolist()
        self.layers = nn.ModuleList([
            SingleStreamREPALayer(layer, drop_path=dpr[i]) for i, layer in enumerate(encoder.layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        for layer in self.layers:
            x = layer(x, position_embeddings)
        return x

def add_qk_norm_affine(layers: nn.ModuleList, config: PretrainConfig) -> None:
    """ Learnable QK-norm scales (initialized to 1) on top of pretrained non-affine QK-norms, as NEPA's finetuning """
    if not config.qk_norm or config.qk_norm_affine:
        return
    for layer in layers:
        attn = layer.attention
        attn.q_norm = nn.LayerNorm(attn.head_dim, eps=config.layer_norm_eps, bias=config.qk_norm_bias)
        attn.k_norm = nn.LayerNorm(attn.head_dim, eps=config.layer_norm_eps, bias=config.qk_norm_bias)

if __name__ == "__main__":
    c = PretrainConfig(
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=2048
    )
    model = REPA(c)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params/1e6:.2f}M")
